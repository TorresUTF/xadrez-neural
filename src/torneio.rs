// =====================================================================
// TORNEIO — métodos de escolher o lance, uns contra os outros
// =====================================================================
//
// Um "jogador" recebe a posição e devolve um lance. Todos usam a mesma
// rede (ou o material); o que muda é COMO a avaliação vira lance:
//
//   ab:3        alfa-beta com profundidade fixa 3
//   ab:200ms    alfa-beta com aprofundamento iterativo, ~200 ms por lance
//   mat:3       alfa-beta com o material no lugar da rede (linha de base)
//   cert:200ms  busca guiada pela certeza (certeza.rs), 200 ms por lance
//   cert:200ms:0.3   ... com bônus de incerteza 0,3 (padrão 0,5)
//
// Para comparar métodos de forma justa, o certo é o MESMO TEMPO por lance
// dos dois lados: vence quem usa melhor o mesmo segundo. Profundidade
// fixa serve para outra pergunta, "quanto a busca acrescenta" (ex.: ab:1
// contra ab:3, o teste do Capablanca).
//
// Cada abertura sorteada é jogada duas vezes, trocando as cores. Os
// finais com valor exato (gabarito de 3 peças, ou Syzygy) encerram a
// partida com esse valor.
//
// Diferença de Elo a partir da fração s de pontos:
//      ΔElo = -400 · log10(1/s - 1)
// A margem de erro vem da variância dos pontos por partida (95%).
// =====================================================================

use crate::autojogo::{abertura, fim, hash, threads, trocar_valor, Config, Verdade, CONFIG_PADRAO};
use crate::avaliacao::Material;
use crate::busca::Busca;
use crate::certeza::{self, Busca2};
use crate::finais::Valor;
use crate::rede::{Aleatorio, Rede};
use shakmaty::{zobrist::Zobrist64, Chess, Move, Position};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug)]
pub enum Limite {
    Profundidade(u32),
    Tempo(Duration),
}

#[derive(Clone, Copy, Debug)]
pub enum Metodo {
    AlfaBeta(Limite),
    Material(Limite),
    Certeza { tempo: Duration, bonus: f32 },
}

pub struct Jogador<'a> {
    pub nome: String,
    pub metodo: Metodo,
    pub rede: &'a Rede,
}

fn limite(texto: &str) -> Option<Limite> {
    match texto.strip_suffix("ms") {
        Some(ms) => ms.parse().ok().map(|ms| Limite::Tempo(Duration::from_millis(ms))),
        None => texto.parse().ok().filter(|&p| p >= 1).map(Limite::Profundidade),
    }
}

// "ab:3", "ab:200ms", "mat:3", "cert:200ms", "cert:200ms:0.3"
pub fn ler(texto: &str) -> Option<Metodo> {
    let partes: Vec<&str> = texto.split(':').collect();
    match partes.as_slice() {
        ["ab", l] => limite(l).map(Metodo::AlfaBeta),
        ["mat", l] => limite(l).map(Metodo::Material),
        ["cert", t] | ["cert", t, _] => {
            let Some(Limite::Tempo(tempo)) = limite(t) else { return None };
            let bonus = match partes.get(2) {
                Some(b) => b.parse().ok()?,
                None => 0.5,
            };
            Some(Metodo::Certeza { tempo, bonus })
        }
        _ => None,
    }
}

impl Jogador<'_> {
    fn lance(&self, pos: &Chess, historico: &[Zobrist64], verdade: &Verdade) -> Move {
        let r = match self.metodo {
            Metodo::AlfaBeta(Limite::Profundidade(p)) => Busca::nova(self.rede, historico.to_vec()).melhor_lance(pos, p),
            Metodo::AlfaBeta(Limite::Tempo(t)) => Busca::nova(self.rede, historico.to_vec()).por_tempo(pos, t),
            Metodo::Material(Limite::Profundidade(p)) => Busca::nova(&Material, historico.to_vec()).melhor_lance(pos, p),
            Metodo::Material(Limite::Tempo(t)) => Busca::nova(&Material, historico.to_vec()).por_tempo(pos, t),
            Metodo::Certeza { tempo, bonus } => {
                let cfg = certeza::Config { tempo, bonus };
                return Busca2::nova(self.rede, verdade).melhor_lance(pos, historico, &cfg).lance.unwrap();
            }
        };
        r.lance.unwrap()
    }
}

// Uma partida a partir de uma abertura; devolve os pontos das brancas.
// Também soma o tempo gasto por cada lado (para conferir a justiça).
fn partida(brancas: &Jogador, pretas: &Jogador, ab: &[Move], verdade: &Verdade, tempos: &mut [Duration; 2]) -> f32 {
    let cfg = Config { max_meios_lances: 300, ..CONFIG_PADRAO };
    let mut pos = Chess::default();
    let mut historico = Vec::new();
    let mut n = 0;
    loop {
        if let Some(v) = fim(&pos, &historico, verdade, n, &cfg) {
            let para_brancas = if pos.turn().is_white() { v } else { trocar_valor(v) };
            return match para_brancas {
                Valor::Vitoria => 1.0,
                Valor::Empate => 0.5,
                Valor::Derrota => 0.0,
            };
        }
        let m = if n < ab.len() {
            ab[n]
        } else {
            let lado = if pos.turn().is_white() { 0 } else { 1 };
            let t = Instant::now();
            let m = [brancas, pretas][lado].lance(&pos, &historico, verdade);
            tempos[lado] += t.elapsed();
            m
        };
        historico.push(hash(&pos));
        pos = pos.play(m).unwrap();
        n += 1;
    }
}

pub fn elo(s: f64) -> f64 {
    let s = s.clamp(1e-3, 1.0 - 1e-3);
    -400.0 * (1.0 / s - 1.0).log10()
}

pub struct Placar {
    pub pontos: Vec<f32>, // de A, uma entrada por partida
    pub tempo_a: Duration,
    pub tempo_b: Duration,
}

impl Placar {
    pub fn fracao(&self) -> f64 {
        self.pontos.iter().map(|&p| p as f64).sum::<f64>() / self.pontos.len() as f64
    }
    // intervalo de 95% para a fração de pontos
    pub fn margem(&self) -> f64 {
        let n = self.pontos.len() as f64;
        let s = self.fracao();
        let var = self.pontos.iter().map(|&p| (p as f64 - s).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
        1.96 * (var / n).sqrt()
    }
}

pub fn jogar(a: &Jogador, b: &Jogador, pares: usize, verdade: &Verdade, semente: u64) -> Placar {
    let mut rng = Aleatorio(semente);
    let aberturas: Vec<Vec<Move>> = (0..pares).map(|_| abertura(&mut rng, 6)).collect();
    // cada abertura: A de brancas e A de pretas; tempos [A, B]
    let resultados = crate::paralelo::mapear(&aberturas, |ab| {
        let (mut t1, mut t2) = ([Duration::ZERO; 2], [Duration::ZERO; 2]);
        let pontos = vec![partida(a, b, ab, verdade, &mut t1), 1.0 - partida(b, a, ab, verdade, &mut t2)];
        (pontos, [t1[0] + t2[1], t1[1] + t2[0]])
    });
    let mut placar = Placar { pontos: Vec::new(), tempo_a: Duration::ZERO, tempo_b: Duration::ZERO };
    for (p, t) in resultados {
        placar.pontos.extend(p);
        placar.tempo_a += t[0];
        placar.tempo_b += t[1];
    }
    placar
}

pub fn torneio(rede: &Rede, nome_rede: &str, texto_a: &str, texto_b: &str, pares: usize, pasta_syzygy: Option<&str>) {
    let ler_ou_sair = |t: &str| {
        ler(t).unwrap_or_else(|| {
            eprintln!("Não entendi o jogador \"{t}\". Exemplos: ab:3, ab:200ms, mat:3, cert:200ms, cert:200ms:0.3");
            std::process::exit(1);
        })
    };
    let a = Jogador { nome: texto_a.into(), metodo: ler_ou_sair(texto_a), rede };
    let b = Jogador { nome: texto_b.into(), metodo: ler_ou_sair(texto_b), rede };
    let verdade = Verdade::nova(pasta_syzygy.map(crate::syzygy::abrir));
    println!("Torneio: {} contra {}, {} partidas ({} aberturas, cores trocadas), rede {nome_rede}, {} threads", a.nome, b.nome, 2 * pares, pares, threads());
    let inicio = Instant::now();
    let p = jogar(&a, &b, pares, &verdade, 0x5EED);
    let (v, e, d) = p.pontos.iter().fold((0, 0, 0), |(v, e, d), &x| {
        if x > 0.75 { (v + 1, e, d) } else if x < 0.25 { (v, e, d + 1) } else { (v, e + 1, d) }
    });
    let s = p.fracao();
    let m = p.margem();
    println!("\n{}: {v} vitórias, {e} empates, {d} derrotas = {:.1}% dos pontos", a.nome, 100.0 * s);
    println!("Diferença de Elo: {:+.0}  (95%: {:+.0} a {:+.0})", elo(s), elo(s - m), elo(s + m));
    let lances_a = p.tempo_a.as_secs_f64();
    let lances_b = p.tempo_b.as_secs_f64();
    println!("Tempo pensando: {} {:.0} s, {} {:.0} s   (total {:.0} s)", a.nome, lances_a, b.nome, lances_b, inicio.elapsed().as_secs_f64());
}

#[cfg(test)]
mod testes {
    use super::*;

    #[test]
    fn le_os_jogadores() {
        assert!(matches!(ler("ab:3"), Some(Metodo::AlfaBeta(Limite::Profundidade(3)))));
        assert!(matches!(ler("ab:200ms"), Some(Metodo::AlfaBeta(Limite::Tempo(t))) if t == Duration::from_millis(200)));
        assert!(matches!(ler("mat:2"), Some(Metodo::Material(Limite::Profundidade(2)))));
        assert!(matches!(ler("cert:100ms:0.3"), Some(Metodo::Certeza { bonus, .. }) if (bonus - 0.3).abs() < 1e-6));
        assert!(matches!(ler("cert:100ms"), Some(Metodo::Certeza { bonus, .. }) if (bonus - 0.5).abs() < 1e-6));
        assert!(ler("cert:3").is_none()); // a busca da certeza só tem limite de tempo
        assert!(ler("ab:0").is_none());
        assert!(ler("xyz:3").is_none());
    }

    #[test]
    fn elo_de_referencia() {
        assert!(elo(0.5).abs() < 1e-9);
        assert!((elo(0.75) - 190.8).abs() < 0.1); // 75% dos pontos ≈ +191 Elo
        assert!((elo(0.25) + elo(0.75)).abs() < 1e-9);
    }

    #[test]
    fn mais_profundidade_ganha_do_material() {
        // Sanidade do torneio: material com profundidade 3 contra 1.
        let rede = Rede::aleatoria(1);
        let verdade = Verdade::nova(None);
        let a = Jogador { nome: "mat:3".into(), metodo: ler("mat:3").unwrap(), rede: &rede };
        let b = Jogador { nome: "mat:1".into(), metodo: ler("mat:1").unwrap(), rede: &rede };
        let p = jogar(&a, &b, 4, &verdade, 1);
        assert_eq!(p.pontos.len(), 8);
        assert!(p.fracao() > 0.5, "{}", p.fracao());
    }
}
