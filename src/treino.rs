// =====================================================================
// TREINO — Etapa 2b: aprendizado supervisionado nos finais de 3 peças
// =====================================================================
//
// Cada exemplo é uma posição + o valor exato dela (vitória, empate ou
// derrota para quem joga), vindo da análise retrógrada (finais.rs).
// O treino é o da Etapa 0, em lotes:
//
//   para cada época:
//     embaralha os exemplos
//     para cada lote de N exemplos:
//       soma os gradientes dos N (retropropagação)
//       ajusta os pesos com o gradiente médio (otimizador Adam, ver rede.rs)
//
// 10% dos exemplos ficam separados (VALIDAÇÃO) e nunca são usados no
// treino: é neles que medimos se a rede aprendeu o xadrez desses finais
// ou só decorou as posições que viu.
// =====================================================================

use crate::finais::{construir_todas, Tabela};
use crate::rede::{entradas, Adam, Aleatorio, Rede, N_SAIDAS};
use shakmaty::Position;
use std::time::Instant;

pub struct Exemplo {
    pub ligadas: Vec<usize>,
    pub classe: usize,
    pub final_: usize, // de qual tabela veio (para medir por final)
}

pub fn exemplos(tabelas: &[Tabela]) -> Vec<Exemplo> {
    let mut v = Vec::new();
    for (t, tabela) in tabelas.iter().enumerate() {
        for (pos, valor) in tabela.exemplos() {
            v.push(Exemplo { ligadas: entradas(pos.board(), pos.turn()), classe: valor.classe(), final_: t });
        }
    }
    v
}

pub fn embaralhar<T>(v: &mut [T], rng: &mut Aleatorio) {
    // Fisher-Yates
    for i in (1..v.len()).rev() {
        let j = ((rng.proximo() as f64 + 1.0) / 2.0 * (i + 1) as f64) as usize;
        v.swap(i, j.min(i));
    }
}

fn alvo(classe: usize) -> [f32; N_SAIDAS] {
    let mut t = [0.0; N_SAIDAS];
    t[classe] = 1.0;
    t
}

pub struct Medida {
    pub erro: f32,           // entropia cruzada média
    pub acerto: f32,         // fração em que a classe mais provável é a certa
    pub p_certa: f32,        // probabilidade média dada à classe certa
}

pub fn medir(rede: &Rede, exemplos: &[&Exemplo]) -> Medida {
    let (mut erro, mut acertos, mut p_certa) = (0.0f64, 0usize, 0.0f64);
    for e in exemplos {
        let p = rede.probabilidades_de(&e.ligadas);
        erro += -(p[e.classe].max(1e-12) as f64).ln();
        p_certa += p[e.classe] as f64;
        let prevista = (0..N_SAIDAS).max_by(|&a, &b| p[a].total_cmp(&p[b])).unwrap();
        acertos += (prevista == e.classe) as usize;
    }
    let n = exemplos.len() as f64;
    Medida { erro: (erro / n) as f32, acerto: (acertos as f64 / n) as f32, p_certa: (p_certa / n) as f32 }
}

const LOTE: usize = 256;
pub const TAXA: f32 = 1e-3;

// Uma época: embaralha e passa por todos os lotes. Devolve o erro médio.
pub fn epoca(rede: &mut Rede, gs: &mut [Rede], adam: &mut Adam, treino: &mut [&Exemplo], rng: &mut Aleatorio) -> f32 {
    embaralhar(treino, rng);
    let mut erro = 0.0f64;
    for lote in treino.chunks(LOTE) {
        erro += rede.treinar_lote(gs, adam, lote, |e| (&e.ligadas, alvo(e.classe)));
    }
    (erro / treino.len() as f64) as f32
}

// Posições clássicas para ver, com os próprios olhos, o que a rede responde
const VITRINE: [(&str, &str); 6] = [
    ("4k3/8/4K3/4P3/8/8/8/8 w - - 0 1", "rei na 6ª à frente do peão"),
    ("4k3/4P3/4K3/8/8/8/8/8 w - - 0 1", "brancas jogam: Kf6! ganha"),
    ("4k3/4P3/4K3/8/8/8/8/8 b - - 0 1", "pretas jogam: afogadas"),
    ("k7/8/K7/P7/8/8/8/8 w - - 0 1", "peão de torre, rei no canto"),
    ("8/8/8/8/8/2k5/2Q5/7K b - - 0 1", "dama pendurada, pretas jogam"),
    ("8/8/8/4k3/8/8/8/R3K3 b - - 0 1", "KRK, pretas jogam"),
];

fn mostrar_vitrine(rede: &Rede, tabelas: &[Tabela]) {
    use shakmaty::{fen::Fen, CastlingMode, Chess};
    println!("\nPosições clássicas (V / E / D para quem joga):");
    for (fen, descricao) in VITRINE {
        let pos: Chess = fen.parse::<Fen>().unwrap().into_position(CastlingMode::Standard).unwrap();
        let certo = tabelas.iter().find_map(|t| t.consultar(&pos)).unwrap();
        let p = rede.probabilidades_de(&entradas(pos.board(), pos.turn()));
        println!(
            "  {:<32} rede {:.2} / {:.2} / {:.2}  v = {:+.2}   gabarito: {:?}",
            descricao,
            p[0],
            p[1],
            p[2],
            p[0] - p[2],
            certo
        );
    }
}

pub fn treinar_finais(epocas: usize, arquivo: &str) {
    let inicio = Instant::now();
    println!("Construindo os finais por análise retrógrada...");
    let tabelas = construir_todas();
    let nomes = ["KQK", "KRK", "KPK"];
    for (t, nome) in tabelas.iter().zip(nomes) {
        let [v, e, d] = t.contagem();
        println!(
            "  {nome}: {} posições ({v} vitórias, {e} empates, {d} derrotas), {} rodadas",
            v + e + d,
            t.rodadas
        );
    }
    println!("  pronto em {:.1} s\n", inicio.elapsed().as_secs_f64());

    let todos = exemplos(&tabelas);
    let mut rede = Rede::aleatoria(0x2545F4914F6CDD1D);
    treinar_supervisionado(&mut rede, todos, &nomes, epocas);
    mostrar_vitrine(&rede, &tabelas);

    match rede.salvar(arquivo) {
        Ok(()) => println!("\nPesos salvos em {arquivo}"),
        Err(e) => println!("\nNão consegui salvar {arquivo}: {e}"),
    }
}


// Treino supervisionado genérico: embaralha, separa 10% para validação,
// treina `epocas` épocas e mostra o resultado por grupo (`nomes[final_]`).
pub fn treinar_supervisionado(rede: &mut Rede, mut todos: Vec<Exemplo>, nomes: &[&str], epocas: usize) {
        let mut rng = Aleatorio(0x9E3779B97F4A7C15);
    embaralhar(&mut todos, &mut rng);
    let corte = todos.len() / 10;
    let (validacao, treino) = todos.split_at(corte);
    let validacao: Vec<&Exemplo> = validacao.iter().collect();
    let mut treino: Vec<&Exemplo> = treino.iter().collect();
    println!("{} exemplos de treino, {} de validação\n", treino.len(), validacao.len());

    let mut gs = rede.gradientes();
    let mut adam = Adam::novo(TAXA, rede);

    let m = medir(rede, &validacao);
    println!("época  0: validação erro {:.4}  acerto {:5.1}%  P(certa) {:.3}", m.erro, 100.0 * m.acerto, m.p_certa);
    for n in 1..=epocas {
        let t0 = Instant::now();
        let erro_treino = epoca(rede, &mut gs, &mut adam, &mut treino, &mut rng);
        let m = medir(rede, &validacao);
        println!(
            "época {n:2}: treino erro {erro_treino:.4} | validação erro {:.4}  acerto {:5.1}%  P(certa) {:.3}  ({:.1} s)",
            m.erro,
            100.0 * m.acerto,
            m.p_certa,
            t0.elapsed().as_secs_f64()
        );
    }

    println!("\nPor grupo (validação):");
    for (t, nome) in nomes.iter().enumerate().filter(|(_, n)| !n.is_empty()) {
        let deste: Vec<&Exemplo> = validacao.iter().copied().filter(|e| e.final_ == t).collect();
        let m = medir(rede, &deste);
        println!("  {nome}: acerto {:5.1}%  P(certa) {:.3}", 100.0 * m.acerto, m.p_certa);
    }

    // Certeza: quantas respostas são quase "duras" (-1, 0 ou +1)?
    let certas = validacao
        .iter()
        .filter(|e| rede.probabilidades_de(&e.ligadas).iter().any(|&p| p > 0.95))
        .count();
    println!(
        "\nRespostas com mais de 95% numa só classe: {:.1}% da validação",
        100.0 * certas as f64 / validacao.len() as f64
    );

}

#[cfg(test)]
mod testes {
    use super::*;
    use crate::finais::Tabela;
    use shakmaty::Role;

    #[test]
    fn uma_epoca_em_krk_ja_aprende() {
        // KRK sozinho, 1 época: a rede tem que acertar a maioria das posições
        // que nunca viu. (O chute "sempre derrota" acertaria 50%.)
        let krk = Tabela::construir(Role::Rook, &[]);
        let mut todos = exemplos(std::slice::from_ref(&krk));
        let mut rng = Aleatorio(2);
        embaralhar(&mut todos, &mut rng);
        let (validacao, treino) = todos.split_at(todos.len() / 10);
        let validacao: Vec<&Exemplo> = validacao.iter().collect();
        let mut treino: Vec<&Exemplo> = treino.iter().collect();
        let mut rede = Rede::aleatoria(1);
        let (mut gs, mut adam) = (rede.gradientes(), Adam::novo(TAXA, &rede));
        epoca(&mut rede, &mut gs, &mut adam, &mut treino, &mut rng);
        let m = medir(&rede, &validacao);
        assert!(m.acerto > 0.9, "acerto {}", m.acerto);
    }
}
