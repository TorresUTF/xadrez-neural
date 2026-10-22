// =====================================================================
// BUSCA GUIADA PELA CERTEZA — pensar onde a rede está em dúvida
// =====================================================================
//
// A ideia do começo do projeto: uma rede perfeita só responderia -1, 0
// ou +1; uma resposta "mole" (ex.: 50% vitória, 50% empate) quer dizer
// "esta posição precisa ser refinada". O alfa-beta ignora isso: olha
// tudo até a mesma profundidade. Aqui a árvore cresce só onde importa.
//
// A árvore fica na memória. Cada nó guarda a distribuição (V, E, D) do
// ponto de vista de quem joga nele. A cada passo:
//
//   1. DESCER: da raiz, escolher em cada nó o filho com maior
//          v + bônus · u
//      onde v = P_V - P_D (do ponto de vista de quem escolhe) e
//      u = 1 - max(P_V, P_E, P_D) é a incerteza do filho. O bônus puxa
//      a busca para filhos promissores E duvidosos. Filhos com valor
//      exato (mate, empate, tablebase) não são mais visitados.
//   2. EXPANDIR a folha alcançada: avaliar todos os filhos dela. Cada
//      filho é avaliado depois da quiescência (como no alfa-beta), para
//      não confiar na rede no meio de uma troca.
//   3. SUBIR: refazer o negamax pelo caminho. O valor de um nó é o do
//      melhor filho, com a perspectiva trocada (V <-> D).
//
// Para quando o tempo acaba ou quando a raiz tem valor exato. Joga o
// filho de maior v, sem bônus.
//
// Parentes conhecidos: proof-number search (Allis, 1994), que expande o
// nó mais fácil de PROVAR, mas só com valores exatos; e o MCTS. Aqui a
// "prova" é substituída pela certeza da rede.
// =====================================================================

use crate::autojogo::{hash, trocar, Dist, Verdade};
use crate::busca::Busca;
use crate::finais::Valor;
use crate::rede::{entradas, Rede, DERROTA, N_SAIDAS, VITORIA};
use shakmaty::{zobrist::Zobrist64, Chess, Move, Position};
use std::time::{Duration, Instant};

pub struct Config {
    pub tempo: Duration,
    pub bonus: f32,
}

struct No {
    lance: Option<Move>, // o lance que levou a este nó (None na raiz)
    pos: Chess,
    dist: Dist,          // do ponto de vista de quem joga em `pos`
    exato: bool,         // valor provado: não precisa (nem pode) refinar
    filhos: Vec<usize>,  // índices na arena; vazio = folha
}

#[allow(dead_code)] // além do lance, para inspecionar a busca (testes)
pub struct Resultado {
    pub lance: Option<Move>,
    pub dist: Dist,         // da raiz, do ponto de vista de quem joga
    pub expansoes: usize,
    pub nos: usize,
    pub profundidade: usize, // da linha principal
}

fn v(d: Dist) -> f32 {
    d[VITORIA] - d[DERROTA]
}

fn incerteza(d: Dist) -> f32 {
    1.0 - d.iter().cloned().fold(0.0, f32::max)
}

fn certo(valor: Valor) -> Dist {
    let mut d = [0.0; N_SAIDAS];
    d[valor.classe()] = 1.0;
    d
}

pub struct Busca2<'a> {
    rede: &'a Rede,
    verdade: &'a Verdade,
    nos: Vec<No>,
}

impl<'a> Busca2<'a> {
    pub fn nova(rede: &'a Rede, verdade: &'a Verdade) -> Busca2<'a> {
        Busca2 { rede, verdade, nos: Vec::new() }
    }

    // Avalia uma posição: valor exato quando existe, senão a rede na
    // folha da quiescência. `caminho` = hashes da partida e da árvore até
    // aqui (para ver repetição). None = o prazo estourou no meio.
    fn avaliar(&self, pos: &Chess, caminho: &[Zobrist64], prazo: Option<Instant>) -> Option<(Dist, bool)> {
        if pos.legal_moves().is_empty() {
            return Some((certo(if pos.is_check() { Valor::Derrota } else { Valor::Empate }), true));
        }
        let h = hash(pos);
        if pos.halfmoves() >= 100 || pos.is_insufficient_material() || caminho.iter().rev().take(pos.halfmoves() as usize).any(|&x| x == h) {
            return Some((certo(Valor::Empate), true));
        }
        if let Some(valor) = self.verdade.valor(pos) {
            return Some((certo(valor), true));
        }
        let mut busca = Busca::nova(self.rede, caminho.to_vec()).com_prazo(prazo);
        let r = busca.quieta(pos);
        if busca.abortou() {
            return None;
        }
        let mut folha = pos.clone();
        for &m in &r.pv {
            folha = folha.play(m).unwrap();
        }
        let d = self.rede.probabilidades_de(&entradas(folha.board(), folha.turn()));
        Some((if r.pv.len() % 2 == 1 { trocar(d) } else { d }, false))
    }

    // O valor de um nó expandido a partir dos filhos (negamax em distribuições)
    fn subir(&mut self, i: usize) {
        let filhos = &self.nos[i].filhos;
        let melhor = filhos
            .iter()
            .map(|&f| (trocar(self.nos[f].dist), self.nos[f].exato))
            .max_by(|a, b| v(a.0).total_cmp(&v(b.0)))
            .unwrap();
        let ganho_provado = melhor.1 && melhor.0[VITORIA] > 0.999;
        let todos_exatos = filhos.iter().all(|&f| self.nos[f].exato);
        self.nos[i].dist = melhor.0;
        self.nos[i].exato = ganho_provado || todos_exatos;
    }

    pub fn melhor_lance(&mut self, pos: &Chess, historico: &[Zobrist64], cfg: &Config) -> Resultado {
        let inicio = Instant::now();
        let prazo = inicio + cfg.tempo;
        // a raiz é sempre expandida (queremos um lance, não só o valor),
        // e a primeira expansão vai até o fim, para haver um lance
        self.nos = vec![No { lance: None, pos: pos.clone(), dist: [0.0; N_SAIDAS], exato: false, filhos: Vec::new() }];
        let mut expansoes = 0;

        loop {
            // 1) descer
            let mut caminho_nos = vec![0];
            let mut caminho_hash: Vec<Zobrist64> = historico.to_vec();
            let mut i = 0;
            while !self.nos[i].filhos.is_empty() {
                caminho_hash.push(hash(&self.nos[i].pos));
                let proximo = self.nos[i]
                    .filhos
                    .iter()
                    .copied()
                    .filter(|&f| !self.nos[f].exato)
                    .max_by(|&a, &b| {
                        let nota = |f: usize| v(trocar(self.nos[f].dist)) + cfg.bonus * incerteza(self.nos[f].dist);
                        nota(a).total_cmp(&nota(b))
                    });
                match proximo {
                    Some(f) => {
                        i = f;
                        caminho_nos.push(f);
                    }
                    None => break, // todos os filhos exatos (o nó já foi marcado exato)
                }
            }

            // 2) expandir (se o prazo estourar no meio, a expansão
            //    incompleta é desfeita e a busca para)
            if self.nos[i].filhos.is_empty() && !(i > 0 && self.nos[i].exato) {
                caminho_hash.push(hash(&self.nos[i].pos));
                let pai = self.nos[i].pos.clone();
                let tamanho = self.nos.len();
                let prazo_aqui = if expansoes == 0 { None } else { Some(prazo) };
                let mut completa = true;
                for m in pai.legal_moves() {
                    let filho = pai.clone().play(m).unwrap();
                    let Some((dist, exato)) = self.avaliar(&filho, &caminho_hash, prazo_aqui) else {
                        completa = false;
                        break;
                    };
                    self.nos.push(No { lance: Some(m), pos: filho, dist, exato, filhos: Vec::new() });
                    let id = self.nos.len() - 1;
                    self.nos[i].filhos.push(id);
                }
                if !completa {
                    self.nos.truncate(tamanho);
                    self.nos[i].filhos.clear();
                    break;
                }
                expansoes += 1;
            }

            // 3) subir
            for &n in caminho_nos.iter().rev() {
                if !self.nos[n].filhos.is_empty() {
                    self.subir(n);
                }
            }

            if self.nos[0].exato || inicio.elapsed() >= cfg.tempo || self.nos[0].filhos.is_empty() {
                break;
            }
        }

        // lance: o filho de maior v, sem bônus
        let melhor = self.nos[0].filhos.iter().copied().max_by(|&a, &b| v(trocar(self.nos[a].dist)).total_cmp(&v(trocar(self.nos[b].dist))));
        let mut profundidade = 0;
        let mut n = melhor;
        while let Some(i) = n {
            profundidade += 1;
            n = self.nos[i].filhos.iter().copied().max_by(|&a, &b| v(trocar(self.nos[a].dist)).total_cmp(&v(trocar(self.nos[b].dist))));
        }
        Resultado {
            lance: melhor.and_then(|i| self.nos[i].lance),
            dist: self.nos[0].dist,
            expansoes,
            nos: self.nos.len(),
            profundidade,
        }
    }
}

#[cfg(test)]
mod testes {
    use super::*;
    use crate::rede::EMPATE;
    use shakmaty::{fen::Fen, uci::UciMove, CastlingMode};

    fn de_fen(fen: &str) -> Chess {
        fen.parse::<Fen>().unwrap().into_position(CastlingMode::Standard).unwrap()
    }

    fn uci(m: Move) -> String {
        UciMove::from_standard(m).to_string()
    }

    #[test]
    fn acha_mate_em_um_e_prova() {
        // Mate em um não depende da rede: o filho Ra8# tem valor exato
        let rede = Rede::aleatoria(1);
        let verdade = Verdade::nova(None);
        let pos = de_fen("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1");
        let cfg = Config { tempo: Duration::from_millis(500), bonus: 0.5 };
        let r = Busca2::nova(&rede, &verdade).melhor_lance(&pos, &[], &cfg);
        assert_eq!(uci(r.lance.unwrap()), "a1a8");
        assert!(r.dist[VITORIA] > 0.999);
        assert_eq!(r.expansoes, 1); // provado na primeira expansão: parou
    }

    #[test]
    fn final_de_3_pecas_e_exato_sem_buscar() {
        // KQK com a dama pendurada e as pretas jogando: Kxd2 toma a dama e
        // sobra rei contra rei, empate exato. Qualquer outro lance deixa um
        // KQK ganho para as brancas (gabarito). Tudo exato: nada a refinar.
        let rede = Rede::aleatoria(2);
        let verdade = Verdade::nova(None);
        let pos = de_fen("8/8/8/8/8/3k4/3Q4/7K b - - 0 1");
        let cfg = Config { tempo: Duration::from_millis(500), bonus: 0.5 };
        let r = Busca2::nova(&rede, &verdade).melhor_lance(&pos, &[], &cfg);
        assert_eq!(uci(r.lance.unwrap()), "d3d2");
        assert!(r.dist[EMPATE] > 0.999);
    }

    #[test]
    fn respeita_o_tempo_e_aprofunda() {
        let rede = Rede::aleatoria(3);
        let verdade = Verdade::nova(None);
        let pos = Chess::default();
        let cfg = Config { tempo: Duration::from_millis(300), bonus: 0.5 };
        let t = Instant::now();
        let r = Busca2::nova(&rede, &verdade).melhor_lance(&pos, &[], &cfg);
        assert!(t.elapsed() < Duration::from_millis(600));
        assert!(r.lance.is_some());
        assert!(r.expansoes > 1);
        assert!(r.profundidade >= 2, "{}", r.profundidade);
        assert!((r.dist.iter().sum::<f32>() - 1.0).abs() < 1e-4);
    }
}
