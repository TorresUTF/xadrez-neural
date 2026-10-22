// =====================================================================
// BUSCA — minimax (na forma negamax) + poda alfa-beta + quiescência
// =====================================================================
//
// Negamax: em vez de alternar "max para as brancas, min para as pretas",
// usamos que  max(a, b) = -min(-a, -b).  Como a nota é sempre do ponto
// de vista de quem joga, a nota de um lance para mim é MENOS a nota da
// posição seguinte para o adversário:
//
//      nota(pos) = max sobre lances m de  -nota(pos depois de m)
//
// Alfa-beta: janela [alfa, beta].
//   alfa = o melhor que EU já garanti em outro ramo.
//   beta = o melhor que o ADVERSÁRIO já garantiu; se eu passar de beta,
//          ele nunca vai deixar a partida chegar aqui -> corto o ramo.
// O resultado é idêntico ao minimax puro, só que visitando menos nós.
// A poda funciona melhor quando os bons lances vêm primeiro (ordenação).
//
// Quiescência: na folha não se avalia direto, porque no meio de uma troca
// a contagem de material mente ("tomei a dama!" sem ver a retomada).
// Continua-se só com capturas e promoções até a posição ficar quieta.
// =====================================================================

use crate::avaliacao::{avaliar, valor};
use shakmaty::{zobrist::Zobrist64, Chess, EnPassantMode, Move, MoveList, Position};

// Nota de mate. Mate em menos lances vale mais: MATE - distância em meios-lances.
pub const MATE: i32 = 1_000_000;
const INFINITO: i32 = MATE + 1;

pub struct Resultado {
    pub lance: Option<Move>,
    pub nota: i32,  // centipeões, do ponto de vista de quem joga
    pub nos: u64,   // posições visitadas (para medir o efeito da poda)
}

pub struct Busca {
    nos: u64,
    // hashes das posições já vistas na partida e no ramo atual (repetição)
    historico: Vec<Zobrist64>,
}

fn hash(pos: &Chess) -> Zobrist64 {
    pos.zobrist_hash(EnPassantMode::Legal)
}

// Ordenação MVV-LVA: primeiro capturar a peça mais valiosa (Most Valuable
// Victim) com a menos valiosa (Least Valuable Attacker). Promoções também cedo.
fn ordenar(lances: &mut MoveList) {
    lances.sort_by_key(|m| {
        let vitima = m.capture().map(valor).unwrap_or(0);
        let promocao = m.promotion().map(valor).unwrap_or(0);
        let bonus = if m.is_capture() || m.is_promotion() { 10_000 } else { 0 };
        -(bonus + 10 * vitima + promocao - valor(m.role()) / 10)
    });
}

impl Busca {
    // `historico_partida`: hashes das posições anteriores da partida,
    // para a busca enxergar empate por repetição.
    pub fn nova(historico_partida: Vec<Zobrist64>) -> Busca {
        Busca { nos: 0, historico: historico_partida }
    }

    pub fn melhor_lance(&mut self, pos: &Chess, profundidade: u32) -> Resultado {
        self.nos = 0;
        let mut lances = pos.legal_moves();
        ordenar(&mut lances);

        let mut melhor = None;
        let mut alfa = -INFINITO;
        self.historico.push(hash(pos));
        for m in lances {
            let filho = pos.clone().play(m).unwrap();
            let nota = -self.negamax(&filho, profundidade - 1, 1, -INFINITO, -alfa);
            if nota > alfa {
                alfa = nota;
                melhor = Some(m);
            }
        }
        self.historico.pop();

        // sem lances: mate ou afogamento
        let nota = if melhor.is_some() { alfa } else if pos.is_check() { -MATE } else { 0 };
        Resultado { lance: melhor, nota, nos: self.nos }
    }

    fn eh_empate(&self, pos: &Chess) -> bool {
        if pos.halfmoves() >= 100 || pos.is_insufficient_material() {
            return true;
        }
        // uma única repetição já conta como empate dentro da busca
        let h = hash(pos);
        self.historico.iter().rev().take(pos.halfmoves() as usize).any(|&x| x == h)
    }

    // `ply` = distância até a raiz, para preferir o mate mais curto
    fn negamax(&mut self, pos: &Chess, profundidade: u32, ply: i32, mut alfa: i32, beta: i32) -> i32 {
        self.nos += 1;

        if self.eh_empate(pos) {
            return 0;
        }
        let mut lances = pos.legal_moves();
        if lances.is_empty() {
            return if pos.is_check() { -(MATE - ply) } else { 0 };
        }
        if profundidade == 0 {
            return self.quiescencia(pos, alfa, beta);
        }

        ordenar(&mut lances);
        self.historico.push(hash(pos));
        for m in lances {
            let filho = pos.clone().play(m).unwrap();
            let nota = -self.negamax(&filho, profundidade - 1, ply + 1, -beta, -alfa);
            if nota >= beta {
                self.historico.pop();
                return beta; // corte beta: o adversário não deixaria chegar aqui
            }
            alfa = alfa.max(nota);
        }
        self.historico.pop();
        alfa
    }

    fn quiescencia(&mut self, pos: &Chess, mut alfa: i32, beta: i32) -> i32 {
        self.nos += 1;

        // "Ficar parado": quem joga pode recusar todas as capturas, então
        // a nota estática é um piso. (Simplificação: em xeque isso não vale
        // de verdade, mas é o padrão nesta etapa.)
        let parado = avaliar(pos);
        if parado >= beta {
            return beta;
        }
        alfa = alfa.max(parado);

        let mut lances = pos.legal_moves();
        lances.retain(|m| m.is_capture() || m.is_promotion());
        ordenar(&mut lances);
        for m in lances {
            let filho = pos.clone().play(m).unwrap();
            let nota = -self.quiescencia(&filho, -beta, -alfa);
            if nota >= beta {
                return beta;
            }
            alfa = alfa.max(nota);
        }
        alfa
    }
}

#[cfg(test)]
mod testes {
    use super::*;
    use shakmaty::{fen::Fen, uci::UciMove, CastlingMode};

    fn de_fen(fen: &str) -> Chess {
        fen.parse::<Fen>().unwrap().into_position(CastlingMode::Standard).unwrap()
    }

    fn lance_uci(pos: &Chess, prof: u32) -> String {
        let r = Busca::nova(vec![]).melhor_lance(pos, prof);
        UciMove::from_standard(r.lance.unwrap()).to_string()
    }

    #[test]
    fn acha_mate_em_um() {
        // mate do corredor: Ra8#
        let pos = de_fen("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1");
        assert_eq!(lance_uci(&pos, 3), "a1a8");
        let r = Busca::nova(vec![]).melhor_lance(&pos, 3);
        assert_eq!(r.nota, MATE - 1);
    }

    #[test]
    fn toma_dama_pendurada() {
        let pos = de_fen("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1");
        assert_eq!(lance_uci(&pos, 1), "d1d5");
    }

    #[test]
    fn quiescencia_ve_a_retomada() {
        // Cxd5 ganha um peão mas perde o cavalo para exd5. Com profundidade 1,
        // só a quiescência impede o motor de achar que ganhou um peão.
        let pos = de_fen("4k3/8/4p3/3p4/8/2N5/8/4K3 w - - 0 1");
        assert_ne!(lance_uci(&pos, 1), "c3d5");
    }

    #[test]
    fn afogamento_vale_zero() {
        let pos = de_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1");
        assert!(pos.is_stalemate());
        assert_eq!(Busca::nova(vec![]).melhor_lance(&pos, 3).nota, 0);
    }

    #[test]
    fn alfa_beta_igual_ao_minimax() {
        // Minimax puro, sem poda, para comparar: a nota tem que ser a mesma.
        fn minimax(pos: &Chess, prof: u32, ply: i32) -> i32 {
            let lances = pos.legal_moves();
            if lances.is_empty() {
                return if pos.is_check() { -(MATE - ply) } else { 0 };
            }
            if prof == 0 {
                return Busca::nova(vec![]).quiescencia(pos, -INFINITO, INFINITO);
            }
            lances
                .iter()
                .map(|&m| -minimax(&pos.clone().play(m).unwrap(), prof - 1, ply + 1))
                .max()
                .unwrap()
        }
        let pos = de_fen("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3");
        let r = Busca::nova(vec![]).melhor_lance(&pos, 3);
        assert_eq!(r.nota, minimax(&pos, 3, 0));
    }
}
