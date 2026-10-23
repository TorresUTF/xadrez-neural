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

use crate::avaliacao::{valor, Avaliador};
use shakmaty::{zobrist::Zobrist64, Chess, EnPassantMode, Move, MoveList, Position};

// Nota de mate. Mate em menos lances vale mais: MATE - distância em meios-lances.
pub const MATE: i32 = 1_000_000;
const INFINITO: i32 = MATE + 1;

// Limite de meios-lances na quiescência. Sem limite, uma rede ainda ruim
// pode achar "boas" longas sequências de capturas e a busca explode.
pub const MAX_QUIESCENCIA: u32 = 8;

pub struct Resultado {
    pub lance: Option<Move>,
    pub nota: i32,  // centipeões, do ponto de vista de quem joga
    pub nos: u64,   // posições visitadas (para medir o efeito da poda)
    // Linha principal (PV): a sequência de lances que a busca espera, dos
    // dois lados. A posição no fim dela é a "folha" cuja avaliação deu a
    // nota da raiz; é ela que o TD-Leaf treina (Etapa 3).
    pub pv: Vec<Move>,
}

pub struct Busca<'a, A: Avaliador> {
    avaliador: &'a A,
    nos: u64,
    // hashes das posições já vistas na partida e no ramo atual (repetição)
    historico: Vec<Zobrist64>,
    // busca por tempo: passou do prazo, a busca desiste (e o resultado
    // incompleto é jogado fora)
    prazo: Option<std::time::Instant>,
    abortou: bool,
}

pub fn hash(pos: &Chess) -> Zobrist64 {
    pos.zobrist_hash(EnPassantMode::Legal)
}

// Ordenação MVV-LVA: primeiro capturar a peça mais valiosa (Most Valuable
// Victim) com a menos valiosa (Least Valuable Attacker). Promoções também cedo.
pub fn ordenar(lances: &mut MoveList) {
    lances.sort_by_key(|m| {
        let vitima = m.capture().map(valor).unwrap_or(0);
        let promocao = m.promotion().map(valor).unwrap_or(0);
        let bonus = if m.is_capture() || m.is_promotion() { 10_000 } else { 0 };
        -(bonus + 10 * vitima + promocao - valor(m.role()) / 10)
    });
}

impl<'a, A: Avaliador> Busca<'a, A> {
    // `historico_partida`: hashes das posições anteriores da partida,
    // para a busca enxergar empate por repetição.
    pub fn nova(avaliador: &'a A, historico_partida: Vec<Zobrist64>) -> Busca<'a, A> {
        Busca { avaliador, nos: 0, historico: historico_partida, prazo: None, abortou: false }
    }

    // posição seguinte e o estado do avaliador nela (ex.: acumulador da rede)
    fn jogar(&self, pos: &Chess, estado: &A::Estado, m: Move) -> (Chess, A::Estado) {
        let filho = pos.clone().play(m).unwrap();
        let estado_filho = self.avaliador.apos_lance(estado, pos, &filho);
        (filho, estado_filho)
    }

    pub fn melhor_lance(&mut self, pos: &Chess, profundidade: u32) -> Resultado {
        self.nos = 0;
        let mut lances = pos.legal_moves();
        ordenar(&mut lances);

        let mut melhor = None;
        let mut alfa = -INFINITO;
        let mut pv = Vec::new();
        let estado = self.avaliador.inicio(pos);
        self.historico.push(hash(pos));
        for m in lances {
            let (filho, est) = self.jogar(pos, &estado, m);
            let mut pv_filho = Vec::new();
            let nota = -self.negamax(&filho, &est, profundidade - 1, 1, -INFINITO, -alfa, &mut pv_filho);
            if self.abortou {
                break;
            }
            if nota > alfa {
                alfa = nota;
                melhor = Some(m);
                pv = std::iter::once(m).chain(pv_filho).collect();
            }
        }
        self.historico.pop();

        // sem lances: mate ou afogamento
        let nota = if melhor.is_some() { alfa } else if pos.is_check() { -MATE } else { 0 };
        Resultado { lance: melhor, nota, nos: self.nos, pv }
    }

    // Só a quiescência a partir da raiz: a avaliação "honesta" de uma
    // posição, depois de resolvidas as trocas pendentes. A PV leva à folha.
    pub fn quieta(&mut self, pos: &Chess) -> Resultado {
        self.nos = 0;
        let estado = self.avaliador.inicio(pos);
        let mut pv = Vec::new();
        let nota = self.quiescencia(pos, &estado, -INFINITO, INFINITO, MAX_QUIESCENCIA, &mut pv);
        Resultado { lance: pv.first().copied(), nota, nos: self.nos, pv }
    }

    // Aprofundamento iterativo: profundidade 1, 2, 3... até acabar o
    // tempo. Quando o prazo passa no meio de uma profundidade, a busca
    // desiste dela e joga o lance da última profundidade completa. Assim
    // o tempo usado é o tempo dado (importante para comparar métodos).
    // A profundidade 1 sempre vai até o fim, para haver um lance.
    pub fn por_tempo(&mut self, pos: &Chess, limite: std::time::Duration) -> Resultado {
        let inicio = std::time::Instant::now();
        let mut nos = 0;
        let mut melhor = self.melhor_lance(pos, 1);
        nos += melhor.nos;
        self.prazo = Some(inicio + limite);
        for prof in 2..=30 {
            if melhor.lance.is_none() || melhor.nota.abs() > MATE - 1000 || inicio.elapsed() >= limite {
                break;
            }
            let r = self.melhor_lance(pos, prof);
            nos += r.nos;
            if self.abortou {
                break;
            }
            melhor = r;
        }
        self.prazo = None;
        self.abortou = false;
        melhor.nos = nos;
        melhor
    }

    // a cada 1024 nós, confere o relógio
    fn sem_tempo(&mut self) -> bool {
        if !self.abortou && self.nos & 1023 == 0 && let Some(p) = self.prazo {
            self.abortou = std::time::Instant::now() >= p;
        }
        self.abortou
    }

    // Para quem usa a busca por dentro (certeza.rs): um prazo, e se ele estourou.
    pub fn com_prazo(mut self, prazo: Option<std::time::Instant>) -> Self {
        self.prazo = prazo;
        self
    }

    pub fn abortou(&self) -> bool {
        self.abortou
    }

    fn eh_empate(&self, pos: &Chess) -> bool {
        if pos.halfmoves() >= 100 || pos.is_insufficient_material() {
            return true;
        }
        // uma única repetição já conta como empate dentro da busca
        let h = hash(pos);
        self.historico.iter().rev().take(pos.halfmoves() as usize).any(|&x| x == h)
    }

    // `ply` = distância até a raiz, para preferir o mate mais curto.
    // `pv` recebe a linha principal a partir daqui (quando a nota fica
    // dentro da janela; num corte ela não importa).
    #[allow(clippy::too_many_arguments)]
    fn negamax(
        &mut self,
        pos: &Chess,
        estado: &A::Estado,
        profundidade: u32,
        ply: i32,
        mut alfa: i32,
        beta: i32,
        pv: &mut Vec<Move>,
    ) -> i32 {
        self.nos += 1;
        if self.sem_tempo() {
            return 0;
        }

        if self.eh_empate(pos) {
            return 0;
        }
        let mut lances = pos.legal_moves();
        if lances.is_empty() {
            return if pos.is_check() { -(MATE - ply) } else { 0 };
        }
        if profundidade == 0 {
            return self.quiescencia(pos, estado, alfa, beta, MAX_QUIESCENCIA, pv);
        }

        ordenar(&mut lances);
        self.historico.push(hash(pos));
        for m in lances {
            let (filho, est) = self.jogar(pos, estado, m);
            let mut pv_filho = Vec::new();
            let nota = -self.negamax(&filho, &est, profundidade - 1, ply + 1, -beta, -alfa, &mut pv_filho);
            if self.abortou {
                self.historico.pop();
                return 0;
            }
            if nota >= beta {
                self.historico.pop();
                return beta; // corte beta: o adversário não deixaria chegar aqui
            }
            if nota > alfa {
                alfa = nota;
                *pv = std::iter::once(m).chain(pv_filho).collect();
            }
        }
        self.historico.pop();
        alfa
    }

    fn quiescencia(
        &mut self,
        pos: &Chess,
        estado: &A::Estado,
        mut alfa: i32,
        beta: i32,
        restante: u32,
        pv: &mut Vec<Move>,
    ) -> i32 {
        self.nos += 1;
        if self.sem_tempo() {
            return 0;
        }

        // "Ficar parado": quem joga pode recusar todas as capturas, então
        // a nota estática é um piso. (Simplificação: em xeque isso não vale
        // de verdade, mas é o padrão nesta etapa.)
        let parado = self.avaliador.nota(estado, pos);
        if parado >= beta {
            return beta;
        }
        alfa = alfa.max(parado);
        if restante == 0 {
            return alfa;
        }

        let mut lances = pos.legal_moves();
        lances.retain(|m| m.is_capture() || m.is_promotion());
        ordenar(&mut lances);
        for m in lances {
            let (filho, est) = self.jogar(pos, estado, m);
            let mut pv_filho = Vec::new();
            let nota = -self.quiescencia(&filho, &est, -beta, -alfa, restante - 1, &mut pv_filho);
            if nota >= beta {
                return beta;
            }
            if nota > alfa {
                alfa = nota;
                *pv = std::iter::once(m).chain(pv_filho).collect();
            }
        }
        alfa
    }
}

#[cfg(test)]
mod testes {
    use super::*;
    use crate::avaliacao::Material;
    use crate::rede::Rede;
    use shakmaty::{fen::Fen, uci::UciMove, CastlingMode};

    fn de_fen(fen: &str) -> Chess {
        fen.parse::<Fen>().unwrap().into_position(CastlingMode::Standard).unwrap()
    }

    fn lance_uci(pos: &Chess, prof: u32) -> String {
        let r = Busca::nova(&Material, vec![]).melhor_lance(pos, prof);
        UciMove::from_standard(r.lance.unwrap()).to_string()
    }

    #[test]
    fn acha_mate_em_um() {
        // mate do corredor: Ra8#
        let pos = de_fen("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1");
        assert_eq!(lance_uci(&pos, 3), "a1a8");
        let r = Busca::nova(&Material, vec![]).melhor_lance(&pos, 3);
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
        assert_eq!(Busca::nova(&Material, vec![]).melhor_lance(&pos, 3).nota, 0);
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
                return Busca::nova(&Material, vec![]).quiescencia(pos, &(), -INFINITO, INFINITO, MAX_QUIESCENCIA, &mut vec![]);
            }
            lances
                .iter()
                .map(|&m| -minimax(&pos.clone().play(m).unwrap(), prof - 1, ply + 1))
                .max()
                .unwrap()
        }
        let pos = de_fen("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3");
        let r = Busca::nova(&Material, vec![]).melhor_lance(&pos, 3);
        assert_eq!(r.nota, minimax(&pos, 3, 0));
    }

    #[test]
    fn busca_com_rede_igual_ao_minimax() {
        // O mesmo teste com a rede (pesos aleatórios): a poda e o acumulador
        // incremental não podem mudar a nota.
        fn minimax(rede: &Rede, pos: &Chess, prof: u32, ply: i32) -> i32 {
            let lances = pos.legal_moves();
            if lances.is_empty() {
                return if pos.is_check() { -(MATE - ply) } else { 0 };
            }
            if prof == 0 {
                let acc = rede.inicio(pos);
                return Busca::nova(rede, vec![]).quiescencia(pos, &acc, -INFINITO, INFINITO, MAX_QUIESCENCIA, &mut vec![]);
            }
            lances
                .iter()
                .map(|&m| -minimax(rede, &pos.clone().play(m).unwrap(), prof - 1, ply + 1))
                .max()
                .unwrap()
        }
        let rede = Rede::aleatoria(11);
        let pos = de_fen("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3");
        let r = Busca::nova(&rede, vec![]).melhor_lance(&pos, 2);
        assert_eq!(r.nota, minimax(&rede, &pos, 2, 0));
    }

    #[test]
    fn por_tempo_usa_o_tempo_dado() {
        // Com 200 ms, a busca tem que gastar perto de 200 ms (nem muito
        // menos, nem muito mais) e devolver um lance legal.
        let rede = Rede::aleatoria(13);
        let pos = Chess::default();
        let t = std::time::Instant::now();
        let r = Busca::nova(&rede, vec![]).por_tempo(&pos, std::time::Duration::from_millis(200));
        let gasto = t.elapsed().as_millis();
        assert!((150..400).contains(&gasto), "{gasto} ms");
        assert!(pos.is_legal(r.lance.unwrap()));
    }

    #[test]
    fn rede_ainda_ve_mate() {
        // mate não depende da avaliação: mesmo com pesos aleatórios
        let rede = Rede::aleatoria(12);
        let pos = de_fen("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1");
        let r = Busca::nova(&rede, vec![]).melhor_lance(&pos, 3);
        assert_eq!(UciMove::from_standard(r.lance.unwrap()).to_string(), "a1a8");
    }

    #[test]
    fn nota_da_raiz_e_a_nota_da_folha_da_pv() {
        // O coração do TD-Leaf: a nota que a busca dá à raiz é exatamente a
        // avaliação estática da folha da linha principal (com o sinal de
        // quem joga na raiz).
        use crate::avaliacao::avaliar;
        for fen in [
            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
            "4k3/8/4p3/3p4/8/2N5/8/4K3 w - - 0 1",
            "r2qk2r/ppp2ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPP2PPP/R2QK2R w KQkq - 0 7",
        ] {
            let pos = de_fen(fen);
            let r = Busca::nova(&Material, vec![]).melhor_lance(&pos, 3);
            let mut folha = pos.clone();
            for &m in &r.pv {
                folha = folha.play(m).unwrap();
            }
            let sinal = if r.pv.len().is_multiple_of(2) { 1 } else { -1 };
            assert_eq!(r.nota, sinal * avaliar(&folha), "{fen}");
            assert!(r.pv.len() >= 3, "{fen}");
        }
    }
}
