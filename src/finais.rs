// =====================================================================
// FINAIS — Etapa 2b: o gabarito exato dos finais de 3 peças
// =====================================================================
//
// Para poucas peças dá para saber o valor de JOGO PERFEITO de toda
// posição: vitória, empate ou derrota para quem joga. As tablebases
// Syzygy fazem isso até 7 peças; aqui construímos as nossas, para
// 3 peças, com o mesmo método: ANÁLISE RETRÓGRADA.
//
//   1. Lista todas as posições legais (rei + peça contra rei).
//   2. As que não têm lances já estão resolvidas:
//        xeque-mate  -> derrota para quem joga
//        afogamento  -> empate
//   3. Repete até nada mudar:
//        se ALGUM lance leva a uma posição de DERROTA do adversário -> VITÓRIA
//        se TODOS os lances levam a VITÓRIA do adversário          -> DERROTA
//   4. O que sobrou sem resolver é EMPATE (ninguém consegue forçar nada).
//
// É o minimax de novo, só que de trás para frente e sem limite de
// profundidade: cada rodada estende o gabarito em um meio-lance.
//
// Finais construídos: KQK, KRK e KPK, sempre com a peça a mais nas brancas.
// Não precisamos do caso "peça a mais nas pretas": a rede vê a posição do
// ponto de vista de quem joga, e a posição espelhada tem exatamente as
// mesmas entradas.
// =====================================================================

use crate::rede::{DERROTA, EMPATE, VITORIA};
use shakmaty::{
    Board, CastlingMode, Chess, Color, FromSetup, Piece, Position, Role, Setup, Square,
};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Valor {
    Vitoria,
    Empate,
    Derrota,
}

impl Valor {
    // valor para o adversário
    fn trocado(self) -> Valor {
        match self {
            Valor::Vitoria => Valor::Derrota,
            Valor::Empate => Valor::Empate,
            Valor::Derrota => Valor::Vitoria,
        }
    }

    // índice da classe na saída da rede
    pub fn classe(self) -> usize {
        match self {
            Valor::Vitoria => VITORIA,
            Valor::Empate => EMPATE,
            Valor::Derrota => DERROTA,
        }
    }
}

// Para onde leva um lance: outra posição do mesmo final (índice) ou um
// valor já conhecido (captura, promoção para outro final).
#[derive(Clone, Copy)]
enum Destino {
    Indice(u32),
    Conhecido(Valor),
}

pub struct Tabela {
    pub peca: Role,
    // índice = ((rei_branco * 64 + rei_preto) * 64 + peca) * 2 + vez
    // None = posição ilegal
    pub valores: Vec<Option<Valor>>,
    pub rodadas: usize,
}

const N_POSICOES: usize = 64 * 64 * 64 * 2;

fn indice(rb: usize, rp: usize, x: usize, vez: Color) -> usize {
    ((rb * 64 + rp) * 64 + x) * 2 + if vez.is_white() { 0 } else { 1 }
}

fn posicao(peca: Role, i: usize) -> Option<Chess> {
    let vez = if i.is_multiple_of(2) { Color::White } else { Color::Black };
    let x = (i / 2) % 64;
    let rp = (i / 128) % 64;
    let rb = i / 8192;
    if rb == rp || rb == x || rp == x {
        return None;
    }
    let mut board = Board::empty();
    board.set_piece_at(Square::new(rb as u32), Piece { color: Color::White, role: Role::King });
    board.set_piece_at(Square::new(rp as u32), Piece { color: Color::Black, role: Role::King });
    board.set_piece_at(Square::new(x as u32), Piece { color: Color::White, role: peca });
    let mut setup = Setup::empty();
    setup.board = board;
    setup.turn = vez;
    // from_setup recusa posições impossíveis: reis colados, peão na
    // 1ª ou 8ª fileira, quem NÃO joga em xeque...
    Chess::from_setup(setup, CastlingMode::Standard).ok()
}

// Índice de uma posição de 3 peças com a peça `peca` nas brancas
fn indice_de(pos: &Chess, peca: Role) -> Option<usize> {
    let b = pos.board();
    if b.occupied().count() != 3 || b.by_piece(Piece { color: Color::White, role: peca }).count() != 1 {
        return None;
    }
    let rb = b.king_of(Color::White)?.to_usize();
    let rp = b.king_of(Color::Black)?.to_usize();
    let x = b.by_piece(Piece { color: Color::White, role: peca }).first()?.to_usize();
    Some(indice(rb, rp, x, pos.turn()))
}

impl Tabela {
    // `outras`: tabelas já prontas, para as promoções do KPK
    pub fn construir(peca: Role, outras: &[&Tabela]) -> Tabela {
        let mut valores: Vec<Option<Valor>> = vec![None; N_POSICOES];
        let mut legal = vec![false; N_POSICOES];
        let mut destinos: Vec<Vec<Destino>> = vec![Vec::new(); N_POSICOES];

        // Passo 1 e 2: posições legais, terminais e para onde cada lance leva
        for i in 0..N_POSICOES {
            let Some(pos) = posicao(peca, i) else { continue };
            legal[i] = true;
            let lances = pos.legal_moves();
            if lances.is_empty() {
                valores[i] = Some(if pos.is_check() { Valor::Derrota } else { Valor::Empate });
                continue;
            }
            for m in lances {
                let filho = pos.clone().play(m).unwrap();
                let destino = if let Some(j) = indice_de(&filho, peca) {
                    Destino::Indice(j as u32)
                } else if filho.is_insufficient_material() {
                    // sobrou rei contra rei (captura) ou promoveu a bispo/cavalo
                    Destino::Conhecido(Valor::Empate)
                } else {
                    // promoveu: o valor está em outra tabela (já pronta)
                    let promovida = m.promotion().expect("lance sai do final sem ser promoção");
                    let t = outras.iter().find(|t| t.peca == promovida).expect("falta a tabela da promoção");
                    let j = indice_de(&filho, promovida).unwrap();
                    Destino::Conhecido(t.valores[j].unwrap())
                };
                destinos[i].push(destino);
            }
        }

        // Passo 3: rodadas até nada mudar. Cada rodada só usa os valores da
        // rodada anterior (os novos são aplicados no fim), então a rodada k
        // resolve exatamente as posições decididas em k meios-lances.
        let mut rodadas = 0;
        loop {
            rodadas += 1;
            let mut novos: Vec<(usize, Valor)> = Vec::new();
            for i in 0..N_POSICOES {
                if !legal[i] || valores[i].is_some() {
                    continue;
                }
                // valor de cada lance para QUEM JOGA aqui = trocado do valor do filho
                let mut todos_conhecidos = true;
                let mut algum_ganha = false;
                let mut todos_perdem = true;
                for d in &destinos[i] {
                    let v = match *d {
                        Destino::Indice(j) => valores[j as usize],
                        Destino::Conhecido(v) => Some(v),
                    };
                    match v.map(Valor::trocado) {
                        Some(Valor::Vitoria) => algum_ganha = true,
                        Some(_) => todos_perdem &= v == Some(Valor::Vitoria),
                        None => {
                            todos_conhecidos = false;
                            todos_perdem = false;
                        }
                    }
                }
                if algum_ganha {
                    novos.push((i, Valor::Vitoria));
                } else if todos_conhecidos && todos_perdem {
                    novos.push((i, Valor::Derrota));
                }
            }
            if novos.is_empty() {
                break;
            }
            for (i, v) in novos {
                valores[i] = Some(v);
            }
        }

        // Passo 4: o resto é empate
        for i in 0..N_POSICOES {
            if legal[i] && valores[i].is_none() {
                valores[i] = Some(Valor::Empate);
            }
        }
        Tabela { peca, valores, rodadas }
    }

    pub fn consultar(&self, pos: &Chess) -> Option<Valor> {
        self.valores[indice_de(pos, self.peca)?]
    }

    // Todas as posições legais com o seu valor
    pub fn exemplos(&self) -> impl Iterator<Item = (Chess, Valor)> + '_ {
        self.valores
            .iter()
            .enumerate()
            .filter_map(move |(i, v)| Some((posicao(self.peca, i)?, (*v)?)))
    }

    // (vitórias, empates, derrotas) para quem joga
    pub fn contagem(&self) -> [usize; 3] {
        let mut c = [0; 3];
        for v in self.valores.iter().flatten() {
            c[v.classe()] += 1;
        }
        c
    }
}

// As três tabelas, na ordem certa (KPK precisa das outras duas)
pub fn construir_todas() -> Vec<Tabela> {
    let kqk = Tabela::construir(Role::Queen, &[]);
    let krk = Tabela::construir(Role::Rook, &[]);
    let kpk = Tabela::construir(Role::Pawn, &[&kqk, &krk]);
    vec![kqk, krk, kpk]
}

#[cfg(test)]
mod testes {
    use super::*;
    use shakmaty::fen::Fen;
    use std::sync::OnceLock;

    fn tabelas() -> &'static Vec<Tabela> {
        static T: OnceLock<Vec<Tabela>> = OnceLock::new();
        T.get_or_init(construir_todas)
    }

    fn valor(fen: &str) -> Valor {
        let pos: Chess = fen.parse::<Fen>().unwrap().into_position(CastlingMode::Standard).unwrap();
        tabelas().iter().find_map(|t| t.consultar(&pos)).unwrap()
    }

    #[test]
    fn kqk_e_krk_brancas_jogando_sempre_ganham() {
        // fato conhecido: com dama ou torre e a vez, sempre se força o mate
        for t in &tabelas()[..2] {
            let mut empates_brancas = 0;
            for (i, v) in t.valores.iter().enumerate() {
                if i % 2 == 0 && *v == Some(Valor::Empate) {
                    empates_brancas += 1;
                }
            }
            assert_eq!(empates_brancas, 0, "{:?}", t.peca);
        }
    }

    #[test]
    fn duracao_maxima() {
        // Mate mais longo: KQK em 10 lances, KRK em 16 (= 20 e 32 meios-lances).
        // As rodadas param uma depois da última mudança.
        assert_eq!(tabelas()[0].rodadas, 20 + 1);
        assert_eq!(tabelas()[1].rodadas, 32 + 1);
    }

    #[test]
    fn kpk_posicoes_classicas() {
        // rei na 6ª à frente do peão: ganha, tenha quem tiver a vez
        assert_eq!(valor("4k3/8/4K3/4P3/8/8/8/8 w - - 0 1"), Valor::Vitoria);
        assert_eq!(valor("4k3/8/4K3/4P3/8/8/8/8 b - - 0 1"), Valor::Derrota);
        // peão na 7ª com o rei atrás: pretas jogando estão afogadas (empate),
        // mas brancas jogando ganham com Kf6! Kd7 Kf7 e a promoção
        assert_eq!(valor("4k3/4P3/4K3/8/8/8/8/8 b - - 0 1"), Valor::Empate);
        assert_eq!(valor("4k3/4P3/4K3/8/8/8/8/8 w - - 0 1"), Valor::Vitoria);
        // peão de torre com o rei defensor no canto: empate
        assert_eq!(valor("k7/8/K7/P7/8/8/8/8 w - - 0 1"), Valor::Empate);
        // peão que corre sozinho para a promoção, rei preto longe demais
        assert_eq!(valor("8/P7/8/8/8/8/7k/K7 b - - 0 1"), Valor::Derrota);
    }

    #[test]
    fn dama_pendurada_com_pretas_jogando_e_empate() {
        assert_eq!(valor("8/8/8/8/8/2k5/2Q5/7K b - - 0 1"), Valor::Empate);
    }
}
