// =====================================================================
// AVALIAÇÃO — o que a busca chama nas folhas
// =====================================================================
//
// Dois avaliadores com a mesma interface (o trait `Avaliador`):
//   Material — Etapa 1, só contagem de material. É a linha de base.
//   Rede     — Etapa 2, a rede 768 -> 384 -> 96 -> 32 -> 3 (vitória/empate/derrota).
//
// A busca não sabe qual dos dois está usando. Cada avaliador pode levar
// um "estado" de lance em lance: a rede leva o acumulador (atualização
// incremental, ver rede.rs); o material não precisa de nada.
//
// Convenção: a nota é sempre do ponto de vista de QUEM JOGA.
// Positivo = bom para quem tem a vez. Isso evita o erro de sinal
// citado no plano: a mesma posição vale +X para um lado e -X para o outro.
//
// Unidade: centipeões (100 = um peão).
// =====================================================================

use crate::rede::{Acumulador, Rede, DERROTA, VITORIA};
use shakmaty::{Chess, Position, Role};

pub trait Avaliador {
    type Estado: Clone;
    // estado da posição da raiz, calculado do zero
    fn inicio(&self, pos: &Chess) -> Self::Estado;
    // estado depois de um lance, a partir do estado de antes
    fn apos_lance(&self, estado: &Self::Estado, antes: &Chess, depois: &Chess) -> Self::Estado;
    // nota do ponto de vista de quem joga
    fn nota(&self, estado: &Self::Estado, pos: &Chess) -> i32;
}

pub struct Material;

impl Avaliador for Material {
    type Estado = ();
    fn inicio(&self, _: &Chess) {}
    fn apos_lance(&self, _: &(), _: &Chess, _: &Chess) {}
    fn nota(&self, _: &(), pos: &Chess) -> i32 {
        avaliar(pos)
    }
}

// A rede dá v = P_V - P_D entre -1 e +1; a busca trabalha com inteiros.
// Multiplicamos por 1000: v = 0,25 vira nota 250.
pub const ESCALA_REDE: f32 = 1000.0;

impl Avaliador for Rede {
    type Estado = Acumulador;
    fn inicio(&self, pos: &Chess) -> Acumulador {
        self.acumulador(pos.board())
    }
    fn apos_lance(&self, acc: &Acumulador, antes: &Chess, depois: &Chess) -> Acumulador {
        self.atualizar(acc, antes.board(), depois.board())
    }
    fn nota(&self, acc: &Acumulador, pos: &Chess) -> i32 {
        let p = self.probabilidades(acc, pos.turn());
        let nota = ((p[VITORIA] - p[DERROTA]) * ESCALA_REDE).round() as i32;
        // Posição decidida e rede com coluna de progresso: a nota passa a
        // medir QUÃO PERTO está o fim, como as notas de mate da busca.
        // Só nos finais (a rosa aprendeu com finais de até 5 peças; no
        // meio-jogo a resposta dela é chute) e só calculada quando vai ser
        // usada (ela custa quase uma coluna inteira a mais).
        let decidida = p[VITORIA].max(p[DERROTA]) > LIMIAR_DECIDIDA;
        let final_ = pos.board().occupied().count() <= MAX_PECAS_PROGRESSO;
        match (decidida && final_).then(|| self.progresso(acc, pos.turn())).flatten() {
            Some(prog) => {
                let meios_lances = (prog * 100.0).clamp(0.0, MAX_MEIOS_LANCES);
                let nota = NOTA_DECIDIDA - (meios_lances * NOTA_POR_MEIO_LANCE).round() as i32;
                if p[VITORIA] > p[DERROTA] { nota } else { -nota }
            }
            None => nota,
        }
    }
}

// Com a coluna de progresso, uma posição com P_V > 95% (ou P_D) não vale
// mais ~950: vale 1500 menos 4 por meio-lance que a rede estima faltar.
// Assim
//   - toda posição decidida fica acima de qualquer posição em dúvida
//     (o mínimo, 1500 - 4·120 = 1020, passa de 950), e
//   - entre duas posições ganhas a busca prefere a mais perto do fim; quem
//     perde prefere a mais longe (a nota é a mesma com o sinal trocado).
// Ganhar 1 meio-lance (4 pontos) pesa mais que o ruído de P_V entre 0,95
// e 1 dentro do grupo ganho, que é justamente o que a rede V/E/D não
// distingue. As notas de mate da busca (perto de 1 000 000) continuam acima.
// Peças no tabuleiro (reis incluídos) até onde a rosa é usada: as que ela
// viu no treino (Syzygy de 3 a 5 peças) e uma a mais.
const MAX_PECAS_PROGRESSO: usize = 6;
const LIMIAR_DECIDIDA: f32 = 0.95;
const NOTA_DECIDIDA: i32 = 1500;
const NOTA_POR_MEIO_LANCE: f32 = 4.0;
const MAX_MEIOS_LANCES: f32 = 120.0;

pub const VALOR_PEAO: i32 = 100;
pub const VALOR_CAVALO: i32 = 320;
pub const VALOR_BISPO: i32 = 330;
pub const VALOR_TORRE: i32 = 500;
pub const VALOR_DAMA: i32 = 900;

pub fn valor(papel: Role) -> i32 {
    match papel {
        Role::Pawn => VALOR_PEAO,
        Role::Knight => VALOR_CAVALO,
        Role::Bishop => VALOR_BISPO,
        Role::Rook => VALOR_TORRE,
        Role::Queen => VALOR_DAMA,
        Role::King => 0, // o rei nunca sai do tabuleiro; não entra na conta
    }
}

const PAPEIS: [Role; 5] = [Role::Pawn, Role::Knight, Role::Bishop, Role::Rook, Role::Queen];

// material(nosso) - material(deles), do ponto de vista de quem joga
pub fn avaliar(pos: &Chess) -> i32 {
    let mut nota = 0;
    for papel in PAPEIS {
        let nossas = pos.our(papel).count() as i32;
        let deles = pos.their(papel).count() as i32;
        nota += valor(papel) * (nossas - deles);
    }
    nota
}

#[cfg(test)]
mod testes {
    use super::*;
    use shakmaty::{fen::Fen, CastlingMode};

    fn de_fen(fen: &str) -> Chess {
        fen.parse::<Fen>().unwrap().into_position(CastlingMode::Standard).unwrap()
    }

    #[test]
    fn posicao_inicial_vale_zero() {
        assert_eq!(avaliar(&Chess::default()), 0);
    }

    #[test]
    fn sinal_troca_com_a_vez() {
        // brancas com uma dama a mais
        let brancas_jogam = de_fen("4k3/8/8/8/8/8/8/3QK3 w - - 0 1");
        let pretas_jogam = de_fen("4k3/8/8/8/8/8/8/3QK3 b - - 0 1");
        assert_eq!(avaliar(&brancas_jogam), VALOR_DAMA);
        assert_eq!(avaliar(&pretas_jogam), -VALOR_DAMA);
    }
}
