// =====================================================================
// AVALIAÇÃO — o que a busca chama nas folhas
// =====================================================================
//
// Dois avaliadores com a mesma interface (o trait `Avaliador`):
//   Material — Etapa 1, só contagem de material. É a linha de base.
//   Rede     — Etapa 2, a rede 768 -> 256 -> 32 -> 3 (vitória/empate/derrota).
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
        ((p[VITORIA] - p[DERROTA]) * ESCALA_REDE).round() as i32
    }
}

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
