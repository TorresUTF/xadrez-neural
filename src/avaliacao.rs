// =====================================================================
// AVALIAÇÃO — Etapa 1: só contagem de material
// =====================================================================
//
// Esta é a linha de base. Na Etapa 2, esta função será trocada pela
// rede neural (768 -> 256 -> 32 -> 3, saída vitória/empate/derrota).
//
// Convenção: a nota é sempre do ponto de vista de QUEM JOGA.
// Positivo = bom para quem tem a vez. Isso evita o erro de sinal
// citado no plano: a mesma posição vale +X para um lado e -X para o outro.
//
// Unidade: centipeões (100 = um peão).
// =====================================================================

use shakmaty::{Chess, Position, Role};

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
