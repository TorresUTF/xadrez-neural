// =====================================================================
// ETAPA 1 — Motor de xadrez sem rede neural (a linha de base)
// =====================================================================
//
// Uso:
//   cargo run --release                 -> o motor joga uma partida contra si mesmo
//   cargo run --release -- jogar 4      -> idem, com profundidade 4
//   cargo run --release -- uci          -> protocolo UCI (para usar numa GUI
//                                          ou contra outra engine, ex.: Maia)
// =====================================================================

mod avaliacao;
mod busca;

use busca::{Busca, MATE};
use shakmaty::{
    fen::Fen, san::SanPlus, uci::UciMove, zobrist::Zobrist64, CastlingMode, Chess, EnPassantMode,
    Outcome, Position,
};
use std::io::{self, BufRead, Write};
use std::time::Instant;

const PROFUNDIDADE_PADRAO: u32 = 3;

fn hash(pos: &Chess) -> Zobrist64 {
    pos.zobrist_hash(EnPassantMode::Legal)
}

fn texto_nota(nota: i32) -> String {
    if nota.abs() > MATE - 1000 {
        let lances = (MATE - nota.abs() + 1) / 2;
        format!("{}M{}", if nota > 0 { "+" } else { "-" }, lances)
    } else {
        format!("{:+.2}", nota as f64 / 100.0)
    }
}

// Três repetições, regra dos 50 lances, material insuficiente, mate ou afogamento.
fn fim_de_partida(pos: &Chess, historico: &[Zobrist64]) -> Option<String> {
    if let Outcome::Known(r) = pos.outcome() {
        return Some(r.to_string());
    }
    let h = hash(pos);
    if historico.iter().filter(|&&x| x == h).count() >= 2 {
        return Some("1/2-1/2 (repetição)".into());
    }
    if pos.halfmoves() >= 100 {
        return Some("1/2-1/2 (50 lances)".into());
    }
    None
}

fn jogar_contra_si(profundidade: u32) {
    let mut pos = Chess::default();
    let mut historico: Vec<Zobrist64> = Vec::new();
    let mut pgn = String::new();
    let inicio = Instant::now();
    let mut nos_total = 0;

    println!("Partida do motor contra si mesmo, profundidade {profundidade}\n");
    let resultado = loop {
        if let Some(fim) = fim_de_partida(&pos, &historico) {
            break fim;
        }
        let r = Busca::nova(historico.clone()).melhor_lance(&pos, profundidade);
        let m = r.lance.unwrap();
        nos_total += r.nos;

        historico.push(hash(&pos));
        let numero = pos.fullmoves();
        let branca = pos.turn().is_white();
        let san = SanPlus::from_move_and_play_unchecked(&mut pos, m);
        if branca {
            pgn.push_str(&format!("{numero}. "));
        }
        pgn.push_str(&format!("{san} "));
        println!(
            "{:>3}{} {:<8} nota {:>7}  ({} nós)",
            numero,
            if branca { "." } else { "..." },
            san.to_string(),
            texto_nota(r.nota),
            r.nos
        );
    };

    let segundos = inicio.elapsed().as_secs_f64();
    println!("\nResultado: {resultado}");
    println!("\n{pgn}");
    println!(
        "\n{nos_total} nós em {segundos:.1} s ({:.0} mil nós/s)",
        nos_total as f64 / segundos / 1000.0
    );
}

// UCI mínimo: o suficiente para uma GUI (ou um script de torneio) usar o motor.
fn uci() {
    let mut pos = Chess::default();
    let mut historico: Vec<Zobrist64> = Vec::new();
    let mut saida = io::stdout();

    for linha in io::stdin().lock().lines() {
        let linha = linha.unwrap();
        let partes: Vec<&str> = linha.split_whitespace().collect();
        match partes.first().copied() {
            Some("uci") => {
                println!("id name xadrez-neural (etapa 1, material)");
                println!("id author Gabriel");
                println!("uciok");
            }
            Some("isready") => println!("readyok"),
            Some("ucinewgame") => {}
            Some("position") => {
                let i_lances = partes.iter().position(|&p| p == "moves");
                let fim_fen = i_lances.unwrap_or(partes.len());
                pos = if partes.get(1) == Some(&"fen") {
                    partes[2..fim_fen]
                        .join(" ")
                        .parse::<Fen>()
                        .unwrap()
                        .into_position(CastlingMode::Standard)
                        .unwrap()
                } else {
                    Chess::default()
                };
                historico.clear();
                if let Some(i) = i_lances {
                    for texto in &partes[i + 1..] {
                        let m = texto.parse::<UciMove>().unwrap().to_move(&pos).unwrap();
                        historico.push(hash(&pos));
                        pos.play_unchecked(m);
                    }
                }
            }
            Some("go") => {
                let profundidade = partes
                    .iter()
                    .position(|&p| p == "depth")
                    .and_then(|i| partes.get(i + 1)?.parse().ok())
                    .unwrap_or(PROFUNDIDADE_PADRAO);
                let r = Busca::nova(historico.clone()).melhor_lance(&pos, profundidade);
                println!("info depth {profundidade} score cp {} nodes {}", r.nota.clamp(-30000, 30000), r.nos);
                match r.lance {
                    Some(m) => println!("bestmove {}", UciMove::from_standard(m)),
                    None => println!("bestmove 0000"),
                }
            }
            Some("quit") => break,
            _ => {}
        }
        saida.flush().unwrap();
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("uci") => uci(),
        _ => {
            let profundidade = args.get(2).and_then(|p| p.parse().ok()).unwrap_or(PROFUNDIDADE_PADRAO);
            jogar_contra_si(profundidade.max(1));
        }
    }
}
