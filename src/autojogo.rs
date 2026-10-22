// =====================================================================
// AUTOJOGO — Etapa 3: a rede aprende jogando contra si mesma (TD-Leaf)
// =====================================================================
//
// A única verdade absoluta numa partida é o RESULTADO no fim. O TD
// (diferença temporal) leva essa verdade para trás, lance a lance:
// a avaliação de cada posição é puxada em direção à avaliação da
// posição seguinte, e a última em direção ao resultado.
//
// Alvo de cada passo t (o "retorno λ"), calculado do fim para o começo:
//
//      G_ultimo = resultado
//      G_t      = (1 - λ) · V_{t+1}  +  λ · G_{t+1}
//
// V_{t+1} é o que a rede achou do passo seguinte. Com λ = 0 cada posição
// só olha a vizinha (TD puro); com λ = 1 todas olham direto o resultado
// (Monte Carlo). Usamos λ = 0,7, como no KnightCap.
//
// Tudo em três classes (V, E, D). Como a vez alterna, "vitória para quem
// joga em t+1" é "derrota para quem joga em t": trocar a perspectiva é
// só trocar V com D.
//
// O "LEAF" (folha): a nota que a busca dá a uma posição é a avaliação da
// FOLHA da linha principal, não da posição em si. Então é a folha que
// recebe o gradiente. Assim a rede aprende a avaliar as posições que a
// busca realmente olha.
//
// Verdade exata quando existe:
//   - a busca achou mate forçado -> alvo é vitória ou derrota certa;
//   - a folha ou a partida chegou a um final de 3 peças -> gabarito da 2b.
// =====================================================================

use crate::avaliacao::Avaliador;
use crate::busca::{Busca, MATE};
use crate::finais::{construir_todas, gabarito, Tabela, Valor};
use crate::rede::{entradas, Adam, Aleatorio, Rede, DERROTA, EMPATE, N_SAIDAS, VITORIA};
use shakmaty::{zobrist::Zobrist64, Chess, Color, EnPassantMode, Move, Outcome, Position};
use std::time::Instant;

type Dist = [f32; N_SAIDAS];

// troca a perspectiva: V <-> D
fn trocar(p: Dist) -> Dist {
    [p[DERROTA], p[EMPATE], p[VITORIA]]
}

fn certeza(v: Valor) -> Dist {
    let mut d = [0.0; N_SAIDAS];
    d[v.classe()] = 1.0;
    d
}

fn hash(pos: &Chess) -> Zobrist64 {
    pos.zobrist_hash(EnPassantMode::Legal)
}

pub struct Config {
    pub profundidade: u32,
    pub lances_aleatorios: usize, // meios-lances sorteados no começo (variedade)
    pub lambda: f32,
    pub max_meios_lances: usize,
}

pub const CONFIG_PADRAO: Config = Config { profundidade: 2, lances_aleatorios: 8, lambda: 0.7, max_meios_lances: 300 };

// Um passo da partida, guardado para o treino
struct Passo {
    folha: Vec<usize>,      // entradas da folha, do ponto de vista de quem joga NA FOLHA
    folha_inverte: bool,    // quem joga na folha é o adversário de quem joga no passo?
    v: Dist,                // avaliação da folha, do ponto de vista de quem joga no passo
    exato: bool,            // v é verdade (mate ou gabarito), não opinião da rede
}

pub struct Exemplo {
    pub ligadas: Vec<usize>,
    pub alvo: Dist,
}

// Fim de partida: o valor para quem joga agora, se acabou
fn fim(pos: &Chess, historico: &[Zobrist64], tabelas: &[Tabela], meios_lances: usize, cfg: &Config) -> Option<Valor> {
    if let Outcome::Known(r) = pos.outcome() {
        return Some(match r.winner() {
            None => Valor::Empate,
            Some(c) if c == pos.turn() => Valor::Vitoria,
            Some(_) => Valor::Derrota,
        });
    }
    if let Some(v) = gabarito(tabelas, pos) {
        return Some(v); // final de 3 peças: o gabarito decide a partida
    }
    let h = hash(pos);
    if historico.iter().filter(|&&x| x == h).count() >= 2 || pos.halfmoves() >= 100 || meios_lances >= cfg.max_meios_lances {
        return Some(Valor::Empate);
    }
    None
}

fn lance_aleatorio(pos: &Chess, rng: &mut Aleatorio) -> Move {
    let lances = pos.legal_moves();
    let k = ((rng.proximo() + 1.0) / 2.0 * lances.len() as f32) as usize;
    lances[k.min(lances.len() - 1)]
}

// Joga uma partida e devolve os exemplos de treino e o resultado (para as brancas)
fn partida_de_treino(rede: &Rede, tabelas: &[Tabela], cfg: &Config, rng: &mut Aleatorio) -> (Vec<Exemplo>, f32) {
    let mut pos = Chess::default();
    let mut historico: Vec<Zobrist64> = Vec::new();
    let mut passos: Vec<Passo> = Vec::new();
    let mut meios_lances = 0;

    let final_ = loop {
        if let Some(v) = fim(&pos, &historico, tabelas, meios_lances, cfg) {
            break (pos.turn(), v);
        }
        let m = if meios_lances < cfg.lances_aleatorios {
            lance_aleatorio(&pos, rng)
        } else {
            let r = Busca::nova(rede, historico.clone()).melhor_lance(&pos, cfg.profundidade);
            let mut folha = pos.clone();
            for &m in &r.pv {
                folha = folha.play(m).unwrap();
            }
            let inverte = r.pv.len() % 2 == 1;
            let (v, exato) = if r.nota.abs() > MATE - 1000 {
                (certeza(if r.nota > 0 { Valor::Vitoria } else { Valor::Derrota }), true)
            } else if let Some(g) = gabarito(tabelas, &folha) {
                let d = certeza(g);
                (if inverte { trocar(d) } else { d }, true)
            } else {
                let d = rede.probabilidades_de(&entradas(folha.board(), folha.turn()));
                (if inverte { trocar(d) } else { d }, false)
            };
            passos.push(Passo { folha: entradas(folha.board(), folha.turn()), folha_inverte: inverte, v, exato });
            r.lance.unwrap()
        };
        historico.push(hash(&pos));
        pos = pos.play(m).unwrap();
        meios_lances += 1;
    };

    // Resultado do ponto de vista de quem jogava no último passo.
    // `final_` é para quem jogaria depois do fim; os passos alternam a vez.
    let (vez_final, valor_final) = final_;
    let resultado_brancas = match (vez_final, valor_final) {
        (_, Valor::Empate) => 0.0,
        (Color::White, Valor::Vitoria) | (Color::Black, Valor::Derrota) => 1.0,
        _ => -1.0,
    };

    // Retorno λ, do fim para o começo. O último passo jogou e passou a vez,
    // então para ele o resultado está na perspectiva trocada.
    let mut exemplos = Vec::with_capacity(passos.len());
    let mut g = trocar(certeza(valor_final));
    let mut v_seguinte: Option<Dist> = None;
    for p in passos.iter().rev() {
        if let Some(vs) = v_seguinte {
            // vs e g estão na perspectiva do passo seguinte: mistura e troca
            let mistura: Dist = std::array::from_fn(|k| (1.0 - cfg.lambda) * vs[k] + cfg.lambda * g[k]);
            g = trocar(mistura);
        }
        if p.exato {
            g = p.v;
        }
        // (folhas exatas também entram: é verdade, não opinião)
        let alvo_folha = if p.folha_inverte { trocar(g) } else { g };
        exemplos.push(Exemplo { ligadas: p.folha.clone(), alvo: alvo_folha });
        v_seguinte = Some(p.v);
    }
    (exemplos, resultado_brancas)
}

// ---------------------------------------------------------------------
// MEDIÇÃO: dois avaliadores jogam pares de partidas (cada abertura
// sorteada é jogada duas vezes, trocando as cores). Devolve os pontos
// do primeiro, de 0 a 1 (vitória 1, empate 0,5).
// ---------------------------------------------------------------------
fn partida<A: Avaliador, B: Avaliador>(
    brancas: &A,
    pretas: &B,
    abertura: &[Move],
    profundidade: u32,
    tabelas: &[Tabela],
) -> f32 {
    let cfg = Config { max_meios_lances: 300, ..CONFIG_PADRAO };
    let mut pos = Chess::default();
    let mut historico = Vec::new();
    let mut n = 0;
    loop {
        if let Some(v) = fim(&pos, &historico, tabelas, n, &cfg) {
            let para_brancas = if pos.turn().is_white() { v } else { trocar_valor(v) };
            return match para_brancas {
                Valor::Vitoria => 1.0,
                Valor::Empate => 0.5,
                Valor::Derrota => 0.0,
            };
        }
        let m = if n < abertura.len() {
            abertura[n]
        } else if pos.turn().is_white() {
            Busca::nova(brancas, historico.clone()).melhor_lance(&pos, profundidade).lance.unwrap()
        } else {
            Busca::nova(pretas, historico.clone()).melhor_lance(&pos, profundidade).lance.unwrap()
        };
        historico.push(hash(&pos));
        pos = pos.play(m).unwrap();
        n += 1;
    }
}

fn trocar_valor(v: Valor) -> Valor {
    match v {
        Valor::Vitoria => Valor::Derrota,
        Valor::Empate => Valor::Empate,
        Valor::Derrota => Valor::Vitoria,
    }
}

fn abertura(rng: &mut Aleatorio, meios_lances: usize) -> Vec<Move> {
    let mut pos = Chess::default();
    let mut lances = Vec::new();
    while lances.len() < meios_lances && !pos.legal_moves().is_empty() {
        let m = lance_aleatorio(&pos, rng);
        lances.push(m);
        pos = pos.play(m).unwrap();
    }
    lances
}

pub fn confronto<A: Avaliador + Sync, B: Avaliador + Sync>(
    a: &A,
    b: &B,
    pares: usize,
    profundidade: u32,
    tabelas: &[Tabela],
    semente: u64,
) -> f32 {
    let mut rng = Aleatorio(semente);
    let aberturas: Vec<Vec<Move>> = (0..pares).map(|_| abertura(&mut rng, 6)).collect();
    let pontos: f32 = std::thread::scope(|s| {
        let tarefas: Vec<_> = aberturas
            .chunks(pares.div_ceil(threads()))
            .map(|parte| {
                s.spawn(move || {
                    parte
                        .iter()
                        .map(|ab| partida(a, b, ab, profundidade, tabelas) + (1.0 - partida(b, a, ab, profundidade, tabelas)))
                        .sum::<f32>()
                })
            })
            .collect();
        tarefas.into_iter().map(|t| t.join().unwrap()).sum()
    });
    pontos / (2 * pares) as f32
}

fn threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
}

// ---------------------------------------------------------------------
// O laço da Etapa 3
// ---------------------------------------------------------------------
pub fn treinar(rodadas: usize, partidas_por_rodada: usize, inicial: Option<&str>, arquivo: &str) {
    use crate::avaliacao::Material;
    let cfg = CONFIG_PADRAO;
    let inicio = Instant::now();
    let tabelas = construir_todas();
    let mut rede = match inicial {
        Some(a) => Rede::carregar(a).unwrap_or_else(|e| panic!("não consegui ler {a}: {e}")),
        None => Rede::aleatoria(0x2545F4914F6CDD1D),
    };
    let original = rede.clone();
    let mut g = Rede::zerada();
    let mut adam = Adam::novo(1e-3);
    let mut rng = Aleatorio(0xD1B54A32D192ED03);
    println!(
        "Autojogo: {rodadas} rodadas de {partidas_por_rodada} partidas, profundidade {}, λ = {}, {} threads",
        cfg.profundidade,
        cfg.lambda,
        threads()
    );
    let pares = 50;
    let medir = |rede: &Rede, rotulo: &str| {
        let contra_material = confronto(rede, &Material, pares, cfg.profundidade, &tabelas, 7);
        let contra_original = confronto(rede, &original, pares, cfg.profundidade, &tabelas, 7);
        println!(
            "  [{rotulo}] pontos contra o material: {:.0}%   contra a rede inicial: {:.0}%   ({:.0} s)",
            100.0 * contra_material,
            100.0 * contra_original,
            inicio.elapsed().as_secs_f64()
        );
    };
    medir(&rede, "início");

    for rodada in 1..=rodadas {
        // 1) gera partidas em paralelo, cada thread com a sua semente
        let sementes: Vec<u64> = (0..partidas_por_rodada).map(|_| (rng.proximo().to_bits() as u64) << 32 | rodada as u64).collect();
        let resultados: Vec<(Vec<Exemplo>, f32)> = std::thread::scope(|s| {
            let rede = &rede;
            let tabelas = &tabelas;
            let cfg = &cfg;
            let tarefas: Vec<_> = sementes
                .chunks(partidas_por_rodada.div_ceil(threads()))
                .map(|parte| {
                    s.spawn(move || {
                        parte
                            .iter()
                            .map(|&sem| partida_de_treino(rede, tabelas, cfg, &mut Aleatorio(sem | 1)))
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            tarefas.into_iter().flat_map(|t| t.join().unwrap()).collect()
        });
        let (mut v, mut e, mut d) = (0, 0, 0);
        for (_, r) in &resultados {
            if *r > 0.0 { v += 1 } else if *r < 0.0 { d += 1 } else { e += 1 }
        }
        let mut exemplos: Vec<Exemplo> = resultados.into_iter().flat_map(|(ex, _)| ex).collect();

        // 2) treina uma passada nos exemplos desta rodada, em lotes
        for i in (1..exemplos.len()).rev() {
            let j = ((rng.proximo() as f64 + 1.0) / 2.0 * (i + 1) as f64) as usize;
            exemplos.swap(i, j.min(i));
        }
        let mut erro = 0.0f64;
        for lote in exemplos.chunks(256) {
            g.zerar();
            for ex in lote {
                erro += rede.retropropagar(&ex.ligadas, ex.alvo, &mut g) as f64;
            }
            rede.aplicar_adam(&mut g, &mut adam, lote.len());
        }
        println!(
            "rodada {rodada:3}: {} partidas (brancas {v} / empates {e} / pretas {d}), {} posições, erro {:.4}",
            partidas_por_rodada,
            exemplos.len(),
            erro / exemplos.len().max(1) as f64
        );
        if rodada % 10 == 0 || rodada == rodadas {
            medir(&rede, &format!("rodada {rodada}"));
            if let Err(err) = rede.salvar(arquivo) {
                println!("  não consegui salvar {arquivo}: {err}");
            }
        }
    }
    println!("Pesos salvos em {arquivo}");
}

#[cfg(test)]
mod testes {
    use super::*;

    #[test]
    fn trocar_perspectiva() {
        assert_eq!(trocar([0.7, 0.2, 0.1]), [0.1, 0.2, 0.7]);
    }

    #[test]
    fn partida_de_treino_gera_alvos_validos() {
        let tabelas = construir_todas();
        let rede = Rede::aleatoria(3);
        let cfg = Config { profundidade: 1, lances_aleatorios: 4, lambda: 0.7, max_meios_lances: 60 };
        let (exemplos, _) = partida_de_treino(&rede, &tabelas, &cfg, &mut Aleatorio(5));
        assert!(!exemplos.is_empty());
        for e in &exemplos {
            // cada alvo é uma distribuição: soma 1, nada negativo
            assert!((e.alvo.iter().sum::<f32>() - 1.0).abs() < 1e-4);
            assert!(e.alvo.iter().all(|&x| x >= 0.0));
        }
    }

    #[test]
    fn lambda_um_e_o_resultado_puro() {
        // Com λ = 1 a opinião da rede some do alvo: só sobra o resultado da
        // partida (ou uma verdade exata do caminho), sempre uma classe certa.
        // Esta partida acaba em empate pelo limite de lances.
        let tabelas = construir_todas();
        let rede = Rede::aleatoria(4);
        let cfg = Config { profundidade: 1, lances_aleatorios: 4, lambda: 1.0, max_meios_lances: 40 };
        let (exemplos, resultado) = partida_de_treino(&rede, &tabelas, &cfg, &mut Aleatorio(9));
        assert_eq!(resultado, 0.0);
        let empates = exemplos.iter().filter(|e| e.alvo[EMPATE] > 0.999).count();
        for e in &exemplos {
            assert!(e.alvo.iter().any(|&x| x > 0.999), "{:?}", e.alvo);
        }
        assert!(empates > exemplos.len() / 2);
    }
}
