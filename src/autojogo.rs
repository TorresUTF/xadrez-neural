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
//   - a folha ou a partida chegou a um final de 3 peças (gabarito da 2b)
//     ou, com a pasta da Syzygy, a um final de até 5 peças.
//
// Com a Syzygy, cada rodada também treina em posições de finais sorteadas,
// tantas quanto as do autojogo. Sem isso a rede esquece os finais: o
// autojogo quase não passa por eles, e os alvos do TD são misturas
// "moles" das opiniões da própria rede, o que desmancha a certeza.
// =====================================================================

use crate::avaliacao::Avaliador;
use crate::busca::{Busca, MATE};
use crate::finais::{construir_todas, gabarito, Tabela, Valor};
use crate::syzygy;
use crate::rede::{entradas, Adam, Aleatorio, Rede, DERROTA, EMPATE, N_SAIDAS, VITORIA};
use shakmaty::{zobrist::Zobrist64, Chess, Color, EnPassantMode, Move, Outcome, Position};
use shakmaty_syzygy::Tablebase;
use std::time::Instant;

pub type Dist = [f32; N_SAIDAS];

// troca a perspectiva: V <-> D
pub fn trocar(p: Dist) -> Dist {
    [p[DERROTA], p[EMPATE], p[VITORIA]]
}

fn certeza(v: Valor) -> Dist {
    let mut d = [0.0; N_SAIDAS];
    d[v.classe()] = 1.0;
    d
}

fn classe_certa(classe: usize) -> Dist {
    let mut d = [0.0; N_SAIDAS];
    d[classe] = 1.0;
    d
}

pub fn hash(pos: &Chess) -> Zobrist64 {
    pos.zobrist_hash(EnPassantMode::Legal)
}

// De onde vem a verdade exata dos finais: o gabarito de 3 peças e,
// se houver, as tablebases Syzygy.
pub struct Verdade {
    tabelas: Vec<Tabela>,
    syzygy: Option<Tablebase<Chess>>,
}

impl Verdade {
    pub fn nova(syzygy: Option<Tablebase<Chess>>) -> Verdade {
        Verdade { tabelas: construir_todas(), syzygy }
    }

    pub fn valor(&self, pos: &Chess) -> Option<Valor> {
        if let Some(v) = gabarito(&self.tabelas, pos) {
            return Some(v);
        }
        let tb = self.syzygy.as_ref()?;
        if pos.board().occupied().count() > tb.max_pieces() {
            return None;
        }
        syzygy::consultar(tb, pos)
    }
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
pub fn fim(pos: &Chess, historico: &[Zobrist64], verdade: &Verdade, meios_lances: usize, cfg: &Config) -> Option<Valor> {
    if let Outcome::Known(r) = pos.outcome() {
        return Some(match r.winner() {
            None => Valor::Empate,
            Some(c) if c == pos.turn() => Valor::Vitoria,
            Some(_) => Valor::Derrota,
        });
    }
    if let Some(v) = verdade.valor(pos) {
        return Some(v); // final com valor exato: a verdade decide a partida
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
fn partida_de_treino(rede: &Rede, verdade: &Verdade, cfg: &Config, rng: &mut Aleatorio) -> (Vec<Exemplo>, f32) {
    let mut pos = Chess::default();
    let mut historico: Vec<Zobrist64> = Vec::new();
    let mut passos: Vec<Passo> = Vec::new();
    let mut meios_lances = 0;

    let final_ = loop {
        if let Some(v) = fim(&pos, &historico, verdade, meios_lances, cfg) {
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
            } else if let Some(g) = verdade.valor(&folha) {
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
    verdade: &Verdade,
) -> f32 {
    let cfg = Config { max_meios_lances: 300, ..CONFIG_PADRAO };
    let mut pos = Chess::default();
    let mut historico = Vec::new();
    let mut n = 0;
    loop {
        if let Some(v) = fim(&pos, &historico, verdade, n, &cfg) {
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

pub fn trocar_valor(v: Valor) -> Valor {
    match v {
        Valor::Vitoria => Valor::Derrota,
        Valor::Empate => Valor::Empate,
        Valor::Derrota => Valor::Vitoria,
    }
}

pub fn abertura(rng: &mut Aleatorio, meios_lances: usize) -> Vec<Move> {
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
    verdade: &Verdade,
    semente: u64,
) -> f32 {
    let mut rng = Aleatorio(semente);
    let aberturas: Vec<Vec<Move>> = (0..pares).map(|_| abertura(&mut rng, 6)).collect();
    let pontos: f32 = crate::paralelo::mapear(&aberturas, |ab| {
        partida(a, b, ab, profundidade, verdade) + (1.0 - partida(b, a, ab, profundidade, verdade))
    })
    .into_iter()
    .sum();
    pontos / (2 * pares) as f32
}

pub use crate::paralelo::threads;

// ---------------------------------------------------------------------
// O laço da Etapa 3
// ---------------------------------------------------------------------
// `taxa`: a taxa do Adam (1e-3 do zero; menor para afinar uma rede pronta).
// `congelar`: na rede crescida, treina só a coluna verde (ver rede.rs).
pub struct Opcoes {
    pub taxa: f32,
    pub congelar: bool,
}

pub fn treinar(rodadas: usize, partidas_por_rodada: usize, inicial: Option<&str>, arquivo: &str, pasta_syzygy: Option<&str>, opcoes: Opcoes) {
    use crate::avaliacao::Material;
    let cfg = CONFIG_PADRAO;
    let inicio = Instant::now();
    let verdade = Verdade::nova(pasta_syzygy.map(syzygy::abrir));
    // Posições fixas de finais (100 por combinação) para medir, a cada 10
    // rodadas, se a rede está esquecendo o que aprendeu com a Syzygy
    let combos = verdade.syzygy.as_ref().map(|tb| syzygy::combinacoes(tb.max_pieces().min(5))).unwrap_or_default();
    let finais_fixos: Vec<crate::treino::Exemplo> = match &verdade.syzygy {
        Some(tb) => syzygy::amostra(tb, &combos, 100, 0x0123456789ABCDEF).0,
        None => Vec::new(),
    };
    let mut rede = match inicial {
        Some(a) => Rede::carregar(a).unwrap_or_else(|e| panic!("não consegui ler {a}: {e}")),
        None => Rede::aleatoria(0x2545F4914F6CDD1D),
    };
    let original = rede.clone();
    if opcoes.congelar && !rede.congelar_antigas() {
        println!("Aviso: a rede tem uma coluna só, não há o que congelar (treinando tudo)");
    }
    let mut gs = rede.gradientes();
    let mut adam = Adam::novo(opcoes.taxa, &rede);
    println!(
        "Rede: {} ({} mil multiplicações por avaliação), taxa {}{}",
        rede.descricao(),
        rede.custo() / 1000,
        opcoes.taxa,
        if rede.congelado > 0 { ", colunas antigas congeladas" } else { "" }
    );
    let mut rng = Aleatorio(0xD1B54A32D192ED03);
    println!(
        "Autojogo: {rodadas} rodadas de {partidas_por_rodada} partidas, profundidade {}, λ = {}, {} threads",
        cfg.profundidade,
        cfg.lambda,
        threads()
    );
    let pares = 50;
    let medir = |rede: &Rede, rotulo: &str| {
        let contra_material = confronto(rede, &Material, pares, cfg.profundidade, &verdade, 7);
        let contra_original = confronto(rede, &original, pares, cfg.profundidade, &verdade, 7);
        let finais = if finais_fixos.is_empty() {
            String::new()
        } else {
            let fixos: Vec<&crate::treino::Exemplo> = finais_fixos.iter().collect();
            let m = crate::treino::medir(rede, &fixos);
            format!("   finais: acerto {:.1}%  P(certa) {:.3}", 100.0 * m.acerto, m.p_certa)
        };
        println!(
            "  [{rotulo}] pontos contra o material: {:.0}%   contra a rede inicial: {:.0}%{finais}   ({:.0} s)",
            100.0 * contra_material,
            100.0 * contra_original,
            inicio.elapsed().as_secs_f64()
        );
    };
    medir(&rede, "início");

    for rodada in 1..=rodadas {
        // 1) gera partidas em paralelo, cada thread com a sua semente
        let sementes: Vec<u64> = (0..partidas_por_rodada).map(|_| (rng.proximo().to_bits() as u64) << 32 | rodada as u64).collect();
        let resultados: Vec<(Vec<Exemplo>, f32)> =
            crate::paralelo::mapear(&sementes, |&sem| partida_de_treino(&rede, &verdade, &cfg, &mut Aleatorio(sem | 1)));
        let (mut v, mut e, mut d) = (0, 0, 0);
        for (_, r) in &resultados {
            if *r > 0.0 { v += 1 } else if *r < 0.0 { d += 1 } else { e += 1 }
        }
        let mut exemplos: Vec<Exemplo> = resultados.into_iter().flat_map(|(ex, _)| ex).collect();
        let do_autojogo = exemplos.len();

        // finais sorteados da Syzygy, tantos quanto as posições do autojogo
        if let Some(tb) = &verdade.syzygy {
            let por_combinacao = do_autojogo.div_ceil(combos.len());
            let semente = (rodada as u64).wrapping_mul(0x9E3779B97F4A7C15);
            let (finais, _) = syzygy::amostra(tb, &combos, por_combinacao, semente);
            exemplos.extend(finais.into_iter().map(|f| Exemplo { ligadas: f.ligadas, alvo: classe_certa(f.classe) }));
        }

        // 2) treina uma passada nos exemplos desta rodada, em lotes
        for i in (1..exemplos.len()).rev() {
            let j = ((rng.proximo() as f64 + 1.0) / 2.0 * (i + 1) as f64) as usize;
            exemplos.swap(i, j.min(i));
        }
        let mut erro = 0.0f64;
        for lote in exemplos.chunks(256) {
            erro += rede.treinar_lote(&mut gs, &mut adam, lote, |ex| (&ex.ligadas, ex.alvo));
        }
        println!(
            "rodada {rodada:3}: {} partidas (brancas {v} / empates {e} / pretas {d}), {do_autojogo} posições + {} de finais, erro {:.4}",
            partidas_por_rodada,
            exemplos.len() - do_autojogo,
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
        let verdade = Verdade::nova(None);
        let rede = Rede::aleatoria(3);
        let cfg = Config { profundidade: 1, lances_aleatorios: 4, lambda: 0.7, max_meios_lances: 60 };
        let (exemplos, _) = partida_de_treino(&rede, &verdade, &cfg, &mut Aleatorio(5));
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
        let verdade = Verdade::nova(None);
        let rede = Rede::aleatoria(4);
        let cfg = Config { profundidade: 1, lances_aleatorios: 4, lambda: 1.0, max_meios_lances: 40 };
        let (exemplos, resultado) = partida_de_treino(&rede, &verdade, &cfg, &mut Aleatorio(9));
        assert_eq!(resultado, 0.0);
        let empates = exemplos.iter().filter(|e| e.alvo[EMPATE] > 0.999).count();
        for e in &exemplos {
            assert!(e.alvo.iter().any(|&x| x > 0.999), "{:?}", e.alvo);
        }
        assert!(empates > exemplos.len() / 2);
    }

    // Só roda com tablebases em ./syzygy (ou na variável SYZYGY): a Syzygy
    // concorda com o gabarito nos finais de 3 peças e responde finais de 5.
    #[test]
    fn verdade_com_syzygy() {
        let dir = std::env::var("SYZYGY").unwrap_or_else(|_| "syzygy".into());
        let mut tb = Tablebase::new();
        if !std::path::Path::new(&dir).is_dir() || tb.add_directory(&dir).unwrap() == 0 {
            eprintln!("(sem tablebases em {dir}: teste pulado)");
            return;
        }
        let verdade = Verdade::nova(Some(tb));
        let pos = |fen: &str| -> Chess {
            fen.parse::<shakmaty::fen::Fen>().unwrap().into_position(shakmaty::CastlingMode::Standard).unwrap()
        };
        assert_eq!(verdade.valor(&pos("8/8/8/8/8/2k5/8/K6Q w - - 0 1")), Some(Valor::Vitoria));
        assert_eq!(verdade.valor(&pos("7r/8/8/3k4/8/8/8/K5R1 w - - 0 1")), Some(Valor::Empate)); // KRvKR
        assert_eq!(verdade.valor(&Chess::default()), None);
        let cfg = Config { profundidade: 1, lances_aleatorios: 4, lambda: 0.7, max_meios_lances: 60 };
        let (exemplos, _) = partida_de_treino(&Rede::aleatoria(3), &verdade, &cfg, &mut Aleatorio(5));
        assert!(exemplos.iter().all(|e| (e.alvo.iter().sum::<f32>() - 1.0).abs() < 1e-4));
    }
}
