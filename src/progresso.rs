// =====================================================================
// PROGRESSO — treinar a coluna rosa: "quantos meios-lances até acabar?"
// =====================================================================
//
// A rede V/E/D diz QUEM ganha, não COMO. Num final ganho todas as
// posições valem "vitória", e a busca não vê diferença entre encurralar
// o rei adversário e girar em volta dele. A coluna de progresso (ver
// rede.rs) aprende a distância até o fim; a busca, entre posições
// ganhas, passa a preferir a mais perto do fim (ver avaliacao.rs).
//
// DE ONDE VEM O RÓTULO
// O ideal seria a distância até o MATE. As tabelas Syzygy não a têm: a
// .rtbz guarda a DTZ, a distância até ZERAR a contagem dos 50 lances
// (uma captura ou lance de peão). A DTZ não serve diretamente como
// progresso, porque recomeça a cada captura: em KQ vs KR, a posição
// logo antes de tomar a torre tem DTZ 1, e logo depois (KQ vs K) DTZ ~15.
// Uma busca que minimizasse a DTZ evitaria tomar a torre!
// Então fazemos o seguinte: a partir de uma posição decidida, jogamos a
// linha recomendada pela Syzygy (que vence pela DTZ) até o mate, e cada
// posição dessa linha recebe como rótulo os meios-lances que faltavam.
// Isso é a duração de uma vitória ótima pela regra dos 50 lances, uma
// distância até o mate que vale igual antes e depois das capturas. De
// brinde, uma linha dá dezenas de exemplos.
// Sem as .rtbz, usamos os finais de 3 peças de finais.rs (KQK e KRK), que
// têm a distância exata até o mate.
//
// Alvo da rede = meios-lances / 100 (até 120 meios-lances).
// =====================================================================

use crate::busca::Busca;
use crate::finais::{gabarito_com_distancia, Tabela, Valor};
use crate::rede::{entradas, Adam, Aleatorio, Alvo, Rede};
use crate::syzygy::{self, Material};
use crate::treino::embaralhar;
use shakmaty::{zobrist::Zobrist64, Chess, EnPassantMode, Position, Role};
use shakmaty_syzygy::Tablebase;
use std::time::Instant;

const MAX_MEIOS_LANCES: u32 = 120;

pub struct Exemplo {
    pub ligadas: Vec<usize>,
    pub meios_lances: u32,
    pub pecas: usize,
}

fn alvo(meios_lances: u32) -> f32 {
    meios_lances.min(MAX_MEIOS_LANCES) as f32 / 100.0
}

fn exemplo(pos: &Chess, meios_lances: u32) -> Exemplo {
    Exemplo { ligadas: entradas(pos.board(), pos.turn()), meios_lances, pecas: pos.board().occupied().count() }
}

// De onde vêm os rótulos
pub enum Fonte {
    Finais(Vec<Tabela>), // KQK e KRK: distância exata até o mate
    Syzygy(Tablebase<Chess>),
}

impl Fonte {
    // Meios-lances até o fim de uma posição decidida, ou None se ela não
    // está coberta ou é empate
    pub fn distancia(&self, pos: &Chess) -> Option<u32> {
        match self {
            Fonte::Finais(t) => match gabarito_com_distancia(t, pos)? {
                (Valor::Empate, _) => None,
                (_, d) => Some(d as u32),
            },
            Fonte::Syzygy(tb) => Some(linha_syzygy(tb, pos)?.first()?.1),
        }
    }

    pub fn valor(&self, pos: &Chess) -> Option<Valor> {
        match self {
            Fonte::Finais(t) => Some(gabarito_com_distancia(t, pos)?.0),
            Fonte::Syzygy(tb) => syzygy::consultar(tb, pos),
        }
    }
}

// A linha da Syzygy a partir de `pos` até o mate, cada posição com os
// meios-lances que faltam. None: posição empatada (pela regra dos 50
// lances), fora das tabelas ou sem a .rtbz.
fn linha_syzygy(tb: &Tablebase<Chess>, pos: &Chess) -> Option<Vec<(Chess, u32)>> {
    if !matches!(syzygy::consultar(tb, pos)?, Valor::Vitoria | Valor::Derrota) {
        return None;
    }
    let mut linha = Vec::new();
    let mut p = pos.clone();
    while !p.legal_moves().is_empty() {
        if linha.len() > 2 * MAX_MEIOS_LANCES as usize || p.is_insufficient_material() {
            return None;
        }
        let (m, _) = tb.best_move(&p).ok()??;
        linha.push(p.clone());
        p.play_unchecked(m);
    }
    if !p.is_check() {
        return None; // afogamento: não era vitória de verdade
    }
    let n = linha.len() as u32;
    Some(linha.into_iter().enumerate().map(|(i, p)| (p, n - i as u32)).collect())
}

// Exemplos de cada combinação de material: posições sorteadas e as
// linhas da Syzygy a partir delas, até `por_combinacao` por combinação
fn amostra_syzygy(tb: &Tablebase<Chess>, combos: &[Material], por_combinacao: usize, semente: u64) -> Vec<Exemplo> {
    let indices: Vec<usize> = (0..combos.len()).collect();
    crate::paralelo::mapear(&indices, |&k| {
        let mut rng = Aleatorio(semente ^ ((k as u64 + 1) << 32) | 1);
        let mut v = Vec::new();
        let mut tentativas = 0;
        while v.len() < por_combinacao && tentativas < 20 * por_combinacao {
            tentativas += 1;
            let Some(pos) = syzygy::posicao_aleatoria(&combos[k], &mut rng) else { continue };
            if let Some(linha) = linha_syzygy(tb, &pos) {
                v.extend(linha.iter().map(|(p, d)| exemplo(p, *d)));
            }
        }
        v
    })
    .into_iter()
    .flatten()
    .collect()
}

fn todos_dos_finais(tabelas: &[Tabela]) -> Vec<Exemplo> {
    tabelas.iter().flat_map(|t| t.exemplos_com_distancia().map(|(p, _, d)| exemplo(&p, d as u32))).collect()
}

// Erro médio, em meios-lances, por número de peças (índice = peças)
fn medir(rede: &Rede, exemplos: &[&Exemplo]) -> (f64, Vec<(usize, f64, usize)>) {
    let mut por_pecas = [(0.0f64, 0usize); 8];
    for e in exemplos {
        let y = rede.progresso_de(&e.ligadas).unwrap();
        let erro = ((y * 100.0) as f64 - e.meios_lances.min(MAX_MEIOS_LANCES) as f64).abs();
        por_pecas[e.pecas.min(7)].0 += erro;
        por_pecas[e.pecas.min(7)].1 += 1;
    }
    let total = por_pecas.iter().map(|p| p.0).sum::<f64>() / exemplos.len().max(1) as f64;
    let grupos = por_pecas.iter().enumerate().filter(|(_, p)| p.1 > 0).map(|(k, p)| (k, p.0 / p.1 as f64, p.1)).collect();
    (total, grupos)
}

pub fn treinar_exemplos(rede: &mut Rede, mut todos: Vec<Exemplo>, epocas: usize, rng: &mut Aleatorio) {
    embaralhar(&mut todos, rng);
    let (validacao, treino) = todos.split_at(todos.len() / 10);
    let validacao: Vec<&Exemplo> = validacao.iter().collect();
    let mut treino: Vec<&Exemplo> = treino.iter().collect();
    println!("{} exemplos de treino, {} de validação", treino.len(), validacao.len());
    let mut gs = rede.gradientes();
    let mut adam = Adam::novo(1e-3, rede);
    println!("época  0: erro médio {:.1} meios-lances", medir(rede, &validacao).0);
    for n in 1..=epocas {
        let t0 = Instant::now();
        embaralhar(&mut treino, rng);
        for lote in treino.chunks(256) {
            rede.treinar_lote(&mut gs, &mut adam, lote, |e| (&e.ligadas, Alvo::Progresso(alvo(e.meios_lances))));
        }
        let (erro, grupos) = medir(rede, &validacao);
        let grupos: Vec<String> = grupos.iter().map(|(k, e, _)| format!("{k} peças {e:.1}")).collect();
        println!("época {n:2}: erro médio {erro:.1} meios-lances ({})  ({:.1} s)", grupos.join(", "), t0.elapsed().as_secs_f64());
    }
}

// ---------------------------------------------------------------------
// O teste que importa: partidas a partir de finais ganhos, a busca
// jogando dos dois lados. Quantas terminam em mate, e em quantos lances?
// ---------------------------------------------------------------------
pub struct Demonstracao {
    pub mates: usize,
    pub partidas: usize,
    pub meios_lances_mates: usize, // soma, só das partidas com mate
    pub otimo: usize,              // soma da distância com jogo perfeito, das mesmas
}

fn jogar_ate_o_fim(rede: &Rede, inicio: &Chess, profundidade: u32) -> Option<usize> {
    let mut pos = inicio.clone();
    let mut historico: Vec<Zobrist64> = Vec::new();
    for meios in 0..300 {
        if pos.is_checkmate() {
            return Some(meios);
        }
        let h = pos.zobrist_hash(EnPassantMode::Legal);
        let repeticoes = historico.iter().filter(|&&x| x == h).count();
        if pos.legal_moves().is_empty() || pos.halfmoves() >= 100 || repeticoes >= 2 || pos.is_insufficient_material() {
            return None;
        }
        let m = Busca::nova(rede, historico.clone()).melhor_lance(&pos, profundidade).lance?;
        historico.push(h);
        pos.play_unchecked(m);
    }
    None
}

pub fn demonstrar(rede: &Rede, fonte: &Fonte, posicoes: &[Chess], profundidade: u32) -> Demonstracao {
    let resultados = crate::paralelo::mapear(posicoes, |p| (jogar_ate_o_fim(rede, p, profundidade), fonte.distancia(p).unwrap_or(0)));
    let mut d = Demonstracao { mates: 0, partidas: posicoes.len(), meios_lances_mates: 0, otimo: 0 };
    for (r, otimo) in resultados {
        if let Some(m) = r {
            d.mates += 1;
            d.meios_lances_mates += m;
            d.otimo += otimo as usize;
        }
    }
    d
}

// Posições ganhas para quem joga (sorteadas), para a demonstração
fn posicoes_ganhas(fonte: &Fonte, combos: &[Material], quantas: usize, semente: u64) -> Vec<Chess> {
    let mut rng = Aleatorio(semente);
    let mut v = Vec::new();
    let mut tentativas = 0;
    while v.len() < quantas && tentativas < 1000 * quantas {
        tentativas += 1;
        let m = &combos[tentativas % combos.len()];
        let Some(pos) = syzygy::posicao_aleatoria(m, &mut rng) else { continue };
        // o lado forte joga, e a vitória não é imediata
        if fonte.valor(&pos) == Some(Valor::Vitoria) && fonte.distancia(&pos).is_some_and(|d| d >= 9) {
            v.push(pos);
        }
    }
    v
}

fn imprimir(rotulo: &str, d: &Demonstracao) {
    let media = |s: usize| s as f64 / d.mates.max(1) as f64;
    println!(
        "  {rotulo:<16} mate em {}/{} partidas, {:.1} meios-lances em média (o jogo perfeito levaria {:.1})",
        d.mates,
        d.partidas,
        media(d.meios_lances_mates),
        media(d.otimo)
    );
}

// O comando: acrescenta a coluna rosa (se ainda não houver), congela o
// resto, treina, salva e mostra a diferença em partidas.
pub fn treinar(inicial: &str, arquivo: &str, pasta_syzygy: Option<&str>, por_combinacao: usize, epocas: usize) {
    let inicio = Instant::now();
    let base = Rede::carregar(inicial).unwrap_or_else(|e| panic!("não consegui ler {inicial}: {e}"));
    let mut rede = if base.tem_progresso() { base } else { base.crescer_progresso(0xB5297A4D) };
    rede.congelar_antigas();
    println!("Rede: {} (treinando só a coluna de progresso)", rede.descricao());
    let mut rng = Aleatorio(0x68E31DA4);

    let (fonte, combos, exemplos) = match pasta_syzygy {
        Some(pasta) => {
            let tb = syzygy::abrir(pasta);
            let combos: Vec<Material> = syzygy::combinacoes(tb.max_pieces().min(5));
            // sem a .rtbz não há best_move: conferimos numa posição qualquer de KQK
            let teste = syzygy::posicao_aleatoria(&combos[0], &mut rng).unwrap();
            if let Err(e) = tb.best_move(&teste) {
                eprintln!("A Syzygy respondeu: {e}");
                eprintln!("Faltam as tabelas de distância (.rtbz). Baixe com:");
                eprintln!("  powershell -ExecutionPolicy Bypass -File baixar_syzygy.ps1 -Dtz");
                std::process::exit(1);
            }
            println!("{} combinações de material, até {por_combinacao} posições de cada (linhas da Syzygy até o mate)", combos.len());
            let exemplos = amostra_syzygy(&tb, &combos, por_combinacao, 0x2F0B5A1C);
            (Fonte::Syzygy(tb), combos, exemplos)
        }
        None => {
            println!("Sem pasta da Syzygy: usando KQK e KRK (distância exata até o mate)");
            let tabelas = vec![Tabela::construir(Role::Queen, &[]), Tabela::construir(Role::Rook, &[])];
            let exemplos = todos_dos_finais(&tabelas);
            let combos = vec![
                Material { brancas: vec![Role::Queen], pretas: vec![] },
                Material { brancas: vec![Role::Rook], pretas: vec![] },
            ];
            (Fonte::Finais(tabelas), combos, exemplos)
        }
    };
    println!("{} exemplos em {:.0} s\n", exemplos.len(), inicio.elapsed().as_secs_f64());
    treinar_exemplos(&mut rede, exemplos, epocas, &mut rng);
    match rede.salvar(arquivo) {
        Ok(()) => println!("\nPesos salvos em {arquivo}"),
        Err(e) => println!("\nNão consegui salvar {arquivo}: {e}"),
    }

    // Só finais com 4 peças ou mais na Syzygy (os de 3 já são triviais lá)
    let grandes: Vec<Material> = combos.iter().filter(|m| m.n_pecas() >= 4).cloned().collect();
    let combos_demo = if grandes.is_empty() { combos } else { grandes };
    let posicoes = posicoes_ganhas(&fonte, &combos_demo, 100, 0x5EED);
    println!("\nPartidas a partir de {} finais ganhos (busca de profundidade 3 dos dois lados):", posicoes.len());
    imprimir("sem progresso:", &demonstrar(&rede.sem_progresso(), &fonte, &posicoes, 3));
    imprimir("com progresso:", &demonstrar(&rede, &fonte, &posicoes, 3));
    println!("({:.0} s no total)", inicio.elapsed().as_secs_f64());
}

#[cfg(test)]
mod testes {
    use super::*;
    use crate::avaliacao::Avaliador;
    use crate::rede::N_SAIDAS;

    // Uma rede que tem certeza de que quem joga ganha (viés grande na
    // saída de valor), com coluna de progresso
    fn rede_ganhando() -> Rede {
        let mut r = Rede::aleatoria(1).crescer_progresso(2);
        let saida = r.colunas[0].densas.last_mut().unwrap();
        saida.w.iter_mut().for_each(|w| *w = 0.0);
        saida.b.copy_from_slice(&[20.0, 0.0, -20.0][..N_SAIDAS]);
        r
    }

    #[test]
    fn progresso_aprende_distancias_do_kqk() {
        // Uma época em parte do KQK: o erro médio tem que cair bastante
        let kqk = Tabela::construir(Role::Queen, &[]);
        let todos: Vec<Exemplo> = kqk.exemplos_com_distancia().step_by(5).map(|(p, _, d)| exemplo(&p, d as u32)).collect();
        let mut rede = Rede::aleatoria(3).crescer_progresso(4);
        rede.congelar_antigas();
        let amostra: Vec<Exemplo> = kqk.exemplos_com_distancia().skip(3).step_by(37).map(|(p, _, d)| exemplo(&p, d as u32)).collect();
        let amostra: Vec<&Exemplo> = amostra.iter().collect();
        let antes = medir(&rede, &amostra).0;
        treinar_exemplos(&mut rede, todos, 2, &mut Aleatorio(5));
        let depois = medir(&rede, &amostra).0;
        assert!(depois < 0.5 * antes && depois < 3.0, "{antes:.1} -> {depois:.1}");
    }

    #[test]
    fn nota_de_posicao_ganha_usa_o_progresso() {
        let mut rede = rede_ganhando();
        let pos = Chess::default();
        let acc = rede.acumulador(pos.board());
        // progresso fixo pelo viés da última camada rosa: 0,25 = 25 meios-lances
        let rosa = rede.colunas.last_mut().unwrap().densas.last_mut().unwrap();
        rosa.w.iter_mut().for_each(|w| *w = 0.0);
        rosa.b[0] = 0.25;
        assert_eq!(rede.nota(&acc, &pos), 1500 - 4 * 25);
        // perto do fim vale mais
        rede.colunas.last_mut().unwrap().densas.last_mut().unwrap().b[0] = 0.05;
        assert_eq!(rede.nota(&acc, &pos), 1500 - 4 * 5);
        // sem a coluna rosa: a nota de sempre
        assert_eq!(rede.sem_progresso().nota(&acc, &pos), 1000);
    }

    #[test]
    fn progresso_acelera_o_mate_no_kqk() {
        // Treina a coluna rosa no KQK com a rede "ganhando" e joga dez
        // finais: com o progresso, mais mates (ou os mesmos, mais rápidos)
        let kqk = Tabela::construir(Role::Queen, &[]);
        let todos: Vec<Exemplo> = kqk.exemplos_com_distancia().map(|(p, _, d)| exemplo(&p, d as u32)).collect();
        let mut rede = rede_ganhando();
        rede.congelar_antigas();
        treinar_exemplos(&mut rede, todos, 3, &mut Aleatorio(6));
        let fonte = Fonte::Finais(vec![kqk]);
        let combos = vec![Material { brancas: vec![Role::Queen], pretas: vec![] }];
        let posicoes = posicoes_ganhas(&fonte, &combos, 10, 7);
        let com = demonstrar(&rede, &fonte, &posicoes, 2);
        let sem = demonstrar(&rede.sem_progresso(), &fonte, &posicoes, 2);
        eprintln!("sem: {} mates em {} meios; com: {} em {} (ótimo {})", sem.mates, sem.meios_lances_mates, com.mates, com.meios_lances_mates, com.otimo);
        assert_eq!(com.mates, 10);
        assert!(com.mates > sem.mates || com.meios_lances_mates < sem.meios_lances_mates);
    }
}
