// =====================================================================
// SYZYGY — Etapa 2b ampliada: gabarito exato para finais de até 5 (ou 6) peças
// =====================================================================
//
// Para 5 peças a análise retrógrada de finais.rs não dá conta (bilhões de
// posições por combinação de peças). As tablebases Syzygy já fizeram
// esse trabalho e estão disponíveis para download:
//
//   até 5 peças: rode baixar_syzygy.ps1 (Windows), que acha a pasta certa em
//                https://tablebase.lichess.ovh/tables/standard/
//   6 peças:     na mesma página, pasta de 6 peças, só os .rtbw (dezenas de GB)
//
// Só os arquivos .rtbw (vitória/empate/derrota) são necessários aqui.
//
// O programa:
//   1. lista todas as combinações de material até N peças (ex.: KRPvKR);
//   2. sorteia posições legais de cada combinação, com a vez sorteada;
//   3. pergunta à Syzygy o valor exato de cada uma;
//   4. treina a rede nelas (o mesmo treino supervisionado de treino.rs).
//
// A regra dos 50 lances: a Syzygy distingue "vitória que o adversário
// salva pelos 50 lances" (cursed win). Pelas regras do xadrez isso é
// empate, e é assim que contamos.
// =====================================================================

use crate::finais::Valor;
use crate::rede::{entradas, Aleatorio};
use crate::treino::Exemplo;
use shakmaty::{Bitboard, Board, CastlingMode, Chess, Color, FromSetup, Piece, Position, Role, Setup, Square};
use shakmaty_syzygy::{Tablebase, Wdl};

const PECAS: [Role; 5] = [Role::Queen, Role::Rook, Role::Bishop, Role::Knight, Role::Pawn];

// Uma combinação de material: as peças além dos reis, de cada cor
#[derive(Clone, Debug)]
pub struct Material {
    pub brancas: Vec<Role>,
    pub pretas: Vec<Role>,
}

impl Material {
    pub fn nome(&self) -> String {
        let letras = |v: &[Role]| v.iter().map(|r| r.upper_char()).collect::<String>();
        format!("K{}vK{}", letras(&self.brancas), letras(&self.pretas))
    }
    pub fn n_pecas(&self) -> usize {
        2 + self.brancas.len() + self.pretas.len()
    }
}

// Todas as combinações com de 3 a `max` peças. Uma combinação e a sua
// espelhada (cores trocadas) são a mesma para a rede, então guardamos só
// uma delas: a que tem o lado "mais forte" nas brancas.
pub fn combinacoes(max: usize) -> Vec<Material> {
    assert!(max >= 3, "finais têm pelo menos 3 peças");
    // multiconjuntos de peças, em ordem não crescente de valor
    fn multiconjuntos(n: usize, a_partir: usize, atual: &mut Vec<Role>, saida: &mut Vec<Vec<Role>>) {
        saida.push(atual.clone());
        if n == 0 {
            return;
        }
        for (i, &peca) in PECAS.iter().enumerate().skip(a_partir) {
            atual.push(peca);
            multiconjuntos(n - 1, i, atual, saida);
            atual.pop();
        }
    }
    let mut lados = Vec::new();
    multiconjuntos(max - 2, 0, &mut Vec::new(), &mut lados);
    let chave = |v: &Vec<Role>| (v.len(), v.iter().map(|r| 5 - PECAS.iter().position(|p| p == r).unwrap()).collect::<Vec<_>>());
    let mut v = Vec::new();
    for b in &lados {
        for p in &lados {
            let n = 2 + b.len() + p.len();
            if n >= 3 && n <= max && chave(b) >= chave(p) {
                v.push(Material { brancas: b.clone(), pretas: p.clone() });
            }
        }
    }
    v
}

fn casa_aleatoria(rng: &mut Aleatorio, livres: Bitboard, peao: bool) -> Option<Square> {
    let permitidas = if peao { livres & !Bitboard::BACKRANKS } else { livres };
    if permitidas.is_empty() {
        return None;
    }
    let k = ((rng.proximo() + 1.0) / 2.0 * permitidas.count() as f32) as usize;
    permitidas.into_iter().nth(k.min(permitidas.count() - 1))
}

// Uma posição legal sorteada com esse material (algumas tentativas)
pub fn posicao_aleatoria(m: &Material, rng: &mut Aleatorio) -> Option<Chess> {
    for _ in 0..100 {
        let mut board = Board::empty();
        let mut livres = Bitboard::FULL;
        let mut pecas: Vec<Piece> = vec![Color::White.king(), Color::Black.king()];
        pecas.extend(m.brancas.iter().map(|&r| Piece { color: Color::White, role: r }));
        pecas.extend(m.pretas.iter().map(|&r| Piece { color: Color::Black, role: r }));
        let mut ok = true;
        for p in pecas {
            match casa_aleatoria(rng, livres, p.role == Role::Pawn) {
                Some(sq) => {
                    board.set_piece_at(sq, p);
                    livres.discard(sq);
                }
                None => ok = false,
            }
        }
        if !ok {
            continue;
        }
        let mut setup = Setup::empty();
        setup.board = board;
        setup.turn = if rng.proximo() > 0.0 { Color::White } else { Color::Black };
        if let Ok(pos) = Chess::from_setup(setup, CastlingMode::Standard) {
            return Some(pos);
        }
    }
    None
}

pub fn consultar(tabelas: &Tablebase<Chess>, pos: &Chess) -> Option<Valor> {
    match tabelas.probe_wdl_after_zeroing(pos).ok()? {
        Wdl::Win => Some(Valor::Vitoria),
        Wdl::Loss => Some(Valor::Derrota),
        Wdl::Draw | Wdl::CursedWin | Wdl::BlessedLoss => Some(Valor::Empate),
    }
}

pub fn abrir(diretorio: &str) -> Tablebase<Chess> {
    let mut tabelas = Tablebase::new();
    match tabelas.add_directory(diretorio) {
        Ok(0) => {
            eprintln!("Nenhuma tabela .rtbw em {diretorio}. Baixe-as antes (veja o começo de src/syzygy.rs).");
            std::process::exit(1);
        }
        Ok(n) => println!("Syzygy: {n} arquivos em {diretorio}, até {} peças", tabelas.max_pieces()),
        Err(e) => {
            eprintln!("Não consegui abrir {diretorio}: {e}");
            std::process::exit(1);
        }
    }
    tabelas
}

// Sorteia `por_combinacao` exemplos de cada combinação, em paralelo. O
// grupo (`final_`) de cada exemplo é o número de peças. Devolve também os
// nomes das combinações sem tabela no diretório.
pub fn amostra(tabelas: &Tablebase<Chess>, combos: &[Material], por_combinacao: usize, semente: u64) -> (Vec<Exemplo>, Vec<String>) {
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let faltando = std::sync::Mutex::new(Vec::new());
    let exemplos: Vec<Exemplo> = std::thread::scope(|s| {
        let tarefas: Vec<_> = combos
            .chunks(combos.len().div_ceil(threads))
            .enumerate()
            .map(|(t, parte)| {
                let faltando = &faltando;
                s.spawn(move || {
                    let mut rng = Aleatorio(semente ^ ((t as u64 + 1) << 32));
                    let mut v = Vec::new();
                    for m in parte {
                        let mut feitos = 0;
                        let mut tentativas = 0;
                        while feitos < por_combinacao && tentativas < 4 * por_combinacao {
                            tentativas += 1;
                            let Some(pos) = posicao_aleatoria(m, &mut rng) else { continue };
                            let Some(valor) = consultar(tabelas, &pos) else {
                                faltando.lock().unwrap().push(m.nome());
                                break; // tabela desta combinação não está no diretório
                            };
                            v.push(Exemplo {
                                ligadas: entradas(pos.board(), pos.turn()),
                                classe: valor.classe(),
                                final_: m.n_pecas(),
                            });
                            feitos += 1;
                        }
                    }
                    v
                })
            })
            .collect();
        tarefas.into_iter().flat_map(|t| t.join().unwrap()).collect()
    });
    (exemplos, faltando.into_inner().unwrap())
}

pub fn gerar_exemplos(tabelas: &Tablebase<Chess>, max: usize, por_combinacao: usize) -> Vec<Exemplo> {
    let combos = combinacoes(max);
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    println!("{} combinações de material, {por_combinacao} posições de cada, {threads} threads", combos.len());
    let (exemplos, f) = amostra(tabelas, &combos, por_combinacao, 0x9E3779B97F4A7C15);
    if !f.is_empty() {
        println!("Aviso: {} combinações sem tabela no diretório ficaram de fora: {}", f.len(), f.join(", "));
    }
    exemplos
}

pub fn treinar(diretorio: &str, max: usize, por_combinacao: usize, epocas: usize, arquivo: &str, inicial: Option<&str>) {
    let inicio = std::time::Instant::now();
    let tabelas = abrir(diretorio);
    let max = max.min(tabelas.max_pieces());
    let todos = gerar_exemplos(&tabelas, max, por_combinacao);
    println!("{} posições rotuladas em {:.1} s\n", todos.len(), inicio.elapsed().as_secs_f64());
    let mut rede = match inicial {
        Some(a) => crate::rede::Rede::carregar(a).unwrap_or_else(|e| panic!("não consegui ler {a}: {e}")),
        None => crate::rede::Rede::aleatoria(0x2545F4914F6CDD1D),
    };
    let nomes = ["", "", "", "3 peças", "4 peças", "5 peças", "6 peças", "7 peças"];
    crate::treino::treinar_supervisionado(&mut rede, todos, &nomes[..=max], epocas);
    match rede.salvar(arquivo) {
        Ok(()) => println!("\nPesos salvos em {arquivo}"),
        Err(e) => println!("\nNão consegui salvar {arquivo}: {e}"),
    }
}

#[cfg(test)]
mod testes {
    use super::*;
    use crate::finais::{construir_todas, gabarito};

    #[test]
    fn combinacoes_ate_5_pecas() {
        let c = combinacoes(5);
        let nomes: Vec<String> = c.iter().map(|m| m.nome()).collect();
        assert!(nomes.contains(&"KQvK".to_string()));
        assert!(nomes.contains(&"KRPvKR".to_string()));
        assert!(!nomes.contains(&"KRvKRP".to_string())); // é a espelhada de KRPvKR
        assert!(c.iter().all(|m| m.n_pecas() >= 3 && m.n_pecas() <= 5));
        // 3 peças: 5 combinações; 4 peças: 15 + 15; 5 peças: 35 + 75 (com espelhos removidos)
        assert_eq!(c.iter().filter(|m| m.n_pecas() == 3).count(), 5);
    }

    #[test]
    fn posicoes_sorteadas_sao_legais_e_tem_o_material() {
        let mut rng = Aleatorio(1);
        for m in combinacoes(5).iter().step_by(7) {
            let pos = posicao_aleatoria(m, &mut rng).unwrap();
            assert_eq!(pos.board().occupied().count(), m.n_pecas(), "{}", m.nome());
            assert!((pos.board().pawns() & Bitboard::BACKRANKS).is_empty());
        }
    }

    // Só roda se houver tablebases em ./syzygy (ou na variável SYZYGY).
    // Confere a Syzygy contra o nosso gabarito de 3 peças: as duas fontes
    // têm que dar exatamente o mesmo valor.
    #[test]
    fn syzygy_concorda_com_o_gabarito_de_3_pecas() {
        let dir = std::env::var("SYZYGY").unwrap_or_else(|_| "syzygy".into());
        if !std::path::Path::new(&dir).is_dir() {
            eprintln!("(sem tablebases em {dir}: teste pulado)");
            return;
        }
        let mut tb = Tablebase::new();
        if tb.add_directory(&dir).unwrap() == 0 {
            eprintln!("(pasta {dir} sem arquivos .rtbw: teste pulado)");
            return;
        }
        let nossas = construir_todas();
        let mut rng = Aleatorio(3);
        let mut conferidas = 0;
        for m in combinacoes(3).iter().filter(|m| m.brancas != [Role::Bishop] && m.brancas != [Role::Knight]) {
            for _ in 0..2000 {
                let pos = posicao_aleatoria(m, &mut rng).unwrap();
                if pos.legal_moves().is_empty() {
                    continue;
                }
                assert_eq!(consultar(&tb, &pos), gabarito(&nossas, &pos), "{}", m.nome());
                conferidas += 1;
            }
        }
        eprintln!("{conferidas} posições conferidas");
        assert!(conferidas > 5000);
    }
}
