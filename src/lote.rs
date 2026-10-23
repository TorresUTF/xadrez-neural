// =====================================================================
// LOTE — busca de força bruta com avaliação em lote (Etapa 5)
// =====================================================================
//
// O alfa-beta (busca.rs) avalia uma posição de cada vez e decide o que
// olhar a seguir pelo que acabou de ver. Isso é ótimo na CPU e péssimo na
// GPU: a GPU só rende quando recebe milhares de posições de uma vez.
//
// Aqui a busca é feita em três passos separados:
//   1. CONSTRUIR a árvore inteira na CPU: todos os lances até a
//      profundidade N (sem poda nenhuma, nenhum lance descartado) e, em
//      cada folha, a árvore de capturas e promoções (quiescência).
//   2. AVALIAR de uma vez todas as posições que precisam de nota (os nós
//      da quiescência, onde vale o "ficar parado"). É o passo da GPU.
//   3. NEGAMAX de baixo para cima com as notas prontas:
//        nó interno:       nota = max sobre filhos de  -nota(filho)
//        nó de quiescência: nota = max(nota estática, max sobre capturas de -nota(filho))
//
// Com as mesmas regras do alfa-beta (empates, mates, quiescência), a nota
// da raiz é EXATAMENTE a do alfa-beta na mesma profundidade: a poda não
// muda o resultado, só quantas posições são vistas (ver o teste
// `mesma_nota_do_alfa_beta`). O preço da força bruta é o tamanho da
// árvore; a aposta é que a GPU avalia tão rápido que compensa.
//
// A árvore fica num vetor só ("arena"): os filhos de um nó ficam lado a
// lado, e o nó guarda onde começam e quantos são.
// =====================================================================

use crate::avaliacao::{nota_da_rede, valor, Material};
use crate::busca::{hash, ordenar, MATE, MAX_QUIESCENCIA};
use crate::rede::{entrada, Rede, N_ENTRADAS};
use shakmaty::{zobrist::Zobrist64, Chess, Move, Position, Role};
use std::time::{Duration, Instant};

// Uma posição para a rede: as entradas ligadas (do ponto de vista de quem
// joga), completadas até 32 com VAZIO, uma entrada que não existe (na GPU
// ela aponta para uma linha de zeros).
pub const MAX_PECAS: usize = 32;
pub const VAZIO: u16 = N_ENTRADAS as u16;
pub type Entrada = [u16; MAX_PECAS];

pub fn entrada_de(pos: &Chess) -> Entrada {
    let mut e = [VAZIO; MAX_PECAS];
    for (x, (sq, p)) in e.iter_mut().zip(pos.board().iter()) {
        *x = entrada(pos.turn(), p, sq) as u16;
    }
    e
}

pub fn pecas(e: &Entrada) -> usize {
    e.iter().take_while(|&&i| i != VAZIO).count()
}

pub fn ligadas(e: &Entrada) -> Vec<usize> {
    e[..pecas(e)].iter().map(|&i| i as usize).collect()
}

// Quem avalia muitas posições de uma vez (a nota é de quem joga, como em
// avaliacao.rs). Rede e Material avaliam uma a uma na CPU; a GPU (gpu.rs)
// avalia o lote inteiro junto.
pub trait AvaliadorLote: Sync {
    fn notas(&self, entradas: &[Entrada]) -> Vec<i32>;
    // o custo fixo de cada chamada (a GPU paga ida e volta mesmo com 1 posição)
    fn latencia(&self) -> Duration {
        Duration::ZERO
    }
}

impl AvaliadorLote for Rede {
    fn notas(&self, entradas: &[Entrada]) -> Vec<i32> {
        entradas
            .iter()
            .map(|e| {
                let l = ligadas(e);
                nota_da_rede(self.probabilidades_de(&l), l.len(), || self.progresso_de(&l))
            })
            .collect()
    }
}

// O material também sai das entradas: (índice / 64) = 6·(0 minha, 1 dele) + papel - 1
impl AvaliadorLote for Material {
    fn notas(&self, entradas: &[Entrada]) -> Vec<i32> {
        const PAPEIS: [Role; 6] = [Role::Pawn, Role::Knight, Role::Bishop, Role::Rook, Role::Queen, Role::King];
        entradas
            .iter()
            .map(|e| {
                e[..pecas(e)]
                    .iter()
                    .map(|&i| {
                        let (dono, papel) = ((i / 64) / 6, PAPEIS[(i / 64 % 6) as usize]);
                        if dono == 0 { valor(papel) } else { -valor(papel) }
                    })
                    .sum()
            })
            .collect()
    }
}

#[derive(Clone, Copy)]
enum Tipo {
    Interno,     // max sobre os filhos
    Quiescencia, // max(nota estática, filhos)
    Fixo,        // nota já conhecida (mate, afogamento, empate)
}

#[derive(Clone, Copy)]
pub struct No {
    tipo: Tipo,
    valor: i32,  // Fixo: a nota.  Quiescência: o índice da avaliação
    filhos: u32, // onde começam os filhos na arena
    n: u32,      // quantos filhos
}

const VAGO: No = No { tipo: Tipo::Fixo, valor: 0, filhos: 0, n: 0 };

#[derive(Default)]
pub struct Arvore {
    nos: Vec<No>,
    pub entradas: Vec<Entrada>, // as posições que precisam de nota estática
    historico: Vec<Zobrist64>,  // da partida + do ramo atual (repetição)
    opcoes: Opcoes,
}

impl Arvore {
    pub fn tamanho(&self) -> usize {
        self.nos.len()
    }

    // as mesmas regras de busca.rs: repetição única dentro da árvore conta como empate
    fn eh_empate(&self, pos: &Chess) -> bool {
        if pos.halfmoves() >= 100 || pos.is_insufficient_material() {
            return true;
        }
        let h = hash(pos);
        self.historico.iter().rev().take(pos.halfmoves() as usize).any(|&x| x == h)
    }

    // reserva espaço para `n` filhos lado a lado e devolve onde começam
    fn reservar(&mut self, n: usize) -> u32 {
        let inicio = self.nos.len() as u32;
        self.nos.extend(std::iter::repeat_n(VAGO, n));
        inicio
    }

    // Igual a negamax() de busca.rs, mas guardando em vez de calculando.
    fn construir(&mut self, pos: &Chess, profundidade: u32, ply: i32) -> No {
        let fixo = |valor| No { tipo: Tipo::Fixo, valor, filhos: 0, n: 0 };
        if self.eh_empate(pos) {
            return fixo(0);
        }
        let mut lances = pos.legal_moves();
        if lances.is_empty() {
            return fixo(if pos.is_check() { -(MATE - ply) } else { 0 });
        }
        if profundidade == 0 {
            return self.quiescencia(pos, self.opcoes.quiescencia);
        }
        ordenar(&mut lances);
        let inicio = self.reservar(lances.len());
        self.historico.push(hash(pos));
        for (k, m) in lances.iter().enumerate() {
            let filho = pos.clone().play(*m).unwrap();
            self.nos[inicio as usize + k] = self.construir(&filho, profundidade - 1, ply + 1);
        }
        self.historico.pop();
        No { tipo: Tipo::Interno, valor: 0, filhos: inicio, n: lances.len() as u32 }
    }

    // Igual a quiescencia() de busca.rs: nota estática + capturas e promoções
    fn quiescencia(&mut self, pos: &Chess, restante: u32) -> No {
        let indice = self.entradas.len() as i32;
        self.entradas.push(entrada_de(pos));
        if restante == 0 {
            return No { tipo: Tipo::Quiescencia, valor: indice, filhos: 0, n: 0 };
        }
        let mut lances = pos.legal_moves();
        lances.retain(|m| m.is_capture() || m.is_promotion());
        ordenar(&mut lances);
        if !self.opcoes.todas_capturas {
            // as que não perdem material, as melhores primeiro, no máximo `capturas`
            let mut boas: Vec<(i32, Move)> = lances.iter().map(|m| (troca(pos, m), *m)).filter(|(t, m)| *t >= 0 || m.is_promotion()).collect();
            boas.sort_by_key(|(t, _)| -t); // estável: empates na ordem de ordenar()
            boas.truncate(self.opcoes.capturas);
            lances.clear();
            lances.extend(boas.into_iter().map(|(_, m)| m));
        }
        let inicio = self.reservar(lances.len());
        for (k, m) in lances.iter().enumerate() {
            let filho = pos.clone().play(*m).unwrap();
            self.nos[inicio as usize + k] = self.quiescencia(&filho, restante - 1);
        }
        No { tipo: Tipo::Quiescencia, valor: indice, filhos: inicio, n: lances.len() as u32 }
    }

    // Passo 3: a nota de um nó, com as notas estáticas prontas
    fn negamax(&self, no: &No, notas: &[i32]) -> i32 {
        let filhos = &self.nos[no.filhos as usize..(no.filhos + no.n) as usize];
        let melhor_filho = filhos.iter().map(|f| -self.negamax(f, notas)).max();
        match no.tipo {
            Tipo::Fixo => no.valor,
            Tipo::Interno => melhor_filho.unwrap(),
            Tipo::Quiescencia => notas[no.valor as usize].max(melhor_filho.unwrap_or(i32::MIN)),
        }
    }
}

// Troca estática (SEE, "static exchange evaluation"): o saldo de material
// da captura `m` se os dois lados continuarem capturando na mesma casa,
// sempre com a peça menos valiosa, e cada um podendo parar quando quiser.
//   ganho[0] = valor da vítima
//   ganho[d] = valor da peça que acabou de capturar - ganho[d-1]
// e de trás para frente: ganho[d-1] = -max(-ganho[d-1], ganho[d])
// (quem joga no lance d só continua se ganhar com isso). Os atacantes
// são recalculados com a casa de quem saiu vazia, assim a torre atrás da
// outra (raio-x) entra na conta.
pub fn troca(pos: &Chess, m: &Move) -> i32 {
    let rei = |r: Role| if r == Role::King { 20_000 } else { valor(r) };
    let (Some(de), alvo) = (m.from(), m.to()) else { return 0 };
    let board = pos.board();
    let mut ocupadas = board.occupied() ^ de;
    let mut ganho = vec![m.capture().map(rei).unwrap_or(0)];
    let mut na_casa = m.role();
    let mut lado = !pos.turn();
    loop {
        let atacantes = board.attacks_to(alvo, lado, ocupadas) & ocupadas;
        let menor = [Role::Pawn, Role::Knight, Role::Bishop, Role::Rook, Role::Queen, Role::King]
            .into_iter()
            .find_map(|r| (atacantes & board.by_role(r)).first().map(|sq| (r, sq)));
        let Some((papel, casa)) = menor else { break };
        ganho.push(rei(na_casa) - ganho.last().unwrap());
        ocupadas ^= casa;
        na_casa = papel;
        lado = !lado;
    }
    for d in (1..ganho.len()).rev() {
        ganho[d - 1] = -(-ganho[d - 1]).max(ganho[d]);
    }
    ganho[0]
}

// Uma árvore por lance da raiz (assim dá para construir em paralelo)
pub struct Raiz {
    pub lances: Vec<Move>,
    pub arvores: Vec<(Arvore, No)>,
}

impl Raiz {
    pub fn nos(&self) -> usize {
        self.arvores.iter().map(|(a, _)| a.tamanho() + 1).sum::<usize>() + 1
    }
    pub fn avaliacoes(&self) -> usize {
        self.arvores.iter().map(|(a, _)| a.entradas.len()).sum()
    }
}

// A árvore de capturas sem filtro explode: numa abertura italiana, só a
// profundidade 1 com 8 meios-lances de capturas já passa de 36 mil
// posições (cada captura ruim, como dama toma peão defendido, abre outra
// árvore inteira). Por isso, por padrão, a quiescência só segue capturas
// que não perdem material pela troca estática (troca() >= 0): a mesma
// abertura cai para ~100 posições. Ainda assim, numa posição cheia de
// trocas iguais (kiwipete) a profundidade 1 passava de 6 milhões: as
// trocas em casas diferentes podem ser feitas em qualquer ordem, e cada
// ordem é um ramo. Então cada nó segue só as `capturas` melhores pela
// troca (padrão 2): no máximo 2^8 = 256 posições por folha.
// A árvore principal continua sem poda nenhuma; o filtro só decide até
// onde olhar depois da folha.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Opcoes {
    pub quiescencia: u32,     // meios-lances de capturas depois da folha
    pub capturas: usize,      // no máximo esta quantidade de capturas por nó
    pub todas_capturas: bool, // sem o filtro da troca nem o máximo (como o alfa-beta)
    pub paralelo: bool,       // cada lance da raiz numa thread
}

impl Default for Opcoes {
    fn default() -> Opcoes {
        OPCOES_PADRAO
    }
}

// Num torneio as partidas já ocupam as threads: paralelo só para medir
pub const OPCOES_PADRAO: Opcoes = Opcoes { quiescencia: MAX_QUIESCENCIA, capturas: 2, todas_capturas: false, paralelo: false };

// Passo 1.
pub fn construir(pos: &Chess, historico: &[Zobrist64], profundidade: u32, opcoes: &Opcoes) -> Raiz {
    let mut lances = pos.legal_moves();
    ordenar(&mut lances);
    let lances: Vec<Move> = lances.into_iter().collect();
    let um = |m: &Move| {
        let mut a = Arvore { historico: historico.to_vec(), opcoes: *opcoes, ..Default::default() };
        a.historico.push(hash(pos));
        let filho = pos.clone().play(*m).unwrap();
        let no = a.construir(&filho, profundidade - 1, 1);
        (a, no)
    };
    let arvores = if opcoes.paralelo { crate::paralelo::mapear(&lances, um) } else { lances.iter().map(um).collect() };
    Raiz { lances, arvores }
}

pub struct Resultado {
    pub lance: Option<Move>,
    pub nota: i32,
    pub profundidade: u32,
    pub nos: usize,
    pub avaliacoes: usize,
}

// Passos 2 e 3: avalia tudo de uma vez e escolhe o lance (o primeiro dos
// melhores, na mesma ordem do alfa-beta)
pub fn escolher(raiz: &Raiz, avaliador: &dyn AvaliadorLote, pos: &Chess) -> (Option<Move>, i32) {
    let todas: Vec<Entrada> = raiz.arvores.iter().flat_map(|(a, _)| a.entradas.iter().copied()).collect();
    let notas = avaliador.notas(&todas);
    let mut inicio = 0;
    let mut melhor: Option<(Move, i32)> = None;
    for (m, (a, no)) in raiz.lances.iter().zip(&raiz.arvores) {
        let fim = inicio + a.entradas.len();
        let nota = -a.negamax(no, &notas[inicio..fim]);
        inicio = fim;
        if melhor.is_none_or(|(_, n)| nota > n) {
            melhor = Some((*m, nota));
        }
    }
    match melhor {
        Some((m, n)) => (Some(m), n),
        None => (None, if pos.is_check() { -MATE } else { 0 }),
    }
}

pub fn melhor_lance(pos: &Chess, historico: &[Zobrist64], profundidade: u32, avaliador: &dyn AvaliadorLote, opcoes: &Opcoes) -> Resultado {
    let raiz = construir(pos, historico, profundidade, opcoes);
    let (lance, nota) = escolher(&raiz, avaliador, pos);
    Resultado { lance, nota, profundidade, nos: raiz.nos(), avaliacoes: raiz.avaliacoes() }
}

// O avaliador da GPU, se o programa foi compilado com --features gpu
#[cfg(feature = "gpu")]
pub fn avaliador_gpu(rede: &Rede) -> Box<dyn AvaliadorLote> {
    Box::new(crate::gpu::abrir(rede))
}

#[cfg(not(feature = "gpu"))]
pub fn avaliador_gpu(_: &Rede) -> Box<dyn AvaliadorLote> {
    eprintln!("Este programa foi compilado sem a GPU. Use: cargo run --release --features gpu -- ...");
    std::process::exit(1);
}

// Limite de posições a avaliar numa jogada (memória: ~70 bytes cada na
// CPU, e a GPU recebe em pedaços)
const MAX_AVALIACOES: usize = 20_000_000;

// Aprofundamento por tempo: profundidade 1, 2, 3... A força bruta não
// pode parar no meio (a nota só sai no fim), então antes de cada
// profundidade estimamos o tempo dela e só começamos se couber:
//      t_próx ≈ latência + (t_atual - latência) · r
// r = quantas vezes a árvore cresceu da profundidade anterior para esta
// (na primeira, 30, o número típico de lances). A latência da GPU é fixa:
// não cresce com a árvore, então sai da conta antes de multiplicar.
pub fn por_tempo(pos: &Chess, historico: &[Zobrist64], limite: Duration, avaliador: &dyn AvaliadorLote, opcoes: &Opcoes) -> Resultado {
    let inicio = Instant::now();
    let latencia = avaliador.latencia().as_secs_f64();
    let mut r = melhor_lance(pos, historico, 1, avaliador, opcoes);
    let mut ultimo = inicio.elapsed().as_secs_f64();
    let mut crescimento = 30.0;
    for prof in 2..=20 {
        if r.lance.is_none() || r.nota.abs() > MATE - 1000 {
            break;
        }
        let estimado = latencia + (ultimo - latencia).max(0.0) * crescimento;
        if inicio.elapsed().as_secs_f64() + estimado > limite.as_secs_f64() || r.avaliacoes as f64 * crescimento > MAX_AVALIACOES as f64 {
            break;
        }
        let t = Instant::now();
        let mut novo = melhor_lance(pos, historico, prof, avaliador, opcoes);
        ultimo = t.elapsed().as_secs_f64();
        crescimento = (novo.avaliacoes as f64 / r.avaliacoes.max(1) as f64).clamp(3.0, 100.0);
        novo.nos += r.nos;
        novo.avaliacoes += r.avaliacoes;
        r = novo;
    }
    r
}

// -----------------------------------------------------------------
// MEDIÇÃO: quanto custa cada passo (comando `medir`)
// -----------------------------------------------------------------
pub const POSICOES_MEDIDAS: [(&str, &str); 4] = [
    ("inicial", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("italiana", "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 6 5"),
    ("tática (kiwipete)", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
    ("final de torres", "8/5pk1/6p1/8/3R4/6P1/r4PK1/8 w - - 0 1"),
];

pub fn posicao_medida(fen: &str) -> Chess {
    use shakmaty::{fen::Fen, CastlingMode};
    fen.parse::<Fen>().unwrap().into_position(CastlingMode::Standard).unwrap()
}

// Passo 1 (CPU): tamanho da árvore e posições geradas por segundo.
// Devolve as posições da italiana na profundidade 3 (para medir a avaliação).
pub fn medir_arvores(opcoes: &Opcoes) -> Vec<Entrada> {
    let threads = crate::paralelo::threads();
    let filtro = if opcoes.todas_capturas { "todas as capturas".to_string() } else { format!("até {} capturas por nó, que não percam material", opcoes.capturas) };
    println!("Passo 1: construir a árvore na CPU (quiescência de até {} meios-lances, {filtro})", opcoes.quiescencia);
    println!("{:<20} {:>5} {:>12} {:>12} {:>10} {:>12} {:>12}", "posição", "prof", "nós", "avaliações", "1 thread", format!("{threads} threads"), "nós/s");
    let mut amostra = Vec::new();
    for (nome, fen) in POSICOES_MEDIDAS {
        let pos = posicao_medida(fen);
        for prof in 1..=6 {
            let t = Instant::now();
            let um = construir(&pos, &[], prof, &Opcoes { paralelo: false, ..*opcoes });
            let t1 = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let raiz = construir(&pos, &[], prof, &Opcoes { paralelo: true, ..*opcoes });
            let tn = t.elapsed().as_secs_f64();
            println!("{nome:<20} {prof:>5} {:>12} {:>12} {:>9.3}s {:>11.3}s {:>12.0}", raiz.nos(), raiz.avaliacoes(), t1, tn, raiz.nos() as f64 / tn);
            if nome == "italiana" && prof == 3 {
                amostra = raiz.arvores.iter().flat_map(|(a, _)| a.entradas.iter().copied()).collect();
            }
            drop(um);
            // a próxima seria ~30 vezes maior
            if raiz.avaliacoes() * 30 > MAX_AVALIACOES || tn * 30.0 > 20.0 {
                break;
            }
        }
    }
    amostra
}

// Passo 2 na CPU: avaliações por segundo da rede, uma posição por vez
pub fn medir_cpu(rede: &Rede, amostra: &[Entrada]) -> f64 {
    let n = amostra.len().min(50_000);
    let t = Instant::now();
    let notas = AvaliadorLote::notas(rede, &amostra[..n]);
    let por_s = notas.len() as f64 / t.elapsed().as_secs_f64();
    let threads = crate::paralelo::threads();
    println!("\nPasso 2 na CPU: {por_s:.0} avaliações/s por thread (~{:.0} com {threads} threads)", por_s * threads as f64);
    por_s * threads as f64
}

// Uma jogada de 200 ms como no torneio (uma thread): até onde chega
pub fn medir_jogada(nome: &str, avaliador: &dyn AvaliadorLote) {
    println!("\nJogada de 200 ms ({nome}, uma thread como no torneio):");
    for (posicao, fen) in POSICOES_MEDIDAS {
        let pos = posicao_medida(fen);
        let t = Instant::now();
        let r = por_tempo(&pos, &[], Duration::from_millis(200), avaliador, &OPCOES_PADRAO);
        println!("  {posicao:<20} profundidade {}, {} avaliações, {:.0} ms", r.profundidade, r.avaliacoes, t.elapsed().as_secs_f64() * 1000.0);
    }
}

#[cfg(test)]
mod testes {
    use super::*;
    use crate::busca::Busca;
    use shakmaty::{fen::Fen, CastlingMode};

    fn de_fen(fen: &str) -> Chess {
        fen.parse::<Fen>().unwrap().into_position(CastlingMode::Standard).unwrap()
    }

    const POSICOES: [&str; 5] = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4", // mate em 1
        "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 6 5",
        "8/8/8/4k3/8/8/8/3QK3 w - - 0 1",
        "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1", // mate do corredor
    ];

    #[test]
    fn entradas_ida_e_volta() {
        let pos = Chess::default();
        let mut a = ligadas(&entrada_de(&pos));
        let mut b = crate::rede::entradas(pos.board(), pos.turn());
        a.sort();
        b.sort();
        assert_eq!(a, b);
        assert_eq!(Material.notas(&[entrada_de(&de_fen(POSICOES[3]))]), vec![900]);
    }

    // a força bruta dá a mesma nota que o alfa-beta, com material e com a rede
    const COMO_ALFA_BETA: Opcoes = Opcoes { todas_capturas: true, paralelo: true, ..OPCOES_PADRAO };

    #[test]
    fn mesma_nota_do_alfa_beta() {
        // sem o filtro as capturas explodem: a posição 2 passa de 50
        // milhões de avaliações na profundidade 3
        for (fen, max) in POSICOES.into_iter().zip([3, 2, 2, 3, 3]) {
            let pos = de_fen(fen);
            for prof in 1..=max {
                let ab = Busca::nova(&Material, vec![]).melhor_lance(&pos, prof);
                let fb = melhor_lance(&pos, &[], prof, &Material, &COMO_ALFA_BETA);
                assert_eq!(ab.nota, fb.nota, "{fen} prof {prof}");
                assert_eq!(ab.lance, fb.lance, "{fen} prof {prof}");
            }
        }
        // com a rede: a busca usa o acumulador incremental e o lote recalcula
        // do zero; a ordem das somas muda o arredondamento, então ±2
        let rede = Rede::aleatoria(7);
        for fen in POSICOES {
            let pos = de_fen(fen);
            let ab = Busca::nova(&rede, vec![]).melhor_lance(&pos, 1);
            let fb = melhor_lance(&pos, &[], 1, &rede, &COMO_ALFA_BETA);
            assert!((ab.nota - fb.nota).abs() <= 2, "{fen}: {} {}", ab.nota, fb.nota);
        }
    }

    #[test]
    fn troca_estatica() {
        // dama toma peão defendido por peão: perde 900, ganha 100
        let pos = de_fen("4k3/8/3p4/4p3/8/8/4Q3/4K3 w - - 0 1");
        let m = pos.legal_moves().into_iter().find(|m| m.to_uci(CastlingMode::Standard).to_string() == "e2e5").unwrap();
        assert_eq!(troca(&pos, &m), 100 - 900);
        // torre toma peão defendido por torre, com outra torre atrás (raio-x):
        // TxP, TxT, TxT = +100 - 500 + 500
        let pos = de_fen("4r1k1/8/8/4p3/8/8/4R3/4R1K1 w - - 0 1");
        let m = pos.legal_moves().into_iter().find(|m| m.to_uci(CastlingMode::Standard).to_string() == "e2e5").unwrap();
        assert_eq!(troca(&pos, &m), 100);
        // peão livre
        let pos = de_fen("4k3/8/8/4p3/8/8/4R3/4K3 w - - 0 1");
        let m = pos.legal_moves().into_iter().find(|m| m.to_uci(CastlingMode::Standard).to_string() == "e2e5").unwrap();
        assert_eq!(troca(&pos, &m), 100);
    }

    #[test]
    fn acha_o_mate() {
        let pos = de_fen(POSICOES[4]);
        let r = por_tempo(&pos, &[], Duration::from_millis(500), &Material, &OPCOES_PADRAO);
        assert_eq!(r.lance.unwrap().to_string(), "Rd1-d8");
        assert_eq!(r.nota, MATE - 1);
    }
}

