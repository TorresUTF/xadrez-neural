// =====================================================================
// REDE — a rede avaliadora, escrita à mão
// =====================================================================
//
// Arquitetura:  768 entradas -> 384 -> 96 -> 32 (CReLU) -> 3 (softmax)
//
// Os tamanhos estão em NIVEIS, logo abaixo. Para experimentar outra
// arquitetura, basta mudar essa lista (pesos salvos de uma arquitetura
// não carregam em outra; o arquivo guarda os tamanhos e avisa).
//
// Saída: (P_V, P_E, P_D) = probabilidades de vitória, empate e derrota
// para QUEM JOGA, assumindo jogo perfeito. A avaliação de -1 a +1 é
//      v = P_V - P_D
// e a maior das três probabilidades mede a certeza da rede.
//
// ENTRADA (768 = 2 cores × 6 peças × 64 casas)
// Cada entrada vale 0 ou 1: "existe uma peça deste tipo nesta casa?".
// A cor não é "branca/preta" e sim "minha/do adversário", e quando as
// pretas jogam o tabuleiro é espelhado. Assim a rede vê toda posição do
// ponto de vista de quem joga, e a mesma posição com as cores trocadas
// dá exatamente a mesma resposta (há um teste para isso).
//
// O TRUQUE DA NNUE: o ACUMULADOR
// Só ~32 das 768 entradas valem 1. Então a 1ª camada, que seria
//      z1 = W1 · x + b1      (384 × 768 multiplicações)
// vira só a soma das colunas de W1 das entradas ligadas:
//      z1 = b1 + soma_{i ligadas} W1[:, i]
// E melhor: um lance muda 2 a 4 entradas. Em vez de recalcular tudo,
// subtrai-se a coluna da peça que saiu e soma-se a da que chegou.
// Esse vetor z1, mantido lance a lance, é o "acumulador".
// Guardamos DOIS acumuladores, um do ponto de vista das brancas e outro
// das pretas, porque a perspectiva troca a cada lance; usa-se o de quem joga.
//
// AS CAMADAS SEGUINTES são "densas": z = W · h + b, todas iguais na forma.
// Por isso o código as trata numa lista, e o passo para trás é um laço
// que percorre essa lista de trás para frente.
//
// ATIVAÇÃO: CReLU(z) = min(max(z, 0), 1)  ("ReLU com teto").
// Barata, e o teto em 1 mantém os números pequenos. Derivada: 1 entre 0 e 1,
// 0 fora. (Na Etapa 0 usamos tanh; a ideia de não linearidade é a mesma.)
// =====================================================================

use shakmaty::{Bitboard, Board, Color, Piece, Role, Square};
use std::io::{self, Read, Write};

pub const N_ENTRADAS: usize = 768;
pub const N_SAIDAS: usize = 3; // 0 = vitória, 1 = empate, 2 = derrota

// Tamanho de cada nível depois da entrada: o 1º é o acumulador, o último a saída.
pub const NIVEIS: [usize; 4] = [384, 96, 32, N_SAIDAS];
pub const N_OCULTA1: usize = NIVEIS[0];
const N_NIVEIS: usize = NIVEIS.len();
const LARGURA: usize = N_OCULTA1; // o nível mais largo (buffers fixos, sem alocação)

pub const VITORIA: usize = 0;
pub const EMPATE: usize = 1;
pub const DERROTA: usize = 2;

// Mesmo gerador xorshift da Etapa 0
pub struct Aleatorio(pub u64);

impl Aleatorio {
    pub fn proximo(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 as f64 / u64::MAX as f64) * 2.0 - 1.0) as f32
    }
}

// Uma camada densa: `saidas` neurônios, cada um ligado a `entradas` valores.
//   w[k * entradas + j] = peso do valor j da camada anterior para o neurônio k
#[derive(Clone)]
pub struct Densa {
    pub entradas: usize,
    pub saidas: usize,
    pub w: Vec<f32>,
    pub b: Vec<f32>,
}

// Os pesos.
//   w1[i * N_OCULTA1 + j] = peso da entrada i para o neurônio j da 1ª camada
//                           (guardado por COLUNA: a coluna da entrada i é contígua,
//                            que é o que o acumulador soma e subtrai)
//   densas[k]             = camada do nível k para o nível k+1
// A mesma struct guarda os GRADIENTES no treino (mesmo formato).
#[derive(Clone)]
pub struct Rede {
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    pub densas: Vec<Densa>,
}

#[derive(Clone)]
pub struct Acumulador {
    // z1 de cada perspectiva: [brancas, pretas]
    pub z1: [[f32; N_OCULTA1]; 2],
}

// Tudo o que o passo para frente calcula e o passo para trás reutiliza.
// z[n] = soma ponderada do nível n, h[n] = CReLU(z[n]); só os primeiros
// NIVEIS[n] números de cada linha são usados.
pub struct Ativacoes {
    pub z: [[f32; LARGURA]; N_NIVEIS],
    pub h: [[f32; LARGURA]; N_NIVEIS],
    pub p: [f32; N_SAIDAS],
}

fn crelu(z: f32) -> f32 {
    z.clamp(0.0, 1.0)
}

fn crelu_derivada(z: f32) -> f32 {
    if z > 0.0 && z < 1.0 { 1.0 } else { 0.0 }
}

// softmax com o truque de subtrair o máximo (evita exp de número grande)
fn softmax(z: &[f32]) -> [f32; N_SAIDAS] {
    let m = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let e: [f32; N_SAIDAS] = std::array::from_fn(|s| (z[s] - m).exp());
    let soma: f32 = e.iter().sum();
    e.map(|x| x / soma)
}

// Produto escalar com 8 somas parciais. Matematicamente é só soma(a_i * b_i),
// mas o compilador não pode reordenar somas de float sozinho (o arredondamento
// mudaria). Com 8 acumuladores independentes ele usa instruções SIMD, que
// fazem 8 multiplicações de uma vez. (O resto, se o tamanho não for múltiplo
// de 8, é somado no fim.)
fn produto(a: &[f32], b: &[f32]) -> f32 {
    let mut parciais = [0.0f32; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let resto: f32 = ca.remainder().iter().zip(cb.remainder()).map(|(x, y)| x * y).sum();
    for (ca, cb) in ca.zip(cb) {
        for i in 0..8 {
            parciais[i] += ca[i] * cb[i];
        }
    }
    parciais.iter().sum::<f32>() + resto
}

fn indice_cor(c: Color) -> usize {
    if c.is_white() { 0 } else { 1 }
}

// Índice (0..768) da entrada "peça p na casa sq", vista pelo lado `lado`.
pub fn entrada(lado: Color, p: Piece, sq: Square) -> usize {
    let relativa = if p.color == lado { 0 } else { 1 }; // 0 = minha, 1 = dele
    let casa = if lado.is_white() { sq } else { sq.flip_vertical() };
    (relativa * 6 + (p.role as usize - 1)) * 64 + casa.to_usize()
}

// Todas as entradas ligadas de um tabuleiro, vistas pelo lado `lado`.
pub fn entradas(board: &Board, lado: Color) -> Vec<usize> {
    board.iter().map(|(sq, p)| entrada(lado, p, sq)).collect()
}

impl Rede {
    pub fn zerada() -> Rede {
        let densas = (1..N_NIVEIS)
            .map(|n| Densa {
                entradas: NIVEIS[n - 1],
                saidas: NIVEIS[n],
                w: vec![0.0; NIVEIS[n - 1] * NIVEIS[n]],
                b: vec![0.0; NIVEIS[n]],
            })
            .collect();
        Rede { w1: vec![0.0; N_ENTRADAS * N_OCULTA1], b1: vec![0.0; N_OCULTA1], densas }
    }

    // Pesos aleatórios uniformes em ±1/sqrt(entradas ativas por neurônio).
    // Ex.: na 1ª camada ~32 entradas ficam ligadas, então ±1/sqrt(32).
    // Isso mantém a soma z na faixa útil da CReLU logo de início.
    pub fn aleatoria(semente: u64) -> Rede {
        let mut rng = Aleatorio(semente);
        let mut r = Rede::zerada();
        let escala1 = 1.0 / (32.0f32).sqrt();
        r.w1.iter_mut().for_each(|w| *w = rng.proximo() * escala1);
        for d in &mut r.densas {
            let escala = 1.0 / (d.entradas as f32).sqrt();
            d.w.iter_mut().for_each(|w| *w = rng.proximo() * escala);
        }
        r
    }

    fn coluna(&self, i: usize) -> &[f32] {
        &self.w1[i * N_OCULTA1..(i + 1) * N_OCULTA1]
    }

    // -----------------------------------------------------------------
    // ACUMULADOR
    // -----------------------------------------------------------------
    pub fn acumulador(&self, board: &Board) -> Acumulador {
        let mut z1 = [[0.0; N_OCULTA1]; 2];
        for lado in Color::ALL {
            z1[indice_cor(lado)] = self.z1_de(&entradas(board, lado));
        }
        Acumulador { z1 }
    }

    // Atualização incremental: só as casas que mudaram entre `antes` e `depois`.
    // Para cada um dos 12 tipos de peça, compara os bitboards:
    //   saiu = antes & !depois,  chegou = depois & !antes
    // Funciona para qualquer lance (captura, roque, en passant, promoção).
    pub fn atualizar(&self, acc: &Acumulador, antes: &Board, depois: &Board) -> Acumulador {
        let mut novo = acc.clone();
        for color in Color::ALL {
            for role in Role::ALL {
                let p = Piece { color, role };
                let (a, d): (Bitboard, Bitboard) = (antes.by_piece(p), depois.by_piece(p));
                for (casas, sinal) in [(a & !d, -1.0), (d & !a, 1.0)] {
                    for sq in casas {
                        for lado in Color::ALL {
                            let col = self.coluna(entrada(lado, p, sq));
                            for (zj, wj) in novo.z1[indice_cor(lado)].iter_mut().zip(col) {
                                *zj += sinal * wj;
                            }
                        }
                    }
                }
            }
        }
        novo
    }

    // z1 calculado do zero a partir das entradas ligadas (sem acumulador)
    fn z1_de(&self, ligadas: &[usize]) -> [f32; N_OCULTA1] {
        let mut z1 = [0.0; N_OCULTA1];
        z1.copy_from_slice(&self.b1);
        for &i in ligadas {
            for (zj, wj) in z1.iter_mut().zip(self.coluna(i)) {
                *zj += wj;
            }
        }
        z1
    }

    // -----------------------------------------------------------------
    // PASSO PARA FRENTE, a partir do z1 (o acumulador de quem joga)
    //   h[0] = CReLU(z1)
    //   z[n] = W_n · h[n-1] + b_n,  h[n] = CReLU(z[n])   (camadas ocultas)
    //   p    = softmax(z[último])                        (saída)
    // -----------------------------------------------------------------
    pub fn para_frente(&self, z1: &[f32; N_OCULTA1]) -> Ativacoes {
        let mut a = Ativacoes { z: [[0.0; LARGURA]; N_NIVEIS], h: [[0.0; LARGURA]; N_NIVEIS], p: [0.0; N_SAIDAS] };
        a.z[0] = *z1;
        a.h[0] = z1.map(crelu);
        for (n, d) in self.densas.iter().enumerate().map(|(k, d)| (k + 1, d)) {
            for k in 0..d.saidas {
                let linha = &d.w[k * d.entradas..(k + 1) * d.entradas];
                a.z[n][k] = d.b[k] + produto(linha, &a.h[n - 1][..d.entradas]);
                a.h[n][k] = crelu(a.z[n][k]);
            }
        }
        a.p = softmax(&a.z[N_NIVEIS - 1][..N_SAIDAS]);
        a
    }

    // (P_V, P_E, P_D) para quem joga
    pub fn probabilidades(&self, acc: &Acumulador, quem_joga: Color) -> [f32; N_SAIDAS] {
        self.para_frente(&acc.z1[indice_cor(quem_joga)]).p
    }

    // (P_V, P_E, P_D) a partir das entradas ligadas, já do ponto de vista de quem joga
    pub fn probabilidades_de(&self, ligadas: &[usize]) -> [f32; N_SAIDAS] {
        self.para_frente(&self.z1_de(ligadas)).p
    }

    // -----------------------------------------------------------------
    // PASSO PARA TRÁS (usado no treino)
    // Recebe as entradas ligadas (já do ponto de vista de quem joga) e o
    // alvo t = (t_V, t_E, t_D). Soma os gradientes em `g` e devolve o erro
    //      L = -soma_k t_k ln p_k        (entropia cruzada)
    // É a mesma regra da cadeia da Etapa 0, repetida camada a camada.
    // -----------------------------------------------------------------
    // os índices k e j seguem as fórmulas, por isso laços com índice
    #[allow(clippy::needless_range_loop)]
    pub fn retropropagar(&self, ligadas: &[usize], alvo: [f32; N_SAIDAS], g: &mut Rede) -> f32 {
        let a = self.para_frente(&self.z1_de(ligadas));
        let erro = -(0..N_SAIDAS).map(|s| alvo[s] * a.p[s].max(1e-12).ln()).sum::<f32>();

        // 1) Saída: com softmax + entropia cruzada, dL/dz = p - t
        //    (o mesmo cancelamento do y - t da Etapa 0)
        let mut delta = [0.0f32; LARGURA];
        for s in 0..N_SAIDAS {
            delta[s] = a.p[s] - alvo[s];
        }

        // 2) De trás para frente, camada densa por camada densa:
        //      dL/dW_n[k][j] = delta_k · h[n-1]_j        (gradiente dos pesos)
        //      dL/db_n[k]    = delta_k
        //      delta_ant_j   = (soma_k W_n[k][j] · delta_k) · CReLU'(z[n-1]_j)
        //    O erro "volta" pelos mesmos pesos que o sinal usou para ir.
        for n in (1..N_NIVEIS).rev() {
            let d = &self.densas[n - 1];
            let gd = &mut g.densas[n - 1];
            let mut anterior = [0.0f32; LARGURA];
            for k in 0..d.saidas {
                let dk = delta[k];
                if dk == 0.0 {
                    continue; // neurônio desligado: não contribui
                }
                gd.b[k] += dk;
                let linha_g = &mut gd.w[k * d.entradas..(k + 1) * d.entradas];
                let linha_w = &d.w[k * d.entradas..(k + 1) * d.entradas];
                for j in 0..d.entradas {
                    linha_g[j] += dk * a.h[n - 1][j];
                    anterior[j] += linha_w[j] * dk;
                }
            }
            for j in 0..d.entradas {
                anterior[j] *= crelu_derivada(a.z[n - 1][j]);
            }
            delta = anterior;
        }

        // 3) Agora delta é o erro na 1ª camada (o acumulador).
        //    dL/dW1[j][i] = delta_j · x_i, e x_i só vale 1 nas entradas
        //    ligadas: as outras ~736 colunas não recebem nada.
        for (gb, dj) in g.b1.iter_mut().zip(&delta) {
            *gb += dj;
        }
        for &i in ligadas {
            for (gj, dj) in g.w1[i * N_OCULTA1..(i + 1) * N_OCULTA1].iter_mut().zip(&delta) {
                *gj += dj;
            }
        }
        erro
    }

    pub fn zerar(&mut self) {
        for v in self.parametros_mut() {
            v.iter_mut().for_each(|x| *x = 0.0);
        }
    }

    // Todos os números da rede, na ordem: w1, b1, depois w e b de cada densa
    fn parametros_mut(&mut self) -> Vec<&mut Vec<f32>> {
        let mut v = vec![&mut self.w1, &mut self.b1];
        for d in &mut self.densas {
            v.push(&mut d.w);
            v.push(&mut d.b);
        }
        v
    }

    // Adam: um passo por lote, com o gradiente médio do lote em `g`.
    pub fn aplicar_adam(&mut self, g: &mut Rede, adam: &mut Adam, n: usize) {
        adam.t += 1;
        let (b1, b2) = (Adam::BETA1, Adam::BETA2);
        // correção de viés: no começo m e v ainda estão perto de zero
        let c1 = 1.0 - b1.powi(adam.t);
        let c2 = 1.0 - b2.powi(adam.t);
        let passo = adam.taxa * c2.sqrt() / c1;
        let tensores = self.parametros_mut().into_iter().zip(g.parametros_mut());
        let momentos = adam.m.parametros_mut().into_iter().zip(adam.v.parametros_mut());
        for ((w, gw), (m, v)) in tensores.zip(momentos) {
            for i in 0..w.len() {
                let gi = gw[i] / n as f32;
                m[i] = b1 * m[i] + (1.0 - b1) * gi;
                v[i] = b2 * v[i] + (1.0 - b2) * gi * gi;
                w[i] -= passo * m[i] / (v[i].sqrt() + 1e-8);
            }
        }
    }

    // -----------------------------------------------------------------
    // Salvar e carregar: assinatura de 4 bytes, os tamanhos dos níveis
    // (u32) e depois os números em sequência, f32 little-endian.
    // -----------------------------------------------------------------
    const ASSINATURA: &'static [u8; 4] = b"XNR2";

    fn cabecalho() -> Vec<u8> {
        let mut c = Self::ASSINATURA.to_vec();
        c.extend((N_NIVEIS as u32).to_le_bytes());
        for n in NIVEIS {
            c.extend((n as u32).to_le_bytes());
        }
        c
    }

    pub fn salvar(&mut self, arquivo: &str) -> io::Result<()> {
        let mut f = io::BufWriter::new(std::fs::File::create(arquivo)?);
        f.write_all(&Self::cabecalho())?;
        for v in self.parametros_mut() {
            for x in v.iter() {
                f.write_all(&x.to_le_bytes())?;
            }
        }
        Ok(())
    }

    pub fn carregar(arquivo: &str) -> io::Result<Rede> {
        let mut bytes = Vec::new();
        std::fs::File::open(arquivo)?.read_to_end(&mut bytes)?;
        let cab = Self::cabecalho();
        if !bytes.starts_with(&cab) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("arquivo de outra versão ou arquitetura (esperava níveis {NIVEIS:?})"),
            ));
        }
        let mut r = Rede::zerada();
        let total: usize = r.parametros_mut().iter().map(|v| v.len()).sum();
        if bytes.len() != cab.len() + 4 * total {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "arquivo de rede com tamanho errado"));
        }
        let mut numeros = bytes[cab.len()..].chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap()));
        for v in r.parametros_mut() {
            v.iter_mut().for_each(|x| *x = numeros.next().unwrap());
        }
        Ok(r)
    }
}

// ---------------------------------------------------------------------
// ADAM: gradiente descendente com "memória"
// Com o passo fixo (w <- w - taxa·g), uma taxa boa para uma rede pode
// fazer outra divergir; foi o que aconteceu ao aprofundar a rede. O Adam
// guarda, para cada peso, uma média do gradiente (m, a "direção") e uma
// média do gradiente ao quadrado (v, o "tamanho típico"), e anda
//      w <- w - taxa · m / sqrt(v)
// Cada peso ganha o seu próprio tamanho de passo. É o otimizador padrão
// de quase todo treino moderno, inclusive de LLMs.
// ---------------------------------------------------------------------
pub struct Adam {
    pub taxa: f32,
    m: Rede,
    v: Rede,
    t: i32,
}

impl Adam {
    const BETA1: f32 = 0.9;
    const BETA2: f32 = 0.999;

    pub fn novo(taxa: f32) -> Adam {
        Adam { taxa, m: Rede::zerada(), v: Rede::zerada(), t: 0 }
    }
}

#[cfg(test)]
mod testes {
    use super::*;
    use shakmaty::{fen::Fen, CastlingMode, Chess, FromSetup, Position};

    fn de_fen(fen: &str) -> Chess {
        fen.parse::<Fen>().unwrap().into_position(CastlingMode::Standard).unwrap()
    }

    #[test]
    fn probabilidades_somam_um() {
        let rede = Rede::aleatoria(1);
        let pos = Chess::default();
        let p = rede.probabilidades(&rede.acumulador(pos.board()), pos.turn());
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert!(p.iter().all(|&x| x > 0.0));
    }

    #[test]
    fn incremental_igual_a_recalcular() {
        // Joga 40 lances "aleatórios" (sempre o k-ésimo da lista) atualizando o
        // acumulador lance a lance, e compara com o recálculo do zero.
        let rede = Rede::aleatoria(2);
        let mut pos = Chess::default();
        let mut acc = rede.acumulador(pos.board());
        let mut rng = Aleatorio(99);
        for _ in 0..40 {
            let lances = pos.legal_moves();
            if lances.is_empty() {
                break;
            }
            let k = ((rng.proximo() + 1.0) / 2.0 * lances.len() as f32) as usize % lances.len();
            let depois = pos.clone().play(lances[k]).unwrap();
            acc = rede.atualizar(&acc, pos.board(), depois.board());
            pos = depois;
        }
        let do_zero = rede.acumulador(pos.board());
        for lado in 0..2 {
            for j in 0..N_OCULTA1 {
                assert!((acc.z1[lado][j] - do_zero.z1[lado][j]).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn incremental_em_lances_especiais() {
        // roque, en passant e promoção com captura
        let rede = Rede::aleatoria(3);
        for (fen, uci) in [
            ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"),
            ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6"),
            ("1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7b8q"),
        ] {
            let pos = de_fen(fen);
            let m = uci.parse::<shakmaty::uci::UciMove>().unwrap().to_move(&pos).unwrap();
            let depois = pos.clone().play(m).unwrap();
            let inc = rede.atualizar(&rede.acumulador(pos.board()), pos.board(), depois.board());
            let zero = rede.acumulador(depois.board());
            for lado in 0..2 {
                for j in 0..N_OCULTA1 {
                    assert!((inc.z1[lado][j] - zero.z1[lado][j]).abs() < 1e-4, "{uci}");
                }
            }
        }
    }

    #[test]
    fn cores_trocadas_mesma_avaliacao() {
        // A posição espelhada (cores trocadas, tabuleiro virado, outra vez de
        // jogar) é a MESMA posição para quem joga: a rede tem que concordar.
        let rede = Rede::aleatoria(4);
        let pos = de_fen("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3");
        let mut setup = pos.to_setup(shakmaty::EnPassantMode::Legal);
        setup.mirror();
        let espelhada: Chess = Chess::from_setup(setup, CastlingMode::Standard).unwrap();
        let p1 = rede.probabilidades(&rede.acumulador(pos.board()), pos.turn());
        let p2 = rede.probabilidades(&rede.acumulador(espelhada.board()), espelhada.turn());
        for s in 0..N_SAIDAS {
            assert!((p1[s] - p2[s]).abs() < 1e-5);
        }
    }

    #[test]
    fn gradiente_bate_com_diferenca_finita() {
        // dL/dw ≈ (L(w + eps) - L(w - eps)) / (2 eps)
        // Se a retropropagação estiver certa, os dois números coincidem.
        let rede = Rede::aleatoria(5);
        let pos = de_fen("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3");
        let ligadas = entradas(pos.board(), pos.turn());
        let alvo = [0.0, 1.0, 0.0];

        let mut g = Rede::zerada();
        rede.retropropagar(&ligadas, alvo, &mut g);

        let erro_com = |r: &Rede| r.retropropagar(&ligadas, alvo, &mut Rede::zerada());
        let eps = 2e-3;
        let mut conferidos = 0;
        // Em cada camada, os 3 primeiros pesos com gradiente não nulo. (Com a
        // CReLU, neurônios desligados dão gradiente exatamente zero, e conferir
        // zero contra zero não prova nada.)
        let mut gs = g.clone();
        let casos: Vec<(usize, usize)> = gs
            .parametros_mut()
            .iter()
            .enumerate()
            .flat_map(|(c, v)| {
                v.iter().enumerate().filter(|(_, x)| x.abs() > 1e-4).take(3).map(move |(i, _)| (c, i))
            })
            .collect();
        // Na quina da CReLU (z exatamente em 0 ou 1) a derivada muda de valor;
        // se um peso deixa algum z a menos de eps de uma quina, a diferença
        // centrada mistura os dois lados. Por isso comparamos também com as
        // diferenças de um lado só: a retropropagação tem que bater com uma delas.
        let base = erro_com(&rede);
        for (camada, i) in casos {
            let mut mais = rede.clone();
            let mut menos = rede.clone();
            mais.parametros_mut()[camada][i] += eps;
            menos.parametros_mut()[camada][i] -= eps;
            let (l_mais, l_menos) = (erro_com(&mais), erro_com(&menos));
            let numericos = [(l_mais - l_menos) / (2.0 * eps), (l_mais - base) / eps, (base - l_menos) / eps];
            let analitico = g.clone().parametros_mut()[camada][i];
            assert!(
                numericos.iter().any(|n| (n - analitico).abs() < 1e-3 + 0.05 * analitico.abs()),
                "tensor {camada}, peso {i}: numérico {numericos:?}, retropropagação {analitico}"
            );
            eprintln!("tensor {camada} peso {i}: numérico {:+.5} retropropagação {analitico:+.5}", numericos[0]);
            conferidos += 1;
        }
        let n_tensores = 2 + 2 * rede.densas.len();
        assert_eq!(conferidos, 3 * n_tensores, "toda camada precisa ter 3 pesos com gradiente");
    }

    #[test]
    fn treino_aprende_um_exemplo() {
        // Sanidade: repetir o mesmo exemplo tem que levar P(alvo) para perto de 1.
        let mut rede = Rede::aleatoria(6);
        let pos = Chess::default();
        let ligadas = entradas(pos.board(), pos.turn());
        let alvo = [0.0, 1.0, 0.0];
        let mut adam = Adam::novo(1e-3);
        for _ in 0..200 {
            let mut g = Rede::zerada();
            rede.retropropagar(&ligadas, alvo, &mut g);
            rede.aplicar_adam(&mut g, &mut adam, 1);
        }
        let p = rede.probabilidades(&rede.acumulador(pos.board()), pos.turn());
        assert!(p[EMPATE] > 0.95, "{p:?}");
    }

    #[test]
    fn salvar_e_carregar() {
        let mut rede = Rede::aleatoria(7);
        let arquivo = std::env::temp_dir().join("xadrez_neural_teste.rede");
        let arquivo = arquivo.to_str().unwrap();
        rede.salvar(arquivo).unwrap();
        let lida = Rede::carregar(arquivo).unwrap();
        assert_eq!(rede.w1, lida.w1);
        assert_eq!(rede.densas[1].w, lida.densas[1].w);
        assert_eq!(rede.densas[2].b, lida.densas[2].b);
        std::fs::remove_file(arquivo).unwrap();
    }
}
