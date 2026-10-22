// =====================================================================
// REDE — Etapa 2: a rede avaliadora, escrita à mão
// =====================================================================
//
// Arquitetura:  768 entradas -> 256 (CReLU) -> 32 (CReLU) -> 3 (softmax)
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
//      z1 = W1 · x + b1      (256 × 768 multiplicações)
// vira só a soma das colunas de W1 das entradas ligadas:
//      z1 = b1 + soma_{i ligadas} W1[:, i]
// E melhor: um lance muda 2 a 4 entradas. Em vez de recalcular tudo,
// subtrai-se a coluna da peça que saiu e soma-se a da que chegou.
// Esse vetor z1, mantido lance a lance, é o "acumulador".
// Guardamos DOIS acumuladores, um do ponto de vista das brancas e outro
// das pretas, porque a perspectiva troca a cada lance; usa-se o de quem joga.
//
// ATIVAÇÃO: CReLU(z) = min(max(z, 0), 1)  ("ReLU com teto").
// Barata, e o teto em 1 mantém os números pequenos. Derivada: 1 entre 0 e 1,
// 0 fora. (Na Etapa 0 usamos tanh; a ideia de não linearidade é a mesma.)
// =====================================================================

use shakmaty::{Bitboard, Board, Color, Piece, Role, Square};
use std::io::{self, Read, Write};

pub const N_ENTRADAS: usize = 768;
pub const N_OCULTA1: usize = 256;
pub const N_OCULTA2: usize = 32;
pub const N_SAIDAS: usize = 3; // 0 = vitória, 1 = empate, 2 = derrota

pub const VITORIA: usize = 0;
#[allow(dead_code)] // usado no treino (Etapa 2b)
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

// Os pesos. As matrizes ficam "achatadas" em Vec:
//   w1[i * N_OCULTA1 + j] = peso da entrada i para o neurônio j da 1ª camada
//                           (guardado por COLUNA: a coluna da entrada i é contígua,
//                            que é o que o acumulador soma e subtrai)
//   w2[k * N_OCULTA1 + j] = peso do neurônio j (1ª camada) para o k (2ª)
//   w3[s * N_OCULTA2 + k] = peso do neurônio k (2ª camada) para a saída s
// A mesma struct guarda os GRADIENTES no treino (mesmo formato).
#[derive(Clone)]
pub struct Rede {
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
    pub w3: Vec<f32>,
    pub b3: Vec<f32>,
}

#[derive(Clone)]
pub struct Acumulador {
    // z1 de cada perspectiva: [brancas, pretas]
    pub z1: [[f32; N_OCULTA1]; 2],
}

#[allow(dead_code)] // z1, h1, z2, h2: lidos só na retropropagação (Etapa 2b)
pub struct Ativacoes {
    pub z1: [f32; N_OCULTA1],
    pub h1: [f32; N_OCULTA1],
    pub z2: [f32; N_OCULTA2],
    pub h2: [f32; N_OCULTA2],
    pub p: [f32; N_SAIDAS],
}

fn crelu(z: f32) -> f32 {
    z.clamp(0.0, 1.0)
}

#[allow(dead_code)]
fn crelu_derivada(z: f32) -> f32 {
    if z > 0.0 && z < 1.0 { 1.0 } else { 0.0 }
}

// softmax com o truque de subtrair o máximo (evita exp de número grande)
fn softmax(z: &[f32; N_SAIDAS]) -> [f32; N_SAIDAS] {
    let m = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let e = z.map(|x| (x - m).exp());
    let soma: f32 = e.iter().sum();
    e.map(|x| x / soma)
}

// Produto escalar com 8 somas parciais. Matematicamente é só soma(a_i * b_i),
// mas o compilador não pode reordenar somas de float sozinho (o arredondamento
// mudaria). Com 8 acumuladores independentes ele usa instruções SIMD, que
// fazem 8 multiplicações de uma vez.
fn produto(a: &[f32], b: &[f32]) -> f32 {
    let mut parciais = [0.0f32; 8];
    for (ca, cb) in a.chunks_exact(8).zip(b.chunks_exact(8)) {
        for i in 0..8 {
            parciais[i] += ca[i] * cb[i];
        }
    }
    parciais.iter().sum()
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
        Rede {
            w1: vec![0.0; N_ENTRADAS * N_OCULTA1],
            b1: vec![0.0; N_OCULTA1],
            w2: vec![0.0; N_OCULTA2 * N_OCULTA1],
            b2: vec![0.0; N_OCULTA2],
            w3: vec![0.0; N_SAIDAS * N_OCULTA2],
            b3: vec![0.0; N_SAIDAS],
        }
    }

    // Pesos aleatórios uniformes em ±1/sqrt(entradas ativas por neurônio).
    // Ex.: na 1ª camada ~32 entradas ficam ligadas, então ±1/sqrt(32).
    // Isso mantém a soma z na faixa útil da CReLU logo de início.
    pub fn aleatoria(semente: u64) -> Rede {
        let mut rng = Aleatorio(semente);
        let mut r = Rede::zerada();
        let escala1 = 1.0 / (32.0f32).sqrt();
        let escala2 = 1.0 / (N_OCULTA1 as f32).sqrt();
        let escala3 = 1.0 / (N_OCULTA2 as f32).sqrt();
        r.w1.iter_mut().for_each(|w| *w = rng.proximo() * escala1);
        r.w2.iter_mut().for_each(|w| *w = rng.proximo() * escala2);
        r.w3.iter_mut().for_each(|w| *w = rng.proximo() * escala3);
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
            let z = &mut z1[indice_cor(lado)];
            z.copy_from_slice(&self.b1);
            for i in entradas(board, lado) {
                for (zj, wj) in z.iter_mut().zip(self.coluna(i)) {
                    *zj += wj;
                }
            }
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

    // -----------------------------------------------------------------
    // PASSO PARA FRENTE, a partir do z1 (o acumulador de quem joga)
    //   h1 = CReLU(z1)
    //   z2 = W2 · h1 + b2,  h2 = CReLU(z2)
    //   z3 = W3 · h2 + b3,  p  = softmax(z3)
    // -----------------------------------------------------------------
    pub fn para_frente(&self, z1: &[f32; N_OCULTA1]) -> Ativacoes {
        let h1 = z1.map(crelu);
        let mut z2 = [0.0; N_OCULTA2];
        for (k, z2k) in z2.iter_mut().enumerate() {
            let linha = &self.w2[k * N_OCULTA1..(k + 1) * N_OCULTA1];
            *z2k = self.b2[k] + produto(linha, &h1);
        }
        let h2 = z2.map(crelu);
        let mut z3 = [0.0; N_SAIDAS];
        for (s, z3s) in z3.iter_mut().enumerate() {
            let linha = &self.w3[s * N_OCULTA2..(s + 1) * N_OCULTA2];
            *z3s = self.b3[s] + produto(linha, &h2);
        }
        Ativacoes { z1: *z1, h1, z2, h2, p: softmax(&z3) }
    }

    // (P_V, P_E, P_D) para quem joga
    pub fn probabilidades(&self, acc: &Acumulador, quem_joga: Color) -> [f32; N_SAIDAS] {
        self.para_frente(&acc.z1[indice_cor(quem_joga)]).p
    }

    // -----------------------------------------------------------------
    // PASSO PARA TRÁS (para as Etapas 2b e 3; aqui só os testes o usam)
    // Recebe as entradas ligadas (já do ponto de vista de quem joga) e o
    // alvo t = (t_V, t_E, t_D). Soma os gradientes em `g` e devolve o erro
    //      L = -soma_k t_k ln p_k        (entropia cruzada)
    // É a mesma regra da cadeia da Etapa 0, com uma camada a mais.
    // -----------------------------------------------------------------
    // os índices s, k, j seguem as fórmulas, por isso laços com índice
    #[allow(dead_code, clippy::needless_range_loop)]
    pub fn retropropagar(&self, ligadas: &[usize], alvo: [f32; N_SAIDAS], g: &mut Rede) -> f32 {
        let mut z1 = [0.0; N_OCULTA1];
        z1.copy_from_slice(&self.b1);
        for &i in ligadas {
            for (zj, wj) in z1.iter_mut().zip(self.coluna(i)) {
                *zj += wj;
            }
        }
        let a = self.para_frente(&z1);
        let erro = -(0..N_SAIDAS).map(|s| alvo[s] * a.p[s].max(1e-12).ln()).sum::<f32>();

        // 1) Saída: com softmax + entropia cruzada, dL/dz3 = p - t
        //    (o mesmo cancelamento do y - t da Etapa 0)
        let delta3: [f32; N_SAIDAS] = std::array::from_fn(|s| a.p[s] - alvo[s]);
        for s in 0..N_SAIDAS {
            for k in 0..N_OCULTA2 {
                g.w3[s * N_OCULTA2 + k] += delta3[s] * a.h2[k];
            }
            g.b3[s] += delta3[s];
        }

        // 2) 2ª camada: o erro volta por W3 e passa pela derivada da CReLU
        let mut delta2 = [0.0; N_OCULTA2];
        for k in 0..N_OCULTA2 {
            let volta: f32 = (0..N_SAIDAS).map(|s| self.w3[s * N_OCULTA2 + k] * delta3[s]).sum();
            delta2[k] = volta * crelu_derivada(a.z2[k]);
            for j in 0..N_OCULTA1 {
                g.w2[k * N_OCULTA1 + j] += delta2[k] * a.h1[j];
            }
            g.b2[k] += delta2[k];
        }

        // 3) 1ª camada: idem, por W2
        let mut delta1 = [0.0; N_OCULTA1];
        for (j, d1) in delta1.iter_mut().enumerate() {
            let volta: f32 = (0..N_OCULTA2).map(|k| self.w2[k * N_OCULTA1 + j] * delta2[k]).sum();
            *d1 = volta * crelu_derivada(a.z1[j]);
            g.b1[j] += *d1;
        }
        // 4) Pesos da 1ª camada: dL/dW1[j][i] = delta1_j * x_i, e x_i só vale 1
        //    nas entradas ligadas. As outras ~736 colunas não recebem nada.
        for &i in ligadas {
            for (gj, dj) in g.w1[i * N_OCULTA1..(i + 1) * N_OCULTA1].iter_mut().zip(&delta1) {
                *gj += dj;
            }
        }
        erro
    }

    fn parametros_mut(&mut self) -> [&mut Vec<f32>; 6] {
        [&mut self.w1, &mut self.b1, &mut self.w2, &mut self.b2, &mut self.w3, &mut self.b3]
    }

    // Gradiente descendente: w <- w - taxa * g / n   (n = exemplos no lote)
    #[allow(dead_code)]
    pub fn aplicar(&mut self, g: &mut Rede, taxa: f32, n: usize) {
        for (w, gw) in self.parametros_mut().into_iter().zip(g.parametros_mut()) {
            for (wi, gi) in w.iter_mut().zip(gw.iter()) {
                *wi -= taxa * gi / n as f32;
            }
        }
    }

    // -----------------------------------------------------------------
    // Salvar e carregar: os números em sequência, f32 little-endian,
    // depois de uma assinatura de 4 bytes.
    // -----------------------------------------------------------------
    const ASSINATURA: &'static [u8; 4] = b"XNR1";

    #[allow(dead_code)]
    pub fn salvar(&mut self, arquivo: &str) -> io::Result<()> {
        let mut f = io::BufWriter::new(std::fs::File::create(arquivo)?);
        f.write_all(Self::ASSINATURA)?;
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
        let mut r = Rede::zerada();
        let total: usize = r.parametros_mut().iter().map(|v| v.len()).sum();
        if bytes.len() != 4 + 4 * total || &bytes[..4] != Self::ASSINATURA {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "arquivo de rede inválido"));
        }
        let mut numeros = bytes[4..].chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap()));
        for v in r.parametros_mut() {
            v.iter_mut().for_each(|x| *x = numeros.next().unwrap());
        }
        Ok(r)
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
        let eps = 1e-2;
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
        for (camada, i) in casos {
            let mut mais = rede.clone();
            let mut menos = rede.clone();
            mais.parametros_mut()[camada][i] += eps;
            menos.parametros_mut()[camada][i] -= eps;
            let numerico = (erro_com(&mais) - erro_com(&menos)) / (2.0 * eps);
            let analitico = g.clone().parametros_mut()[camada][i];
            assert!(
                (numerico - analitico).abs() < 1e-3 + 0.05 * analitico.abs(),
                "camada {camada}, peso {i}: numérico {numerico}, retropropagação {analitico}"
            );
            eprintln!("camada {camada} peso {i}: numérico {numerico:+.5} retropropagação {analitico:+.5}");
            conferidos += 1;
        }
        assert_eq!(conferidos, 18, "toda camada precisa ter 3 pesos com gradiente");
    }

    #[test]
    fn treino_aprende_um_exemplo() {
        // Sanidade: repetir o mesmo exemplo tem que levar P(alvo) para perto de 1.
        let mut rede = Rede::aleatoria(6);
        let pos = Chess::default();
        let ligadas = entradas(pos.board(), pos.turn());
        let alvo = [0.0, 1.0, 0.0];
        for _ in 0..200 {
            let mut g = Rede::zerada();
            rede.retropropagar(&ligadas, alvo, &mut g);
            rede.aplicar(&mut g, 0.05, 1);
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
        assert_eq!(rede.w2, lida.w2);
        assert_eq!(rede.b3, lida.b3);
        std::fs::remove_file(arquivo).unwrap();
    }
}
