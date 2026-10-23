// =====================================================================
// REDE — a rede avaliadora, escrita à mão
// =====================================================================
//
// Arquitetura padrão:  768 entradas -> 384 -> 96 -> 32 (CReLU) -> 3 (softmax)
//
// Desde a "rede crescida" (mais abaixo), a rede é um acumulador seguido
// de uma ou mais COLUNAS. Cada coluna lê um pedaço do acumulador, tem as
// suas camadas densas e termina em 3 números; os 3 números de todas as
// colunas são somados antes do softmax. A rede padrão tem uma coluna só.
// O arquivo guarda a arquitetura, então qualquer uma salva carrega.
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
//
// A REDE CRESCIDA (ideia do Gabriel): aproveitar uma rede treinada
// ("azul") e dar a ela uma coluna nova ("verde") ao lado:
//
//      acumulador azul (384) ──────────────> 96 -> 32 -> 3 ─┐
//                 │                                          + -> softmax
//      acumulador verde (384) ─┴─ (768) ─> 64 -> 32 -> 3 ───┘
//
// A coluna verde lê os dois acumuladores (o azul já sabe muita coisa).
// Os pesos da última camada verde começam em ZERO, então a soma começa
// igual à saída da azul: a rede crescida nasce jogando exatamente como a
// original, e o treino só pode melhorá-la a partir dali. (A mesma ideia
// das "progressive networks" da DeepMind e das "zero convolutions" do
// ControlNet.) No começo a azul fica CONGELADA (gradiente zero), para a
// verde aprender o que falta sem estragar o que a azul já sabe.
//
// A COLUNA DE PROGRESSO ("rosa"): a saída V/E/D não diz COMO ganhar.
// Num KQ vs K toda posição vale "vitória", então a busca não vê
// diferença entre encurralar o rei e girar a dama em volta dele. A
// coluna rosa responde outra pergunta: "quantos meios-lances faltam
// para a vitória (ou derrota) se concretizar?". Ela tem um acumulador
// próprio pequeno (128) e lê também o azul e o verde (congelados):
//
//      azul (384) + verde (384) + rosa (128) = 896 -> 32 -> 32 -> 1
//
// A saída dela é um número só, e não entra no softmax. Uma coluna cuja
// última camada tem 1 saída é de progresso; com 3, de valor.
// O alvo vem das tablebases: a DTZ da Syzygy (meios-lances até zerar a
// contagem dos 50 lances), ou a distância até o mate nos finais de 3
// peças de finais.rs. Unidade: 1,0 = 100 meios-lances.
// =====================================================================

use shakmaty::{Bitboard, Board, Color, Piece, Role, Square};
use std::io::{self, Read, Write};

pub const N_ENTRADAS: usize = 768;
pub const N_SAIDAS: usize = 3; // 0 = vitória, 1 = empate, 2 = derrota

// Arquitetura padrão: tamanho de cada nível depois da entrada; o 1º é o
// acumulador, o último a saída.
pub const NIVEIS: [usize; 4] = [384, 96, 32, N_SAIDAS];
// Camadas densas da coluna verde da rede crescida (depois vem a saída)
pub const VERDE: [usize; 2] = [64, 32];
// O acumulador mais largo aceito (buffers fixos, sem alocação na busca)
// Coluna de progresso: acumulador próprio e camadas (a última dá 1 número)
pub const ACC_PROGRESSO: usize = 128;
pub const PROGRESSO: [usize; 3] = [32, 32, 1];
pub const MAX_ACUMULADOR: usize = 1024;
const LARGURA: usize = MAX_ACUMULADOR; // nenhum nível é mais largo que isso

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

impl Densa {
    fn zerada(entradas: usize, saidas: usize) -> Densa {
        Densa { entradas, saidas, w: vec![0.0; entradas * saidas], b: vec![0.0; saidas] }
    }

    // pesos uniformes em ±1/sqrt(entradas): mantém z na faixa útil da CReLU
    fn sortear(&mut self, rng: &mut Aleatorio) {
        let escala = 1.0 / (self.entradas as f32).sqrt();
        self.w.iter_mut().for_each(|w| *w = rng.proximo() * escala);
    }
}

// Uma coluna: lê h[inicio..fim] do acumulador e passa pelas `densas`;
// a última densa dá os 3 números da saída (sem CReLU).
#[derive(Clone)]
pub struct Coluna {
    pub inicio: usize,
    pub fim: usize,
    pub densas: Vec<Densa>,
    pub congelada: bool, // só no treino: não muda (não vai para o arquivo)
}

impl Coluna {
    // coluna de progresso (1 número) em vez de valor (V/E/D)
    pub fn de_progresso(&self) -> bool {
        self.densas.last().unwrap().saidas == 1
    }
}

// O que o treino quer de uma posição: a distribuição V/E/D, ou o
// progresso (meios-lances até acabar / 100)
#[derive(Clone, Copy)]
pub enum Alvo {
    Valor([f32; N_SAIDAS]),
    Progresso(f32),
}

impl From<[f32; N_SAIDAS]> for Alvo {
    fn from(t: [f32; N_SAIDAS]) -> Alvo {
        Alvo::Valor(t)
    }
}

// Os pesos.
//   w1[i * largura + j] = peso da entrada i para o neurônio j do acumulador
//                         (guardado por COLUNA: a coluna da entrada i é contígua,
//                          que é o que o acumulador soma e subtrai)
//   colunas             = as colunas depois do acumulador
//   congelado           = só no treino: os neurônios 0..congelado do
//                         acumulador não mudam
// A mesma struct guarda os GRADIENTES no treino (mesmo formato).
#[derive(Clone)]
pub struct Rede {
    pub largura: usize,
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    pub colunas: Vec<Coluna>,
    pub congelado: usize,
}

#[derive(Clone)]
pub struct Acumulador {
    // z1 de cada perspectiva: [brancas, pretas]; só os primeiros
    // `largura` números são usados
    pub z1: [[f32; MAX_ACUMULADOR]; 2],
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

fn invalido(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

impl Rede {
    // Rede de zeros com a arquitetura dada: largura do acumulador e, para
    // cada coluna, (início, fim, tamanhos das densas, terminando em N_SAIDAS)
    pub fn zerada(largura: usize, colunas: &[(usize, usize, Vec<usize>)]) -> Rede {
        let colunas = colunas
            .iter()
            .map(|(inicio, fim, niveis)| {
                let mut anterior = fim - inicio;
                let densas = niveis
                    .iter()
                    .map(|&n| {
                        let d = Densa::zerada(anterior, n);
                        anterior = n;
                        d
                    })
                    .collect();
                Coluna { inicio: *inicio, fim: *fim, densas, congelada: false }
            })
            .collect();
        Rede { largura, w1: vec![0.0; N_ENTRADAS * largura], b1: vec![0.0; largura], colunas, congelado: 0 }
    }

    // A arquitetura padrão (uma coluna, NIVEIS)
    pub fn padrao() -> Rede {
        Rede::zerada(NIVEIS[0], &[(0, NIVEIS[0], NIVEIS[1..].to_vec())])
    }

    // Uma rede de zeros com a mesma forma (para gradientes e para o Adam)
    pub fn zerada_como(&self) -> Rede {
        let mut r = self.clone();
        r.zerar();
        r
    }

    // A arquitetura (sem os pesos), no formato que `zerada` recebe
    pub fn arquitetura(&self) -> Vec<(usize, usize, Vec<usize>)> {
        self.colunas.iter().map(|c| (c.inicio, c.fim, c.densas.iter().map(|d| d.saidas).collect())).collect()
    }

    // Pesos aleatórios uniformes em ±1/sqrt(entradas ativas por neurônio).
    // Ex.: na 1ª camada ~32 entradas ficam ligadas, então ±1/sqrt(32).
    // Isso mantém a soma z na faixa útil da CReLU logo de início.
    pub fn aleatoria(semente: u64) -> Rede {
        let mut rng = Aleatorio(semente);
        let mut r = Rede::padrao();
        let escala1 = 1.0 / (32.0f32).sqrt();
        r.w1.iter_mut().for_each(|w| *w = rng.proximo() * escala1);
        for d in r.colunas.iter_mut().flat_map(|c| &mut c.densas) {
            d.sortear(&mut rng);
        }
        r
    }

    // A rede crescida: esta rede vira a coluna azul, e nasce uma coluna
    // verde com um acumulador novo da mesma largura. A verde lê os dois
    // acumuladores e passa por VERDE e pela saída, que começa em zero:
    // a rede crescida dá exatamente as mesmas probabilidades que esta.
    pub fn crescer(&self, semente: u64) -> Rede {
        assert!(!self.tem_progresso(), "crescer uma rede que já tem coluna de progresso");
        let (velha, nova) = (self.largura, 2 * self.largura);
        assert!(nova <= MAX_ACUMULADOR, "acumulador de {nova} passa do máximo ({MAX_ACUMULADOR})");
        let mut arq = self.arquitetura();
        arq.push((0, nova, VERDE.iter().copied().chain([N_SAIDAS]).collect()));
        let mut r = Rede::zerada(nova, &arq);
        let mut rng = Aleatorio(semente);
        let escala1 = 1.0 / (32.0f32).sqrt();
        for i in 0..N_ENTRADAS {
            r.w1[i * nova..i * nova + velha].copy_from_slice(self.coluna(i));
            r.w1[i * nova + velha..(i + 1) * nova].iter_mut().for_each(|w| *w = rng.proximo() * escala1);
        }
        r.b1[..velha].copy_from_slice(&self.b1);
        for (c, original) in r.colunas.iter_mut().zip(&self.colunas) {
            c.densas.clone_from(&original.densas);
        }
        let verde = r.colunas.last_mut().unwrap();
        let n = verde.densas.len();
        for d in &mut verde.densas[..n - 1] {
            d.sortear(&mut rng);
        }
        r
    }

    pub fn tem_progresso(&self) -> bool {
        self.colunas.iter().any(Coluna::de_progresso)
    }

    // A coluna rosa: o acumulador ganha ACC_PROGRESSO neurônios novos, e
    // nasce uma coluna de progresso que lê o acumulador inteiro. As
    // colunas de valor não mudam (continuam lendo o mesmo pedaço), então
    // V/E/D sai igual ao da rede original.
    pub fn crescer_progresso(&self, semente: u64) -> Rede {
        assert!(!self.tem_progresso(), "a rede já tem coluna de progresso");
        let (velha, nova) = (self.largura, self.largura + ACC_PROGRESSO);
        assert!(nova <= MAX_ACUMULADOR, "acumulador de {nova} passa do máximo ({MAX_ACUMULADOR})");
        let mut arq = self.arquitetura();
        arq.push((0, nova, PROGRESSO.to_vec()));
        let mut r = Rede::zerada(nova, &arq);
        let mut rng = Aleatorio(semente);
        let escala1 = 1.0 / (32.0f32).sqrt();
        for i in 0..N_ENTRADAS {
            r.w1[i * nova..i * nova + velha].copy_from_slice(self.coluna(i));
            r.w1[i * nova + velha..(i + 1) * nova].iter_mut().for_each(|w| *w = rng.proximo() * escala1);
        }
        r.b1[..velha].copy_from_slice(&self.b1);
        for (c, original) in r.colunas.iter_mut().zip(&self.colunas) {
            c.densas.clone_from(&original.densas);
        }
        for d in &mut r.colunas.last_mut().unwrap().densas {
            d.sortear(&mut rng);
        }
        r
    }

    // A mesma rede sem as colunas de progresso (para comparar)
    pub fn sem_progresso(&self) -> Rede {
        let mut r = self.clone();
        r.colunas.retain(|c| !c.de_progresso());
        r
    }

    // Congela tudo menos a última coluna e a parte do acumulador que só
    // ela usa (na rede crescida: congela a azul, treina a verde).
    // Devolve false se a rede tem uma coluna só (não há o que congelar).
    pub fn congelar_antigas(&mut self) -> bool {
        let n = self.colunas.len();
        if n < 2 {
            return false;
        }
        self.congelado = self.colunas[..n - 1].iter().map(|c| c.fim).max().unwrap();
        self.colunas[..n - 1].iter_mut().for_each(|c| c.congelada = true);
        true
    }

    fn coluna(&self, i: usize) -> &[f32] {
        &self.w1[i * self.largura..(i + 1) * self.largura]
    }

    // Quantas multiplicações por avaliação, fora o acumulador (que é
    // incremental): o custo de verdade de avaliar uma posição na busca
    pub fn custo(&self) -> usize {
        self.colunas.iter().flat_map(|c| &c.densas).map(|d| d.w.len()).sum()
    }

    pub fn descricao(&self) -> String {
        let colunas: Vec<String> = self
            .colunas
            .iter()
            .map(|c| {
                let niveis: Vec<String> = c.densas.iter().map(|d| d.saidas.to_string()).collect();
                let tipo = if c.de_progresso() { " (progresso)" } else { "" };
                format!("[{}..{}] {} -> {}{tipo}", c.inicio, c.fim, c.fim - c.inicio, niveis.join(" -> "))
            })
            .collect();
        format!("768 -> {} | {}", self.largura, colunas.join(" + "))
    }

    // -----------------------------------------------------------------
    // ACUMULADOR
    // -----------------------------------------------------------------
    pub fn acumulador(&self, board: &Board) -> Acumulador {
        let mut z1 = [[0.0; MAX_ACUMULADOR]; 2];
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
    fn z1_de(&self, ligadas: &[usize]) -> [f32; MAX_ACUMULADOR] {
        let mut z1 = [0.0; MAX_ACUMULADOR];
        let z = &mut z1[..self.largura];
        z.copy_from_slice(&self.b1);
        for &i in ligadas {
            for (zj, wj) in z.iter_mut().zip(self.coluna(i)) {
                *zj += wj;
            }
        }
        z1
    }

    // -----------------------------------------------------------------
    // PASSO PARA FRENTE, a partir do z1 (o acumulador de quem joga)
    //   h0   = CReLU(z1)
    //   em cada coluna, a partir de h = h0[inicio..fim]:
    //     z = W · h + b,  h = CReLU(z)     (camadas ocultas)
    //   saída = soma, sobre as colunas, do z da última camada
    //   p     = softmax(saída)
    // Na busca isto roda milhões de vezes: só buffers fixos, sem alocação.
    // -----------------------------------------------------------------
    fn saida(&self, z1: &[f32; MAX_ACUMULADOR]) -> [f32; N_SAIDAS] {
        let h0 = self.h0(z1);
        let mut soma = [0.0f32; N_SAIDAS];
        for c in self.colunas.iter().filter(|c| !c.de_progresso()) {
            for (s, x) in soma.iter_mut().zip(self.coluna_para_frente(c, &h0)) {
                *s += x;
            }
        }
        soma
    }

    fn h0(&self, z1: &[f32; MAX_ACUMULADOR]) -> [f32; MAX_ACUMULADOR] {
        let mut h0 = [0.0f32; MAX_ACUMULADOR];
        for (h, z) in h0[..self.largura].iter_mut().zip(z1) {
            *h = crelu(*z);
        }
        h0
    }

    // Os números da última camada de uma coluna (os primeiros `saidas`)
    fn coluna_para_frente(&self, c: &Coluna, h0: &[f32; MAX_ACUMULADOR]) -> [f32; N_SAIDAS] {
        let (mut a, mut b) = ([0.0f32; LARGURA], [0.0f32; LARGURA]);
        // `ent` é a entrada da camada, `sai` a saída; depois trocam de papel
        let (mut ent, mut sai) = (&mut a, &mut b);
        ent[..c.fim - c.inicio].copy_from_slice(&h0[c.inicio..c.fim]);
        let ultima = c.densas.len() - 1;
        for (n, d) in c.densas.iter().enumerate() {
            for (k, (s, b)) in sai.iter_mut().zip(&d.b).enumerate() {
                let z = b + produto(&d.w[k * d.entradas..(k + 1) * d.entradas], &ent[..d.entradas]);
                *s = if n == ultima { z } else { crelu(z) };
            }
            std::mem::swap(&mut ent, &mut sai);
        }
        std::array::from_fn(|s| ent[s])
    }

    fn progresso_z(&self, z1: &[f32; MAX_ACUMULADOR]) -> Option<f32> {
        let c = self.colunas.iter().find(|c| c.de_progresso())?;
        Some(self.coluna_para_frente(c, &self.h0(z1))[0])
    }

    // Meios-lances até acabar / 100, para quem joga (None: a rede não tem
    // coluna de progresso). Só faz sentido em posição decidida.
    pub fn progresso(&self, acc: &Acumulador, quem_joga: Color) -> Option<f32> {
        self.progresso_z(&acc.z1[indice_cor(quem_joga)])
    }

    pub fn progresso_de(&self, ligadas: &[usize]) -> Option<f32> {
        self.progresso_z(&self.z1_de(ligadas))
    }

    // (P_V, P_E, P_D) para quem joga
    pub fn probabilidades(&self, acc: &Acumulador, quem_joga: Color) -> [f32; N_SAIDAS] {
        softmax(&self.saida(&acc.z1[indice_cor(quem_joga)]))
    }

    // (P_V, P_E, P_D) a partir das entradas ligadas, já do ponto de vista de quem joga
    pub fn probabilidades_de(&self, ligadas: &[usize]) -> [f32; N_SAIDAS] {
        softmax(&self.saida(&self.z1_de(ligadas)))
    }

    // -----------------------------------------------------------------
    // PASSO PARA TRÁS (usado no treino)
    // Recebe as entradas ligadas (já do ponto de vista de quem joga) e o
    // alvo t = (t_V, t_E, t_D). Soma os gradientes em `g` e devolve o erro
    //      L = -soma_k t_k ln p_k        (entropia cruzada)
    // Com alvo de progresso t, só a coluna de progresso participa, e o erro é
    //      L = (y - t)² / 2              (dL/dy = y - t)
    // É a mesma regra da cadeia da Etapa 0, repetida camada a camada.
    // Partes congeladas não recebem gradiente (e por isso não mudam).
    // -----------------------------------------------------------------
    // os índices k e j seguem as fórmulas, por isso laços com índice
    #[allow(clippy::needless_range_loop)]
    pub fn retropropagar(&self, ligadas: &[usize], alvo: impl Into<Alvo>, g: &mut Rede) -> f32 {
        let alvo = alvo.into();
        // as colunas que este alvo treina
        let participa = |c: &Coluna| c.de_progresso() == matches!(alvo, Alvo::Progresso(_));
        // Passo para frente guardando os z de cada camada de cada coluna
        let z0 = self.z1_de(ligadas);
        let h0: Vec<f32> = z0[..self.largura].iter().map(|&z| crelu(z)).collect();
        let mut soma = [0.0f32; N_SAIDAS];
        let zs: Vec<Vec<Vec<f32>>> = self
            .colunas
            .iter()
            .map(|c| {
                let mut zs: Vec<Vec<f32>> = Vec::with_capacity(c.densas.len());
                if !participa(c) {
                    return zs;
                }
                let mut h = h0[c.inicio..c.fim].to_vec();
                for d in &c.densas {
                    let z: Vec<f32> =
                        (0..d.saidas).map(|k| d.b[k] + produto(&d.w[k * d.entradas..(k + 1) * d.entradas], &h)).collect();
                    h = z.iter().map(|&x| crelu(x)).collect();
                    zs.push(z);
                }
                for (s, x) in soma.iter_mut().zip(zs.last().unwrap()) {
                    *s += x;
                }
                zs
            })
            .collect();
        // (a coluna de progresso tem 1 saída: soma[0] é o y dela)
        let (erro, delta_saida) = match alvo {
            // 1) Saída: com softmax + entropia cruzada, dL/dz = p - t
            //    (o mesmo cancelamento do y - t da Etapa 0). Como a saída é a
            //    SOMA das colunas, cada coluna recebe esse mesmo erro.
            Alvo::Valor(t) => {
                let p = softmax(&soma);
                let erro = -(0..N_SAIDAS).map(|s| t[s] * p[s].max(1e-12).ln()).sum::<f32>();
                (erro, (0..N_SAIDAS).map(|s| p[s] - t[s]).collect::<Vec<f32>>())
            }
            Alvo::Progresso(t) => (0.5 * (soma[0] - t) * (soma[0] - t), vec![soma[0] - t]),
        };
        let mut delta0 = vec![0.0f32; self.largura]; // erro no acumulador

        // 2) Em cada coluna, de trás para frente, camada densa por camada densa:
        //      dL/dW_n[k][j] = delta_k · h[n-1]_j        (gradiente dos pesos)
        //      dL/db_n[k]    = delta_k
        //      delta_ant_j   = (soma_k W_n[k][j] · delta_k) · CReLU'(z[n-1]_j)
        //    O erro "volta" pelos mesmos pesos que o sinal usou para ir.
        for ((c, zs), gc) in self.colunas.iter().zip(&zs).zip(&mut g.colunas) {
            if !participa(c) || (c.congelada && c.fim <= self.congelado) {
                continue; // nada aqui nem atrás daqui muda
            }
            let mut delta = delta_saida.clone();
            for n in (0..c.densas.len()).rev() {
                let d = &c.densas[n];
                let gd = &mut gc.densas[n];
                // o z e o h do nível anterior (n = 0: o pedaço do acumulador)
                let z_ant: &[f32] = if n == 0 { &z0[c.inicio..c.fim] } else { &zs[n - 1] };
                let mut anterior = vec![0.0f32; d.entradas];
                for k in 0..d.saidas {
                    let dk = delta[k];
                    if dk == 0.0 {
                        continue; // neurônio desligado: não contribui
                    }
                    let linha_w = &d.w[k * d.entradas..(k + 1) * d.entradas];
                    if !c.congelada {
                        gd.b[k] += dk;
                        let linha_g = &mut gd.w[k * d.entradas..(k + 1) * d.entradas];
                        for j in 0..d.entradas {
                            linha_g[j] += dk * crelu(z_ant[j]);
                        }
                    }
                    for j in 0..d.entradas {
                        anterior[j] += linha_w[j] * dk;
                    }
                }
                for j in 0..d.entradas {
                    anterior[j] *= crelu_derivada(z_ant[j]);
                }
                delta = anterior;
            }
            for (d0, dj) in delta0[c.inicio..c.fim].iter_mut().zip(&delta) {
                *d0 += dj;
            }
        }

        // 3) Agora delta0 é o erro no acumulador.
        //    dL/dW1[j][i] = delta_j · x_i, e x_i só vale 1 nas entradas
        //    ligadas: as outras ~736 colunas não recebem nada.
        let livres = self.congelado..self.largura;
        for (gb, dj) in g.b1[livres.clone()].iter_mut().zip(&delta0[livres.clone()]) {
            *gb += dj;
        }
        for &i in ligadas {
            let base = i * self.largura;
            for (gj, dj) in g.w1[base + livres.start..base + livres.end].iter_mut().zip(&delta0[livres.clone()]) {
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

    // Todos os números da rede, na ordem: w1, b1, depois w e b de cada
    // densa, coluna por coluna
    fn parametros_mut(&mut self) -> Vec<&mut Vec<f32>> {
        let mut v = vec![&mut self.w1, &mut self.b1];
        for d in self.colunas.iter_mut().flat_map(|c| &mut c.densas) {
            v.push(&mut d.w);
            v.push(&mut d.b);
        }
        v
    }

    fn parametros(&self) -> Vec<&Vec<f32>> {
        let mut v = vec![&self.w1, &self.b1];
        for d in self.colunas.iter().flat_map(|c| &c.densas) {
            v.push(&d.w);
            v.push(&d.b);
        }
        v
    }

    // Um passo de treino com um lote, usando todos os núcleos:
    //   1) o lote é dividido entre as threads; cada uma acumula o
    //      gradiente das suas posições num `g` próprio (gs[k]);
    //   2) os parâmetros são divididos em faixas; cada thread soma os
    //      gradientes parciais na sua faixa e aplica o Adam ali.
    // A soma dos gradientes parciais é o gradiente do lote, e o Adam age
    // número a número, então o resultado é o mesmo de `aplicar_adam`
    // com o laço simples (a menos de arredondamento).
    // Devolve o erro total do lote.
    pub fn treinar_lote<T: Sync, A: Into<Alvo>>(
        &mut self,
        gs: &mut [Rede],
        adam: &mut Adam,
        lote: &[T],
        exemplo: impl Fn(&T) -> (&[usize], A) + Sync,
    ) -> f64 {
        let por_thread = lote.len().div_ceil(gs.len()).max(1);
        let usados = lote.len().div_ceil(por_thread);
        let rede = &*self;
        let erro = std::thread::scope(|s| {
            let exemplo = &exemplo;
            let tarefas: Vec<_> = gs
                .iter_mut()
                .zip(lote.chunks(por_thread))
                .map(|(g, parte)| {
                    s.spawn(move || {
                        g.zerar();
                        parte
                            .iter()
                            .map(|e| {
                                let (ligadas, alvo) = exemplo(e);
                                rede.retropropagar(ligadas, alvo, g) as f64
                            })
                            .sum::<f64>()
                    })
                })
                .collect();
            tarefas.into_iter().map(|t| t.join().unwrap()).sum()
        });

        adam.t += 1;
        let (b1, b2) = (Adam::BETA1, Adam::BETA2);
        let c1 = 1.0 - b1.powi(adam.t);
        let c2 = 1.0 - b2.powi(adam.t);
        let passo = adam.taxa * c2.sqrt() / c1;
        let n = lote.len() as f32;
        let parciais: Vec<Vec<&Vec<f32>>> = gs[..usados].iter().map(|g| g.parametros()).collect();
        let threads = crate::paralelo::threads();
        let tensores = self.parametros_mut().into_iter().zip(adam.m.parametros_mut()).zip(adam.v.parametros_mut());
        std::thread::scope(|s| {
            for (t, ((w, m), v)) in tensores.enumerate() {
                let faixa = w.len().div_ceil(threads).max(4096);
                let pedacos = w.chunks_mut(faixa).zip(m.chunks_mut(faixa)).zip(v.chunks_mut(faixa));
                for (k, ((w, m), v)) in pedacos.enumerate() {
                    let parciais = &parciais;
                    s.spawn(move || {
                        let inicio = k * faixa;
                        for i in 0..w.len() {
                            let gi = parciais.iter().map(|g| g[t][inicio + i]).sum::<f32>() / n;
                            m[i] = b1 * m[i] + (1.0 - b1) * gi;
                            v[i] = b2 * v[i] + (1.0 - b2) * gi * gi;
                            w[i] -= passo * m[i] / (v[i].sqrt() + 1e-8);
                        }
                    });
                }
            }
        });
        erro
    }

    // Um gradiente por thread, para o treinar_lote
    pub fn gradientes(&self) -> Vec<Rede> {
        (0..crate::paralelo::threads()).map(|_| self.zerada_como()).collect()
    }

    // Adam: um passo por lote, com o gradiente médio do lote em `g`.
    // É a versão de referência, simples; o treino usa `treinar_lote`, que
    // faz o mesmo em paralelo (o teste confere que os dois concordam).
    #[cfg(test)]
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
    // Salvar e carregar. Formato "XNR3": assinatura de 4 bytes, depois
    // (u32) a largura do acumulador, o número de colunas e, para cada
    // coluna, início, fim, número de densas e o tamanho de cada uma; por
    // fim os números em sequência, f32 little-endian.
    // O formato antigo "XNR2" (só a arquitetura padrão) ainda carrega:
    // os números estão na mesma ordem.
    // -----------------------------------------------------------------
    const ASSINATURA: &'static [u8; 4] = b"XNR3";
    const ASSINATURA_V2: &'static [u8; 4] = b"XNR2";

    fn cabecalho(&self) -> Vec<u8> {
        let mut numeros = vec![self.largura, self.colunas.len()];
        for (inicio, fim, niveis) in self.arquitetura() {
            numeros.extend([inicio, fim, niveis.len()]);
            numeros.extend(niveis);
        }
        let mut c = Self::ASSINATURA.to_vec();
        for n in numeros {
            c.extend((n as u32).to_le_bytes());
        }
        c
    }

    fn cabecalho_v2() -> Vec<u8> {
        let mut c = Self::ASSINATURA_V2.to_vec();
        c.extend((NIVEIS.len() as u32).to_le_bytes());
        for n in NIVEIS {
            c.extend((n as u32).to_le_bytes());
        }
        c
    }

    pub fn salvar(&self, arquivo: &str) -> io::Result<()> {
        let mut f = io::BufWriter::new(std::fs::File::create(arquivo)?);
        f.write_all(&self.cabecalho())?;
        for v in self.parametros() {
            for x in v.iter() {
                f.write_all(&x.to_le_bytes())?;
            }
        }
        Ok(())
    }

    // Lê a arquitetura do cabeçalho XNR3; devolve a rede zerada e o tamanho do cabeçalho
    fn ler_cabecalho(bytes: &[u8]) -> io::Result<(Rede, usize)> {
        let mut pos = 4;
        let mut ler = || -> io::Result<usize> {
            let b = bytes.get(pos..pos + 4).ok_or_else(|| invalido("cabeçalho cortado".into()))?;
            pos += 4;
            Ok(u32::from_le_bytes(b.try_into().unwrap()) as usize)
        };
        let largura = ler()?;
        if largura == 0 || largura > MAX_ACUMULADOR {
            return Err(invalido(format!("acumulador de {largura} (o máximo é {MAX_ACUMULADOR})")));
        }
        let n_colunas = ler()?;
        if n_colunas == 0 || n_colunas > 16 {
            return Err(invalido(format!("{n_colunas} colunas?")));
        }
        let mut arq = Vec::new();
        for _ in 0..n_colunas {
            let (inicio, fim, n) = (ler()?, ler()?, ler()?);
            let niveis = (0..n).map(|_| ler()).collect::<io::Result<Vec<usize>>>()?;
            let ok = inicio < fim
                && fim <= largura
                && matches!(niveis.last(), Some(&N_SAIDAS) | Some(&1))
                && niveis.iter().all(|&x| (1..=LARGURA).contains(&x));
            if !ok {
                return Err(invalido(format!("coluna inválida: {inicio}..{fim} {niveis:?}")));
            }
            arq.push((inicio, fim, niveis));
        }
        if arq.iter().all(|(_, _, n)| n.last() == Some(&1)) {
            return Err(invalido("rede sem coluna de valor".into()));
        }
        let tamanho = 4 + 4 * (2 + arq.iter().map(|(_, _, n)| 3 + n.len()).sum::<usize>());
        Ok((Rede::zerada(largura, &arq), tamanho))
    }

    pub fn carregar(arquivo: &str) -> io::Result<Rede> {
        let mut bytes = Vec::new();
        std::fs::File::open(arquivo)?.read_to_end(&mut bytes)?;
        let (mut r, tamanho) = if bytes.starts_with(Self::ASSINATURA) {
            Self::ler_cabecalho(&bytes)?
        } else if bytes.starts_with(&Self::cabecalho_v2()) {
            (Rede::padrao(), Self::cabecalho_v2().len())
        } else {
            return Err(invalido(format!("arquivo de outra versão ou arquitetura (formato XNR2 só com níveis {NIVEIS:?})")));
        };
        let total: usize = r.parametros().iter().map(|v| v.len()).sum();
        if bytes.len() != tamanho + 4 * total {
            return Err(invalido("arquivo de rede com tamanho errado".into()));
        }
        let mut numeros = bytes[tamanho..].chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap()));
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
// (Um peso congelado tem gradiente sempre zero: m e v ficam em zero e o
// passo m / (sqrt(v) + 1e-8) também; ele não se mexe.)
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

    pub fn novo(taxa: f32, rede: &Rede) -> Adam {
        Adam { taxa, m: rede.zerada_como(), v: rede.zerada_como(), t: 0 }
    }
}

#[cfg(test)]
mod testes {
    use super::*;
    use shakmaty::{fen::Fen, CastlingMode, Chess, FromSetup, Position};

    fn de_fen(fen: &str) -> Chess {
        fen.parse::<Fen>().unwrap().into_position(CastlingMode::Standard).unwrap()
    }

    // Uma rede crescida com a saída verde já "treinada" (não nula), para
    // os testes enxergarem a coluna verde
    fn crescida_ativa(semente: u64) -> Rede {
        let mut r = Rede::aleatoria(semente).crescer(semente + 1);
        let mut rng = Aleatorio(semente + 2);
        r.colunas.last_mut().unwrap().densas.last_mut().unwrap().sortear(&mut rng);
        r
    }

    // A rede padrão e duas crescidas: os testes gerais valem para todas
    fn redes(semente: u64) -> Vec<Rede> {
        vec![Rede::aleatoria(semente), Rede::aleatoria(semente).crescer(semente + 1), crescida_ativa(semente)]
    }

    #[test]
    fn probabilidades_somam_um() {
        for rede in redes(1) {
            let pos = Chess::default();
            let p = rede.probabilidades(&rede.acumulador(pos.board()), pos.turn());
            assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-5);
            assert!(p.iter().all(|&x| x > 0.0));
        }
    }

    #[test]
    fn incremental_igual_a_recalcular() {
        // Joga 40 lances "aleatórios" (sempre o k-ésimo da lista) atualizando o
        // acumulador lance a lance, e compara com o recálculo do zero.
        for rede in redes(2) {
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
                for j in 0..rede.largura {
                    assert!((acc.z1[lado][j] - do_zero.z1[lado][j]).abs() < 1e-3);
                }
            }
        }
    }

    #[test]
    fn incremental_em_lances_especiais() {
        // roque, en passant e promoção com captura
        for rede in redes(3) {
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
                    for j in 0..rede.largura {
                        assert!((inc.z1[lado][j] - zero.z1[lado][j]).abs() < 1e-4, "{uci}");
                    }
                }
            }
        }
    }

    #[test]
    fn cores_trocadas_mesma_avaliacao() {
        // A posição espelhada (cores trocadas, tabuleiro virado, outra vez de
        // jogar) é a MESMA posição para quem joga: a rede tem que concordar.
        for rede in redes(4) {
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
    }

    #[test]
    fn crescida_nasce_igual_a_original() {
        // Saída verde zerada: a soma das colunas é só a azul, posição a posição
        let azul = Rede::aleatoria(20);
        let crescida = azul.crescer(21);
        assert_eq!(crescida.largura, 768);
        assert_eq!(crescida.colunas.len(), 2);
        let mut pos = Chess::default();
        for _ in 0..30 {
            let ligadas = entradas(pos.board(), pos.turn());
            let (p1, p2) = (azul.probabilidades_de(&ligadas), crescida.probabilidades_de(&ligadas));
            for s in 0..N_SAIDAS {
                assert!((p1[s] - p2[s]).abs() < 1e-6, "{p1:?} {p2:?}");
            }
            let lances = pos.legal_moves();
            if lances.is_empty() {
                break;
            }
            pos = pos.clone().play(lances[lances.len() / 2]).unwrap();
        }
        // o custo: 384·96 + 96·32 + 32·3 da azul, mais 768·64 + 64·32 + 32·3 da verde
        assert_eq!(azul.custo(), 384 * 96 + 96 * 32 + 32 * 3);
        assert_eq!(crescida.custo(), azul.custo() + 768 * 64 + 64 * 32 + 32 * 3);
    }

    #[test]
    fn gradiente_bate_com_diferenca_finita() {
        // dL/dw ≈ (L(w + eps) - L(w - eps)) / (2 eps)
        // Se a retropropagação estiver certa, os dois números coincidem.
        for rede in [Rede::aleatoria(5), crescida_ativa(5)] {
            let pos = de_fen("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3");
            let ligadas = entradas(pos.board(), pos.turn());
            let alvo = [0.0, 1.0, 0.0];

            let mut g = rede.zerada_como();
            rede.retropropagar(&ligadas, alvo, &mut g);

            let erro_com = |r: &Rede| r.retropropagar(&ligadas, alvo, &mut r.zerada_como());
            let eps = 2e-3;
            let mut conferidos = 0;
            // Em cada tensor, os 3 primeiros pesos com gradiente não nulo. (Com a
            // CReLU, neurônios desligados dão gradiente exatamente zero, e conferir
            // zero contra zero não prova nada.) Na rede crescida, w1 e b1 têm a
            // parte azul e a verde: conferimos 3 de cada.
            let (largura, crescida) = (rede.largura, rede.colunas.len() > 1);
            let mut gs = g.clone();
            let casos: Vec<(usize, usize)> = gs
                .parametros_mut()
                .iter()
                .enumerate()
                .flat_map(|(c, v)| {
                    // nas crescidas, a metade verde do acumulador: j >= largura/2
                    let verde = |i: &usize| c < 2 && crescida && i % largura >= largura / 2;
                    let (azuis, verdes): (Vec<usize>, Vec<usize>) =
                        v.iter().enumerate().filter(|(_, x)| x.abs() > 1e-4).map(|(i, _)| i).partition(|i| !verde(i));
                    let mut escolhidos: Vec<(usize, usize)> = azuis.into_iter().take(3).map(|i| (c, i)).collect();
                    escolhidos.extend(verdes.into_iter().take(3).map(|i| (c, i)));
                    escolhidos
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
                let analitico = g.parametros()[camada][i];
                assert!(
                    numericos.iter().any(|n| (n - analitico).abs() < 1e-3 + 0.05 * analitico.abs()),
                    "tensor {camada}, peso {i}: numérico {numericos:?}, retropropagação {analitico}"
                );
                conferidos += 1;
            }
            let n_densas: usize = rede.colunas.iter().map(|c| c.densas.len()).sum();
            let extra = if rede.colunas.len() > 1 { 2 } else { 0 }; // as metades verdes de w1 e b1
            assert_eq!(conferidos, 3 * (2 + extra + 2 * n_densas), "toda camada precisa ter 3 pesos com gradiente");
        }
    }

    #[test]
    fn congelada_nao_muda_e_verde_aprende() {
        let mut rede = Rede::aleatoria(30).crescer(31);
        let original = rede.clone();
        assert!(rede.congelar_antigas());
        assert_eq!(rede.congelado, 384);
        let pos = Chess::default();
        let ligadas = entradas(pos.board(), pos.turn());
        let alvo = [0.0, 1.0, 0.0];
        let antes = rede.probabilidades_de(&ligadas)[EMPATE];
        let mut adam = Adam::novo(1e-3, &rede);
        for _ in 0..100 {
            let mut g = rede.zerada_como();
            rede.retropropagar(&ligadas, alvo, &mut g);
            rede.aplicar_adam(&mut g, &mut adam, 1);
        }
        // azul intacta: a parte azul do acumulador e as densas da coluna azul
        for i in 0..N_ENTRADAS {
            assert_eq!(&rede.w1[i * 768..i * 768 + 384], &original.w1[i * 768..i * 768 + 384]);
        }
        assert_eq!(&rede.b1[..384], &original.b1[..384]);
        for (a, b) in rede.colunas[0].densas.iter().zip(&original.colunas[0].densas) {
            assert_eq!(a.w, b.w);
            assert_eq!(a.b, b.b);
        }
        // e a verde aprendeu o exemplo
        let depois = rede.probabilidades_de(&ligadas)[EMPATE];
        assert!(depois > antes + 0.1, "{antes} -> {depois}");
        // rede de uma coluna: não há o que congelar
        assert!(!Rede::aleatoria(1).congelar_antigas());
    }

    #[test]
    fn treino_aprende_um_exemplo() {
        // Sanidade: repetir o mesmo exemplo tem que levar P(alvo) para perto de 1.
        let mut rede = Rede::aleatoria(6);
        let pos = Chess::default();
        let ligadas = entradas(pos.board(), pos.turn());
        let alvo = [0.0, 1.0, 0.0];
        let mut adam = Adam::novo(1e-3, &rede);
        for _ in 0..200 {
            let mut g = rede.zerada_como();
            rede.retropropagar(&ligadas, alvo, &mut g);
            rede.aplicar_adam(&mut g, &mut adam, 1);
        }
        let p = rede.probabilidades(&rede.acumulador(pos.board()), pos.turn());
        assert!(p[EMPATE] > 0.95, "{p:?}");
    }

    #[test]
    fn lote_em_paralelo_igual_ao_laco_simples() {
        // O treino paralelo (gradiente dividido entre threads + Adam por
        // faixas) tem que dar os mesmos pesos que o laço simples.
        let mut rng = Aleatorio(9);
        let lote: Vec<(Vec<usize>, [f32; N_SAIDAS])> = (0..100)
            .map(|k| {
                let ligadas = (0..5).map(|_| ((rng.proximo() + 1.0) / 2.0 * 767.0) as usize).collect();
                let mut alvo = [0.0; N_SAIDAS];
                alvo[k % N_SAIDAS] = 1.0;
                (ligadas, alvo)
            })
            .collect();
        for mut simples in [Rede::aleatoria(8), crescida_ativa(8)] {
            let mut paralela = simples.clone();
            let (mut adam_s, mut adam_p) = (Adam::novo(1e-3, &simples), Adam::novo(1e-3, &simples));
            let mut gs = simples.gradientes();
            for _ in 0..3 {
                let mut g = simples.zerada_como();
                let erro_s: f64 = lote.iter().map(|(l, a)| simples.retropropagar(l, *a, &mut g) as f64).sum();
                simples.aplicar_adam(&mut g, &mut adam_s, lote.len());
                let erro_p = paralela.treinar_lote(&mut gs, &mut adam_p, &lote, |(l, a)| (l, *a));
                assert!((erro_s - erro_p).abs() < 1e-3 * erro_s.abs(), "{erro_s} {erro_p}");
            }
            for (a, b) in simples.parametros().into_iter().zip(paralela.parametros()) {
                for (x, y) in a.iter().zip(b) {
                    assert!((x - y).abs() < 1e-5, "{x} {y}");
                }
            }
        }
    }

    #[test]
    fn salvar_e_carregar() {
        for (k, rede) in redes(7).into_iter().enumerate() {
            let arquivo = std::env::temp_dir().join(format!("xadrez_neural_teste_{k}.rede"));
            let arquivo = arquivo.to_str().unwrap();
            rede.salvar(arquivo).unwrap();
            let lida = Rede::carregar(arquivo).unwrap();
            assert_eq!(rede.arquitetura(), lida.arquitetura());
            for (a, b) in rede.parametros().into_iter().zip(lida.parametros()) {
                assert_eq!(a, b);
            }
            std::fs::remove_file(arquivo).unwrap();
        }
    }

    #[test]
    fn carrega_o_formato_antigo() {
        // Um arquivo XNR2 (como auto3.rede): o cabeçalho antigo e os mesmos
        // números na mesma ordem
        let rede = Rede::aleatoria(8);
        let mut bytes = Rede::cabecalho_v2();
        for v in rede.parametros() {
            for x in v {
                bytes.extend(x.to_le_bytes());
            }
        }
        let arquivo = std::env::temp_dir().join("xadrez_neural_teste_v2.rede");
        let arquivo = arquivo.to_str().unwrap();
        std::fs::write(arquivo, bytes).unwrap();
        let lida = Rede::carregar(arquivo).unwrap();
        assert_eq!(lida.arquitetura(), rede.arquitetura());
        assert_eq!(lida.w1, rede.w1);
        assert_eq!(lida.colunas[0].densas[2].b, rede.colunas[0].densas[2].b);
        std::fs::remove_file(arquivo).unwrap();
    }
}
