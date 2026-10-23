// =====================================================================
// GPU — a mesma rede, avaliando milhares de posições de uma vez (Etapa 5)
// =====================================================================
//
// Na CPU a rede soma as colunas de W1 das peças no tabuleiro (o
// acumulador). Na GPU o jeito eficiente é outro: escrever o lote como uma
// matriz e fazer multiplicações de matrizes, que é o que ela faz melhor.
//
//   X  (B × 769): linha b = posição b; X[b][i] = 1 se a entrada i está
//                 ligada. A coluna 768 é a entrada VAZIO (lote.rs), que
//                 completa as 32 casas; a linha 768 de W1 é zero.
//   Z1 = X · W1 + b1          (B × largura)   = os acumuladores do lote
//   H0 = CReLU(Z1)
//   em cada coluna: H = H0[:, início..fim], e em cada densa
//        Z = H · Wᵀ + b,  H = CReLU(Z)   (na última, sem CReLU)
//   P  = softmax(soma das colunas de valor)
//
// É a mesma conta da CPU (rede.rs), linha por linha; só a ordem das somas
// muda, então os números batem até ~1e-5 (teste `igual_a_cpu`).
// X é montado na própria GPU: só sobem os índices (B × 32 inteiros) e um
// "scatter" escreve os 1s. Subir X pronto custaria 24 vezes mais.
//
// Só compila com --features gpu (a biblioteca Burn, que fala com a GPU
// pelo wgpu: Vulkan/DirectX 12 no Windows, Metal no Mac).
// =====================================================================

use crate::avaliacao::nota_da_rede;
use crate::lote::{pecas, AvaliadorLote, Entrada, MAX_PECAS};
use crate::rede::{Rede, N_ENTRADAS, N_SAIDAS};
use burn::prelude::*;
use burn::tensor::activation::softmax;
use burn::tensor::IndexingUpdateOp;
use std::sync::Mutex;
use std::time::Instant;

struct Camada {
    w: Tensor<2>, // entradas × saídas (a transposta da Densa da CPU)
    b: Tensor<2>, // 1 × saídas
}

struct Coluna {
    inicio: usize,
    fim: usize,
    camadas: Vec<Camada>,
    progresso: bool,
}

struct Pesos {
    w1: Tensor<2>, // (768 + 1) × largura
    b1: Tensor<2>, // 1 × largura
    colunas: Vec<Coluna>,
}

pub struct RedeGpu {
    device: Device,
    // Um lote por vez na GPU (num torneio várias partidas chamam juntas;
    // elas esperam a vez aqui em vez de disputar a memória da GPU)
    pesos: Mutex<Pesos>,
    pub lote: usize, // posições por envio
}

// Posições por envio. Com a rede de 1024 neurônios, um lote de 16 mil usa
// ~50 MB para X e ~70 MB para Z1: cabe com folga nos 8 GB.
pub const LOTE_PADRAO: usize = 16_384;

pub fn placa() -> Device {
    Device::wgpu(DeviceKind::DefaultDevice)
}

// A rede na placa, conferindo antes se a placa responde (sem placa, o
// wgpu entra em pânico numa thread dele; aqui vira uma mensagem)
pub fn abrir(rede: &Rede) -> RedeGpu {
    let teste = std::panic::catch_unwind(|| {
        let gpu = RedeGpu::nova(rede, placa());
        gpu.avaliar(&[[crate::lote::VAZIO; MAX_PECAS]]);
        gpu
    });
    teste.unwrap_or_else(|_| {
        eprintln!("\nNão consegui usar a placa de vídeo pelo wgpu (Vulkan/DirectX 12). Confira o driver da placa.");
        std::process::exit(1);
    })
}

// Na CPU, sem placa de vídeo (para os testes conferirem as contas)
pub fn cpu() -> Device {
    Device::flex()
}

fn matriz(dados: Vec<f32>, linhas: usize, colunas: usize, device: &Device) -> Tensor<2> {
    Tensor::from_data(TensorData::new(dados, [linhas, colunas]), device)
}

impl RedeGpu {
    pub fn nova(rede: &Rede, device: Device) -> RedeGpu {
        let l = rede.largura;
        let mut w1 = rede.w1.clone();
        w1.extend(std::iter::repeat_n(0.0, l)); // a linha de VAZIO
        let colunas = rede
            .colunas
            .iter()
            .map(|c| Coluna {
                inicio: c.inicio,
                fim: c.fim,
                progresso: c.de_progresso(),
                camadas: c
                    .densas
                    .iter()
                    .map(|d| {
                        // Densa: w[k * entradas + j] (saídas × entradas); aqui a transposta
                        let mut t = vec![0.0; d.w.len()];
                        for k in 0..d.saidas {
                            for j in 0..d.entradas {
                                t[j * d.saidas + k] = d.w[k * d.entradas + j];
                            }
                        }
                        Camada { w: matriz(t, d.entradas, d.saidas, &device), b: matriz(d.b.clone(), 1, d.saidas, &device) }
                    })
                    .collect(),
            })
            .collect();
        let pesos = Pesos {
            w1: matriz(w1, N_ENTRADAS + 1, l, &device),
            b1: matriz(rede.b1.clone(), 1, l, &device),
            colunas,
        };
        RedeGpu { device, pesos: Mutex::new(pesos), lote: LOTE_PADRAO }
    }

    // (P_V, P_E, P_D) e o progresso (se a rede tiver) de cada posição
    pub fn avaliar(&self, entradas: &[Entrada]) -> (Vec<[f32; N_SAIDAS]>, Option<Vec<f32>>) {
        let mut probs = Vec::with_capacity(entradas.len());
        let mut progresso: Option<Vec<f32>> = None;
        for pedaco in entradas.chunks(self.lote.max(1)) {
            let (p, prog) = self.avaliar_pedaco(pedaco);
            probs.extend(p);
            if let Some(prog) = prog {
                progresso.get_or_insert_with(Vec::new).extend(prog);
            }
        }
        (probs, progresso)
    }

    fn avaliar_pedaco(&self, entradas: &[Entrada]) -> (Vec<[f32; N_SAIDAS]>, Option<Vec<f32>>) {
        let b = entradas.len();
        let indices: Vec<i32> = entradas.iter().flatten().map(|&i| i as i32).collect();
        let pesos = self.pesos.lock().unwrap();
        let dev = &self.device;
        let indices = Tensor::<2, Int>::from_data(TensorData::new(indices, [b, MAX_PECAS]), dev);
        let x = Tensor::<2>::zeros([b, N_ENTRADAS + 1], dev).scatter(1, indices, Tensor::ones([b, MAX_PECAS], dev), IndexingUpdateOp::Add);
        let h0 = (x.matmul(pesos.w1.clone()) + pesos.b1.clone()).clamp(0.0, 1.0);

        let mut valor: Option<Tensor<2>> = None;
        let mut progresso = None;
        for c in &pesos.colunas {
            let mut h = h0.clone().slice(s![.., c.inicio..c.fim]);
            let ultima = c.camadas.len() - 1;
            for (n, camada) in c.camadas.iter().enumerate() {
                h = h.matmul(camada.w.clone()) + camada.b.clone();
                if n < ultima {
                    h = h.clamp(0.0, 1.0);
                }
            }
            if c.progresso {
                progresso = Some(h);
            } else {
                valor = Some(match valor {
                    Some(v) => v + h,
                    None => h,
                });
            }
        }
        let p: Vec<f32> = softmax(valor.expect("rede sem coluna de valor"), 1).into_data().try_into_vec().unwrap();
        let p = p.chunks_exact(N_SAIDAS).map(|x| [x[0], x[1], x[2]]).collect();
        let progresso = progresso.map(|t| t.into_data().try_into_vec().unwrap());
        (p, progresso)
    }
}

impl AvaliadorLote for RedeGpu {
    fn notas(&self, entradas: &[Entrada]) -> Vec<i32> {
        let (probs, progresso) = self.avaliar(entradas);
        probs
            .iter()
            .zip(entradas)
            .enumerate()
            .map(|(k, (&p, e))| nota_da_rede(p, pecas(e), || progresso.as_ref().map(|v| v[k])))
            .collect()
    }
}

// -----------------------------------------------------------------
// MEDIÇÃO (comando `medir`): avaliações por segundo na GPU, por tamanho
// de lote e por tamanho de rede
// -----------------------------------------------------------------
fn por_segundo(gpu: &RedeGpu, amostra: &[Entrada]) -> f64 {
    gpu.avaliar(&amostra[..gpu.lote.min(amostra.len())]); // aquecer (compilar os kernels)
    let t = Instant::now();
    let mut n = 0;
    // repete a amostra até passar de 1 s
    while t.elapsed().as_secs_f64() < 1.0 {
        n += gpu.avaliar(amostra).0.len();
    }
    n as f64 / t.elapsed().as_secs_f64()
}

// Uma rede de pesos aleatórios com uma coluna: largura -> densas -> 3
fn rede_de_teste(largura: usize, densas: &[usize]) -> Rede {
    let mut niveis = densas.to_vec();
    niveis.push(N_SAIDAS);
    let mut r = Rede::zerada(largura, &[(0, largura, niveis)]);
    let mut rng = crate::rede::Aleatorio(1);
    r.w1.iter_mut().for_each(|w| *w = rng.proximo() * 0.17);
    for d in r.colunas.iter_mut().flat_map(|c| &mut c.densas) {
        let escala = 1.0 / (d.entradas as f32).sqrt();
        d.w.iter_mut().for_each(|w| *w = rng.proximo() * escala);
    }
    r
}

pub fn medir(rede: &Rede, gpu: &RedeGpu, amostra: &[Entrada]) {
    let device = gpu.device.clone();
    println!("\nPasso 2 na GPU ({:?}), {} posições da italiana:", device, amostra.len());
    let mut gpu = RedeGpu::nova(rede, device.clone());
    for lote in [1024, 4096, 16_384, 65_536] {
        gpu.lote = lote;
        println!("  lote de {lote:>6}: {:>12.0} avaliações/s", por_segundo(&gpu, amostra));
    }
    // Na GPU a primeira camada é a multiplicação inteira (769 × largura),
    // não a atualização do acumulador: ela entra na conta
    let mil = |r: &Rede| (r.custo() + (N_ENTRADAS + 1) * r.largura) / 1000;
    println!("\nRedes maiores (pesos aleatórios, lote de {LOTE_PADRAO}); avaliações/s e mil multiplicações por avaliação na GPU:");
    let v = por_segundo(&RedeGpu::nova(rede, device.clone()), amostra);
    println!("  {:>12.0} {:>6}  a sua: {}", v, mil(rede), rede.descricao());
    for (largura, densas) in [(1024, vec![32, 32]), (2048, vec![256, 32]), (4096, vec![512, 32])] {
        let r = rede_de_teste(largura, &densas);
        let v = por_segundo(&RedeGpu::nova(&r, device.clone()), amostra);
        println!("  {:>12.0} {:>6}  {}", v, mil(&r), r.descricao());
    }
}

#[cfg(test)]
mod testes {
    use super::*;
    use crate::lote::{entrada_de, ligadas};
    use shakmaty::{fen::Fen, CastlingMode, Chess, Position};

    fn posicoes() -> Vec<Chess> {
        let mut v = vec![Chess::default()];
        for fen in [
            "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
            "8/8/8/4k3/8/8/8/3QK3 w - - 0 1",
            "8/8/8/4k3/8/8/8/3QK3 b - - 0 1",
            "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1",
        ] {
            v.push(fen.parse::<Fen>().unwrap().into_position(CastlingMode::Standard).unwrap());
        }
        // e as posições depois de cada lance da inicial
        let filhos: Vec<Chess> = Chess::default().legal_moves().iter().map(|m| Chess::default().play(*m).unwrap()).collect();
        v.extend(filhos);
        v
    }

    #[test]
    fn igual_a_cpu() {
        // uma rede com as três colunas (azul, verde, rosa)
        let rede = Rede::aleatoria(3).crescer(4).crescer_progresso(5);
        let mut rede = rede;
        // a verde e a rosa nascem com a última camada zerada: sorteia para o teste valer
        let mut rng = crate::rede::Aleatorio(9);
        for c in rede.colunas.iter_mut() {
            for d in c.densas.iter_mut() {
                d.w.iter_mut().for_each(|w| *w = rng.proximo() * 0.3);
            }
        }
        let gpu = RedeGpu { lote: 7, ..RedeGpu::nova(&rede, cpu()) }; // lote pequeno: testa os pedaços
        let pos = posicoes();
        let entradas: Vec<Entrada> = pos.iter().map(entrada_de).collect();
        let (probs, progresso) = gpu.avaliar(&entradas);
        let progresso = progresso.unwrap();
        for (k, e) in entradas.iter().enumerate() {
            let l = ligadas(e);
            let cpu = rede.probabilidades_de(&l);
            for s in 0..N_SAIDAS {
                assert!((cpu[s] - probs[k][s]).abs() < 1e-4, "posição {k}: {cpu:?} {:?}", probs[k]);
            }
            assert!((rede.progresso_de(&l).unwrap() - progresso[k]).abs() < 1e-4);
        }
        // e as notas
        let cpu = AvaliadorLote::notas(&rede, &entradas);
        let gpu = gpu.notas(&entradas);
        for (a, b) in cpu.iter().zip(&gpu) {
            assert!((a - b).abs() <= 1, "{cpu:?}\n{gpu:?}");
        }
    }
}
