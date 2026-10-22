// =====================================================================
// PARALELO — dividir trabalho entre os núcleos sem deixar nenhum parado
// =====================================================================
//
// Dividir a lista em pedaços iguais, um por thread, parece justo, mas
// não é: uma partida pode durar 20 lances e outra 150. A thread que
// pegou as curtas acaba cedo e fica parada esperando as outras.
//
// Aqui cada thread pega o PRÓXIMO item livre de um contador atômico
// (uma "fila" sem trava). Quem termina cedo pega mais trabalho. A ordem
// dos resultados é a mesma da lista de entrada.
// =====================================================================

use std::sync::atomic::{AtomicUsize, Ordering};

pub fn threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
}

pub fn mapear<T: Sync, R: Send>(itens: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let proximo = AtomicUsize::new(0);
    let mut saida: Vec<(usize, R)> = std::thread::scope(|s| {
        let tarefas: Vec<_> = (0..threads().min(itens.len()))
            .map(|_| {
                s.spawn(|| {
                    let mut meus = Vec::new();
                    loop {
                        let i = proximo.fetch_add(1, Ordering::Relaxed);
                        if i >= itens.len() {
                            return meus;
                        }
                        meus.push((i, f(&itens[i])));
                    }
                })
            })
            .collect();
        tarefas.into_iter().flat_map(|t| t.join().unwrap()).collect()
    });
    saida.sort_by_key(|(i, _)| *i);
    saida.into_iter().map(|(_, r)| r).collect()
}

#[cfg(test)]
mod testes {
    use super::*;

    #[test]
    fn mantem_a_ordem() {
        let v: Vec<u64> = (0..1000).collect();
        assert_eq!(mapear(&v, |x| x * x), v.iter().map(|x| x * x).collect::<Vec<_>>());
        assert!(mapear(&Vec::<u64>::new(), |x| *x).is_empty());
    }
}
