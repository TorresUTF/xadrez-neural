# xadrez-neural

Uma rede neural de xadrez escrita do zero em Rust, sem bibliotecas de rede neural. A rede avalia uma posição com três probabilidades (vitória, empate, derrota para quem joga); a avaliação de −1 a +1 é P(vitória) − P(derrota).

## Como usar

```
cargo test --release                    # testes
cargo run --release                     # o motor joga contra si mesmo, profundidade 3, avaliando material
cargo run --release -- jogar 4 rede     # profundidade 4, avaliando com a rede (pesos aleatórios)
cargo run --release -- jogar 4 x.rede   # idem, com pesos lidos de um arquivo
cargo run --release -- uci rede         # protocolo UCI, para GUIs e torneios
```

O `.cargo/config.toml` compila para o processador da máquina (`target-cpu=native`), o que deixa a rede cerca de 5 vezes mais rápida.

## Estrutura

- `src/busca.rs`: negamax com poda alfa-beta e quiescência. Não sabe qual avaliação está usando.
- `src/avaliacao.rs`: o trait `Avaliador` e a avaliação por material (Etapa 1, a linha de base).
- `src/rede.rs`: a rede 768 → 256 → 32 → 3 escrita à mão: acumulador incremental, passo para frente, retropropagação, salvar e carregar.

As regras do xadrez vêm da biblioteca [`shakmaty`](https://crates.io/crates/shakmaty).

## Roteiro

| Etapa | O que é | Status |
| --- | --- | --- |
| 0 | Rede XOR escrita à mão | Concluída |
| 1 | Motor: alfa-beta + quiescência, avaliação por material | Concluída |
| 2 | Rede 768 → 256 → 32 → 3 (softmax V/E/D) no lugar do material, ainda sem treino | Concluída |
| 2b | Treino supervisionado em finais de até 5 peças com tablebases Syzygy | A fazer |
| 3 | Autojogo com TD-Leaf | A fazer |
| 4 | Medição contra a linha de base e a Maia-1100 | A fazer |
| 5 | Política + MCTS, transformer (opcional) | A fazer |
