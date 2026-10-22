# xadrez-neural

Uma rede neural de xadrez escrita do zero em Rust, sem bibliotecas de rede neural. A rede avalia uma posição com três probabilidades (vitória, empate, derrota para quem joga); a avaliação de −1 a +1 é P(vitória) − P(derrota).

## Como usar

```
cargo test --release                    # testes
cargo run --release                     # o motor joga contra si mesmo, profundidade 3, avaliando material
cargo run --release -- jogar 4 rede     # profundidade 4, avaliando com a rede (pesos aleatórios)
cargo run --release -- jogar 4 x.rede   # idem, com pesos lidos de um arquivo
cargo run --release -- uci rede         # protocolo UCI, para GUIs e torneios
cargo run --release -- finais 20 finais.rede   # Etapa 2b: treina nos finais de 3 peças
cargo run --release -- syzygy syzygy 5 20000 10 syzygy.rede   # finais de 3 a 5 peças com tablebases Syzygy
cargo run --release -- autojogo 200 64 auto.rede syzygy.rede syzygy  # Etapa 3: autojogo com TD-Leaf, finais pela Syzygy
```

Para usar as tablebases: rode `powershell -ExecutionPolicy Bypass -File baixar_syzygy.ps1`, que baixa os arquivos `.rtbw` de até 5 peças para a pasta `syzygy`, e rode `cargo test --release` (o teste `syzygy_concorda_com_o_gabarito_de_3_pecas` confere a Syzygy contra o nosso gabarito).

O `.cargo/config.toml` compila para o processador da máquina (`target-cpu=native`), o que deixa a rede cerca de 5 vezes mais rápida.

## Estrutura

- `src/busca.rs`: negamax com poda alfa-beta e quiescência. Não sabe qual avaliação está usando.
- `src/avaliacao.rs`: o trait `Avaliador` e a avaliação por material (Etapa 1, a linha de base).
- `src/rede.rs`: a rede 768 → 384 → 96 → 32 → 3 escrita à mão (tamanhos em `NIVEIS`): acumulador incremental, passo para frente, retropropagação, otimizador Adam, salvar e carregar.
- `src/finais.rs`: gabarito exato de KQK, KRK e KPK por análise retrógrada (o mesmo método das tablebases Syzygy).
- `src/treino.rs`: treino supervisionado em lotes, com 10% das posições separadas para validação.
- `src/syzygy.rs`: sorteia posições de cada combinação de material até 5 (ou 6) peças e rotula com as tablebases Syzygy.
- `src/autojogo.rs`: Etapa 3. Autojogo em paralelo com TD-Leaf(λ) em três classes, finais de 3 peças decididos pelo gabarito, e confrontos para medir (rede contra material, rede contra a inicial).

As regras do xadrez vêm da biblioteca [`shakmaty`](https://crates.io/crates/shakmaty).

## Roteiro

| Etapa | O que é | Status |
| --- | --- | --- |
| 0 | Rede XOR escrita à mão | Concluída |
| 1 | Motor: alfa-beta + quiescência, avaliação por material | Concluída |
| 2 | Rede (hoje 768 → 384 → 96 → 32 → 3, softmax V/E/D) no lugar do material | Concluída |
| 2b | Treino supervisionado em finais de 3 peças com gabarito exato (99,4% de acerto em posições nunca vistas) | Concluída |
| 3 | Autojogo com TD-Leaf | Código pronto, falta treinar por mais tempo |
| 4 | Medição contra a linha de base e a Maia-1100 | A fazer |
| 5 | Política + MCTS, transformer (opcional) | A fazer |
