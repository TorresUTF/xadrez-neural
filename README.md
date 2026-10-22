# xadrez-neural

Uma rede neural de xadrez escrita do zero em Rust, sem bibliotecas de rede neural. A rede avalia uma posição com três probabilidades (vitória, empate, derrota para quem joga); a avaliação de −1 a +1 é P(vitória) − P(derrota).

## Etapa atual: 1, motor sem rede

Minimax (negamax) com poda alfa-beta e busca de quiescência, avaliando só material. É a linha de base contra a qual a rede será medida. As regras do xadrez vêm da biblioteca [`shakmaty`](https://crates.io/crates/shakmaty).

```
cargo test --release               # testes (mate em 1, quiescência, alfa-beta = minimax...)
cargo run --release                # o motor joga contra si mesmo, profundidade 3
cargo run --release -- jogar 4     # idem, profundidade 4
cargo run --release -- uci         # protocolo UCI, para GUIs e torneios
```

Avaliando só material, o motor não tem preferência entre lances que não ganham peça, então joga o primeiro da lista e tende a repetir posições. Isso é esperado: é exatamente o que a rede vai corrigir.

## Roteiro

| Etapa | O que é | Status |
| --- | --- | --- |
| 0 | Rede XOR escrita à mão | Concluída |
| 1 | Motor: alfa-beta + quiescência, avaliação por material | Em andamento |
| 2 | Rede 768 → 256 → 32 → 3 (softmax V/E/D) no lugar do material | A fazer |
| 2b | Treino supervisionado em finais de até 5 peças com tablebases Syzygy | A fazer |
| 3 | Autojogo com TD-Leaf | A fazer |
| 4 | Medição contra a linha de base e a Maia-1100 | A fazer |
| 5 | Política + MCTS, transformer (opcional) | A fazer |
