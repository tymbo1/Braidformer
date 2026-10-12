# BraidFormer A0/A1 1024-value solvability-rung result

Status: **complete**. This is the next calibrated value-space rung after the completed 64-, 128-, and 256-value tests.

Frozen protocol commit: `18025c11921b0ef11295c768d073913ee6752bf4`.

## Final measurements

| Metric | A0 | A1 |
| --- | ---: | ---: |
| Parameters | 41,866,240 | 41,870,336 |
| Reserved answer tokens | 1,024 | 1,024 |
| Optimizer steps | 2,000 | 2,000 |
| Token positions | 256,000 | 256,000 |
| Final training target loss | 0.000636 | 0.000600 |
| Training greedy exact match | 1.0000 | 1.0000 |
| Final held-out target loss | 8.802002 | 9.573342 |
| Held-out greedy exact match | 0.1406 | 0.0703 |
| CUDA training seconds | 182.948 | 287.715 |
| Elapsed wall seconds | 216.747 | 338.984 |
| CUDA tokens/second | 1399.3 | 889.8 |

## Interpretation

Both unchanged 42M models again passed the primary training-memorization test: they reached 100% autoregressive one-token retrieval on the repeated training set with the 1,024-token candidate answer pool.

On the diagnostic held-out set, A0 reached `0.1406` and A1 `0.0703`. The paired A1 − A0 target-loss delta was `+0.771340` and the A1 − A0 exact-match delta was `-0.0703`. A0 remained ahead on this diagnostic, although the exact-match gap was slightly narrower than at values256. The held-out set measures transfer of the copy/retrieve operation, not unseen fact learning.

Across the ladder, held-out exact match moved from values256 `0.4297` to values1024 `0.1406` for A0, and from `0.3516` to `0.0703` for A1. This is a substantial shared difficulty effect. Neither model has yet lost training memorization, so the training-capacity boundary remains unreached at this budget.

A1 used `1.573×` A0's CUDA time and achieved `0.637×` A0's average token throughput. This was not a matched-compute comparison.

No values8000 rung, association-count change, distractor-distance change, target-length change, decay analysis, predictive learning-state or competence-ledger branch, scaling, A2, or memory experiment was started.

## Selected validation checkpoints

| Step | A0 held-out loss | A1 held-out loss | A1 − A0 | A0 exact | A1 exact |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 9.710166 | 10.559525 | +0.849359 | 0.1484 | 0.0625 |
| 500 | 8.800333 | 9.566602 | +0.766269 | 0.1484 | 0.0781 |
| 1,000 | 8.799296 | 9.569386 | +0.770089 | 0.1484 | 0.0781 |
| 1,500 | 8.801396 | 9.571914 | +0.770518 | 0.1406 | 0.0703 |
| 2,000 | 8.802002 | 9.573342 | +0.771340 | 0.1406 | 0.0703 |

## Integrity and scope

Both runs completed with finite losses and gradients, no CUDA/OOM failure, and SHA-256-verified final checkpoints containing optimizer and RNG state. Gradient checkpointing remains unimplemented and disabled.

- A0 checkpoint: `29061eaf41ba714695c1fbcf351989f31baf7c46374a6ce17f6d173436ac5372`
- A1 checkpoint: `a2a81646797826285a9b8f0760ca39d5706d63b3e30143cdc1e1ee2cc251b790`
- Synthetic data hash: `8ba5b9af3ff24d26a2575cb84a26a1442c94d9a1a5d80d4108ef927aff7f5d20`
- Raw logs, checkpoints, and sidecars remain in ignored `results/runs/` directories.
