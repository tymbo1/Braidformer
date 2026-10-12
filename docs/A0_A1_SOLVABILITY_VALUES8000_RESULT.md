# BraidFormer A0/A1 8000-value solvability-rung result

Status: **complete**. This finishes the frozen answer-space axis after the completed 64-, 128-, 256-, and 1024-value tests.

Frozen protocol commit: `eee5e58839e4440a7e9fa21c743decdb1723efbb`.

## Final measurements

| Metric | A0 | A1 |
| --- | ---: | ---: |
| Parameters | 41,866,240 | 41,870,336 |
| Reserved answer tokens | 8,000 | 8,000 |
| Optimizer steps | 2,000 | 2,000 |
| Token positions | 256,000 | 256,000 |
| Final training target loss | 0.000646 | 0.000628 |
| Training greedy exact match | 1.0000 | 1.0000 |
| Final held-out target loss | 9.947532 | 10.239311 |
| Held-out greedy exact match | 0.0391 | 0.0391 |
| CUDA training seconds | 185.684 | 287.884 |
| Elapsed wall seconds | 219.341 | 328.504 |
| CUDA tokens/second | 1378.7 | 889.2 |

## Interpretation

Both unchanged 42M models passed the primary training-memorization test again: they reached 100% autoregressive one-token retrieval on the repeated training set with the 8,000-token candidate answer pool.

On the diagnostic held-out set, both models reached `0.0391` (5/128). The paired A1 − A0 target-loss delta was `+0.291780`, while the exact-match delta was `+0.0000`. Thus exact-match separation disappears at this rung, but A0 remains closer to the correct answer in target loss.

Relative to values1024, held-out exact match fell from 0.1406 to 0.0391 for A0 and from 0.0703 to 0.0391 for A1. This is a shared generalization collapse in the unchanged one-association recipe. Both models still memorize the training examples, so this rung does not establish a training-memorization boundary.

A1 used `1.550×` A0's CUDA time and achieved `0.646×` A0's average token throughput. This was not a matched-compute comparison.

The value-space axis is now complete. The next authorized ladder change is association count; distractor distance, target length, decay analysis, predictive learning-state or competence-ledger work, scaling, A2, and memory remain deferred.

## Selected validation checkpoints

| Step | A0 held-out loss | A1 held-out loss | A1 − A0 | A0 exact | A1 exact |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 10.637713 | 11.274628 | +0.636914 | 0.0391 | 0.0312 |
| 500 | 9.940380 | 10.216232 | +0.275852 | 0.0391 | 0.0391 |
| 1,000 | 9.944639 | 10.232891 | +0.288252 | 0.0391 | 0.0391 |
| 1,500 | 9.947644 | 10.238060 | +0.290416 | 0.0391 | 0.0391 |
| 2,000 | 9.947532 | 10.239311 | +0.291780 | 0.0391 | 0.0391 |

## Integrity and scope

Both runs completed with finite losses and gradients, no CUDA/OOM failure, and SHA-256-verified final checkpoints containing optimizer and RNG state. Gradient checkpointing remains unimplemented and disabled.

- A0 checkpoint: `05703ee89ccf6cbef2d62cffe75ee32705af9be3ad6062612d8c14029fb1d4cf`
- A1 checkpoint: `550c669c663d63db56765df4ee4512edbecbc5bcd6552bcb922bd236b41a12e7`
- Synthetic data hash: `bf361a9dd1519f5441bff61a5e26f854adfd72ee4650f3c9525170ab0af376b1`
- Raw logs, checkpoints, and sidecars remain in ignored `results/runs/` directories.
