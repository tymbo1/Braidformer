# BraidFormer A0/A1 four-association, 8000-value solvability result

Status: **complete**. This is the second association-count rung after the completed one-association value-space ladder and two-association test.

Frozen protocol commit: `5c4dfb12c8e90462336adebe8c3cf45fa843e920`.

## Final measurements

| Metric | A0 | A1 |
| --- | ---: | ---: |
| Parameters | 41,866,240 | 41,870,336 |
| Associations | 4 | 4 |
| Reserved answer tokens | 8,000 | 8,000 |
| Sequence length | 17 | 17 |
| Optimizer steps | 2,000 | 2,000 |
| Token positions | 544,000 | 544,000 |
| Final training target loss | 0.000684 | 0.000630 |
| Training greedy exact match | 1.0000 | 1.0000 |
| Final held-out target loss | 10.506805 | 10.871704 |
| Held-out greedy exact match | 0.0000 | 0.0000 |
| CUDA training seconds | 186.286 | 349.060 |
| Elapsed wall seconds | 220.208 | 390.155 |
| CUDA tokens/second | 2920.2 | 1558.5 |

## Interpretation

Both unchanged 42M models passed the primary training-memorization test: they reached 100% autoregressive one-token retrieval on the repeated training set with four explicit bindings and the 8,000-token candidate answer pool.

On the diagnostic held-out set, both models reached `0.0000` (`0/128`). The paired A1 − A0 target-loss delta was `+0.364899`, and the exact-match delta was `+0.0000`. Exact-match separation is therefore at the floor, while target loss still distinguishes the models.

Relative to the two-association 8000-value rung, held-out exact match fell from 0.0156 to 0.0000 for A0 and remained at 0.0000 for A1 under this 2,000-optimizer-step recipe. Both models still memorize the fixed training set, so this is a held-out retrieval/generalization floor rather than a training-memorization boundary.

A1 used `1.874×` A0's CUDA time and achieved `0.535×` A0's average token throughput. This was not a matched-compute comparison. The four-association rung itself used the same 544,000 token-position budget for A0 and A1.

The next association-count rung is eight associations, but it is not frozen yet. Distractor distance, target length, decay analysis, predictive learning-state or competence-ledger work, scaling, A2, and memory remain deferred.

## Selected validation checkpoints

| Step | A0 held-out loss | A1 held-out loss | A1 − A0 | A0 exact | A1 exact |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 11.250305 | 11.834167 | +0.583862 | 0.0000 | 0.0000 |
| 500 | 10.504791 | 10.817596 | +0.312805 | 0.0000 | 0.0000 |
| 1,000 | 10.505707 | 10.853271 | +0.347565 | 0.0000 | 0.0000 |
| 1,500 | 10.505920 | 10.869019 | +0.363098 | 0.0000 | 0.0000 |
| 2,000 | 10.506805 | 10.871704 | +0.364899 | 0.0000 | 0.0000 |

## Integrity and scope

Both runs completed with finite losses and gradients, no CUDA/OOM failure, and SHA-256-verified final checkpoints containing optimizer and RNG state. Gradient checkpointing remains unimplemented and disabled.

- A0 checkpoint: `0dedb8cde6edb3a80248ebef225db43032d274c9c3fc1efd485f547a1066b13e`
- A1 checkpoint: `035520e3891f9a36bc449f0cb1860fcab4ec2f1fd528c4e15c65ba593f1ea68d`
- Synthetic data hash: `4976c24949a6aaa7656153bc9059c6caee2c900d9f42fff22bbf2928fe778e50`
- Raw logs, checkpoints, and sidecars remain in ignored `results/runs/` directories.
