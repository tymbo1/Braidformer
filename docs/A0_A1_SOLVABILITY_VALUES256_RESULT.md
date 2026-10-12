# BraidFormer A0/A1 256-value solvability-rung result

Status: **complete**. This is the next calibrated value-space rung after the completed 64- and 128-value one-association tests.

Frozen protocol commit: `5b02db578f147a9801d092d9a4fffea23962d5ac`.

## Final measurements

| Metric | A0 | A1 |
| --- | ---: | ---: |
| Parameters | 41,866,240 | 41,870,336 |
| Reserved answer tokens | 256 | 256 |
| Optimizer steps | 2,000 | 2,000 |
| Token positions | 256,000 | 256,000 |
| Final training target loss | 0.000584 | 0.000571 |
| Training greedy exact match | 1.0000 | 1.0000 |
| Final held-out target loss | 5.813058 | 6.850369 |
| Held-out greedy exact match | 0.4297 | 0.3516 |
| CUDA training seconds | 184.821 | 285.427 |
| Elapsed wall seconds | 222.427 | 327.392 |
| CUDA tokens/second | 1385.1 | 896.9 |

## Interpretation

Both unchanged 42M models again passed the primary memorization test: they reached 100% autoregressive one-token retrieval on the repeated training set with the 256-token candidate answer pool.

On the diagnostic held-out set, A0 reached `0.4297` and A1 `0.3516`. The paired A1 − A0 target-loss delta was `+1.037312` and the A1 − A0 exact-match delta was `-0.0781`. A0 remained ahead on this diagnostic at the same seed and budget. The held-out set measures transfer of the simple copy/retrieve operation, not unseen fact learning.

Relative to the completed 128-value rung, held-out exact match moved from 0.6484 to 0.4297 for A0 and from 0.5938 to 0.3516 for A1. This is a clear difficulty effect and a wider A0-minus-A1 separation, but neither model has yet failed the training memorization criterion or reached a zero held-out boundary.

A1 used `1.544×` A0's CUDA time and achieved `0.648×` A0's average token throughput. This was not a matched-compute comparison.

No values1024/8000 rung, association-count change, distractor-distance change, target-length change, decay analysis, predictive-training branch, scaling, A2, or memory experiment was started.

## Selected validation checkpoints

| Step | A0 held-out loss | A1 held-out loss | A1 − A0 | A0 exact | A1 exact |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 6.421840 | 7.972145 | +1.550305 | 0.4297 | 0.2656 |
| 500 | 5.813258 | 6.947936 | +1.134678 | 0.4297 | 0.3438 |
| 1,000 | 5.810979 | 6.877240 | +1.066261 | 0.4297 | 0.3516 |
| 1,500 | 5.812501 | 6.854749 | +1.042248 | 0.4297 | 0.3516 |
| 2,000 | 5.813058 | 6.850369 | +1.037312 | 0.4297 | 0.3516 |

## Integrity and scope

Both runs completed with finite losses and gradients, no CUDA/OOM failure, and SHA-256-verified final checkpoints containing optimizer and RNG state. Gradient checkpointing remains unimplemented and disabled.

- A0 checkpoint: `c26ec8a4d5829364d953b4f65e079d0a72d667f5d798345b10757909222cb092`
- A1 checkpoint: `7f306e41b6a0995fea8fe19e44b9acbd45516950c4a9834c3977f5993dd98216`
- Synthetic data hash: `fd2c8dd49e6eaa890f1bbf1a1a97677664c31c0d2af69ab296606d40641d033a`
- Raw logs, checkpoints, and sidecars remain in ignored `results/runs/` directories.
