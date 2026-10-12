# BraidFormer A0/A1 two-association, 8000-value solvability result

Status: **complete**. This is the first association-count rung after the completed one-association value-space ladder.

Frozen protocol commit: `3079c74eb0e321316520093ed4ac3a3515006f4e`.

## Final measurements

| Metric | A0 | A1 |
| --- | ---: | ---: |
| Parameters | 41,866,240 | 41,870,336 |
| Associations | 2 | 2 |
| Reserved answer tokens | 8,000 | 8,000 |
| Sequence length | 11 | 11 |
| Optimizer steps | 2,000 | 2,000 |
| Token positions | 352,000 | 352,000 |
| Final training target loss | 0.000662 | 0.000629 |
| Training greedy exact match | 1.0000 | 1.0000 |
| Final held-out target loss | 10.385956 | 10.544830 |
| Held-out greedy exact match | 0.0156 | 0.0000 |
| CUDA training seconds | 182.455 | 316.698 |
| Elapsed wall seconds | 216.137 | 358.368 |
| CUDA tokens/second | 1929.2 | 1111.5 |

## Interpretation

Both unchanged 42M models passed the primary training-memorization test: they reached 100% autoregressive one-token retrieval on the repeated training set with two explicit bindings and the 8,000-token candidate answer pool.

On the diagnostic held-out set, A0 reached `0.0156` (2/128) and A1 `0.0000` (0/128). The paired A1 − A0 target-loss delta was `+0.158875`, and the exact-match delta was `-0.0156`. A0 therefore retained the diagnostic advantage at two associations.

Relative to the one-association 8000-value rung, A0 held-out exact match fell from 0.0391 to 0.0156 and A1 from 0.0391 to 0.0000 under this 2,000-optimizer-step recipe. Both models still memorize the fixed training set, so this is a held-out retrieval/generalization effect rather than a training-memorization boundary.

A1 used `1.736×` A0's CUDA time and achieved `0.577×` A0's average token throughput. This was not a matched-compute comparison. The two-association rung itself used the same 352,000 token-position budget for A0 and A1.

The next association-count rung is four associations, but it is not frozen yet. Distractor distance, target length, decay analysis, predictive learning-state or competence-ledger work, scaling, A2, and memory remain deferred.

## Selected validation checkpoints

| Step | A0 held-out loss | A1 held-out loss | A1 − A0 | A0 exact | A1 exact |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 11.170555 | 11.691589 | +0.521034 | 0.0156 | 0.0000 |
| 500 | 10.393402 | 10.503326 | +0.109924 | 0.0156 | 0.0000 |
| 1,000 | 10.387917 | 10.534454 | +0.146538 | 0.0156 | 0.0000 |
| 1,500 | 10.386124 | 10.543274 | +0.157150 | 0.0156 | 0.0000 |
| 2,000 | 10.385956 | 10.544830 | +0.158875 | 0.0156 | 0.0000 |

## Integrity and scope

Both runs completed with finite losses and gradients, no CUDA/OOM failure, and SHA-256-verified final checkpoints containing optimizer and RNG state. Gradient checkpointing remains unimplemented and disabled.

- A0 checkpoint: `90350c157f884679d58d807efad30a043cee503b906cbe28c1420024bb9b9371`
- A1 checkpoint: `edd73e658ee4725da7333d94cc7c12fa95afab1df6e6479711a790b5e672cc03`
- Synthetic data hash: `f8a1bb028658570d86b41c20162af4309c0ad5dde5d45f718e027d185aa23962`
- Raw logs, checkpoints, and sidecars remain in ignored `results/runs/` directories.
