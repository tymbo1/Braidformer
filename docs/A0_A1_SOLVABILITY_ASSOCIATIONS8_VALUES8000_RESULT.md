# BraidFormer A0/A1 eight-association, 8000-value solvability result

Status: **complete**. This finishes the predeclared association-count axis after the completed one-association value-space ladder and two- and four-association tests.

Frozen protocol commit: `cde7170085108aced51cbd79e602cad2d00acf0e`.

## Final measurements

| Metric | A0 | A1 |
| --- | ---: | ---: |
| Parameters | 41,866,240 | 41,870,336 |
| Associations | 8 | 8 |
| Reserved answer tokens | 8,000 | 8,000 |
| Sequence length | 29 | 29 |
| Optimizer steps | 2,000 | 2,000 |
| Token positions | 928,000 | 928,000 |
| Final training target loss | 0.000713 | 0.000635 |
| Training greedy exact match | 1.0000 | 1.0000 |
| Final held-out target loss | 10.917267 | 10.812836 |
| Held-out greedy exact match | 0.0000 | 0.0000 |
| CUDA training seconds | 187.689 | 347.492 |
| Elapsed wall seconds | 222.120 | 389.554 |
| CUDA tokens/second | 4944.3 | 2670.6 |

## Interpretation

Both unchanged 42M models passed the primary training-memorization test: they reached 100% autoregressive one-token retrieval on the repeated training set with eight explicit bindings and the 8,000-token candidate answer pool.

On the diagnostic held-out set, both models reached `0.0000` (`0/128`). The paired A1 − A0 target-loss delta was `-0.104431`, and the exact-match delta was `+0.0000`. Exact-match separation is at the floor, while target loss still distinguishes the models.

Relative to the four-association 8000-value rung, held-out exact match remained at 0.0000 for both architectures under this 2,000-optimizer-step recipe. Both models still memorize the fixed training set, so the association-count axis reaches a held-out floor without establishing a training-memorization boundary.

A1 used `1.851×` A0's CUDA time and achieved `0.542×` A0's average token throughput. This was not a matched-compute comparison. The eight-association rung itself used the same 928,000 token-position budget for A0 and A1.

The predeclared association-count axis is complete. The next work should be selected from the remaining calibrated axes—such as distractor distance or target length—or from the later non-invasive A1 forensics. Predictive learning-state, competence-ledger, scaling, A2, and memory remain deferred.

## Selected validation checkpoints

| Step | A0 held-out loss | A1 held-out loss | A1 − A0 | A0 exact | A1 exact |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 11.532318 | 11.956879 | +0.424561 | 0.0000 | 0.0000 |
| 500 | 10.794830 | 10.722656 | -0.072174 | 0.0000 | 0.0000 |
| 1,000 | 10.872681 | 10.787628 | -0.085052 | 0.0000 | 0.0000 |
| 1,500 | 10.909943 | 10.808197 | -0.101746 | 0.0000 | 0.0000 |
| 2,000 | 10.917267 | 10.812836 | -0.104431 | 0.0000 | 0.0000 |

## Integrity and scope

Both runs completed with finite losses and gradients, no CUDA/OOM failure, and SHA-256-verified final checkpoints containing optimizer and RNG state. Gradient checkpointing remains unimplemented and disabled.

- A0 checkpoint: `bb241a5adeded44d52707f95dcfd0508b7df849f2962a24a6c911dbd4a7c4dc9`
- A1 checkpoint: `dc1c0c1fd4c418f5c8cc509f2035bd53705d7fd03832ed2dc095244df66d914b`
- Synthetic data hash: `a1ed0fd53aa8fcc6a4d42a3d27fdbf26286ee6dd6e297b42c40d81783935042e`
- Raw logs, checkpoints, and sidecars remain in ignored `results/runs/` directories.
