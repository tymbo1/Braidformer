# BraidFormer A0/A1 distractor-distance solvability result

Status: **complete**. This records the one-association, 8000-value distractor-distance axis after the completed value-space and association-count tests.

Frozen execution commit: `3e8e8623d596d0878d0e7f49dea280d37335fcd7`.

## Final distance curve

The distance-0 row is the previously completed one-association values8000 baseline. Distances 4, 16, and 64 are the fresh runs in this axis.

| Distractor distance | Sequence length | Token positions | A0 held-out EM | A1 held-out EM | A0 held-out loss | A1 held-out loss | A1 − A0 loss | A1/A0 CUDA |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 8 | 256,000 | 0.0391 | 0.0391 | 9.947532 | 10.239311 | +0.291780 | 1.550× |
| 4 | 12 | 384,000 | 0.0469 | 0.0156 | 9.779089 | 10.466507 | +0.687418 | 1.721× |
| 16 | 24 | 768,000 | 0.0469 | 0.0000 | 9.673668 | 10.824158 | +1.150490 | 1.907× |
| 64 | 72 | 2,304,000 | 0.0469 | 0.0000 | 9.734173 | 10.974030 | +1.239856 | 2.221× |

## Interpretation

Both unchanged 42M models reached 100% training exact match at every distance rung. The distance manipulation therefore did not create a training-memorization failure.

A0 remained essentially flat: its held-out exact match was 4.69% at distances 4, 16, and 64, with target loss 9.7791, 9.6737, and 9.7342. A1 fell from 1.56% at distance 4 to 0% at distances 16 and 64, while its target loss rose from 10.4665 to 10.8242 and 10.9740.

This separates the present behaviors more clearly than the association axis alone. Both models still show the previously observed binding/generalization weakness, but A1 also shows a distance-sensitive degradation under this one-seed, fixed-step protocol. The result is consistent with a recurrent-state retention or state-utilization weakness; it does not by itself distinguish decay, interference, optimization, or readout failure.

The distance-0 baseline is not an equal-token control for the longer arms. Every fresh rung keeps 2,000 optimizer steps and effective batch 16, so longer distances process more token positions. The comparison is therefore a matched-step retention probe, with actual CUDA time and token exposure reported explicitly.

## Training memorization and resource summary

| Distance | A0 train EM | A1 train EM | A0 CUDA seconds | A1 CUDA seconds |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1.0000 | 1.0000 | 185.684 | 287.884 |
| 4 | 1.0000 | 1.0000 | 184.032 | 316.648 |
| 16 | 1.0000 | 1.0000 | 180.823 | 344.882 |
| 64 | 1.0000 | 1.0000 | 182.303 | 404.906 |

## Selected validation checkpoints

| Distance | Step | A0 held-out loss | A1 held-out loss | A1 − A0 | A0 exact | A1 exact |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 100 | 10.637713 | 11.274628 | +0.636914 | 0.0391 | 0.0312 |
| 0 | 500 | 9.940380 | 10.216232 | +0.275852 | 0.0391 | 0.0391 |
| 0 | 1,000 | 9.944639 | 10.232891 | +0.288252 | 0.0391 | 0.0391 |
| 0 | 1,500 | 9.947644 | 10.238060 | +0.290416 | 0.0391 | 0.0391 |
| 0 | 2,000 | 9.947532 | 10.239311 | +0.291780 | 0.0391 | 0.0391 |
| 4 | 100 | 10.844955 | 11.376724 | +0.531769 | 0.0469 | 0.0000 |
| 4 | 500 | 9.772680 | 10.397224 | +0.624544 | 0.0469 | 0.0156 |
| 4 | 1,000 | 9.775895 | 10.448051 | +0.672157 | 0.0469 | 0.0156 |
| 4 | 1,500 | 9.778861 | 10.464081 | +0.685220 | 0.0469 | 0.0156 |
| 4 | 2,000 | 9.779089 | 10.466507 | +0.687418 | 0.0469 | 0.0156 |
| 16 | 100 | 10.731719 | 11.355637 | +0.623918 | 0.0469 | 0.0000 |
| 16 | 500 | 9.652233 | 10.723419 | +1.071186 | 0.0469 | 0.0000 |
| 16 | 1,000 | 9.666409 | 10.787109 | +1.120700 | 0.0469 | 0.0000 |
| 16 | 1,500 | 9.672414 | 10.818512 | +1.146098 | 0.0469 | 0.0000 |
| 16 | 2,000 | 9.673668 | 10.824158 | +1.150490 | 0.0469 | 0.0000 |
| 64 | 100 | 10.737452 | 11.557541 | +0.820089 | 0.0469 | 0.0078 |
| 64 | 500 | 9.662354 | 10.822418 | +1.160065 | 0.0469 | 0.0000 |
| 64 | 1,000 | 9.712355 | 10.915680 | +1.203324 | 0.0469 | 0.0000 |
| 64 | 1,500 | 9.731005 | 10.965698 | +1.234694 | 0.0469 | 0.0000 |
| 64 | 2,000 | 9.734173 | 10.974030 | +1.239856 | 0.0469 | 0.0000 |

## Integrity and scope

All six fresh runs completed with finite losses and gradients, no CUDA/OOM failure, and SHA-256-verified final checkpoints containing optimizer and RNG state. The verifier checked every optimizer step, token counter, evaluation/checkpoint cadence, config hash, protocol hash, generator hash, data hash, checkpoint metadata, and final checkpoint digest.

- Fresh raw artifacts remain in ignored `results/runs/` directories.
- The distance-0 baseline remains the tracked result in `results/a0_a1_solvability_values8000.json`.
- No A1 architecture change, relational curriculum, A2 store, predictive-learning branch, scaling run, or memory work was introduced.
