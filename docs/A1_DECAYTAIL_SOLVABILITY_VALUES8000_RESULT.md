# A1 decay-tail intervention on the 8000-value distance axis

Status: **complete; all three runs and final checkpoints verified**.

Frozen intervention commit: `515f9d99e359789ee65e0291d25904ab8717768c`.

The intervention changes only fresh A1 state-transition initialization: 75% of each layer remains at the original decay initialization and 25% is split across initial τ=16, 32, 64, and 128 token channels. The original A1 controls remain unchanged.

| Distance | Control EM | Tail EM | Δ EM | Control loss | Tail loss | Δ loss | Train EM |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 0.0156 | 0.0078 | -0.0078 | 10.466507 | 10.702339 | +0.235832 | 1.0000 |
| 16 | 0.0000 | 0.0000 | +0.0000 | 10.824158 | 10.891266 | +0.067108 | 1.0000 |
| 64 | 0.0000 | 0.0000 | +0.0000 | 10.974030 | 10.857605 | -0.116425 | 1.0000 |

## Immediate task-level result

All three intervention models reached 100% training exact match, so the initialization change did not prevent memorization. It did not improve held-out exact match on this seed: distance 4 fell from 1.5625% to 0.78125%, while distances 16 and 64 remained at 0%. Target loss was worse at distances 4 and 16, but improved at distance 64 by 0.116425 despite no exact-match gain.

This is not yet the final mechanistic conclusion. The fixed state probe below is the predeclared discriminator: improved probe retention with unchanged task performance would mean storage improved but query/readout remained limiting; unchanged probe retention would weaken the decay-causality hypothesis.

## Integrity

Each run has 2,000 contiguous step records, the expected 100-step validation cadence, matching generator/data hashes, a frozen protocol hash, and a sidecar-verified final checkpoint.
