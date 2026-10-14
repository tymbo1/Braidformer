# A1 protected associative-lane distance-axis result

Status: **complete; minimal protected-lane branch closed negative**.

This report completes the frozen protected-lane falsification protocol after
the original ordinary distance-4 arm stopped on a filesystem checkpoint-write
failure. Storage was remediated before continuation. The failed arm was resumed
from its verified step-300 checkpoint into a fresh directory; the missing
step-400 file was never treated as evidence.

## Question and controls

The experiment asked whether a 64-wide parallel associative lane could preserve
one key/value binding through distractors and improve held-out retrieval as
distance increased. Protected and ordinary lane variants used the same A1 base,
seed 17, 8,000-value universe, one association, 2,000 optimizer steps,
optimizer, data splits, and checkpoint/evaluation schedule. Original A0 and
original A1 remained untouched controls.

The protocol and implementation were frozen at `32fd4d0`, `d6358bf`, and
`b48ddc2`. The storage continuation protocol was frozen at `289b397`.

## Final results

| Distance | Protected held-out EM | Ordinary held-out EM | Protected loss | Ordinary loss | Protected − ordinary loss |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3.90625% | 3.90625% | 10.248867 | 10.265778 | −0.016911 |
| 4 | 0.78125% | 0.78125% | 10.414185 | 10.554665 | −0.140480 |
| 16 | 0% | 0% | 10.949066 | 10.671906 | +0.277161 |
| 64 | 0% | 0% | 10.891357 | 10.842102 | +0.049255 |

All eight completed arms reached 100% training exact match. Protected never
improved held-out exact match. Its target loss was slightly lower at distances
0 and 4, then higher at distances 16 and 64; the direction was not consistent
with a protected-persistence benefit. Value-candidate loss showed the same
non-monotonic pattern.

The complete machine-readable aggregate is
`results/a1_protected_lane_distance_axis_values8000.json`; its read-only
aggregator is `scripts/analyze_a1_protected_lane_completed_axis.py`.

## Mechanistic reading

The protected lane did perform the narrow operation it was designed to test.
At distances 16 and 64, its post-value and post-distractor memory norms were
identical, while the ordinary lane's memory norms changed substantially. The
protected read gates were high (`0.969` at distance 16 and `0.964` at distance
64), so the null result is not explained by the protected branch simply
forgetting to hold its state.

However, preserved lane state did not become a transferable held-out answer.
The result therefore rejects the narrow hypothesis:

> protecting a single learned key/value state path is sufficient to restore
> A1's long-distance retrieval.

The strongest supported interpretation remains composite: state preservation
addresses one component of distractor interference, but binding formation and
query-conditioned integration/use of the preserved representation remain
insufficient. A protected lane by itself is not an adequate associative
primitive. This branch does not establish that every structured memory design
will fail, and it does not authorize stacking another architectural change.

## Integrity and interruption handling

The resumed distance-4 ordinary arm completed at step 2,000 with a verified
final checkpoint. The other seven arms also completed with integrity-confirmed
final checkpoints. The aggregate structurally verified 158 checkpoint records:
20 for each fresh arm and 18 for the resumed arm, including its preserved
step-300 source checkpoint. The original interrupted audit remains at
`results/a1_protected_lane_distance_axis_interrupted.json`; its valid source
steps were 100, 200, and 300, and its missing step-400 checkpoint remains a
failure artifact rather than a scientific result.

No A0/A1 definition was changed, no analysis modified weights, no secrets or
raw run directories were tracked, and nothing was pushed to GitHub.
