# A0/A1 state-replay intervention result

Status: **complete; reversible inference-only intervention**.

Intervention plan: `docs/A0_A1_STATE_REPLAY_INTERVENTION_PLAN.md` (sha256 `508e553810158e2c9e1403f590b0be37365728c1978d2468ff2cee7ebdb6d25e`).
Source protocol commit: `e133e5279761b73b390f8d52feacd63c5289368c`; source runner sha256 `2a1a1c84f1ebfabc11cf66f0defa2e2ad4db2d1741c89c3d12660d3ac066d3e5`.

The intervention copied each target value's layer-7 recurrent state from its stored-value position to the corresponding target-output position. No model parameter or checkpoint was changed.

## Held-out A1 baseline versus replay

| L | target position | baseline value top1 | replay value top1 | Δ top1 | baseline value CE | replay value CE | Δ mean rank |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 1 | 0.0078 | 0.0078 | +0.0000 | 9.5760 | 9.5861 | -4.1 |
| 2 | 2 | 0.0000 | 0.0000 | +0.0000 | 9.3654 | 9.3662 | +0.7 |
| 4 | 1 | 0.0000 | 0.0000 | +0.0000 | 9.3608 | 9.3786 | +4.5 |
| 4 | 2 | 0.0000 | 0.0000 | +0.0000 | 9.1124 | 9.1151 | -0.8 |
| 4 | 3 | 0.0000 | 0.0000 | +0.0000 | 9.6288 | 9.6331 | -1.3 |
| 4 | 4 | 0.0000 | 0.0000 | +0.0000 | 9.3452 | 9.3492 | +3.2 |
| 8 | 1 | 0.0000 | 0.0000 | +0.0000 | 9.6014 | 9.6432 | -1.9 |
| 8 | 2 | 0.0000 | 0.0000 | +0.0000 | 9.4634 | 9.4800 | -5.6 |
| 8 | 3 | 0.0000 | 0.0000 | +0.0000 | 9.4516 | 9.4619 | -10.1 |
| 8 | 4 | 0.0000 | 0.0000 | +0.0000 | 9.4748 | 9.4892 | +0.1 |
| 8 | 5 | 0.0000 | 0.0000 | +0.0000 | 9.3906 | 9.4050 | +3.5 |
| 8 | 6 | 0.0000 | 0.0000 | +0.0000 | 9.2255 | 9.2383 | +3.5 |
| 8 | 7 | 0.0000 | 0.0000 | +0.0000 | 9.4135 | 9.4308 | +21.0 |
| 8 | 8 | 0.0000 | 0.0000 | +0.0000 | 9.3194 | 9.3270 | -2.4 |

## Predeclared prediction

The prediction was that replay would improve value rank/top-1 at one or more positions, especially at lengths 4 and 8, if temporal retention/state transformation were causal. The machine-readable deltas above are authoritative. A null or worsening result means this particular layer-7 state-preservation operation did not restore retrieval; no second intervention is authorized in this campaign.

## Finding

The prediction failed. Replay produced no top-1 improvement at any of the 14
target positions. Mean value-range cross-entropy became slightly worse at
every position, while rank changes were small and mixed. The before/after
parameter digests were identical for all three A1 checkpoints.

This rejects the narrow hypothesis that copying the layer-7 recurrent state
from each stored-value token directly into its matching output position is
sufficient to restore retrieval. It does not disprove every temporal or
state-transformation mechanism: the intervention changes the state after the
query, preserves the target output gate, and tests only one layer and one
state-replay rule. No further intervention is authorized; item 5 must adjudicate
the evidence without stacking modifications.

## Integrity

All six A0/A1 source runs were sidecar/hash verified. A1 parameter digests before and after each functional intervention were unchanged; source/target positions and all baseline/patched tensors were finite. Raw checkpoints/logs remain ignored under `results/runs/`.
