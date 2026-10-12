# A0/A1 query/readout localization result

Status: **complete; read-only frozen-checkpoint analysis**.

Analysis plan: `docs/A0_A1_QUERY_READOUT_LOCALIZATION_PLAN.md` (sha256 `1ef1dda8188eeb3f49c97ebc40589c70fa76930c7af2eeeac59e8ac5dd96ef82`).
Source protocol commit: `e133e5279761b73b390f8d52feacd63c5289368c`; source runner sha256 `2a1a1c84f1ebfabc11cf66f0defa2e2ad4db2d1741c89c3d12660d3ac066d3e5`.

The native metrics use teacher-forced target positions. Diagnostic readouts are four-fold cross-validated within the heldout set; no same-sample fitted ceiling is reported.

## Native held-out output metrics

| arch | L | target position | full-vocab top1 | value-range top1 | value-range CE | value-range mean rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A0 | 2 | 1 | 0.0000 | 0.0000 | 9.4044 | 3111.0 |
| A0 | 2 | 2 | 0.0078 | 0.0078 | 9.4520 | 2957.4 |
| A1 | 2 | 1 | 0.0078 | 0.0078 | 9.5760 | 3282.0 |
| A1 | 2 | 2 | 0.0000 | 0.0000 | 9.3654 | 3760.6 |
| A0 | 4 | 1 | 0.0234 | 0.0234 | 8.6561 | 1360.0 |
| A0 | 4 | 2 | 0.0781 | 0.0781 | 8.2270 | 1281.5 |
| A0 | 4 | 3 | 0.0078 | 0.0078 | 9.0325 | 2638.2 |
| A0 | 4 | 4 | 0.0078 | 0.0078 | 9.0503 | 2811.0 |
| A1 | 4 | 1 | 0.0000 | 0.0000 | 9.3608 | 3434.1 |
| A1 | 4 | 2 | 0.0000 | 0.0000 | 9.1124 | 3388.0 |
| A1 | 4 | 3 | 0.0000 | 0.0000 | 9.6288 | 3643.1 |
| A1 | 4 | 4 | 0.0000 | 0.0000 | 9.3452 | 3676.0 |
| A0 | 8 | 1 | 0.0312 | 0.0312 | 8.7548 | 2472.7 |
| A0 | 8 | 2 | 0.0234 | 0.0234 | 8.6801 | 2542.2 |
| A0 | 8 | 3 | 0.0000 | 0.0000 | 8.8746 | 3059.3 |
| A0 | 8 | 4 | 0.0078 | 0.0078 | 8.8705 | 2898.4 |
| A0 | 8 | 5 | 0.0000 | 0.0000 | 8.9599 | 2902.5 |
| A0 | 8 | 6 | 0.0000 | 0.0000 | 8.8842 | 2719.9 |
| A0 | 8 | 7 | 0.0000 | 0.0000 | 8.7986 | 2557.9 |
| A0 | 8 | 8 | 0.0078 | 0.0078 | 8.8600 | 2989.5 |
| A1 | 8 | 1 | 0.0000 | 0.0000 | 9.6014 | 3994.2 |
| A1 | 8 | 2 | 0.0000 | 0.0000 | 9.4634 | 3880.7 |
| A1 | 8 | 3 | 0.0000 | 0.0000 | 9.4516 | 3804.7 |
| A1 | 8 | 4 | 0.0000 | 0.0000 | 9.4748 | 4025.9 |
| A1 | 8 | 5 | 0.0000 | 0.0000 | 9.3906 | 3762.1 |
| A1 | 8 | 6 | 0.0000 | 0.0000 | 9.2255 | 3737.5 |
| A1 | 8 | 7 | 0.0000 | 0.0000 | 9.4135 | 4004.4 |
| A1 | 8 | 8 | 0.0000 | 0.0000 | 9.3194 | 4142.7 |

## Selected fixed-pre-query and diagnostic readouts

The fixed columns use the same pre-query probe at each later site. The site-specific column fits the probe at that site. `target` means the site corresponding to the requested target position.

| arch | L | path | target position | native value top1 | fixed pre top1 | fixed post top1 | fixed target top1 | site-specific target top1 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A0 | 2 | final_norm | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A0 | 2 | final_norm | 2 | 0.0078 | 0.0078 | 0.0000 | 0.0000 | 0.0078 |
| A1 | 2 | final_norm | 1 | 0.0078 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A1 | 2 | final_norm | 2 | 0.0000 | 0.0234 | 0.0078 | 0.0000 | 0.0000 |
| A1 | 2 | recurrent_state_0 | 1 | 0.0078 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A1 | 2 | recurrent_state_0 | 2 | 0.0000 | 0.0156 | 0.0156 | 0.0078 | 0.0000 |
| A1 | 2 | recurrent_state_7 | 1 | 0.0078 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A1 | 2 | recurrent_state_7 | 2 | 0.0000 | 0.0078 | 0.0000 | 0.0000 | 0.0000 |
| A0 | 4 | final_norm | 1 | 0.0234 | 0.0000 | 0.0000 | 0.0000 | 0.0156 |
| A0 | 4 | final_norm | 2 | 0.0781 | 0.0156 | 0.0078 | 0.0000 | 0.0156 |
| A0 | 4 | final_norm | 3 | 0.0078 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A0 | 4 | final_norm | 4 | 0.0078 | 0.0156 | 0.0000 | 0.0000 | 0.0078 |
| A1 | 4 | final_norm | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A1 | 4 | final_norm | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0078 |
| A1 | 4 | final_norm | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A1 | 4 | final_norm | 4 | 0.0000 | 0.0156 | 0.0000 | 0.0000 | 0.0000 |
| A1 | 4 | recurrent_state_0 | 1 | 0.0000 | 0.0078 | 0.0000 | 0.0000 | 0.0000 |
| A1 | 4 | recurrent_state_0 | 2 | 0.0000 | 0.0078 | 0.0078 | 0.0000 | 0.0078 |
| A1 | 4 | recurrent_state_0 | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A1 | 4 | recurrent_state_0 | 4 | 0.0000 | 0.0156 | 0.0078 | 0.0078 | 0.0000 |
| A1 | 4 | recurrent_state_7 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A1 | 4 | recurrent_state_7 | 2 | 0.0000 | 0.0078 | 0.0078 | 0.0078 | 0.0078 |
| A1 | 4 | recurrent_state_7 | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A1 | 4 | recurrent_state_7 | 4 | 0.0000 | 0.0078 | 0.0000 | 0.0000 | 0.0000 |
| A0 | 8 | final_norm | 1 | 0.0312 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A0 | 8 | final_norm | 2 | 0.0234 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A0 | 8 | final_norm | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0078 |
| A0 | 8 | final_norm | 4 | 0.0078 | 0.0078 | 0.0000 | 0.0000 | 0.0000 |
| A0 | 8 | final_norm | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A0 | 8 | final_norm | 6 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A0 | 8 | final_norm | 7 | 0.0000 | 0.0000 | 0.0000 | 0.0078 | 0.0156 |
| A0 | 8 | final_norm | 8 | 0.0078 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A1 | 8 | final_norm | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A1 | 8 | final_norm | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A1 | 8 | final_norm | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A1 | 8 | final_norm | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A1 | 8 | final_norm | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A1 | 8 | final_norm | 6 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A1 | 8 | final_norm | 7 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A1 | 8 | final_norm | 8 | 0.0000 | 0.0234 | 0.0000 | 0.0000 | 0.0000 |
| A1 | 8 | recurrent_state_0 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A1 | 8 | recurrent_state_0 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A1 | 8 | recurrent_state_0 | 3 | 0.0000 | 0.0078 | 0.0078 | 0.0000 | 0.0000 |
| A1 | 8 | recurrent_state_0 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A1 | 8 | recurrent_state_0 | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A1 | 8 | recurrent_state_0 | 6 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A1 | 8 | recurrent_state_0 | 7 | 0.0000 | 0.0156 | 0.0156 | 0.0156 | 0.0078 |
| A1 | 8 | recurrent_state_0 | 8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A1 | 8 | recurrent_state_7 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A1 | 8 | recurrent_state_7 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A1 | 8 | recurrent_state_7 | 3 | 0.0000 | 0.0156 | 0.0078 | 0.0000 | 0.0078 |
| A1 | 8 | recurrent_state_7 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A1 | 8 | recurrent_state_7 | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A1 | 8 | recurrent_state_7 | 6 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| A1 | 8 | recurrent_state_7 | 7 | 0.0000 | 0.0078 | 0.0000 | 0.0000 | 0.0000 |
| A1 | 8 | recurrent_state_7 | 8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Interpretation boundary

A fixed-probe drop from pre-query to post-query/target supports a changed or damaged value code; a strong site-specific probe with weak native output would instead support a readout/routing gap. These are correlational frozen-checkpoint measurements. Queue item 4 is required for causal evidence.

## Findings

- On the heldout split, A1's native value-range top-1 is zero at every target
  position for lengths 4 and 8. Its mean value-candidate target ranks are
  roughly `3,388--3,676` at length 4 and `3,737--4,143` at length 8, out of
  8,000 candidates.
- Four-fold fixed-pre-query transfer remains at or near floor through the
  query and target sites. For example, at length 4, A1 recurrent state 0 for
  target 1 moves from mean rank `2,636` before the query to `2,869` after it
  and `2,955` at target 1; recurrent state 7 moves from `2,670` to `3,072`
  and `3,096`. At length 8 the same paths remain near random value ranking and
  do not recover at target sites.
- Site-specific diagnostic probes also remain at floor on the selected A1
  paths (at most `0.0078` top-1 in the displayed summaries), so there is no
  strong evidence that a linearly decodable value survives while only the
  native output head fails. This weakens H5, the pure readout/routing
  hypothesis.
- A0's diagnostic probes likewise do not expose a clean hidden-value code;
  its better native output therefore cannot be described as a simple
  readout-only advantage. The result is compatible with a model-specific
  transformation/attention computation that is not captured by a stable
  linear value coordinate.

The result favours a representation/binding or recurrent-transformation
failure over a pure final-readout failure, but does not distinguish temporal
decay from query-conditioned corruption or other state interference. Queue
item 4 is therefore a reversible inference-time state intervention, not a
training change.

## Integrity

All six source runs were sidecar/hash verified before loading. Expected parameter counts, capture sites, finite logits, finite representations, balanced folds, and finite diagnostic metrics were required. Raw checkpoints/logs remain ignored under `results/runs/`.
