# A0/A1 layerwise representation forensics result

Status: **complete; read-only frozen-checkpoint analysis**.

Analysis plan: `docs/A0_A1_LAYERWISE_REPRESENTATION_FORENSICS_PLAN.md` (sha256 `025b7d31a77d1d3d9c5a48fc4b46357a2322de7fa2355d98dc256e2cbd235115`).
Source protocol commit: `e133e5279761b73b390f8d52feacd63c5289368c`; source runner sha256 `2a1a1c84f1ebfabc11cf66f0defa2e2ad4db2d1741c89c3d12660d3ac066d3e5`.

The machine-readable aggregate contains per-site key/value/binding probes, fixed pre-query transfer probes, target-position probes, geometry, and all source checkpoint hashes. Probe weights were temporary in-memory diagnostics; no checkpoint or model parameter was modified.

## Summary by representation path

| arch | L | path | key heldout-fit top1 | value heldout-fit top1 | value fixed-pre-query transfer top1 | target-position heldout acc. |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| A0 | 2 | block_output_0 | 1.0000 | 1.0000 | 0.0078 | 1.0000 |
| A0 | 2 | block_output_1 | 1.0000 | 1.0000 | 0.0098 | 1.0000 |
| A0 | 2 | block_output_2 | 1.0000 | 1.0000 | 0.0088 | 1.0000 |
| A0 | 2 | block_output_3 | 1.0000 | 1.0000 | 0.0078 | 1.0000 |
| A0 | 2 | block_output_4 | 1.0000 | 1.0000 | 0.0078 | 1.0000 |
| A0 | 2 | block_output_5 | 1.0000 | 1.0000 | 0.0088 | 1.0000 |
| A0 | 2 | block_output_6 | 1.0000 | 1.0000 | 0.0088 | 1.0000 |
| A0 | 2 | block_output_7 | 1.0000 | 1.0000 | 0.0088 | 1.0000 |
| A0 | 2 | final_norm | 1.0000 | 1.0000 | 0.0088 | 1.0000 |
| A1 | 2 | block_output_0 | 1.0000 | 1.0000 | 0.0029 | 1.0000 |
| A1 | 2 | block_output_1 | 1.0000 | 1.0000 | 0.0039 | 1.0000 |
| A1 | 2 | block_output_2 | 1.0000 | 1.0000 | 0.0039 | 1.0000 |
| A1 | 2 | block_output_3 | 1.0000 | 1.0000 | 0.0029 | 1.0000 |
| A1 | 2 | block_output_4 | 1.0000 | 1.0000 | 0.0049 | 1.0000 |
| A1 | 2 | block_output_5 | 1.0000 | 1.0000 | 0.0039 | 1.0000 |
| A1 | 2 | block_output_6 | 1.0000 | 1.0000 | 0.0049 | 1.0000 |
| A1 | 2 | block_output_7 | 1.0000 | 1.0000 | 0.0039 | 1.0000 |
| A1 | 2 | final_norm | 1.0000 | 1.0000 | 0.0029 | 1.0000 |
| A1 | 2 | recurrent_state_0 | 1.0000 | 1.0000 | 0.0127 | 0.5195 |
| A1 | 2 | recurrent_state_1 | 1.0000 | 1.0000 | 0.0166 | 0.8555 |
| A1 | 2 | recurrent_state_2 | 1.0000 | 1.0000 | 0.0137 | 0.5508 |
| A1 | 2 | recurrent_state_3 | 1.0000 | 1.0000 | 0.0146 | 0.7070 |
| A1 | 2 | recurrent_state_4 | 1.0000 | 1.0000 | 0.0166 | 0.8164 |
| A1 | 2 | recurrent_state_5 | 1.0000 | 1.0000 | 0.0107 | 0.8320 |
| A1 | 2 | recurrent_state_6 | 1.0000 | 1.0000 | 0.0156 | 0.7812 |
| A1 | 2 | recurrent_state_7 | 1.0000 | 1.0000 | 0.0146 | 0.8359 |
| A0 | 4 | block_output_0 | 1.0000 | 1.0000 | 0.0029 | 0.9570 |
| A0 | 4 | block_output_1 | 1.0000 | 1.0000 | 0.0026 | 1.0000 |
| A0 | 4 | block_output_2 | 1.0000 | 1.0000 | 0.0026 | 0.9688 |
| A0 | 4 | block_output_3 | 1.0000 | 1.0000 | 0.0026 | 0.9727 |
| A0 | 4 | block_output_4 | 1.0000 | 1.0000 | 0.0026 | 0.9824 |
| A0 | 4 | block_output_5 | 1.0000 | 1.0000 | 0.0033 | 0.9883 |
| A0 | 4 | block_output_6 | 1.0000 | 1.0000 | 0.0039 | 0.9082 |
| A0 | 4 | block_output_7 | 1.0000 | 1.0000 | 0.0033 | 0.7676 |
| A0 | 4 | final_norm | 1.0000 | 1.0000 | 0.0036 | 0.7754 |
| A1 | 4 | block_output_0 | 1.0000 | 1.0000 | 0.0023 | 0.9746 |
| A1 | 4 | block_output_1 | 1.0000 | 1.0000 | 0.0023 | 0.9922 |
| A1 | 4 | block_output_2 | 1.0000 | 1.0000 | 0.0010 | 0.9980 |
| A1 | 4 | block_output_3 | 1.0000 | 1.0000 | 0.0010 | 0.9492 |
| A1 | 4 | block_output_4 | 1.0000 | 1.0000 | 0.0003 | 0.8340 |
| A1 | 4 | block_output_5 | 1.0000 | 1.0000 | 0.0003 | 0.9199 |
| A1 | 4 | block_output_6 | 1.0000 | 1.0000 | 0.0003 | 0.6230 |
| A1 | 4 | block_output_7 | 1.0000 | 1.0000 | 0.0003 | 0.7188 |
| A1 | 4 | final_norm | 1.0000 | 1.0000 | 0.0003 | 0.6875 |
| A1 | 4 | recurrent_state_0 | 1.0000 | 1.0000 | 0.0036 | 0.3867 |
| A1 | 4 | recurrent_state_1 | 1.0000 | 1.0000 | 0.0020 | 0.3242 |
| A1 | 4 | recurrent_state_2 | 1.0000 | 1.0000 | 0.0042 | 0.5820 |
| A1 | 4 | recurrent_state_3 | 1.0000 | 1.0000 | 0.0020 | 0.6484 |
| A1 | 4 | recurrent_state_4 | 1.0000 | 1.0000 | 0.0023 | 0.5195 |
| A1 | 4 | recurrent_state_5 | 1.0000 | 1.0000 | 0.0003 | 0.4043 |
| A1 | 4 | recurrent_state_6 | 1.0000 | 1.0000 | 0.0020 | 0.3711 |
| A1 | 4 | recurrent_state_7 | 1.0000 | 1.0000 | 0.0029 | 0.2812 |
| A0 | 8 | block_output_0 | 1.0000 | 1.0000 | 0.0012 | 1.0000 |
| A0 | 8 | block_output_1 | 1.0000 | 1.0000 | 0.0010 | 1.0000 |
| A0 | 8 | block_output_2 | 1.0000 | 1.0000 | 0.0012 | 1.0000 |
| A0 | 8 | block_output_3 | 1.0000 | 1.0000 | 0.0014 | 1.0000 |
| A0 | 8 | block_output_4 | 1.0000 | 1.0000 | 0.0015 | 0.9990 |
| A0 | 8 | block_output_5 | 1.0000 | 1.0000 | 0.0013 | 1.0000 |
| A0 | 8 | block_output_6 | 1.0000 | 1.0000 | 0.0019 | 0.9990 |
| A0 | 8 | block_output_7 | 1.0000 | 1.0000 | 0.0017 | 0.9766 |
| A0 | 8 | final_norm | 1.0000 | 1.0000 | 0.0014 | 0.9961 |
| A1 | 8 | block_output_0 | 1.0000 | 1.0000 | 0.0020 | 1.0000 |
| A1 | 8 | block_output_1 | 1.0000 | 1.0000 | 0.0007 | 1.0000 |
| A1 | 8 | block_output_2 | 1.0000 | 1.0000 | 0.0003 | 1.0000 |
| A1 | 8 | block_output_3 | 1.0000 | 1.0000 | 0.0004 | 1.0000 |
| A1 | 8 | block_output_4 | 1.0000 | 1.0000 | 0.0002 | 1.0000 |
| A1 | 8 | block_output_5 | 1.0000 | 1.0000 | 0.0002 | 0.9990 |
| A1 | 8 | block_output_6 | 1.0000 | 1.0000 | 0.0004 | 0.9902 |
| A1 | 8 | block_output_7 | 1.0000 | 1.0000 | 0.0004 | 0.9424 |
| A1 | 8 | final_norm | 1.0000 | 1.0000 | 0.0005 | 0.9434 |
| A1 | 8 | recurrent_state_0 | 1.0000 | 1.0000 | 0.0008 | 0.2900 |
| A1 | 8 | recurrent_state_1 | 1.0000 | 1.0000 | 0.0002 | 0.4482 |
| A1 | 8 | recurrent_state_2 | 1.0000 | 1.0000 | 0.0010 | 0.4385 |
| A1 | 8 | recurrent_state_3 | 1.0000 | 1.0000 | 0.0010 | 0.5264 |
| A1 | 8 | recurrent_state_4 | 1.0000 | 1.0000 | 0.0010 | 0.3779 |
| A1 | 8 | recurrent_state_5 | 1.0000 | 1.0000 | 0.0017 | 0.1914 |
| A1 | 8 | recurrent_state_6 | 1.0000 | 1.0000 | 0.0016 | 0.3418 |
| A1 | 8 | recurrent_state_7 | 1.0000 | 1.0000 | 0.0005 | 0.3389 |

## Interpretation boundary

Site-fitted held-out probes measure whether a linear decoder can recover information at a representation site; fixed pre-query transfer probes measure whether the same value coordinate survives later transformations. A drop in both supports information loss or destructive transformation. Strong site-fitted value probes with weak model output would shift attention toward query-conditioned use, routing, or readout. These are correlational findings; queue item 4 is required for causal evidence.

## Integrity

All six source runs were sidecar/hash verified before loading. Expected A0/A1 parameter counts, capture sites, recurrent layers, finite tensors, and finite probe metrics were required. Raw checkpoints/logs remain ignored under `results/runs/`.

## Findings and limitations

- The heldout-fitted key/value rows are near-perfect throughout because they are
  fitted and evaluated on the same 128 heldout examples. They are therefore
  decodability ceilings only; they do not show that the model learned a
  cross-example retrieval rule.
- The train-fitted-to-heldout value probes are near floor across the A0 and A1
  sites at lengths 4 and 8, and the fixed pre-query transfer probes are also
  near floor. The corresponding pair probes are not recovered by later target
  sites. This is consistent with a failure to preserve a stable, transferable
  value coordinate, but does not by itself distinguish information erasure from
  a changed representation coordinate or a query/readout failure.
- A1's recurrent-state target-position probes become substantially weaker with
  recurrent depth at length 4 (heldout position accuracy falls from `0.3867`
  at recurrent state 0 to `0.2812` at state 7). This is a position diagnostic,
  not a value-retrieval result, so it is evidence about state transformation,
  not proof of the mechanism.
- The present analysis transfers the fixed pre-query probe from the training
  split to the heldout split. Because the heldout examples use a separate key
  range and only 128 examples are available, that cross-split result conflates
  retention with representation-transfer/generalization. Queue item 3 will add
  within-heldout pre-query-to-post-query/target transfer and direct logit
  localization before any causal claim is made.

Accordingly, this item leaves H2, H3, H4, and H5 open. It weakens a simple
claim that the output-stage failure alone explains the result, but it does not
yet establish whether A1 loses the required value before the query, corrupts
it during query processing, or retains it in a coordinate the trained output
route cannot use.
