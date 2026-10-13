# A1 query-conditioned binding postdiagnostic result

Status: **complete; no stable transferable binding repair detected**.

The exact frozen layerwise representation and query/readout analyses were
rerun on both query-conditioned A1 checkpoints at target lengths 2/4/8.

## Provenance

- Plan/code freeze: `ed55b28f4b29dd4ec7cb3ad5c0a61ba68e7fc0ef`.
- Training result aggregate:
  `results/a0_a1_relational_curriculum_query_binding_intervention.json`
  (sha256 `5f63c688985ac049047f5e8b34b6c359b1eeeea28de5069728d71a80dcbad9a2`).
- Postdiagnostic aggregate:
  `results/a0_a1_relational_curriculum_query_binding_postdiagnostics.json`
  (sha256 `03ddd4e90b8cbe78fd264180b99669a36d33bf9ae707e667ef71981509bcbeb9`).
- Six layerwise and six query/readout runs completed; no source checkpoint or
  parameter was modified.

## Findings

Staged A1 was already near the held-out-fitted value ceiling in the original
control. Query-conditioned training did not improve its fixed pre-query value
transfer: mean deltas were −0.0065, −0.0020, and −0.0006 at target lengths
2/4/8. Native query/readout value top-1 moved only +0.0078, +0.0020, and 0;
site-specific target transfer remained approximately zero.

Shuffled A1 showed a larger held-out-fitted ceiling, with mean value deltas
about +0.163, +0.210, and +0.270 at target lengths 2/4/8. However, fixed
pre-query transfer moved only +0.0021, +0.0001, and +0.0002. Query/readout
native and site-specific transfer remained at or near zero.

The increase in shuffled held-out-fit decodability is therefore a geometry or
ceiling change, not evidence of a stable arbitrary binding coordinate. The
diagnostic prediction failed in both arms.

## Final interpretation

The intervention improved some continuous native metrics without restoring
systematic retrieval. The evidence continues to support recurrent
transformation/interference plus compositional binding/generalisation failure;
it does not support treating a simple query-position auxiliary score as a
sufficient repair. No second intervention is authorized in this branch.
