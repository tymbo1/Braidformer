# A1 value-trace intervention postdiagnostic plan

Status: **frozen read-only post-training diagnostic plan**.

## Question

The value-trace auxiliary objective improved native target loss on the hard
target-length tests but did not restore exact match. This required follow-up
reruns the two diagnostics that motivated the intervention:

1. layerwise key/value/binding representation forensics; and
2. four-fold cross-validated query/readout localization.

The purpose is to determine whether the auxiliary objective changed the
recurrent representation in the predicted direction, even if native decoding
remains below the exact-match boundary.

## Frozen scope and controls

- Use only the two completed intervention checkpoints: A1 value-trace staged
  and A1 value-trace shuffled, target lengths 2/4/8.
- Compare against the already committed matched A1 staged/shuffled controls;
  do not retrain controls.
- Reuse the exact layerwise and query/readout methods, deterministic examples,
  target positions, value pool, ridge settings, and fold assignment from the
  preceding committed diagnostics.
- No checkpoint, model parameter, architecture, configuration, or training
  artifact is modified. All probe weights are temporary in-memory objects.

Pinned sources:

```text
value-trace aggregate:
  results/a0_a1_relational_curriculum_value_trace_intervention.json
  sha256: 13dbace59f46a16ff6fc30e00532500b582f5a1ad5a8df8210297604e2532112
layerwise control aggregate sha256:
  b36749b40024b1f84b21fddbe825f60105f28cfe4bf99b21673d649f347268b3
query/readout control aggregate sha256:
  73dc7df64d9d57099d3f31ede2628ba673ed8cf716249fbf17b83164a8d77e50
```

## Measurements

Run the exact prior layerwise probe on every intervention checkpoint and record
key identity, held-out-fitted value, fixed pre-query transfer, binding pair, and
target-position metrics at every block/recurrent path. Run the exact prior
query/readout diagnostic and record native full-vocabulary/value-range logits,
four-fold fixed pre-query transfer, and site-specific value readouts.

For every metric, report intervention minus matched-control deltas by arm,
target length, path, and target position. Do not treat held-out-fitted probe
ceilings as train-to-held-out generalization.

## Interpretation boundary

- If the auxiliary objective raises cross-validated fixed/site-specific value
  readouts and native target loss improves, it supports the hypothesis that
  recurrent binding is trainable in the existing state.
- If it raises only held-out-fitted ceilings, it changes representation
  geometry without producing a stable transferable binding code.
- If diagnostics remain near control while native CE improves, the auxiliary
  objective may improve confidence/calibration without repairing the measured
  representation.
- These are read-only correlational diagnostics after the one authorized
  intervention; no further modification is authorized by this plan.

## Integrity and output

Verify intervention summary/checkpoint sidecars and hashes, source aggregate
hashes, parameter counts, capture paths, balanced folds, finite tensors and
metrics, and `read_only: true`. Fail nonzero on any source drift or malformed
output.

Output:

- `results/a0_a1_relational_curriculum_value_trace_postdiagnostics.json`
