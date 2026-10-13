# A1 value-trace intervention postdiagnostic result

Status: **complete; read-only rerun of the motivating diagnostics**.

## Provenance and integrity

- Frozen postdiagnostic plan:
  `docs/A0_A1_RELATIONAL_CURRICULUM_VALUE_TRACE_POSTDIAGNOSTIC_PLAN.md`.
- Analysis commit: `1ac08489939275b0571f1c626e82060632ae79a3`.
- Aggregate: `results/a0_a1_relational_curriculum_value_trace_postdiagnostics.json`
  (sha256 `70b147d7632d6e8855abf1e70223a1bd865ed2f2eb9cef532ba7a90105f99328`).
- Six layerwise and six query/readout runs completed on the two auxiliary-trained
  A1 checkpoints at target lengths 2/4/8. All intervention checkpoint sidecars,
  finite tensors/metrics, capture paths, and source control hashes passed.
- No checkpoint or model parameter was modified. Full test suite remains
  `71 passed`; `git diff --check` passed.

## Layerwise representation deltas

These are intervention minus matched original-control values. The
held-out-fitted value metric is a decodability ceiling; the fixed metric is the
pre-query train-fitted transfer measure.

| arm | L | held-out-fitted value Δ mean | fixed-transfer Δ mean | interpretation |
| --- | ---: | ---: | ---: | --- |
| staged | 2 | 0.0000 | −0.0062 | already at ceiling; transfer did not improve |
| staged | 4 | −0.0002 | −0.0016 | no meaningful representation repair |
| staged | 8 | −0.0003 | −0.0000 | no meaningful representation repair |
| shuffled | 2 | +0.2114 | +0.0035 | ceiling rose, transfer remained near floor |
| shuffled | 4 | +0.1536 | +0.0006 | geometry/ceiling changed, binding transfer did not |
| shuffled | 8 | +0.0888 | +0.0001 | smaller ceiling change, transfer remained near floor |

The shuffled branch gained held-out-fitted decodability, while staged was
already near 1.0 in the corresponding control. Neither branch acquired a
material fixed pre-query transfer improvement. This is exactly the distinction
between changing representation geometry and learning a stable cross-example
binding coordinate.

## Query/readout deltas

Four-fold cross-validated target-site readouts remained near floor. Across the
selected A1 paths, intervention-minus-control native value top-1 averaged only
`+0.0039`, `+0.0039`, and `+0.0029` for staged target lengths 2/4/8, and
`0.0000` for all three shuffled lengths. Site-specific target-readout deltas
were similarly small: staged means `+0.0026`, `+0.0013`, `0.0000`; shuffled
means `0.0000` at all lengths. Fixed target transfer remained approximately
zero, with no coherent improvement.

Thus the native-loss improvement is not accompanied by a useful diagnostic
key/value readout. The auxiliary objective improved confidence/calibration or
some non-linearly represented information, but it did not establish a
linearly transferable query-conditioned binding.

## Final adjudication

- The intervention **partially succeeded** at reducing native held-out loss,
  especially on longer target outputs, so optimization pressure and training
  history are real modifiers.
- It **failed as a retrieval repair**: hard exact match remained at zero and
  cross-validated fixed/site-specific readouts remained near floor.
- H2 recurrent transformation/interference and H4 compositional
  binding/generalization remain the strongest supported mechanism pair.
- H1 temporal decay remains a contributor but is not sufficient; H3 query
  conditioning remains unlocalized; H5 pure final readout remains weakened.
- No second mechanism-targeted intervention is authorized in this campaign.

The bounded campaign is complete. The single best next experiment outside this
campaign is a fresh, pre-registered comparison of a **query-conditioned**
key→value auxiliary objective against this unconditioned value-trace objective,
using the same sealed held-out tests. It should explicitly score the query key
and target value jointly; that is the missing factor left unresolved here.
