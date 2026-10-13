# A1 relational-curriculum value-trace intervention result

Status: **complete; one authorized mechanism-targeted training intervention**.

## Result in one sentence

The layer-7 value-trace auxiliary objective improved native target loss on the
hard values8000/distance/target-length tests, but it did not restore exact
retrieval and did not add parameters or create a stable transfer-binding code.
It is a partial confidence/optimization improvement, not a mechanism repair.

## Provenance and integrity

- Frozen plan: `docs/A0_A1_RELATIONAL_CURRICULUM_VALUE_TRACE_INTERVENTION_PLAN.md`.
- Protocol/code commit: `fdcd032d640053084b84c204b8e737fe72242464`.
- Run aggregate commit: `1657da5ec8d3c20ae71eb6bcd0c49f7ff3789083`.
- Aggregate: `results/a0_a1_relational_curriculum_value_trace_intervention.json`
  (sha256 `13dbace59f46a16ff6fc30e00532500b582f5a1ad5a8df8210297604e2532112`).
- Two fresh A1 runs completed: staged and shuffled, each 2,000 steps and
  928,000 tokens. Checkpoint sidecars and digests passed verification.
- Parameter count remained 41,870,336; no new parameters were introduced and
  original A0/A1 definitions and source checkpoints were untouched.
- Full test suite after the protocol/run work: `71 passed`; `git diff --check`
  passed.

## Native held-out outcome versus matched controls

Negative loss deltas are improvements. Exact match is native greedy exact
match on the held-out split.

| arm | test | control loss | intervention loss | Δ loss | control EM | intervention EM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| staged | values8000 | 9.7759 | 9.3330 | −0.4428 | 0.0000 | 0.0000 |
| staged | distance16 | 9.6227 | 9.2743 | −0.3484 | 0.0000 | 0.0000 |
| staged | target length 2 | 9.9860 | 9.4583 | −0.5277 | 0.0000 | 0.0000 |
| staged | target length 4 | 9.8129 | 9.3857 | −0.4272 | 0.0000 | 0.0000 |
| staged | target length 8 | 9.7002 | 9.3730 | −0.3272 | 0.0000 | 0.0000 |
| shuffled | values8000 | 9.7602 | 9.5311 | −0.2291 | 0.0000 | 0.0000 |
| shuffled | distance16 | 11.9042 | 11.6756 | −0.2286 | 0.0000 | 0.0000 |
| shuffled | target length 2 | 10.5598 | 9.7213 | −0.8385 | 0.0000 | 0.0000 |
| shuffled | target length 4 | 10.6984 | 9.5673 | −1.1310 | 0.0000 | 0.0000 |
| shuffled | target length 8 | 10.9066 | 9.6416 | −1.2649 | 0.0000 | 0.0000 |

The auxiliary objective therefore met the continuous native-loss part of the
prediction on the hard target-length tests, but failed the decisive exact
retrieval part. Easier value-space effects were mixed: staged EM rose at
values128/256/1024 while CE was not uniformly improved; shuffled easier-rung
metrics worsened.

Training CUDA time was 401.57 seconds staged and 401.78 seconds shuffled,
versus 183.70/184.34 seconds for the original A1 curriculum controls. The
improvement is not a compute-budget explanation: both runs used the same step
and token budget, while the auxiliary branch cost additional computation.

## Adjudication before postdiagnostics

The intervention partially supports the idea that learning pressure on the
recurrent state can improve native confidence and hard-task loss. It does not
show that the existing state has learned a systematic arbitrary-address
binding algorithm. Exact-match remains at zero on all hard target-length tests,
so the intervention is recorded as a failed repair and no second training
intervention is authorized.

The required layerwise and query/readout postdiagnostics are recorded in
`docs/A0_A1_RELATIONAL_CURRICULUM_VALUE_TRACE_POSTDIAGNOSTIC_RESULT.md`.
