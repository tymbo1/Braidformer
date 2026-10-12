# A0/A1 relational-curriculum representation-probe result

Status: **complete; read-only frozen-checkpoint analysis**.

## Result in one sentence

Staged relational exposure substantially improved A1's representation-level
decodability relative to the shuffled control, especially through later
recurrent states, but the improvement was not a stable train-to-held-out
binding coordinate and did not restore native hard-target exact match. The A0
control did not show the same large site-fitted representation separation.

## Provenance and integrity

- Analysis plan: `docs/A0_A1_RELATIONAL_CURRICULUM_REPRESENTATION_PROBE_PLAN.md`
  (sha256 `e799641cd02301d3e4ae48f419051733f014d8e1d6688cec2b00f58b644fad16`).
- Analysis commit: `ba3b48a0213a0809056d9dbf92cc5319f9c4f23f`.
- Reused forensic script sha256:
  `8a581b6fa34bcfa33a1e93ad3235195f570e719640e06c337cc9514810371495`.
- Source curriculum aggregate sha256:
  `66f68cfb3a2f84f5e2193dfcdd606c9c5ef9e8e111aee3e99254ce5c975d8d71`.
- Probe aggregate:
  `results/a0_a1_relational_curriculum_representation_probes.json` (sha256
  `b36749b40024b1f84b21fddbe825f60105f28cfe4bf99b21673d649f347268b3`).
- Twelve analyses completed: A1/A0 × staged/shuffled × target lengths 2/4/8.
- All four source checkpoint sidecars and SHA-256 digests were verified before
  loading. The aggregate contains 97,864 finite numeric values; no checkpoint
  or model parameter was modified.

## Method

The existing frozen layerwise forensic method captured every block output, each
A1 recurrent state, and final normalization at `pre_query`, `post_query`, and
each target-output position. Temporary ridge probes measured key identity,
associated value, paired binding, fixed pre-query transfer, target-position
decodability, and representation geometry. The important distinction is:

- **held-out-fit held-out** is a decodability ceiling because the probe is fit
  on the held-out representations;
- **train-fit held-out** tests whether a probe learned from the training
  examples transfers to held-out examples;
- **fixed pre-query transfer** tests whether the pre-query value coordinate
  survives later sites.

## A1 staged versus shuffled

The table reports mean associated-value top-1 across sites and target
positions. “HF” is the held-out-fitted ceiling; “TF→HO” is a train-fitted
probe evaluated on held-out examples; “fixed” is the pre-query train-fitted
probe transferred to the later held-out site.

| target length | path | staged HF | shuffled HF | staged TF→HO | shuffled TF→HO | staged fixed | shuffled fixed |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | recurrent_state_0 | 1.0000 | 1.0000 | 0.009 | 0.012 | 0.0059 | 0.0127 |
| 2 | recurrent_state_7 | 1.0000 | 0.0693 | 0.003 | 0.002 | 0.0029 | 0.0000 |
| 2 | final_norm | 1.0000 | 0.1660 | 0.004 | 0.001 | 0.0020 | 0.0000 |
| 4 | recurrent_state_0 | 1.0000 | 1.0000 | 0.003 | 0.001 | 0.0013 | 0.0016 |
| 4 | recurrent_state_7 | 1.0000 | 0.0837 | 0.002 | 0.000 | 0.0007 | 0.0000 |
| 4 | final_norm | 1.0000 | 0.1572 | 0.003 | 0.000 | 0.0010 | 0.0000 |
| 8 | recurrent_state_0 | 1.0000 | 1.0000 | 0.006 | 0.000 | 0.0009 | 0.0003 |
| 8 | recurrent_state_7 | 1.0000 | 0.1019 | 0.000 | 0.000 | 0.0001 | 0.0000 |
| 8 | final_norm | 1.0000 | 0.2140 | 0.003 | 0.000 | 0.0009 | 0.0000 |

The separation is largest in the later A1 recurrent and final representations:
the staged checkpoint remains linearly decodable by a probe fitted on held-out
examples, while the shuffled checkpoint's held-out-fitted ceiling falls. This
is consistent with curriculum-dependent representation formation or
recurrent transformation.

However, the train-fitted held-out values remain near zero at later sites, and
the fixed pre-query transfer values are also near zero for both arms. Thus the
result does **not** show that staged A1 learned a robust arbitrary-address
binding operation. It shows that staged training leaves a representation from
which a separately fitted decoder can recover the held-out value more easily;
the shared train-to-held-out coordinate remains inadequate.

The recurrent-state target-position probe also favoured staged A1 at the
longer lengths (`0.918` versus `0.775` at length 4 and `0.803` versus `0.401`
at length 8 in `recurrent_state_7`), while block-output/final position probes
were near ceiling in both arms. This is a secondary state-organization signal,
not evidence that output routing is solved.

## A0 control

At `block_output_0`, `block_output_7`, and `final_norm`, A0's held-out-fitted
value decodability was `1.0000` for both curriculum orders at all three target
lengths. Its fixed pre-query transfer differences were small and mixed (for
example, final normalization staged minus shuffled was `−0.0078`, `+0.0039`,
and `+0.0006` at lengths 2, 4, and 8). This is materially different from the
large A1 staged-versus-shuffled later-state separation and supports an
A1-specific interaction between learning order and recurrent transformation.

## Adjudication

1. **Curriculum/developmental history (H8): strengthened as a representation
   modifier.** The A1 staged and shuffled checkpoints differ in how much value
   information is recoverable at later recurrent sites under a held-out-fitted
   probe, despite identical exposure hashes and training budgets.
2. **Recurrent transformation/interference (H2): strengthened.** The largest
   A1 separation appears after recurrent depth, and the staged gain is not a
   simple pre-query storage gain.
3. **Stable arbitrary binding: not demonstrated.** Train-fitted held-out and
   fixed-transfer probes remain near floor; paired binding transfer is not
   recovered.
4. **Pure final readout failure: unresolved, not sufficient.** The staged
   checkpoint has stronger later-state decodability than the shuffled one, but
   the native output still has zero exact match on the hard target-length
   tests. A query/readout-localization experiment is required to distinguish
   coordinate drift from failure to use retained information.
5. **Temporal decay alone: not restored.** This result is compatible with the
   earlier decay-tail negative and does not overturn it.

The next queued experiment is query/readout localization on these frozen
checkpoints. It will compare the retained representation, query transition,
and output logits without changing A0/A1 or training another model.
