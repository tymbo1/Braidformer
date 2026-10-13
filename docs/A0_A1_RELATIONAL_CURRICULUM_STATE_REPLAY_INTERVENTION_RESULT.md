# A0/A1 relational-curriculum state-replay intervention result

Status: **complete; reversible inference-only intervention**.

## Result in one sentence

Copying each stored value's layer-7 recurrent state into its matching target
position changed the A1 state as intended but did not rescue retrieval: no
target position gained top-1 accuracy, and aggregate value-range
cross-entropy worsened in every staged or shuffled run.

## Provenance and integrity

- Intervention plan:
  `docs/A0_A1_RELATIONAL_CURRICULUM_STATE_REPLAY_INTERVENTION_PLAN.md`.
- Analysis commit: `f3164bb559e270df1c4b5166ac424b4c01b05070`.
- Reused replay implementation sha256:
  `d364b06f88338e92c2c600700079f0656d90108672e21e9205339fdb6cc43d3e`.
- Source curriculum aggregate sha256:
  `66f68cfb3a2f84f5e2193dfcdd606c9c5ef9e8e111aee3e99254ce5c975d8d71`.
- Preceding representation aggregate sha256:
  `b36749b40024b1f84b21fddbe825f60105f28cfe4bf99b21673d649f347268b3`.
- Preceding query/readout aggregate sha256:
  `73dc7df64d9d57099d3f31ede2628ba673ed8cf716249fbf17b83164a8d77e50`.
- Intervention aggregate:
  `results/a0_a1_relational_curriculum_state_replay_intervention.json`
  (sha256 `4729924feec50b8dadbcf27afed0c7c2fa8aae2dff97e95b9b717c0457fd3dd7`).
- Six A1 interventions completed: staged/shuffled × target lengths 2/4/8.
  A0 staged/shuffled checkpoints were verified as untouched controls.
- All six functional runs had identical parameter digests before and after
  intervention. Source/target positions and all tensors/metrics were finite.

## Intervention

At A1 layer 7 only, for each target position `j`, the state after the stored
value token at position `2+j` replaced the ordinary state at target-output
position `4+L+j`. Output gating, output projection, residual path, feed-forward
path, all other layers, and all other timesteps were unchanged. The same
held-out examples were evaluated with the native unpatched forward pass.

## Aggregate native deltas

Values are replay minus baseline, averaged across target positions. Negative
cross-entropy and rank would indicate improvement; positive values indicate
worsening. Rank is within the 8,000-value candidate range.

| arm | target length | Δ cross-entropy | Δ mean rank | Δ top-1 |
| --- | ---: | ---: | ---: | ---: |
| staged | 2 | +0.0187 | +2.7 | 0.0000 |
| staged | 4 | +0.0134 | +5.6 | 0.0000 |
| staged | 8 | +0.0010 | +0.7 | 0.0000 |
| shuffled | 2 | +0.0019 | −1.9 | 0.0000 |
| shuffled | 4 | +0.0045 | +0.3 | 0.0000 |
| shuffled | 8 | +0.0034 | −1.3 | 0.0000 |

The replayed states were not no-ops. For example, staged A1's replay/original
state cosine was `0.8521` at length 4 target 1 and `0.7429` at length 8 target
1; the corresponding state norms also changed. The functional intervention
therefore altered the intended representation, but that alteration moved the
native output away from the target rather than restoring it.

## Adjudication

The predeclared prediction failed. This rejects the narrow causal hypothesis
that preserving the layer-7 stored-value state at the matching output position
is sufficient to restore A1 retrieval. It does not reject every recurrent or
temporal mechanism: the intervention tests one layer, one state variable, and
one post-query replay rule, and it does not construct a query-conditioned key→
value binding.

Combined with the query/readout result, the evidence now favours a distributed
recurrent transformation/binding failure over a simple late state-erasure or
final-readout defect. No second causal intervention is authorized in this
campaign. The next item is mechanism adjudication, followed by the campaign
stop report unless a previously documented queue condition requires otherwise.
