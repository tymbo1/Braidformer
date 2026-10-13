# A0/A1 relational-curriculum state-replay intervention plan

Status: **frozen causal diagnostic plan; inference-only execution**.

## Question

The preceding diagnostics show that staged A1 improves native confidence and a
held-out-fitted representation ceiling, but cross-validated fixed and
site-specific value readouts remain at floor. The leading remaining mechanism
is destructive recurrent transformation/interference. This intervention asks
whether preserving the layer-7 state associated with each stored value is
sufficient to restore the native target output.

## Frozen scope and controls

- Use the four verified relational-curriculum checkpoints: A1 staged and A1
  shuffled are intervened; A0 staged and A0 shuffled are verified as untouched
  controls and are not patched.
- Run target lengths 2, 4, and 8 on the fixed held-out split, with one
  association, values8000, and distance zero.
- Use the existing functional state-replay implementation in
  `scripts/analyze_state_replay_intervention.py`, whose sha256 is
  `d364b06f88338e92c2c600700079f0656d90108672e21e9205339fdb6cc43d3e`.
- Keep the native A1 model, weights, equations, gates, output projection,
  residual path, feed-forward path, and all non-patched timesteps unchanged.
- No checkpoint or model parameter is modified. A parameter digest must match
  before and after every functional intervention.

The exact intervention is layer 7 only. For target length `L` and target
position `j`:

```text
s[layer=7, target_output_position=4+L+j]
    <- s[layer=7, stored_value_position=2+j]
```

The source state is the state after processing the stored value token for that
target; the target state is replaced after its ordinary recurrent update and
before output gating/projection. The intervention is discarded after the
forward pass. The same held-out examples receive an unpatched native baseline.

Source provenance is pinned to curriculum aggregate sha256
`66f68cfb3a2f84f5e2193dfcdd606c9c5ef9e8e111aee3e99254ce5c975d8d71`, preceding
representation aggregate sha256
`b36749b40024b1f84b21fddbe825f60105f28cfe4bf99b21673d649f347268b3`, and
query/readout aggregate sha256
`73dc7df64d9d57099d3f31ede2628ba673ed8cf716249fbf17b83164a8d77e50`.

## Measurements

At every target position, compare unpatched and replayed native logits over
the full vocabulary and the 8,000-value candidate range:

- cross-entropy and correct-token probability;
- top-1 accuracy and correct-token logit margin;
- mean and median target rank;
- source/replayed state norms and cosine similarity;
- source/target positions and finite-tensor checks.

## Predeclared prediction

If recurrent-state transformation at this layer is causally responsible for
the retrieval weakness, replay should improve value-candidate rank and
cross-entropy at one or more A1 target positions, with the clearest rescue at
lengths 4 and 8 and especially the first target position. A top-1 improvement
would be especially strong evidence, but rank/CE improvement is the primary
continuous signal.

If replay produces no top-1 improvement and no coherent rank/CE improvement at
lengths 4/8, the narrow hypothesis “layer-7 stored-value state preservation
is sufficient” is rejected. Do not stack a second intervention. A null result
does not reject every recurrent/state hypothesis; query-conditioned binding or
another layer/state geometry could still matter.

## Integrity and output

The analysis must verify all four source summaries and checkpoint sidecars,
configuration hashes, parameter counts, seed/token controls, exposure hash,
source/target positions, finite baseline/patched tensors, and unchanged
parameter digests. It must fail nonzero on missing artifacts, hash drift,
non-finite values, or checkpoint writes. The result is mechanistic evidence,
not a production fix.

Outputs:

- `results/a0_a1_relational_curriculum_state_replay_intervention.json`
- `docs/A0_A1_RELATIONAL_CURRICULUM_STATE_REPLAY_INTERVENTION_RESULT.md`
- updated `docs/RELATIONAL_CURRICULUM_CAMPAIGN_STATUS.md`
