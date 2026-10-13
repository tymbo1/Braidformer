# A0/A1 query-conditioned binding campaign status

Status: **protocol frozen; training not yet executed**.

This is a new branch after the completed bounded mechanism campaign. It does
not modify or reopen the frozen A0/A1 controls.

## Hypothesis

The prior layer-7 value-trace objective improved native loss but did not create
a transferable key→value code. This branch tests whether explicit supervision
of the query-conditioned operation—using the layer-7 state at the query-key
position jointly with the query-key embedding—repairs held-out retrieval.

## Frozen intervention

Two fresh A1 runs will use the exact relational-curriculum-v1 staged and
shuffled schedules. The training-only auxiliary objective has no trainable
parameters and is:

```text
joint = normalize(layer7_state_at_query_key) * normalize(query_key_embedding)
binding_logits = joint @ normalize(existing_value_embeddings).T / 0.1
loss = native_causal_loss + 0.1 * cross_entropy(binding_logits, target_value)
```

Seed, configuration, optimizer, exposure, 2,000 steps, 928,000 tokens, and
integrity controls are inherited unchanged from the completed curriculum and
value-trace experiments. Native evaluation uses the existing battery.

## Queue

1. **Complete.** Freeze protocol, runner, and CPU validation tests.
2. **Queued.** Run fresh staged and shuffled A1 query-binding arms.
3. **Queued.** Verify checkpoint/result integrity and compare native metrics with
   original A1 and unconditioned value-trace arms.
4. **Queued.** Rerun the exact layerwise and query/readout diagnostics.
5. **Queued.** Adjudicate whether exact retrieval and transferable binding
   improved; do not stack another auxiliary objective if the prediction fails.

## Controls and exclusions

Original A0, original A1, and the completed unconditioned value-trace runs are
read-only controls. No A2, scaling, persistent memory, predictive learning,
inheritance, new dataset, or production architecture change is in scope.

## Protocol artifacts

- Protocol:
  `docs/A0_A1_RELATIONAL_CURRICULUM_QUERY_BINDING_INTERVENTION_PROTOCOL.md`
- Runner:
  `scripts/run_relational_curriculum_query_binding_intervention.py`
- Unit tests:
  `tests/test_query_binding_intervention.py`

The exact freeze and execution commits will be appended before training.
