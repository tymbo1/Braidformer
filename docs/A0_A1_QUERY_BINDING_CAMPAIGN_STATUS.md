# A0/A1 query-conditioned binding campaign status

Status: **protocol corrected and frozen; training not yet completed**.

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
2. **Attempted and corrected.** The first gated staged run reached step 1,754
   and stopped on an implementation guard that assumed one target label. The
   existing stress curriculum legitimately contains target lengths 2/4/8; no
   result was accepted. The partial raw run was moved to
   `/tmp/braidformer_query_binding_failed_20261013_1809/` for preservation.
   The corrected objective scores every target token against the same
   pre-answer query-conditioned state, with no change to model, data, or
   predeclared objective.
3. **Queued.** Run fresh staged and shuffled A1 query-binding arms.
4. **Queued.** Verify checkpoint/result integrity and compare native metrics with
   original A1 and unconditioned value-trace arms.
5. **Queued.** Rerun the exact layerwise and query/readout diagnostics.
6. **Queued.** Adjudicate whether exact retrieval and transferable binding
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

Protocol/code freeze commit: `e7259e7b56023d4f4e83e9e17b7eefe194b25da1`.
Protocol sha256:
`c47025b588d2b0341cda8f7f1a8cb6725abc323031eff4400f913dd99f184e22`.
Correction commit: **pending commit of the multi-target implementation fix**.
