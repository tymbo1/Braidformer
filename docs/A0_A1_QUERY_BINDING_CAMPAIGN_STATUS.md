# A0/A1 query-conditioned binding campaign status

Status: **complete; postdiagnostics adjudicated and branch stopped**.

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
3. **Complete.** Fresh staged and shuffled A1 query-binding arms completed
   under the corrected protocol. Native hard-task loss improved in the staged
   arm and mixed in the shuffled arm; hard exact retrieval remained at floor.
4. **Complete.** Checkpoint sidecars, parameter count, aggregate status, and
   native comparisons against original A1 and value-trace controls passed.
5. **Complete.** Exact layerwise and query/readout diagnostics reran on all
   six checkpoint/target-length combinations.
6. **Complete.** Native loss improved selectively, but exact retrieval and
   transferable binding did not; no second auxiliary objective is authorized.

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
`bd0a8c0af7d34d8dcc21469edcd1e4bec68c647a1991e117afc594a04817edc7`.
Correction commit: `8484cd608f4a5ac6860fb4e7f8a23c2149563512`.
The corrected execution commit is recorded below.

Corrected execution commit: `4ca85c7ac389d91f164874e6e80914b322da2357`.
Intervention aggregate:
`results/a0_a1_relational_curriculum_query_binding_intervention.json`
(sha256 `5f63c688985ac049047f5e8b34b6c359b1eeeea28de5069728d71a80dcbad9a2`).
New checkpoint hashes: staged
`baf6af6f6653e71578246cffff3103a086baa1bcbbbc1b21c98ba3370ad359c2`;
shuffled
`d453b5014de1933b51fff9b5a014c7d5a108147d76fbf47ac78594376d56ce8f`.
Postdiagnostic plan/code freeze commit:
`ed55b28f4b29dd4ec7cb3ad5c0a61ba68e7fc0ef`.
Postdiagnostic plan sha256:
`0f8b561b186ee0be1bf89a74201fc9cda5d35869adb94aff8aab82e41cd1bf52`.
Postdiagnostic aggregate:
`results/a0_a1_relational_curriculum_query_binding_postdiagnostics.json`
(sha256 `03ddd4e90b8cbe78fd264180b99669a36d33bf9ae707e667ef71981509bcbeb9`).
Result reports:
`docs/A0_A1_RELATIONAL_CURRICULUM_QUERY_BINDING_INTERVENTION_RESULT.md` and
`docs/A0_A1_RELATIONAL_CURRICULUM_QUERY_BINDING_POSTDIAGNOSTIC_RESULT.md`.
