# A1 relational-curriculum query-conditioned binding intervention protocol

Status: **frozen pre-registered training protocol**.

## Purpose

The completed mechanism campaign found that an unconditioned layer-7 value
trace can improve native loss without producing a transferable arbitrary
key→value code. This branch tests the narrower hypothesis that A1 needs
explicit supervision of the query-conditioned binding operation itself.

The intervention is deliberately separate from, and does not reopen, the
completed bounded mechanism campaign. A0 and original A1 remain untouched
controls.

## Hypothesis

If A1's dominant failure includes weak query-conditioned binding, then scoring
the value only from the layer-7 state immediately after the query key should
improve held-out retrieval more than the prior unconditioned value trace,
especially for long distractor distances and target lengths 4/8.

The binding score is explicitly joint in the query key and recurrent state:

```text
query_state = layer7_state[position_of_query_key]
key_vector = token_embedding[query_key]
joint = normalize(query_state) * normalize(key_vector)
binding_logits = joint @ normalize(value_embeddings).T / 0.1
loss = native_causal_loss + 0.1 * cross_entropy(binding_logits, target_value)
```

The query-key position is the token immediately following the unique query
marker. The target is the final non-masked value label. The state is therefore
captured before the answer marker/target and cannot read the target token from
the future. The existing tied token-embedding table supplies both key and
candidate-value vectors; no auxiliary head or trainable parameter is added.

The auxiliary objective is active only during training. Evaluation uses the
native A1 causal-language-model forward path and the existing frozen battery.

## Arms and matched controls

Run two fresh A1 arms:

- staged relational curriculum;
- shuffled relational curriculum.

Both use seed 17, the exact relational-curriculum-v1 A1 configuration,
schedule, exposure, optimizer, learning-rate schedule, 2,000 optimizer steps,
928,000 training tokens, and existing checkpoint/integrity tooling.

The comparison set is:

1. original A1 staged/shuffled controls from
   `results/a0_a1_relational_curriculum_v1.json`;
2. the completed unconditioned layer-7 value-trace arms from
   `results/a0_a1_relational_curriculum_value_trace_intervention.json`;
3. the new query-conditioned binding arms.

The same held-out retrieval battery is used for final native evaluation. No
held-out metric selects a checkpoint, changes the schedule, or changes the
objective. A0 is not retrained.

## Predeclared predictions

Primary prediction: relative to both original A1 and unconditioned value-trace
controls, the query-conditioned objective improves native held-out exact match
or token-level retrieval on at least one hard axis, with the largest expected
effect on distance16/64 and target lengths 4/8.

Mechanistic prediction: the exact layerwise and query/readout diagnostics show
an improvement in fixed or site-specific cross-validated key/value transfer,
not merely a higher held-out-fitted decodability ceiling.

Falsification: if native loss improves but exact retrieval remains at floor and
fixed/site-specific transfer remains near floor in both arms, the intervention
does not repair the hypothesized binding mechanism. Record that as a negative
result and do not stack another auxiliary objective in this branch without a
new protocol.

## Integrity gates

The runner must:

- require the committed protocol hash before training;
- verify the original curriculum aggregate and value-trace aggregate hashes;
- preserve A1's 41,870,336 trainable parameters and state-dict keys;
- use the unchanged A1 config and exposure hash;
- fail on non-finite loss, gradients, metrics, or checkpoint metadata;
- verify checkpoint sidecars and final digests;
- leave A0, original A1, and all prior checkpoints untouched;
- keep raw runs under ignored `results/runs/` paths.

After training, rerun the exact layerwise representation and query/readout
diagnostics on both new checkpoints. The postdiagnostic must distinguish a
transferable binding improvement from an in-sample decodability ceiling.

## Outputs

- `results/a0_a1_relational_curriculum_query_binding_intervention.json`
- `docs/A0_A1_RELATIONAL_CURRICULUM_QUERY_BINDING_INTERVENTION_RESULT.md`
- `results/a0_a1_relational_curriculum_query_binding_postdiagnostics.json`
- `docs/A0_A1_RELATIONAL_CURRICULUM_QUERY_BINDING_POSTDIAGNOSTIC_RESULT.md`
