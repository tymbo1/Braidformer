# A1 protected-lane exact-address intervention protocol

Status: **frozen for read-only execution**.

This is the next mechanistic test after the protected-lane query/binding
localization result. It is an inference-only functional intervention over the
existing verified checkpoints. It does not train, alter, or save any model
weights and does not modify the frozen A0 or original A1 definitions.

## Question

Can a structurally exact key-conditioned address path make the already
preserved protected value usable at the answer marker?

The intervention bypasses only the learned lane key projection, cosine
similarity, and scalar read gate at the answer marker. It uses the exact token
identity of the recorded key and query key to decide whether to pass the
existing stored `memory_value` through the existing trained lane `readout`.
The stored value state, trained readout, base A1 hidden state, embeddings,
output head, and all other recurrence remain unchanged.

This is deliberately an oracle-like mechanistic isolation, not a production
architecture. Because the canonical one-binding examples repeat the same key
token at record and query, exact identity is a clean upper-isolation test for
the learned address route.

## Frozen sources and data

- Source aggregate: `results/a1_protected_lane_distance_axis_values8000.json`.
- Source checkpoints: the verified final checkpoints for protected and ordinary
  modes at distances `0`, `4`, `16`, and `64`.
- Seed `17`, one association, 8,000 values, target length `1`, held-out split
  of 128 examples.
- Existing protected-lane model and synthetic generator source hashes must
  match the recorded values before loading.
- Expected lane parameter count: `41,969,346`.

## Functional intervention

For the frozen layout

```text
TASK, RECORD, KEY, VALUE, DISTRACTOR×d, QUERY, KEY, ANSWER, VALUE
```

the analyzer computes the normal frozen forward trace, then replaces only the
answer-marker lane gate with:

\[
g_{\mathrm{exact}} = \mathbf{1}[\mathrm{record\_key}=\mathrm{query\_key}].
\]

The patched answer representation is:

\[
h_{\mathrm{patched}}
= h_{\mathrm{base}} + W_{\mathrm{readout}}(g_{\mathrm{exact}}m_v),
\]

where `m_v` is the original frozen lane memory value at the answer marker.
No key/value state is rewritten, no diagnostic probe is inserted into the
model, and no parameter is updated.

The native and patched logits are evaluated from the same captured base hidden
state and the same captured memory value. The exact gate must be one for every
canonical example; that condition is asserted as an integrity check.

## Measurements

For native and exact-address paths, report:

- full-vocabulary and value-range cross-entropy;
- top-1 accuracy, target probability, target-vs-best-incorrect margin, and
  target rank;
- target-logit, probability, and rank deltas from native to patched output;
- native learned gate and exact-address gate;
- stored-value norm/cosine and lane-readout norm;
- checkpoint hashes, parameter digests, finite-value checks, and source
  provenance.

The analyzer also records the prior localization result path so the exact
address result can be read alongside the memory-value and query-conditioned
probe evidence. It does not refit a model or claim that a diagnostic probe is
an architectural output.

## Predeclared predictions

| Result | Interpretation |
| --- | --- |
| Exact address improves protected held-out rank/CE, especially at distances 16/64, while ordinary gains less | Learned address/key matching is a primary bottleneck conditional on protected storage. |
| Exact address changes little and both modes remain at floor | The learned scalar address gate is not the dominant blocker; value-to-output integration or binding representation remains insufficient. |
| Exact address improves logits but not exact/rank metrics | The route adds a local output perturbation but does not restore a usable value code. |
| Ordinary improves similarly to protected | Storage persistence is not the distinguishing factor for the observed gain. |

No result automatically authorizes a training run or architecture change. A
positive result would require a separate protocol for a learned address path;
a null result closes this exact-address hypothesis and leaves readout/binding
as the stronger target.

## Integrity requirements

The analyzer must verify every source checkpoint sidecar and SHA-256, expected
parameter counts, source hashes, canonical exact-key equality, finite tensors
and metrics, and unchanged parameter digests. It may write only the new JSON
aggregate and report. It must not write checkpoints or modify raw run files.
After execution, run `git diff --check`, the full test suite, and the
untracked-artifact audit before committing the result and ledger update.
