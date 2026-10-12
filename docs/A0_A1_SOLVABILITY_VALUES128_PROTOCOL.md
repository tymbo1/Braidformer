# A0/A1 solvability ladder: 128-value rung

Status: **frozen for review; no training started under this protocol**.

This is the next rung after the completed seed-17 one-association calibration.
It keeps the task, model architectures, optimizer, token budget, and training
geometry unchanged while expanding the reserved answer-token set from 64 to
128. The purpose is to test whether the first-rung result survives a modestly
harder output representation before adding multiple associations, distractors,
longer contexts, or larger models.

## Question and pass criterion

Can the unchanged A0 and A1 models memorize 128 fixed one-key/one-value
examples when each answer is drawn from a 128-token candidate set?

The primary calibration criterion is final autoregressive greedy exact match
and answer-token accuracy on the repeated training set, reported separately
for A0 and A1. The held-out set uses disjoint keys and mappings and remains
diagnostic; it is not a winner threshold.

No A0-versus-A1 winner threshold is declared for this rung. Failure to reach
near-perfect training accuracy is a signal to inspect optimization, output
representation, or state capacity before increasing task difficulty or model
size.

## Frozen settings

- A0 remains the existing 41,866,240-parameter Transformer.
- A1 remains the existing 41,870,336-parameter `parallel_scan` model.
- Fresh seed 17 initialization for each architecture.
- Context length 512, hidden size 512, eight layers, 32,000-token vocabulary,
  tied embeddings, and existing normalization/bias settings remain unchanged.
- Each example remains exactly eight tokens:
  `TASK, RECORD, KEY, VALUE, QUERY, KEY, ANSWER, VALUE`.
- There are no filler or distractor tokens, no multiple records, and one
  one-token answer.
- The fixed training set contains 128 examples and repeats deterministically.
- The candidate answer set contains 128 reserved value tokens. The full 32,000
  output vocabulary remains active; the candidate set is a task property.
- Micro-batch 2, gradient accumulation 8, effective batch 16, FP16 autocast.
- AdamW, learning rate `3e-4`, existing betas, epsilon, weight decay, and
  gradient clipping.
- Fresh 2,000-step cosine schedule with 20 warmup steps and 256,000 sequence
  token positions.
- Evaluation occurs every 100 steps and at step 2,000 on the repeated training
  set and a disjoint held-out set of 128 examples.
- Checkpoints are written atomically every 100 completed optimizer steps and at
  the final step, with SHA-256 sidecars and optimizer/RNG state.

## Deterministic data identity

The generator identity is `synthetic_solvability_values128_v1`. It preserves
the first-rung key and split construction while changing only the value-pool
cardinality from 64 to 128. The data hash, generator specification, config
hash, protocol hash, runtime seed, checkpoint hash, and all step/validation
records are recorded.

## Power-loss recovery

An interrupted run must not be resumed by appending to its directory. The
latest verified interior checkpoint is supplied with `--resume`, and the
continuation writes to a fresh run directory. The interrupted directory and
its logs remain preserved as evidence. A resumed checkpoint must match the
config, protocol, seed, data hash, generator specification, parameter count,
and token progress; otherwise the runner fails closed.

Because the checkpoint interval is 100 steps, a hard power loss can discard at
most the uncheckpointed suffix after the last completed checkpoint. No result
may be reported as complete unless the final checkpoint and sidecar verify.

## Execution boundary

Only the two seed-17 128-value arms are in scope. This protocol does not
authorize seeds 23/29, 256/1024/8000-value rungs, multiple associations,
distractor-distance changes, longer answers, model scaling, initialization
ablations, A2, persistent memory, or SNAIL/SSRE/WNF integration.

The completed 64-value calibration and all TinyStories results remain
unchanged.
