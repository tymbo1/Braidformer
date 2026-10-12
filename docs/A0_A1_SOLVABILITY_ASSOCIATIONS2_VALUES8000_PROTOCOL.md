# A0/A1 solvability ladder: two-association, 8000-value rung

Status: **frozen for execution; no training started under this protocol**.

This is the first association-count rung after the completed seed-17
one-association value-space ladder through 8,000 candidate values. It keeps
the models, value pool, seed, fixed-example count, optimizer, and optimizer
step budget unchanged while replacing one binding with two explicit bindings.
The purpose is to test whether the recurrent state loses retrieval quality as
it must maintain more than one key/value association.

## Question and pass criterion

Can the unchanged A0 and A1 models memorize 128 fixed examples containing two
key/value bindings and answer a query for either binding when values are drawn
from the 8,000-token candidate set?

The primary calibration criterion is final autoregressive greedy exact match
and answer-token accuracy on the repeated training set, reported separately
for A0 and A1. The held-out set uses disjoint keys and mappings and remains
diagnostic; it is not a winner threshold. Report held-out exact match and
target loss even if exact match reaches zero.

## Frozen settings

- A0 remains the existing 41,866,240-parameter Transformer.
- A1 remains the existing 41,870,336-parameter `parallel_scan` model.
- Fresh seed 17 initialization for each architecture.
- Context length 512, hidden size 512, eight layers, 32,000-token vocabulary,
  tied embeddings, and existing normalization/bias settings remain unchanged.
- Each example is exactly eleven tokens:
  `TASK, RECORD, KEY1, VALUE1, RECORD, KEY2, VALUE2, QUERY, KEY, ANSWER, VALUE`.
- The query key is selected deterministically from the two preceding records;
  the final value is the corresponding bound value. There are no additional
  distractor tokens or records.
- The fixed training set contains 128 examples and repeats deterministically.
- The candidate answer set contains 8,000 reserved value tokens. The full
  32,000-token output vocabulary remains active; the candidate set is a task
  property.
- Micro-batch 2, gradient accumulation 8, effective batch 16, FP16 autocast.
- AdamW, learning rate `3e-4`, existing betas, epsilon, weight decay, and
  gradient clipping.
- Fresh 2,000-step cosine schedule with 20 warmup steps.
- The 11-token sequence necessarily changes the actual budget to
  `2,000 × 11 × 16 = 352,000` sequence-token positions. Optimizer steps,
  effective batch, schedule, and evaluation cadence remain fixed; this is not
  an equal-token comparison with the one-association rung.
- Evaluation occurs every 100 steps and at step 2,000 on the repeated
  training set and a disjoint held-out set of 128 examples.
- Checkpoints are written atomically every 100 completed optimizer steps and at
  the final step, with SHA-256 sidecars and optimizer/RNG state.

## Deterministic data identity

The generator identity is `synthetic_solvability_associations2_values8000_v1`.
It uses the same split count, key domains, value base, and seed-derivation
discipline as the value-space ladder, but derives two distinct records and a
deterministic query per example. The data hash, generator specification,
config hash, protocol hash, runtime seed, checkpoint hash, and all
step/validation records are recorded.

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

Only the two fresh seed-17 two-association arms with 8,000 candidate values
are in scope. This protocol does not authorize four/eight associations,
additional seeds, distractor-distance changes, longer answers, model scaling,
initialization ablations, decay analysis, predictive learning-state or
competence-ledger work, A2, persistent memory, or SNAIL/SSRE/WNF work.

The completed one-association value-space ladder and all TinyStories results
remain unchanged.
