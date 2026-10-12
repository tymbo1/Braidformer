# A0/A1 solvability ladder: distance-4, 8000-value rung

Status: **frozen for execution; no training started under this protocol**.

This is the first distractor-distance rung after the completed one-association
value-space ladder and association-count axis. It keeps one key/value binding
and the 8,000-value answer pool, then inserts four neutral markers between the
stored value and the query. The purpose is to test retention over a longer
sequence without changing the binding count.

## Question and pass criterion

Can unchanged A0 and A1 preserve and retrieve one binding after four neutral
sequence positions? The primary calibration criterion is final autoregressive
greedy exact match and answer-token accuracy on the repeated 128-example
training set. The disjoint held-out set remains diagnostic; report its exact
match and target loss even if exact match reaches zero.

## Frozen settings

- A0 remains the existing 41,866,240-parameter Transformer.
- A1 remains the existing 41,870,336-parameter `parallel_scan` model.
- Fresh seed 17 initialization for each architecture.
- Context length 512, hidden size 512, eight layers, 32,000-token vocabulary,
  tied embeddings, and existing normalization/bias settings remain unchanged.
- Each example is exactly twelve tokens:
  `TASK, RECORD, KEY, VALUE, DISTRACTOR×4, QUERY, KEY, ANSWER, VALUE`.
- `DISTRACTOR` is a reserved neutral marker distinct from all task markers,
  keys, and candidate values. It is not a candidate answer.
- The key, value, query, and answer mapping is identical to the completed
  distance-0 one-association values8000 rung. Only the four neutral positions
  are added.
- The fixed training set contains 128 examples and repeats deterministically;
  the held-out split uses disjoint keys and mappings.
- The candidate answer set contains 8,000 reserved value tokens. The full
  32,000-token output vocabulary remains active.
- Micro-batch 2, gradient accumulation 8, effective batch 16, FP16 autocast.
- AdamW, learning rate `3e-4`, existing betas, epsilon, weight decay, and
  gradient clipping.
- Fresh 2,000-step cosine schedule with 20 warmup steps.
- The twelve-token sequence necessarily changes the actual budget to
  `2,000 × 12 × 16 = 384,000` sequence-token positions. Optimizer steps,
  effective batch, schedule, and evaluation cadence remain fixed; this is not
  an equal-token comparison with distance 0.
- Evaluation occurs every 100 steps and at step 2,000 on the repeated
  training set and disjoint held-out set.
- Checkpoints are written atomically every 100 completed optimizer steps and at
  the final step, with SHA-256 sidecars and optimizer/RNG state.

## Deterministic data identity

The generator identity is `synthetic_solvability_distance4_values8000_v1`.
The distance rung deliberately derives its binding from the completed
distance-0 values8000 generator, so each example can be paired directly with
its no-distractor counterpart. The data hash, generator specification, config
hash, protocol hash, runtime seed, checkpoint hash, and all step/validation
records are recorded.

## Power-loss recovery

An interrupted run must not be resumed by appending to its directory. Supply
the latest verified interior checkpoint with `--resume`; the continuation must
write to a fresh run directory. The interrupted directory and logs remain
preserved as evidence. A resumed checkpoint must match config, protocol, seed,
data hash, generator specification, parameter count, and token progress.

Because the checkpoint interval is 100 steps, a hard power loss can discard at
most the uncheckpointed suffix after the last completed checkpoint. No result
may be reported as complete unless the final checkpoint and sidecar verify.

## Execution boundary

Only the two fresh seed-17 distance-4 arms with one association and 8,000
candidate values are in scope. This protocol does not authorize additional
seeds, multiple associations, other distance values, longer answers, model
changes, decay analysis, predictive learning-state or competence-ledger work,
A2, persistent memory, or SNAIL/SSRE/WNF work.

