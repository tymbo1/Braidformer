# A0/A1 synthetic recall and copying protocol

Status: **frozen for execution; no training started under this protocol**.

This is the next bounded experiment after the completed TinyStories
schedule-disentanglement work. It tests known potential weaknesses of
recurrent/state-space models without changing either frozen architecture. It
is a task-specific learning experiment, not an evaluation of the existing
TinyStories checkpoints.

## Hypotheses

The battery measures whether A1 can learn and use distant, exact information
as well as A0 at the same token/update budget:

1. **Exact copying:** reproduce an eight-token random payload after a long
   distractor span.
2. **Associative recall:** retrieve a four-token value for a queried key from
   eight key/value records.
3. **Selective retrieval:** retrieve a four-token value for a queried key from
   thirty-two records and intervening random distractors.

The primary measurements are autoregressive exact answer match and
answer-token accuracy on a fixed held-out test set. Target-only cross-entropy
and teacher-forced diagnostics are recorded as secondary metrics. No
predeclared architecture winner is assumed.

## Frozen common settings

- A0: existing 41,866,240-parameter Transformer.
- A1: existing 41,870,336-parameter recurrent/state-space model with the
  verified `parallel_scan` path.
- Fresh initialization for seeds `17`, `23`, and `29`.
- Context length 512; micro-batch 2; accumulation 8; effective batch 16.
- FP16 autocast; AdamW with the existing betas, epsilon, weight decay, and
  gradient clipping; learning rate `3e-4`.
- Identical 10-step warmup and 1,000-step cosine schedule for both models.
- Exactly 8,192,000 synthetic token positions per run.
- Validation every 100 optimizer steps and at step 1,000; final test once.
- A1 uses `parallel_scan`; no architecture, recurrence, optimizer, or
  implementation changes are permitted during execution.
- CUDA time, wall time, throughput, peak VRAM, losses, accuracy, failures,
  and checkpoint integrity are recorded. CUDA time is measured, not matched.
- Gradient checkpointing remains unimplemented and is not enabled.

## Synthetic data generator v1

Examples are fixed-length integer-token sequences using a reserved marker and
data-token range inside the existing 32,000-token vocabulary. TinyStories and
its tokenizer are intentionally not used. Each sequence ends with its answer;
training loss is masked to answer positions only. The generator is deterministic
from `(base seed, split, task, example index)`.

Training examples are balanced round-robin across the three tasks. Validation
uses 64 examples per task and test uses 128 examples per task. Training,
validation, and test seed domains are disjoint. The generator specification,
fixed evaluation examples, and script/config hashes are recorded in the result
JSON; no raw synthetic corpus is added to Git.

The copy payload has eight random tokens. Associative records contain eight
unique keys and four-token values. Selective-retrieval records contain thirty-
two unique keys and four-token values, with random distractor tokens filling
the context. Keys, values, markers, and distractors are generated from
disjoint reserved ranges. Query markers appear immediately before the answer.

## Metrics and interpretation

For each task and split, record target loss and teacher-forced diagnostics. On
the final test set, also perform autoregressive greedy generation from the
query prefix and record answer-token accuracy and exact whole-answer match.
Also record the validation curve, final test metrics,
parameter count, actual CUDA/wall time, throughput, VRAM, numerical/CUDA/OOM
failures, seed, configuration hash, generator hash, and final checkpoint
SHA-256 integrity.

The comparison is equal-token and equal-update, not equal CUDA time, FLOPs, or
energy. A1's task accuracy may be lower, equal, or higher; any result remains
specific to this generator and model scale. Failure on one task is evidence
against broad claims, not evidence that the architecture is unusable in every
setting.

## Execution boundary

The two synthetic configs and this protocol are committed before execution.
Only the six A0/A1 task-battery runs are in scope. No TinyStories checkpoint,
schedule-disentanglement artifact, A2 implementation, long-context sweep,
second corpus, scaling experiment, post-training, persistent memory, or WNF
work is included.
