# A0/A1 fixed-set solvability calibration protocol

Status: **frozen for execution; no training started under this protocol**.

This is the first rung of the solvability ladder following the completed
synthetic recall battery. It asks whether the unchanged 42M-parameter models
can memorize a trivial one-association operation when the long-context and
multi-record difficulties are removed. It is a calibration experiment, not a
new architectural comparison or a replacement for the frozen recall result.

## Question and pass criterion

Can each model memorize 128 fixed one-key/one-value examples when those same
examples are repeated throughout training?

The primary pass criterion is final **autoregressive greedy exact-match and
answer-token accuracy on the repeated training set**, reported separately for
A0 and A1. A model that cannot reach near-perfect training accuracy has not
passed the memorization calibration. The held-out set uses unseen keys and is
diagnostic only; random key/value mappings are not expected to generalize.

No A0-vs-A1 winner threshold is declared for this calibration. Failure is a
signal to inspect optimization, initialization, objective construction, or
state capacity before increasing model size.

## Frozen settings

- A0 remains the existing 41,866,240-parameter Transformer.
- A1 remains the existing 41,870,336-parameter `parallel_scan` model.
- Fresh seed 17 initialization for each architecture.
- The model configuration remains context length 512, hidden size 512,
  eight layers, 32,000-token vocabulary, tied embeddings, and existing
  normalization/bias settings.
- Actual examples contain exactly eight tokens:
  `TASK, RECORD, KEY, VALUE, QUERY, KEY, ANSWER, VALUE`.
- There are no filler or distractor tokens, no multiple records, and one
  one-token answer.
- The fixed training set contains 128 examples and is repeated deterministically.
- Micro-batch 2, gradient accumulation 8, effective batch 16, FP16 autocast.
- AdamW with the existing betas, epsilon, weight decay, and gradient clipping;
  learning rate `3e-4`.
- Fresh 2,000-step cosine schedule with 20 warmup steps (`warmup_ratio=0.01`).
- The run processes 256,000 actual sequence-token positions. CUDA time is
  measured, not matched.
- Evaluation occurs every 100 steps and at step 2,000 on both the repeated
  training set and a disjoint held-out set of 128 examples.
- Gradient checkpointing remains unimplemented and disabled.

The full 32,000-token output vocabulary is intentionally retained in this
first calibration so that a small answer candidate set does not silently
change the model parameter count. The 64-token candidate value set is a task
property, not a vocabulary reduction.

## Deterministic data

The generator version is `synthetic_overfit_calibration_v1`. Training examples
use unique keys from one reserved range and values drawn from a fixed set of
64 reserved value tokens. Held-out examples use a disjoint key range and a
disjoint deterministic seed domain. The held-out mapping is not a memorization
target and is not used to declare a calibration pass.

Loss is masked to the final answer token. Greedy evaluation starts from the
seven-token prefix and generates exactly one token, so it does not expose the
answer during measurement. Generator, configuration, protocol, run, CUDA,
VRAM, loss, accuracy, failure, and checkpoint hashes are recorded.

## Execution boundary

Only the two seed-17 calibration runs are in scope. Do not start seeds 23/29,
the difficulty ladder, longer-context tests, model scaling, initialization
ablations, A2, persistent memory, SNAIL/SSRE, or WNF from this protocol.

The existing synthetic recall battery, TinyStories experiments, checkpoints,
and results remain unchanged.
