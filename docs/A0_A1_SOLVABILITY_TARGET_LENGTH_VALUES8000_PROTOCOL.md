# A0/A1 solvability target-length axis: 8000-value one-association task

Status: **frozen for execution; no training started under this protocol**.

This is the next controlled axis after the completed value-space,
association-count, distractor-distance, and read-only A1 decay intervention
work. It changes the bound answer from one value token to a sequence of 2, 4,
or 8 value tokens while keeping the key/value task and the original A0/A1
architectures unchanged.

## Question

Does answer length expose a distinct output/retrieval weakness after the
one-token distance curve has already shown A1's long-range degradation?

Each example binds one key to a deterministic sequence of independently drawn
tokens from the same reserved 8,000-value pool:

`TASK, RECORD, KEY, VALUE[1:L], QUERY, KEY, ANSWER, VALUE[1:L]`

Only the answer suffix is scored. Held-out examples use disjoint keys and
deterministic mappings. Whole-sequence exact match requires every target token
to be correct; answer-token accuracy is computed across all target positions.

## Frozen controls

- A0 is the existing 41,866,240-parameter Transformer.
- A1 is the existing 41,870,336-parameter parallel-scan recurrent model.
- Both use the original uniform τ≈9.5-token initialization; the decay-tail
  intervention is not part of this axis.
- Fresh seed 17 initialization for every arm.
- One association, 8,000 reserved value tokens, and zero distractor markers.
- Context length 512, hidden size 512, eight layers, tied embeddings, FP16
  autocast, AdamW, existing betas/epsilon/weight decay/clipping, and the same
  20-step warmup plus 2,000-step cosine schedule.
- Micro-batch 2, gradient accumulation 8, effective batch size 16.
- Fixed training set of 128 examples, repeated deterministically; held-out
  set of 128 examples with disjoint keys.
- Evaluation every 100 steps and at step 2,000; atomic checkpoints every 100
  steps and at the final step.

The axis is matched by optimizer steps and task construction, not by equal
token count. Longer target sequences therefore process more token positions;
actual CUDA time, throughput, and token exposure remain reported.

## Runs

| Target length | Sequence length | Training token positions | Arms |
| ---: | ---: | ---: | --- |
| 2 | 10 | 320,000 | A0, A1 |
| 4 | 14 | 448,000 | A0, A1 |
| 8 | 22 | 704,000 | A0, A1 |

The generator identities are respectively
`synthetic_solvability_targetlength2_values8000_v1`,
`synthetic_solvability_targetlength4_values8000_v1`, and
`synthetic_solvability_targetlength8_values8000_v1`.

## Interpretation

- If both models' whole-sequence exact match declines similarly, target
  length is a shared output difficulty.
- If A1 loses token accuracy or exact match faster than A0 while both retain
  training memorization, recurrent retrieval/readout is implicated beyond
  the one-token distance effect.
- If A1's token accuracy remains reasonable but whole-sequence exact match
  collapses, the axis is exposing compounding autoregressive/readout error.
- This axis does not authorize a decay change, relational store, curriculum,
  model scaling, or A2 architecture.

## Integrity and recovery

The runner records generator/config/protocol hashes, CUDA and wall time,
contiguous step records, validation cadence, and verified optimizer/RNG
checkpoints. A power-loss recovery must resume from a verified interior
checkpoint into a fresh output directory; no interrupted directory may be
appended to.
