# A1 recurrent/state-space-majority proposal

Status: the A1 architecture is implemented and CPU-verified. It has not been
trained, GPU-smoke-tested, or compared empirically with A0.

## Research question

Can a recurrent/state-space-majority decoder-only model reach comparable
validation quality to the A0 Transformer control at the same TinyStories token
budget while changing the sequence-mixing cost profile?

This is a falsifiable architecture hypothesis. The implementation does not
claim that recurrence is better, faster, or more memory-efficient until a
matched measurement exists.

## Fixed architecture

The A1 proposal is deliberately narrow so the first comparison changes one
major factor:

| Component | A1 choice |
| --- | --- |
| Model | decoder-only causal language model |
| Sequence mixer | 8 gated diagonal state-space recurrent blocks |
| Retained attention | 0 layers; no attention module is present |
| Hidden size | 512 |
| Layers | 8 |
| State size | 512 scalar state channels per block |
| Vocabulary | 32,000 |
| Context | 512 tokens with the same learned absolute position embedding as A0 |
| Feed-forward width | 2,048, GELU |
| Normalization | pre-layer normalization, epsilon `1e-5` |
| Dropout | 0.0 |
| Linear biases | enabled |
| Word embeddings | tied input/output weights |
| Experts/depth controller/retrieval | absent |

Each recurrent mixer projects the normalized hidden state to candidate,
input-gate, and output-gate channels. A learned diagonal transition coefficient
per hidden channel updates the state sequentially over the supplied context:

```text
state[t] = decay * state[t-1]
            + (1 - decay) * input_gate[t] * tanh(candidate[t])
output[t] = output_gate[t] * state[t]
```

The state is initialized to zero independently for every input sequence. It is
not carried between packed windows or batches. The output is passed through a
dense hidden-size projection and the same A0 feed-forward residual path.

The shared `num_attention_heads: 8` configuration field is retained only for
the common BraidNano configuration shape and parameter-accounting comparison;
it does not instantiate attention in A1.

## Parameter accounting

A1 uses the same embeddings, output head, feed-forward path, layer count, and
hidden dimensions as A0. The recurrent mixer replaces A0's QKV-plus-output
attention projections with a 3-hidden input projection plus a hidden-size
output projection. Those dense projection counts are equal. A1 adds one
learned diagonal transition scalar per hidden channel per block.

| Group | A0 | A1 |
| --- | ---: | ---: |
| Token embedding | 16,384,000 | 16,384,000 |
| Position embedding | 262,144 | 262,144 |
| Eight sequence/FF blocks | 25,219,072 | 25,223,168 |
| Final LayerNorm | 1,024 | 1,024 |
| Tied LM-head increment | 0 | 0 |
| **Unique trainable parameters** | **41,866,240** | **41,870,336** |

The difference is **4,096 parameters**, or approximately **0.0098%** relative
to A0. The exact A1 construction and proposal are recorded in
`configs/braidnano_a1_recurrent_state_space_proposal.json` (SHA-256:
`b8444b2c899a76737c18383f94310cb4699b3385c627a2d7e2c2ce62b2eef6ff`).

## Context and training compatibility

A1 accepts the same `[batch, sequence]` token input and returns the same logits
and shifted next-token cross-entropy interface as A0. The proposed comparison
therefore reuses:

- the pinned TinyStories revision and 32,000-token tokenizer;
- deterministic packed streams and split manifest;
- seed 17 initially, with seeds 23 and 29 reserved for replication;
- context length 512;
- micro-batch 2, accumulation 8, effective batch 16;
- AdamW, learning rate `3e-4`, cosine schedule, 1% warmup, weight decay 0.1;
- 1,000 optimizer steps / 8,192,000 token positions;
- validation every 100 steps and the existing integrity policy.

The proposal remains `planned_only: true`. The existing training runner is
intentionally A0-only and rejects this A1 configuration, so no training can
start accidentally. A future A1 runner change must reuse its checkpoint,
validation, logging, and circuit-breaker behavior without changing the
approved comparison budget.

Gradient checkpointing remains not implemented and not enabled, as in A0.

## Expected computational difference

For sequence length `L`, hidden size `H`, and feed-forward width `I`, both
models retain roughly the same per-token dense projection work. A1 replaces
A0's quadratic attention interaction term with a recurrent state update:

- A0 sequence mixing includes an `O(L^2 H)` attention interaction and is highly
  parallel across positions.
- A1 sequence mixing adds only `O(L H)` state updates after its dense
  projections, but those updates are sequential across positions within each
  layer.

Therefore A1 may reduce activation/attention-memory growth at longer context,
but it may be slower on a GPU because the recurrence limits time-dimension
parallelism. No throughput, FLOP, or memory advantage is assumed; all three
must be measured.

## Falsifiable comparison hypotheses

The first comparison should report A0 and A1 at matched seed, data, tokenizer,
token positions, optimizer, schedule, context, effective batch, precision, and
evaluation budget. The primary exploratory decision rule is:

1. **Quality parity:** A1 final validation loss is within `0.05` of the
   measured A0 seed-17 result at the same token budget.
2. **Efficiency claim:** only if quality parity holds, A1 may claim an
   efficiency advantage if measured peak reserved VRAM or measured training
   cost is at least 10% lower. Tokens/sec and wall time must also be reported,
   because recurrent serialization can reverse a memory advantage.
3. **Falsification:** failure to reach the quality band, or failure to produce
   a measured efficiency gain at parity, falsifies the corresponding A1
   hypothesis for this pilot. It does not establish that all recurrent models
   fail.

These thresholds are a preregistered pilot comparison rule, not an empirical
result. Seeds 23 and 29 remain necessary before a stronger claim.

## Readiness findings

| Finding | Classification | State |
| --- | --- | --- |
| A1 model construction, causal loss, backward pass, and parameter accounting | REQUIRED | Verified by CPU tests |
| A1 training entrypoint with shared checkpoint/recovery/logging behavior | BLOCKER for training | Not implemented; current runner is A0-only |
| Host CUDA smoke test for A1 at micro-batch 2 / context 512 | BLOCKER for training | Not run; must precede authorization |
| A1 fixed-prompt evaluation wiring | RECOMMENDED | Existing evaluator is A0-specific; extend only before A1 training |
| Gradient checkpointing | DEFERRED | Not implemented or enabled; unchanged from A0 |
| Attention, MoE, conditional depth, controller, persistent memory | DEFERRED | Explicitly outside A1 |

No A1 training is authorized by this proposal. Review is required before the
training entrypoint or GPU smoke test is extended for A1.
