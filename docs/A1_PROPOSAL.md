# A1 recurrent/state-space-majority proposal

Status: the A1 architecture is implemented and CPU-verified. Its approved
host-GPU smoke test, bounded ten-update recurrence stability check, and
authorized seed-17 1,000-step pilot are complete. The measured result is
recorded in [`docs/A1_TRAINING_RESULT.md`](A1_TRAINING_RESULT.md).

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

## Reference and accelerated recurrence paths

The actual A1 equations satisfy the associative-scan condition. The decay is a
learned per-channel constant, and the candidate and gates are functions of the
current token representation only; none depends on the evolving state. Writing

```text
s[t] = a[t] * s[t-1] + b[t]
```

allows affine updates to compose associatively:

```text
(a2, b2) o (a1, b1) = (a2 * a1, a2 * b1 + b2)
```

The implementation retains the auditable Python loop as the default reference
(`use_parallel_scan=False`) and adds an opt-in tensor doubling scan
(`use_parallel_scan=True`). Both paths use the same parameters and have the
same parameter count. A future state-dependent gate would invalidate this
scan; it must not be substituted silently.

The planned A1 JSON configuration now selects `recurrent_path: "parallel_scan"`
explicitly. The runner and evaluator pass that selection into the model
factory; direct model construction still defaults to the sequential reference
so tests and diagnostic comparisons cannot silently change path.

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
`0ddd4e1aa69e24dc33bd6846aaed645199ef26990d5eee33dfa911098056e7ab`).

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

The proposal remains `planned_only: true`. The generalized training runner now
selects A0 or A1 only from the explicit stage/model-type pair and selects the
configured A1 recurrence path, and its A1
dry-run passed the pinned data, tokenizer, validation, logging, checkpoint, and
recovery checks. The execution gate still prevents training unless an explicit
pilot authorization is supplied. The fixed-prompt evaluator uses the same
prompts and decoding settings for either architecture and chooses separate
result paths.

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
but its reference recurrence may be slower on a GPU because it limits
time-dimension parallelism. No throughput, FLOP, or memory advantage is
assumed; all three must be measured.

The matched one-update RTX 3060 Ti benchmark used identical initial weights,
synthetic batches, FP16 autocast, micro-batch 2, accumulation 8, context 512,
and AdamW for both paths:

| Metric | Sequential reference | Associative scan |
| --- | ---: | ---: |
| Optimizer-step wall time | 8.3435 s | 0.3714 s |
| Throughput | 981.85 tokens/s | 22,055.47 tokens/s |
| Peak allocated VRAM | 1,057,416,704 bytes | 1,316,685,312 bytes |
| Peak reserved VRAM | 1,243,611,136 bytes | 1,526,726,656 bytes |

The scan was **22.46x faster** in this implementation benchmark, at the cost
of higher peak memory. Both paths had finite loss and gradients and completed
one AdamW update. At FP16 context 512, forward differences were max absolute
`0.001953125`, mean absolute `0.000254959`, and relative L2 `0.00076814`;
gradient differences were max absolute `0.00121737`, mean absolute
`1.80e-6`, and relative L2 `0.03355`. CPU FP32 tests pass the tighter forward
and gradient-agreement tolerances. The FP16 differences are recorded as
rounding from a different associative evaluation order, not hidden as exact
bitwise equality.

This is an implementation-path result, not an A1 training result or an A0
architectural comparison. The benchmark record is
`results/a1_recurrence_benchmark.json`.

The approved smoke test exposed the expected implementation risk: one
accumulated optimizer update processed 8,192 synthetic token positions in
11.5694 seconds, or **708.08 tokens/sec**, with 1,294,976,512 allocated and
1,526,726,656 reserved bytes on the RTX 3060 Ti. The Python timestep loop is a
likely bottleneck, but this preparation deliberately does not optimize or
redesign it. The measurement is a readiness result, not an A1 training result
or a matched A0 benchmark; the full comparison must use the same runner and
measurement method.

## Ten-update numerical stability check

The sequential reference and parallel scan were initialized from identical
weights and empty AdamW states, then consumed the same 80 pre-generated
micro-batches (10 updates, 8,192 tokens per update) with seed 17, FP16
autocast, the approved cosine warmup schedule, micro-batch 2, context 512, and
accumulation 8. No checkpoint was written and the 1,000-step pilot was not
started.

| Measurement | Result |
| --- | ---: |
| Non-finite losses or gradients | None in either path |
| Maximum absolute loss difference across 10 updates | `6.795e-5` |
| Maximum relative loss difference across 10 updates | `6.487e-6` |
| Final parameter relative-L2 difference | `0.001411` (0.141%) |
| Final gradient relative-L2 difference | `0.04960` (4.96%) |
| Reference path wall time / throughput | `82.036 s` / `998.59 tokens/s` |
| Parallel path wall time / throughput | `4.009 s` / `20,433.84 tokens/s` |
| Reference / parallel peak allocated VRAM | `1,392,904,192` / `1,656,395,776` bytes |

The loss and parameter trajectories remained close over the bounded check. The
larger full-gradient difference is retained as a measured FP16 operation-order
effect; it is not reported as bitwise equivalence. This evidence is sufficient
to select the parallel scan as the explicit A1 pilot path, while retaining the
sequential implementation as the reference diagnostic path. It is not an A1
training result and does not establish quality or architectural superiority.

The raw record is `results/a1_scan_stability.json`.

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
| A1 model construction, causal loss, backward pass, and parameter accounting | RECOMMENDED | Verified by CPU tests |
| A1 runner selection and shared checkpoint/recovery/logging behavior | RECOMMENDED | Generalized, dry-run passed, and pilot completed with verified checkpoint |
| Host CUDA smoke test for A1 at micro-batch 2 / context 512 | RECOMMENDED | Passed: finite loss/gradients, one AdamW update, no OOM |
| A1 fixed-prompt evaluation wiring | RECOMMENDED | Generalized; fixed-prompt evaluation remains separate from this training pilot |
| Associative scan validity, path benchmark, and ten-update stability | RECOMMENDED | Verified; no non-finite values, 0.141% final parameter relative-L2 difference, and explicit parallel-path selection |
| Final A1 pilot authorization | COMPLETE | Explicitly authorized; seed-17 1,000-step pilot completed |
| Gradient checkpointing | DEFERRED | Not implemented or enabled; unchanged from A0 |
| Attention, MoE, conditional depth, controller, persistent memory | DEFERRED | Explicitly outside A1 |

The seed-17 A1 pilot is complete. No A2, scaling, persistent-memory, or other
architectural extension is authorized by this result; those remain separate
future experiments.
