# A2 research-options assessment

Status: **assessment only**. No A2 implementation, training, scaling, or
persistent-memory work was started by this document.

The frozen A0/A1 milestone is recorded in
[`docs/A0_A1_EQUAL_CUDA_REPLICATIONS.md`](A0_A1_EQUAL_CUDA_REPLICATIONS.md)
and committed as `00bb31a`, which has been pushed to `origin/main`. The
existing A0/A1 source, configurations, checkpoints, six run records, and
decision thresholds remain unchanged.

## Frozen evidence

- A0: 41,866,240 parameters; conventional eight-block decoder-only Transformer.
- A1: 41,870,336 parameters; eight gated diagonal state-space blocks, with the
  verified `parallel_scan` path.
- Seeds 17, 23, and 29 all met the predeclared equal-CUDA-time rule.
- Paired validation-loss differences `A1 - A0` were `-0.1619`, `-0.2008`, and
  `-0.1504`; mean `-0.1710`, sample standard deviation `0.0264`.
- This is evidence for the specified TinyStories, 42M-parameter,
  equal-measured-CUDA-time protocol. It is not evidence that A1 is generally
  superior, more FLOP-efficient, more energy-efficient, or architecturally
  novel.

## Equation-level prior-art mapping

The checked-in A1 mixer first projects the normalized token representation and
splits it into candidate, input-gate, and output-gate channels. With
`d = sigmoid(state_transition_logit)`, its per-channel update is:

```text
b[t] = (1 - d) * sigmoid(input_gate[t]) * tanh(candidate[t])
s[t] = d * s[t - 1] + b[t]
y[t] = sigmoid(output_gate[t]) * s[t]
```

The decay `d` is learned and channel-wise but constant over positions. The
candidate and gates depend on the current representation, not on the evolving
state. This is why the update is affine in `s[t - 1]` and admits the verified
associative scan. State is reset to zero for every supplied sequence. A1 then
uses the same residual feed-forward path, embeddings, positional embeddings,
normalization, and tied language-model head as A0.

| Method | Overlap with A1 | Material difference | Current novelty assessment |
| --- | --- | --- | --- |
| S4D / diagonal SSMs | Diagonal state dynamics and state-space sequence processing. | S4D is framed as a structured continuous-time SSM with diagonal `A`, `B`/`C` parameterization, discretization, and a convolution-kernel computation; its initialization is tied to the S4/HiPPO line. A1 uses a simple real learned decay plus gated nonlinear input/output and does not implement that S4D parameterization. | A1 is adjacent to the diagonal-SSM family, not evidence of a new family. See [Gu et al.](https://arxiv.org/abs/2206.11893). |
| Mamba / selective SSM | Input-conditioned recurrent processing and a hardware-conscious scan are closely related engineering themes. | Mamba makes `Delta`, `B`, and `C` functions of the input and uses a selective SSM with expanded state dimensions. A1 has current-input gates but a fixed transition coefficient and a simpler hidden-width affine recurrence; it is not a Mamba block. | A1 should not be called Mamba or claimed as a new selective-SSM mechanism without a deeper equation and implementation comparison. See [Gu and Dao](https://arxiv.org/abs/2312.00752). |
| RetNet | Both expose parallel and recurrent forms, use decaying state, and motivate efficient sequence processing. | RetNet's retention state is built from content-dependent `Q/K/V` projections and a decaying matrix, with parallel, recurrent, and chunkwise-recurrent forms. A1 has no retention `Q/K/V` state or chunkwise representation. | The proposed dual-timescale/chunk idea has clear prior-art overlap; any A2 contribution must be defined more narrowly than “two communicating states.” See [Sun et al.](https://arxiv.org/abs/2307.08621). |
| RWKV | Both use per-channel temporal decay, recurrent state, gating, and a parallelizable training formulation. | RWKV's time-mixing is a normalized weighted key/value recurrence with receptance, key, value, token shift, and a separate channel-mixing design. A1 uses candidate/input/output gates and a direct diagonal state, without RWKV's WKV numerator/denominator state. | A1 is a related gated recurrent design, not an RWKV reimplementation. See [Peng et al.](https://arxiv.org/abs/2305.13048). |

This is a focused literature mapping, not an exhaustive patent, publication, or
licensing search. The defensible present claim is: **A1 is an original
implementation and an independently measured result for this repository's
protocol; architectural novelty remains unestablished.** Before any paper or
public novelty claim, the unresolved work is an equation-level comparison of
initialization, state dimensionality, discretization, gating, scan semantics,
and block layout against canonical implementations, followed by matched
ablations.

## Three bounded next-step options

### Option 1 — longer training at the frozen 42M scale

**Hypothesis.** The observed A1 gap may be an early-optimization effect. With
more compute, A0 may catch up, the gap may persist, or the gap may widen.

**Controls.** Keep the A0/A1 source, TinyStories revision, tokenizer, split
manifest, context 512, micro-batch 2, accumulation 8, AdamW, base learning
rate `3e-4`, FP16 settings, A1 `parallel_scan`, and seed policy unchanged.
Start fresh from initialization for every arm; do not resume an existing pilot
or use an intermediate checkpoint from a differently scheduled run as the
definitive comparison. Use a normalized cosine schedule for the new total
steps and validate every 100 steps plus the final step.

**Proposed bounded budget.** Use a four-times extension of the committed
per-arm CUDA target: `4 * 202.106895447 = 808.427582` CUDA seconds. Based on
the existing calibration, the planning counts are:

| Arm | Steps | Token positions | Warmup |
| --- | ---: | ---: | ---: |
| A0 | 4,000 | 32,768,000 | 40 |
| A1 | 2,010 | 16,465,920 | 21 |

The A1 count is `floor(808.427582 / 0.402105196) = 2,010`; actual CUDA time
must be measured and reported rather than inferred. With three seeds this is
six fresh runs and approximately 1.35 GPU-hours of update-time budget before
validation and I/O overhead.

**Meaningful result.** Use the final validation loss at the matched measured
CUDA budget as the primary comparison. Report normalized validation curves as
descriptive secondary evidence, not as definitive equal-budget checkpoints.
A1 persistence would require the predeclared `A1 - A0 <= -0.05` rule at the
final comparison; A0 catch-up would be indicated by the final difference
entering the `±0.05` parity band or reversing across seeds. Also report time
to any predeclared validation-loss target if both models reach it.

**Risks.** TinyStories may saturate or reward memorization; a longer cosine
schedule changes optimization dynamics; the equal-CUDA step counts are based
on calibration and still require actual timing; and the cost is materially
larger than the completed milestone.

### Option 2 — optimize A1 execution without changing its mathematics

**Hypothesis.** A1's remaining speed and memory gap is partly caused by the
generic tensor-doubling scan and materialized intermediates, not by the
recurrence itself. Kernel fusion, memory reuse, or bounded recomputation might
improve execution while preserving the learned function.

**Controls.** Keep the recurrence equations, parameter count, state reset,
optimizer, and training protocol unchanged. Retain the sequential reference
path. Any candidate must pass forward/gradient agreement at context 512,
finite-value checks, and the existing ten-update stability procedure before
being considered for a new benchmark. Compare only the same device-resident
CUDA-event scope and record VRAM separately.

**Proposed bounded budget.** Engineering and measurement only: one repeat of
the existing 20-update CUDA timing calibration and one ten-update numerical
stability check. No quality training is part of this option. A practical
adoption threshold would be at least 10% lower mean A1 CUDA time per update or
peak reserved VRAM, with no failure of the existing numerical tolerances.

**Meaningful result.** A measurable execution improvement at unchanged
equations, parameters, outputs within declared tolerances, and no new
instability. If the threshold is not reached, retain the current scan and
defer further kernel work.

**Risks.** Floating-point operation order can change training trajectories;
fusion may require custom kernels or make portability worse; recomputation can
trade memory for wall time; and implementation work can expand into an
unbounded infrastructure project. The frozen A1 result must remain the
comparison reference.

### Option 3 — dual-state or multiscale recurrence

**Hypothesis.** Separate fast/local and slow/global states could represent
different temporal scales more effectively than one state of the same total
width, while retaining a parallelizable causal computation.

**Controls.** Preserve total hidden/state width and the A1 embedding, FFN,
normalization, context, optimizer, and data protocol. Predeclare the state
allocation, communication schedule, parameter budget, and reset semantics.
Match A1's parameter count as closely as possible (the existing tolerance is
approximately `0.01%`). Verify causality and zero-state reset. If cross-state
messages depend on evolving states, do not assume the current associative scan
remains valid: prove a new associative operator or retain a reference path and
measure it separately.

**Proposed bounded budget.** First perform a design review, parameter count,
CPU forward/backward/causality tests, and one CUDA smoke/timing test. Only if
that candidate is internally valid should a fresh seed-17 42M comparison be
considered at the committed `202.106895447`-second CUDA target, with its step
count determined by measurement. A three-seed replication would require a
separate approval; no such training is authorized by this document.

**Meaningful result.** Against frozen A1, a later authorized experiment would
need `A2 - A1 <= -0.05` at matched measured CUDA time, or parity with a
predeclared at-least-10% reduction in CUDA time or peak reserved VRAM. A single
pilot could motivate replication but would not establish superiority.

**Risks.** State partitioning can reduce usable capacity; cross-state
communication adds compute and may destroy scan associativity; chunk or global
state handling can introduce leakage; and the broad concept overlaps with
known multi-scale/chunkwise recurrent designs, especially RetNet. This option
has the highest attribution and prior-art risk and should not be presented as
novel without a more specific mechanism and comparison.

## Bounded recommendation

Keep the A0/A1 milestone frozen. If the immediate objective is engineering
efficiency, Option 2 is the smallest next investigation and does not require a
quality claim. If the immediate scientific question is whether A0 eventually
catches up, Option 1 is the cleanest next experiment. Option 3 should remain a
design-only proposal until the equation-level prior-art review and scan
validity are resolved.

No option above authorizes training. Any future run needs a separately approved
final configuration and must preserve the current A0/A1 records.
