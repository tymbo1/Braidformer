# Bounded A1 equal-compute comparison proposal

Status: **proposal only**. No additional training is authorized by this
document.

The completed A0 and A1 pilots matched 8,192,000 training token positions, but
they did not match wall-clock time, GPU execution time, or measured FLOPs:

| Measurement | A0 Transformer | A1 parallel scan |
| --- | ---: | ---: |
| Training token positions | 8,192,000 | 8,192,000 |
| Training wall time, excluding validation/checkpoint overhead | 208.55 s | 404.41 s |
| Mean training throughput | 39,320.88 tokens/s | 20,256.62 tokens/s |
| Total elapsed wall time | 231.62 s | 436.41 s |

Equal tokens are therefore not equal compute. Equal wall time is also not
automatically equal GPU compute: host time includes synchronization, input
movement, validation, checkpointing, and runtime overhead, while different
architectures can issue different amounts of GPU work during the same wall
interval. FLOPs are not recorded in the existing pilot logs.

## Proposed bounded comparison legs

### Leg 1 — equal-token reference

This is already complete and remains the control condition:

- A0 and A1, seed 17 initially;
- 1,000 optimizer steps and 8,192,000 token positions;
- the approved TinyStories streams, tokenizer, context, effective batch,
  AdamW settings, and validation prefix;
- validation at steps 100 through 1,000;
- no architecture, optimizer, or data changes.

The existing result is reported as an equal-token comparison only.

### Leg 2 — equal training-wall budget

Use the measured A0 training-only wall time, **208.55 seconds**, as the
planning budget. Exclude validation and final-checkpoint overhead from this
budget, but record total elapsed time separately.

| Arm | Proposed optimizer steps | Proposed token positions | Basis |
| --- | ---: | ---: | --- |
| A0 | 1,000 | 8,192,000 | Existing measured A0 budget |
| A1 | 515 | 4,218,880 | `floor(208.55 × 20,256.62 / 8,192)` |

The A1 step count is a measured-throughput planning estimate, not an empirical
result. The actual run must record its training wall time; if the completed
step budget does not fall within ±5% of the target, label the result as a
wall-time approximation rather than silently calling it equal-wall.

For this leg, normalize the cosine schedule to each arm's planned optimizer
steps: retain base learning rate `3e-4`, use 1% warmup rounded up, and reach
zero at that arm's final step. This avoids giving the shorter A1 run an
unfinished fraction of the A0 1,000-step schedule. Validation remains at every
100 optimizer steps plus the final step (A1: 100, 200, 300, 400, 500, 515).

### Leg 3 — equal measured GPU-compute budget

This is the preferred scientific comparison, but its budget is not yet
available. Add bounded CUDA-event instrumentation around forward, backward,
gradient clipping, and the optimizer update, excluding host input preparation,
validation, checkpoint I/O, and unrelated synchronization. Sum those GPU event
durations over optimizer steps.

1. Measure the A0 reference GPU-compute total under the approved conditions.
2. Give A1 the same measured CUDA-event budget, stopping only at a completed
   optimizer step.
3. Record the actual CUDA-event time, wall time, tokens, validation losses,
   throughput, and peak VRAM for both arms.

The A0 GPU-compute budget is intentionally **TBD until measured**. It must not
be inferred from the 208.55-second host time, parameter count, or theoretical
complexity. If theoretical FLOPs are later added, report them as a separate
counter with its counting convention; do not substitute FLOPs for measured
GPU execution time without labeling the distinction.

Use the same normalized learning-rate rule as Leg 2, the same fixed validation
prefix, and validation every 100 optimizer steps plus the final completed step.
Preserve only the final model/optimizer/RNG checkpoint for each arm; raw logs
and checkpoints remain excluded from Git.

## Decision criteria

For each leg, report the final validation-loss difference
`Delta = validation_loss_A1 - validation_loss_A0` at the same comparison
budget:

- **Equal-compute advantage:** `Delta <= -0.05`, with finite losses/gradients,
  no CUDA/OOM failure, and intact checkpoints.
- **Parity:** `abs(Delta) <= 0.05`; report comparable quality without claiming
  superiority.
- **Failure at that budget:** `Delta > 0.05`, or a critical integrity/runtime
  failure.

The `0.05` margin is the existing exploratory protocol threshold, not a claim
of statistical significance. Seed 17 is the bounded first comparison. Seeds 23
and 29 should be run only after review, with per-seed results and the mean
reported; three seeds do not justify overstated certainty.

The fixed-prompt generation result remains qualitative and is not part of the
equal-compute quality criterion. A1's current lower loss at equal tokens must
not be described as an equal-compute advantage until Leg 3 is measured.
