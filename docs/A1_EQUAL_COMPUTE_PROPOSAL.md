# Bounded A0/A1 equal-compute comparison proposal

Status: **executed under explicit authorization; protocol rules unchanged**.
Measured results are recorded in
[`docs/A0_A1_EQUAL_CUDA_RESULT.md`](A0_A1_EQUAL_CUDA_RESULT.md).

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
interval. FLOPs were not recorded in the existing pilot logs.

## Bounded CUDA timing calibration

A bounded timing calibration was run after the pilots. It loaded the preserved
final checkpoints and optimizer states, performed two unmeasured warm-up
updates and twenty measured in-memory updates per arm, then discarded the
model and optimizer. It did not write a checkpoint, persist an optimizer
update, or start a training pilot. The machine-readable record is
[`results/a0_a1_cuda_timing_calibration.json`](../results/a0_a1_cuda_timing_calibration.json),
and the bounded probe is
[`scripts/measure_a0_a1_cuda_timing.py`](../scripts/measure_a0_a1_cuda_timing.py).

The timing scope was the default CUDA stream from `optimizer.zero_grad` through
`optimizer.step`, with all calibration batches already resident on the GPU.
Host-to-device copies, data loading, validation, checkpoint I/O, and unrelated
synchronization were excluded from the CUDA event. A host `perf_counter` was
recorded for the same device-resident scope.

| Calibration measurement | A0 Transformer | A1 parallel scan |
| --- | ---: | ---: |
| Mean CUDA-event time/update | 0.202106895 s | 0.402105196 s |
| Median CUDA-event time/update | 0.196771317 s | 0.403978989 s |
| Total measured CUDA-event time, 20 updates | 4.042137909 s | 8.042103912 s |
| Mean host time/update, same scope | 0.202128821 s | 0.402126285 s |
| CUDA-event throughput | 40,533.01 tokens/s | 20,372.78 tokens/s |
| Peak allocated VRAM | 1,329,843,712 bytes | 1,669,996,032 bytes |
| Peak reserved VRAM | 1,520,435,200 bytes | 1,860,173,824 bytes |
| Finite losses and gradients | Pass | Pass |

The measured A1 update took `1.98956693x` the A0 CUDA-event time. The timing
probe used the final pilot states only to obtain steady-state execution timing;
its losses are not quality measurements and its in-memory updates are not a
training result.

### What each cost measure means

- **CUDA elapsed time** is the primary matched-compute measure: summed CUDA
  event duration for the specified device update scope.
- **Host wall time** is a separate operational measure. The calibration host
  times are nearly identical to the CUDA events because batches were device
  resident and each update was synchronized. The completed pilot wall times
  were `208.55 s` for A0 and `404.41 s` for A1, excluding validation/checkpoint
  overhead as recorded by the pilot summaries.
- **Estimated FLOPs** are an analytical lower-bound counter, not measured GPU
  work. Under the record's convention, the lower bound is `2.248415379456e12`
  training FLOPs per A0 optimizer step and `2.042256949248e12` per A1 step
  (approximately `2.2484e15` and `1.0252e15` respectively across the proposed
  1,000-step and 502-step runs).
  It excludes embedding, normalization, nonlinearities, softmax/masking,
  optimizer work, memory traffic, kernel overhead, and A1 recurrence/scan
  elementwise operations. It is not used to set the budget or acceptance
  decision.
- **Energy** was not measured. No synchronized power telemetry was sampled,
  so energy cannot be inferred from CUDA time or the FLOP estimate and is not
  an acceptance criterion for this bounded pilot.

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

### Leg 2 — equal training-wall budget (secondary)

Use the measured A0 training-only wall time, **208.55 seconds**, as the
planning budget. Exclude validation and final-checkpoint overhead from this
budget, but record total elapsed time separately.

| Arm | Proposed optimizer steps | Proposed token positions | Basis |
| --- | ---: | ---: | --- |
| A0 | 1,000 | 8,192,000 | Existing measured A0 budget |
| A1 | 515 | 4,218,880 | `floor(208.55 × 20,256.62 / 8,192)` |

The A1 step count is a measured-throughput planning estimate, not an empirical
result. This remains a secondary wall-time proposal; it is not the next run
selected by this document. If it is later authorized, the actual run must
record its training wall time. If the completed step budget does not fall
within ±5% of the target, it must be labeled a wall-time approximation rather
than silently called equal-wall.

For this leg, normalize the cosine schedule to each arm's planned optimizer
steps: retain base learning rate `3e-4`, use 1% warmup rounded up, and reach
zero at that arm's final step. This avoids giving the shorter A1 run an
unfinished fraction of the A0 1,000-step schedule. Validation remains at every
100 optimizer steps plus the final step (A1: 100, 200, 300, 400, 500, 515).

### Leg 3 — equal measured GPU-compute budget (primary proposal)

The calibration establishes the fixed device-time target. Use the A0
calibration mean multiplied by the existing 1,000-step budget:

`T_cuda = 1,000 × 0.20210689544677735 s = 202.10689544677734 s`.

The exact proposed step budgets are:

| Arm | Optimizer steps | Token positions | Calibration-based CUDA target |
| --- | ---: | ---: | ---: |
| A0 | 1,000 | 8,192,000 | 202.106895447 s |
| A1 | 502 | 4,112,384 | 201.856808200 s |

The A1 count is `floor(T_cuda / 0.40210519561767577) = 502`. Its projected
device time is `0.250087247 s` (0.1237%) below the A0 target, so the two fixed
budgets are matched within the calibration precision without using an
intermediate checkpoint. The actual CUDA-event total must still be recorded in
each authorized run; the calibration is a planning measurement, not a quality
run.

The primary equal-compute run must use these rules:

- Start both arms from fresh seed-17 initialization at step zero. Do not load
  either completed pilot checkpoint, and do not use an intermediate checkpoint
  from a differently scheduled run as a definitive comparison.
- Keep the approved TinyStories revision, tokenizer, split manifest, packed
  streams, context length 512, micro-batch 2, accumulation 8, effective batch
  size 16, FP16 execution, AdamW, betas, epsilon, weight decay, gradient clip,
  and A1 `parallel_scan` path unchanged.
- Use base learning rate `3e-4` and cosine schedule normalized to the declared
  step budget. Warmup is linear for `ceil(0.01 × planned_steps)`: 10 steps for
  A0 and 6 steps for A1. Cosine decay begins after warmup and reaches zero at
  the final planned step.
- Consume the same deterministic training stream from token zero. Validation
  uses the fixed validation prefix and the existing 131,072-token evaluation
  budget.
- Validate at A0 steps `100, 200, 300, 400, 500, 600, 700, 800, 900, 1000`;
  validate at A1 steps `100, 200, 300, 400, 500, 502`. The final step is always
  evaluated, even when it is not an interval multiple.
- Record per-step CUDA-event duration for the same device update scope, total
  CUDA time, host training wall time, total elapsed wall time, tokens,
  validation losses, throughput, peak allocated/reserved VRAM, finite-value
  status, and final checkpoint hashes. Checkpoint and optimizer/RNG state
  integrity must be verified, but raw logs and checkpoints remain excluded
  from Git.

The comparison uses the final validation loss at the two declared final steps.
It does not compare an A1 step-502 result with an A0 step-502 intermediate
checkpoint, nor does it use the existing equal-token pilot checkpoints as
definitive equal-compute results.

For budget classification, report each arm's actual summed CUDA time against
`T_cuda`. A result is **CUDA-budget matched** only when both arms complete their
declared step budgets and each total is within ±5% of `T_cuda`; otherwise report
the measurements but do not make an equal-compute claim. This is a reporting
criterion, not permission to alter the step counts or schedule during a run.

## Decision criteria

For the primary Leg 3 comparison, first require a valid CUDA-budget match as
defined above. Then report the final validation-loss difference
`Delta = validation_loss_A1 - validation_loss_A0` at the same measured CUDA
budget:

- **Equal-compute advantage:** `Delta <= -0.05`, with finite losses/gradients,
  no CUDA/OOM failure, intact checkpoints, and a valid CUDA-budget match.
- **Parity:** `abs(Delta) <= 0.05`; report comparable quality without claiming
  superiority. The same runtime and checkpoint conditions still apply.
- **Failure at that budget:** `Delta > 0.05`, or a critical integrity/runtime
  failure. A budget mismatch is reported separately and prevents an
  equal-compute claim.

The `0.05` margin is the existing exploratory protocol threshold, not a claim
of statistical significance. Seed 17 is the bounded first comparison. Seeds 23
and 29 should be run only after review, with per-seed results and the mean
reported; three seeds do not justify overstated certainty.

The fixed-prompt generation result remains qualitative and is not part of the
equal-compute quality criterion. A1's current lower loss at equal tokens and
the CUDA timing calibration must not be described as an equal-compute quality
advantage until the protocol above is authorized and executed.
