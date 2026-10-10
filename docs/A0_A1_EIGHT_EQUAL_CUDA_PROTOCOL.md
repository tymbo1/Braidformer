# A0/A1 8× matched-CUDA-time protocol

Status: **frozen before execution for seeds 17, 23, and 29**.

This protocol continues the committed 1× and 4× A0/A1 evidence chain. It
changes only the declared training duration. A0 and A1 source, architectures,
data, tokenizer, optimizer family, batch geometry, precision, and measured
CUDA timing scope remain unchanged.

## Fixed compute budget and calibration basis

The original 1× calibration measured mean CUDA-event times of
`0.20210689544677735` seconds per A0 optimizer update and
`0.40210519561767577` seconds per A1 update. The 1× target was
`202.10689544677734` seconds per arm. This protocol defines:

```text
T8 = 8 × 202.10689544677734 = 1,616.8551635742188 CUDA seconds per arm
```

The A0 budget is exactly eight times the original 1× step/token budget. The A1
budget is derived from the established calibration rather than equalizing token
counts:

| Arm | Steps | Tokens/step | Training tokens | Warmup | Calibration projection |
| --- | ---: | ---: | ---: | ---: | ---: |
| A0 Transformer | 8,000 | 8,192 | 65,536,000 | 80 | 1,616.855163574 s |
| A1 parallel scan | 4,020 | 8,192 | 32,931,840 | 41 | 1,616.462886383 s |

The A1 step count is `floor(T8 / 0.40210519561767577) = 4,020`, with a
calibration projection `−0.0243%` from the target. These projections do not
replace measured execution. A completed comparison is valid matched-CUDA
evidence only when both arms complete and each total CUDA-event time is within
the existing ±5% tolerance of `T8`.

## Frozen configurations

Each configuration is planned-only and is committed before execution:

- `configs/braidnano_a0_eight_equal_cuda_seed17.json`
  (`c0baa323b5910d40cda13eefc8bf7cbe0c92cd5a91e7936546ce4912c2c42a55`)
- `configs/braidnano_a1_eight_equal_cuda_seed17.json`
  (`f741291638804bb8e21a85f8e7552fa4b41ee494f8c537dae7dc189f441eeecb`)
- `configs/braidnano_a0_eight_equal_cuda_seed23.json`
  (`74438d7dd25d53248afd27f0bf58bf306c821adfeeddcd49135a2920a93c9e62`)
- `configs/braidnano_a1_eight_equal_cuda_seed23.json`
  (`214967c544800e4711af6ea14ce14e3879efaa87219b88917fa18c9483c1775b`)
- `configs/braidnano_a0_eight_equal_cuda_seed29.json`
  (`c13aed77fce2a5ebe50a6d0d4a5ade6fc5fecd2efc85d69ab81c4eab1eca60ab`)
- `configs/braidnano_a1_eight_equal_cuda_seed29.json`
  (`60c5c197302093b1b1b5ed3e123381162ef0264f55fd436fbd25b370d112c7cb`)

For every seed, A0 and A1 start from fresh initialization at step zero in
isolated ignored run directories. No prior checkpoint is loaded or overwritten.
The existing 1×/4× source, configurations, checkpoints, and result records are
preserved unchanged.

## Controlled settings

Both arms use the pinned TinyStories revision, tokenizer, split manifest, and
packed streams; context length 512; micro-batch 2; gradient accumulation 8;
effective batch size 16; FP16 autocast; AdamW with the established betas,
epsilon, weight decay, and gradient clipping; and the same fixed validation
token budget. A0 remains the existing decoder-only Transformer. A1 remains the
existing eight-block gated diagonal state-space model using the verified
`parallel_scan` path. No architecture or optimization strategy is changed.

Each arm uses base learning rate `3e-4` and cosine decay normalized to its own
declared total steps, with 1% linear warmup rounded up: 80 steps for A0 and 41
for A1. The schedule reaches zero at that arm's final step. Validation remains
every 100 optimizer steps and at the final step: A0 steps 100 through 8,000;
A1 steps 100 through 4,000 plus 4,020. Gradient checkpointing remains present
in the inherited configuration shape but is not implemented or enabled.

## Measurements and stop rules

Record for every arm: validation and training curves; optimizer steps; tokens;
learning rate; gradient norms; per-step and accumulated CUDA-event time;
training and total wall time; throughput; peak allocated/reserved VRAM;
finite-value status; warnings; exact seed; configuration hash; data/tokenizer
hashes; and final checkpoint hash/integrity including optimizer and RNG state.
Stop on non-finite loss or gradients, unrecoverable CUDA error, OOM, or
checkpoint-integrity failure. Raw logs and checkpoints remain excluded from
Git.

CUDA time is summed over the default CUDA stream from `optimizer.zero_grad`
through `optimizer.step` with device-resident batches. Host input preparation,
validation, checkpoint I/O, and unrelated synchronization are excluded from
that primary timing measure. Wall time, analytical FLOP labels, and energy are
reported separately; none substitutes for the CUDA-time budget. Energy is not
expected to be available from the existing instrumentation.

## Matched-time curve and AUC analysis

The training behavior is not changed for curve analysis. After each run,
validation records are paired with cumulative CUDA time equal to the sum of
the preceding per-step CUDA-event durations. For each budget level, use
piecewise-linear interpolation only between observed validation points.

Report matched validation loss at 25%, 50%, 75%, and 100% of that level's
planned CUDA target, matching the existing 1×/4× analysis convention. Report
the A1-minus-A0 gap at every anchor. If an anchor is outside either arm's
observed validation-time domain, report it as unavailable rather than
extrapolating.

For each seed and budget level, compute a paired AUC over the common observed
interval:

```text
common_start = max(first A0 validation CUDA time, first A1 validation CUDA time)
common_end = min(planned target, actual A0 total CUDA time, actual A1 total CUDA time)
```

Integrate the two piecewise-linear validation-loss curves over
`[common_start, common_end]`. Report both raw loss-seconds and normalized AUC
(`AUC / (common_end - common_start)`, the time-averaged validation loss).
If the common interval is invalid, do not report an AUC. No crossover is fit or
extrapolated from these curves.

## Decision and comparison criteria

The primary 8× result is the final validation-loss difference at the completed
matched-CUDA budget, with the interpolated anchor curves and AUC as secondary
trajectory summaries. Retain the established exploratory rule:

- **A1 advantage:** `validation_loss_A1 - validation_loss_A0 <= -0.05`, with
  valid CUDA-budget matching, finite values, no critical runtime failure, and
  intact checkpoints.
- **Parity:** absolute final difference `<= 0.05` under the same conditions.
- **No supported advantage:** positive difference above `0.05`, or an invalid
  matched-CUDA budget/integrity result.

Compare the three-seed 8× mean and variation against the existing three-seed
1× and 4× means and variations. Classify the mean advantage as narrowing,
approximately stable, widening, or inconsistent only from the measured data.
Three budget levels remain limited evidence; do not claim an asymptotic
crossover unless a future analysis has a defensible non-extrapolative basis.

No A2, scaling, persistent memory, SSRE, or unrelated architecture work is
part of this protocol.
