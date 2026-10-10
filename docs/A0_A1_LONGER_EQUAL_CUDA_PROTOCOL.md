# A0/A1 longer matched-CUDA-time protocol

Status: **frozen for one seed-17 experiment; committed before execution**.
This protocol authorizes exactly one fresh A0 run and one fresh A1 run. It does
not authorize additional seeds, scaling, A2, SSRE, or persistent memory.

The original A0/A1 configurations, source, checkpoints, and results remain
unchanged. The two protocol-specific configurations below are derived copies
that change only the declared step and training-token budgets:

- `configs/braidnano_a0_longer_equal_cuda_seed17.json`
- `configs/braidnano_a1_longer_equal_cuda_seed17.json`

Frozen configuration hashes:

- A0 derived config: `1d9aa4f30ee2cefc7a98db0ecf0287c6985b525ff4629ef0638fa368464e785f`
- A1 derived config: `242207c968bbe07203b347e91198200cc248ab66179ee75faa0730f6f1b5e4ef`
- Original A0 config remains `6dd6b619be65c50437121235a0a9ae0a0493e5d210cc2f512a45d5079818d7e4`.
- Original A1 config remains `0ddd4e1aa69e24dc33bd6846aaed645199ef26990d5eee33dfa911098056e7ab`.

## Fixed compute budget

The existing calibration target was `202.10689544677734` CUDA seconds for the
completed 1,000-step A0 comparison. The longer-run target is exactly four times
that value:

```text
T4 = 808.4275817871094 CUDA seconds per arm
```

Planning uses the existing mean update times of `0.20210689544677735` seconds
for A0 and `0.40210519561767577` seconds for A1:

| Arm | Steps | Tokens/step | Training tokens | Planned warmup | Projected CUDA time |
| --- | ---: | ---: | ---: | ---: | ---: |
| A0 | 4,000 | 8,192 | 32,768,000 | 40 | 808.427581787 s |
| A1 | 2,010 | 8,192 | 16,465,920 | 21 | 808.231443192 s |

The A1 planning total is `0.0243%` below `T4`. These are planning values, not
substitutes for measured execution. The experiment is a valid matched-CUDA
comparison only if both completed arms have accumulated CUDA-event totals within
`±5%` of `T4`.

## Matched model and training settings

Both arms use the existing approved TinyStories revision, tokenizer, split
manifest, packed streams, context length 512, micro-batch 2, gradient
accumulation 8, effective batch size 16, FP16 autocast, AdamW, betas, epsilon,
weight decay, gradient clipping, and validation-token budget. A0 is the
existing Transformer. A1 is the existing eight-block gated diagonal
state-space model with the verified `parallel_scan` path. No architecture or
optimization strategy is changed.

Both runs start from fresh seed-17 initialization at step zero. No existing
pilot checkpoint is loaded or overwritten. Run directories are isolated and
ignored by Git.

## Learning-rate schedule

Both arms use base learning rate `3e-4` and a schedule normalized to that arm's
declared total steps:

- A0: linear warmup for steps 1–40 (`ceil(0.01 * 4000)`), then cosine decay to
  zero at step 4,000.
- A1: linear warmup for steps 1–21 (`ceil(0.01 * 2010)`), then cosine decay to
  zero at step 2,010.

At matched fractional progress through the declared run, both arms therefore
have the same normalized schedule progress. No schedule is resumed from an
earlier pilot, and no schedule parameter may be changed after observing a
result. Gradient checkpointing remains configured in the inherited JSON shape
but is not implemented or enabled.

## Validation and recorded curves

The existing validation interval of every 100 optimizer steps is retained, plus
the final step when it is not an interval multiple. The approximate quarter
CUDA-budget observations are the nearest existing validation intervals:

| Budget fraction | A0 validation step | A1 validation step | Approximate fraction of A1 steps |
| ---: | ---: | ---: | ---: |
| 25% | 1,000 | 500 | 24.88% |
| 50% | 2,000 | 1,000 | 49.75% |
| 75% | 3,000 | 1,500 | 74.63% |
| 100% | 4,000 | 2,010 | 100% |

Thus the full validation series is A0 steps `100, 200, ..., 4000` and A1
steps `100, 200, ..., 2000, 2010`. The quarter points are reporting anchors;
the underlying validation metrics come from the same fixed schedule and data
stream, not from differently scheduled intermediate checkpoints.

For each arm, record training and validation loss, gradient norm, learning rate,
tokens processed, optimizer step, per-step and accumulated CUDA-event time,
training wall time, total elapsed wall time, throughput, peak allocated and
reserved VRAM, finite-value status, and warnings. Save and SHA-256 verify the
final checkpoint including optimizer and RNG state. Raw logs and checkpoints
remain excluded from Git.

## Decision and stop rules

The primary quality comparison is final validation loss at the completed
matched-CUDA budget. Report the quarter-point and 100-step curves as measured
learning trajectories, while keeping the final matched-budget result primary.

- **Valid matched-CUDA run:** both arms complete and each total CUDA time is
  within `±5%` of `T4`.
- **Invalid budget match:** report all measurements, but do not claim an
  equal-CUDA comparison if either arm falls outside the tolerance.
- Stop on non-finite loss or gradients, unrecoverable CUDA failure, OOM, or
  checkpoint-integrity failure.
- Do not launch seed 23, seed 29, A2, scaling, or any follow-up automatically.

Energy and true executed FLOPs are not acceptance metrics here. CUDA elapsed
time is the declared matched-compute measure; wall time, analytical FLOP labels,
and energy telemetry, if available, must be reported separately.
