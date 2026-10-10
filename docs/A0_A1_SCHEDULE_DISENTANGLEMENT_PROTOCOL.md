# A0/A1 schedule-disentanglement protocol

Status: **prepared and frozen for review; training not started**.

This is a bounded follow-up to the completed native 8× experiment. It keeps
the A0 and A1 implementations, TinyStories revision, tokenizer, context,
batch geometry, optimizer, precision, seeds, validation policy, and integrity
controls unchanged. It tests whether the native 8× result depends materially
on the two architectures following different learning-rate trajectories.

## Important control reduction

The frozen batch geometry processes

```text
2 micro-batch sequences × 8 accumulation steps × 512 tokens = 8,192 tokens/optimizer step
```

for both architectures. Consequently, a schedule normalized to token fraction
and one normalized to optimizer-step fraction are mathematically identical when
they use the same step budget. Running both as separate conditions would spend
the same compute twice without creating a distinct control.

The experiment therefore has three conditions in total:

1. **Native 8× reference:** the already completed result in
   `b6632cb`, using each arm's own normalized step schedule. No new training is
   performed for this condition.
2. **CUDA-fraction schedule control:** fresh 8× A0/A1 runs with one cosine
   schedule indexed by cumulative CUDA execution time.
3. **Shared token/step schedule control:** fresh A0/A1 runs with identical
   4,020-step / 32,931,840-token budgets and one shared step schedule. This is
   both an equal-token and equal-update comparison; actual CUDA time is still
   recorded but is not the acceptance metric.

This reduction is deliberate. Separating token and optimizer-step schedules
would require changing the frozen tokens-per-step geometry, which would add a
new optimization variable rather than isolate the existing confound.

## Frozen common settings

- Fresh seeds: `17`, `23`, and `29`.
- A0: existing 41,866,240-parameter decoder-only Transformer.
- A1: existing 41,870,336-parameter gated diagonal state-space model using
  `parallel_scan`.
- TinyStories revision, tokenizer, split manifest, and packed streams unchanged.
- Context length 512; micro-batch 2; gradient accumulation 8; effective batch
  size 16; FP16 autocast.
- AdamW with the existing betas, epsilon, weight decay, and gradient clipping;
  base learning rate `3e-4`.
- Validation every 100 optimizer steps and at the final step.
- Atomic checkpoints with optimizer and RNG state; SHA-256 sidecars; raw logs
  and checkpoints remain ignored and local.
- Stop on non-finite loss/gradients, unrecoverable CUDA failure, OOM, or
  checkpoint-integrity failure.
- No architecture, recurrence, optimizer, dataset, tokenizer, or A0/A1 source
  changes are allowed during execution.

## Condition 2: CUDA-fraction schedule control

The existing 8× measured-CUDA target remains
`T8 = 1,616.8551635742188` seconds per arm, with ±5% acceptance.

| Arm | Steps | Tokens | Calibration step CUDA s | Schedule warmup |
| --- | ---: | ---: | ---: | ---: |
| A0 | 8,000 | 65,536,000 | 0.20210689544677735 | 1% of `T8` = 16.168552 s |
| A1 | 4,020 | 32,931,840 | 0.40210519561767577 | 1% of `T8` = 16.168552 s |

The schedule is the same normalized warmup-plus-cosine function of CUDA
progress for both arms:

```text
progress_k = min(1, (completed_cuda_seconds_before_step_k
                     + calibrated_step_cuda_seconds) / T8)
```

The learning rate for step `k` is computed from `progress_k`; the calibrated
one-step term assigns the first update a non-zero warmup position while keeping
the schedule predeclared. Actual cumulative CUDA time remains the recorded
comparison measure. Small deviations between calibrated and measured step
times are retained as measured behavior and reported.

Configs:

- `configs/braidnano_a0_schedule_cuda_control.json` —
  `17d05a6ca330e477faeb738afe571614d3caa7ff95addda308a5fc2fb4653c2c`
- `configs/braidnano_a1_schedule_cuda_control.json` —
  `d1eefadc7e169702f5e58ba4a32365b381fcc41c2212e3d7fd9a7ca32bcb4eee`

Each config is canonicalized with seed 17. Seeds 23 and 29 use the existing
explicit `--seed` runtime override in isolated run directories; the summary,
run metadata, checkpoint, and result record preserve the exact runtime seed.

## Condition 3: shared token/step schedule control

Both arms run exactly 4,020 optimizer steps and 32,931,840 token positions.
Both therefore receive the same step-indexed cosine schedule: 41 warmup steps,
then decay to zero at step 4,020. This is also a common token-indexed schedule
because every step consumes exactly 8,192 tokens.

| Arm | Steps | Tokens | Warmup | CUDA acceptance |
| --- | ---: | ---: | ---: | --- |
| A0 | 4,020 | 32,931,840 | 41 steps | none; report measured time |
| A1 | 4,020 | 32,931,840 | 41 steps | none; report measured time |

Configs:

- `configs/braidnano_a0_equal_token_schedule_control.json` —
  `40ead0d50538f60ff4ca00527fe44ce0bc067c7d84a267e1596a88abab601336`
- `configs/braidnano_a1_equal_token_schedule_control.json` —
  `81638dc2b8b7e4f0bbe690d75d099f031268f9611289d2e743f0aced381a9d8f`

The primary comparison is validation loss at the common token/update budget.
CUDA time, wall time, throughput, VRAM, and checkpoint integrity remain
secondary recorded measurements.

## Measurements and comparison

For every new run record final and checkpoint validation loss, training curves,
learning rate, gradient norms, tokens, measured CUDA time, wall time,
throughput, peak allocated/reserved VRAM, numerical/CUDA/OOM failures, exact
runtime seed, config hash, and checkpoint hash/integrity.

Compare each new condition against the existing native 8× result and report:

- final A1-minus-A0 validation-loss delta;
- validation loss versus CUDA time and versus tokens;
- learning-rate trajectories;
- matched-time and equal-token AUC where the interval is valid;
- whether A1's advantage remains across all three seeds.

The exploratory advantage threshold remains `A1 − A0 <= −0.05` for the CUDA
matched condition and for the equal-token/update condition. No asymptotic fit,
crossover claim, or general architectural claim is permitted.

## Execution gate

This protocol and the four configs are committed before execution. Training
requires a separate explicit authorization after review of these exact
conditions. No 16× run, A2, scaling, persistent-memory integration, or
unrelated architecture work is included.
