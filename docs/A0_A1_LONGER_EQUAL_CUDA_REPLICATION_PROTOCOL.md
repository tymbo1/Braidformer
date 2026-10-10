# A0/A1 4× matched-CUDA-time replication protocol

Status: **frozen before execution for seeds 23 and 29**.

This is a replication of the committed 4× seed-17 protocol in
`docs/A0_A1_LONGER_EQUAL_CUDA_PROTOCOL.md`. It changes only the fresh random
seed. The A0 and A1 architectures, data, tokenizer, step budgets, optimizer,
learning-rate schedules, validation policy, CUDA timing scope, and acceptance
tolerance remain unchanged.

## Frozen arms

| Seed | Arm | Steps | Training tokens | Warmup | Run directory |
| ---: | --- | ---: | ---: | ---: | --- |
| 23 | A0 Transformer | 4,000 | 32,768,000 | 40 | `results/runs/a0_seed23_longer_equal_cuda/` |
| 23 | A1 parallel scan | 2,010 | 16,465,920 | 21 | `results/runs/a1_seed23_longer_equal_cuda/` |
| 29 | A0 Transformer | 4,000 | 32,768,000 | 40 | `results/runs/a0_seed29_longer_equal_cuda/` |
| 29 | A1 parallel scan | 2,010 | 16,465,920 | 21 | `results/runs/a1_seed29_longer_equal_cuda/` |

Each arm starts from fresh initialization. Existing seed-17 4× runs, all 1×
runs, checkpoints, configurations, and results are preserved and are not used
as resumes or overwritten.

## Controls and measurements

The runs use the approved TinyStories revision and tokenizer, context 512,
micro-batch 2, gradient accumulation 8, FP16 autocast, AdamW, learning rate
`3e-4`, weight decay and clipping from the inherited configuration, and cosine
decay normalized to that arm's declared total steps. Validation remains every
100 optimizer steps and at the final step. A1 uses the verified `parallel_scan`
path.

The matched-CUDA target is `808.427581787` CUDA seconds per arm, inherited from
the original 1× calibration and 4× protocol. CUDA event time is measured from
`optimizer.zero_grad` through `optimizer.step` with device-resident batches;
host wall time, analytical FLOP labels, and energy are recorded separately and
are not substituted for the CUDA-time budget.

Each run records the complete validation series, training loss, tokens,
throughput, accumulated CUDA time, wall time, peak allocated/reserved VRAM,
finite-value status, warnings, and a SHA-256-verified final checkpoint with
optimizer and RNG state. Gradient checkpointing remains configured in the JSON
shape but is not implemented or enabled.

Both arms of each seed are valid matched-CUDA evidence only when they complete
and each total lies within ±5% of the `808.427581787`-second target. No
additional seed, 8× run, A2 implementation, scaling experiment, or
persistent-memory integration is part of this protocol.
