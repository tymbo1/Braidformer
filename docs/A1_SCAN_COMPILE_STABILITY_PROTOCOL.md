# A1 scan compilation stability protocol

Status: **frozen for engineering-only execution; no quality training started**.

This is the required follow-up to the compile preflight. It tests whether the
default-mode compiled A1 path remains finite and usable across ten optimizer
updates while preserving the unchanged recurrence and optimizer semantics. It
does not modify the normal runner or authorize a quality-training run.

## Frozen comparison

- Existing A1 configuration and parallel-scan equations.
- Fresh seed 17 initialization for both paths.
- One shared CPU-generated batch sequence with batch seed 2017.
- Ten optimizer updates per path; each update uses eight microbatches of
  micro-batch 2 and context length 512.
- Existing AdamW, FP16 autocast, gradient clipping, and learning rate.
- Eager path: the current A1 parallel scan.
- Compiled path: torch.compile in default mode; CUDA graphs are not enabled.
- Update 1 includes compiler warm-up. CUDA-event timing for the performance
  comparison covers updates 2 through 10 only.
- No checkpoint, optimizer state, log, or model parameter is persisted.

## Measurements and gates

Record per-update loss and gradient norm, warm-up time, steady-state CUDA time,
throughput, peak VRAM, final-parameter relative L2 difference, finite-value
checks, and any runtime/CUDA failure.

The predeclared engineering gates are:

1. both paths complete all ten updates with finite values and no runtime or
   CUDA-graph output-lifetime error;
2. compiled steady-state CUDA time is at least 10% lower than eager;
3. trajectory drift is reported per update and adjudicated before any runner
   integration. Passing gates 1 and 2 does not authorize normal-runner use.

Any compiled-path adoption requires a separate decision after this result.
This protocol contains no A2, architecture, data, quality-training, or
checkpoint work.
