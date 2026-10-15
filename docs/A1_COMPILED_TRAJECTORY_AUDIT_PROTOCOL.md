# A1 compiled trajectory audit protocol

Status: **frozen and prepared for engineering-only execution; not executed; no
quality training authorized**.

This audit is the next gate after the ten-update compile stability result. It
tests whether the full default-mode compiled A1 path follows the eager
reference over a longer, matched sequence of optimizer updates. It is an
execution-semantic audit, not a quality-training experiment and not an
architecture change.

## Frozen comparison

- Existing A1 configuration
  `configs/braidnano_a1_recurrent_state_space_proposal.json`, with the
  unchanged parallel-scan equations and 42M-parameter count.
- Seed `17` for model initialization and deterministic execution.
- A single CPU-generated input sequence with batch seed `5017`. The sequence
  contains `50 × 8` microbatches, each of micro-batch size `2` and context
  length `512`. The exact sequence is moved to CUDA once and reused by both
  paths in the same order.
- Fifty optimizer updates per path, each using eight microbatches. Update 1
  includes the compiled graph warm-up; updates 2–50 are the post-warm-up
  trajectory.
- Fresh model instances loaded from the same captured initial state.
- Fresh AdamW optimizers with the existing A1 hyperparameters, empty initial
  optimizer state, FP16 autocast, gradient clipping, and no scheduler.
- Eager path: the unwrapped A1 model.
- Compiled path: `build_forward_model(..., compile_model=True)`, which uses
  `torch.compile(mode="default")` while the unwrapped base model owns
  parameters, optimizer state, and state keys.
- No checkpoint, model artifact, validation result, or durable training log
  is written. The script requires `--execute-approved` before it will perform
  the in-memory updates.

## Measurements

Record for every update and path:

- mean training loss and gradient norm;
- CUDA-event update time and wall time;
- parameter L2 norm and a fixed 1,024-element parameter probe;
- finite-value status and optimizer step state.

Record for the comparison:

- per-update relative loss and gradient-norm differences;
- per-update parameter-probe difference;
- final full-parameter relative L2 difference;
- model state-key digests and wrapped-key checks;
- optimizer-state structure and final step checks;
- steady-state throughput, compiler warm-up time, and peak VRAM.

## Predeclared audit gates

The result is considered trajectory-consistent only if all of the following
hold; a failed gate is still recorded as an engineering result and does not
authorize quality training or default-runner adoption:

1. both paths complete exactly 50 finite updates with the same batch digest and
   optimizer step sequence;
2. maximum per-update relative loss difference is at most `1e-4`;
3. maximum per-update relative gradient-norm difference is at most `1e-2`;
4. final parameter relative L2 difference is at most `0.02`;
5. model state-key digests and optimizer-state structures match, with no
   compiled wrapper keys in the base model.

The performance timing is diagnostic. It is reported after excluding update 1
from the steady-state aggregate, but no new speedup threshold is used by this
trajectory audit. Any longer-horizon drift is adjudicated from the raw
per-update trajectories before further runner or training authorization.
