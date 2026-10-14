# A1 scan compilation preflight protocol

Status: **frozen for engineering-only execution; no quality training started**.

This preflight evaluates whether PyTorch Inductor can accelerate the existing
A1 parallel-scan training path without changing its equations, parameters,
optimizer, data, or model architecture. It is an implementation benchmark, not
a new A1 training result and not an authorization to replace the current
runner path.

## Question

Can torch.compile in default mode reduce the steady-state CUDA time of the
existing 42M-parameter A1 path while keeping losses and gradients finite?

The candidate wraps the unchanged A1RecurrentStateSpaceLM configured with
use_parallel_scan=True. Default compiler mode is used deliberately: the
reduce-overhead CUDA-graph mode overwrote compiled outputs across the existing
eight-microbatch gradient-accumulation pattern during exploratory preflight,
even with cudagraph_mark_step_begin(). No CUDA-graph mode is part of this
protocol.

## Frozen scope

- Existing A1 source, recurrence equations, parallel-scan path, parameter
  count, config, optimizer, precision, and batch geometry are unchanged.
- Configuration: configs/braidnano_a1_recurrent_state_space_proposal.json.
- Seed 17; eight pre-generated CPU batches shared by eager and compiled paths;
  micro-batch 2; context 512; accumulation 8; FP16 autocast.
- AdamW and the existing learning rate, weight decay, and gradient clipping
  are retained.
- Each path performs one in-memory warm-up optimizer update and three measured
  in-memory optimizer updates. No checkpoint, log, or model parameter is
  persisted by the runner.
- CUDA-event time is measured only for the three steady-state updates;
  compiler warm-up time and peak VRAM are reported separately.

## Measurements

Report for eager and compiled paths:

- device, PyTorch/CUDA versions, parameter count, and source/config hashes;
- warm-up and steady-state CUDA time, tokens per second, and peak VRAM;
- mean loss and gradient norm over measured updates;
- final-parameter relative-L2 difference and loss/gradient deltas;
- finite-value, checkpoint-written, and training-pilot-started flags.

The exploratory engineering threshold is at least 10% lower steady-state
CUDA time with finite values and no checkpoint writes. Passing this preflight
does not authorize runner integration. A separate ten-update stability
protocol must verify eager/compiled trajectory drift and accumulation behavior
before any compiled path is considered for normal training.

## Execution boundary

The runner may write only the preflight aggregate and report named by its CLI.
It must not write checkpoints, modify source model parameters on disk, or
change the canonical A1 benchmark results. No A2, longer training, new data,
persistent memory, or architecture change is included.
