# A1 end-to-end compile benchmark protocol

Status: frozen for engineering-only execution; no quality training authorized.

This benchmark measures how much of the isolated A1 mixer compile speedup
survives a complete eight-microbatch A1 language-model update. It does not
change the A1 architecture, equations, optimizer, data, or normal runner.

## Frozen question

For the same A1 parallel-scan model and update geometry, compare:

1. an isolated mixer forward with eager execution;
2. an isolated mixer forward with the mixer forward compiled;
3. an isolated mixer accumulated training update with eager execution;
4. an isolated mixer accumulated training update with the mixer forward
   compiled;
5. the complete A1 model with eager execution;
6. the complete A1 model with only its eight state-space mixer forwards
   compiled; and
7. the complete A1 model with the whole forward compiled in default mode.

The forward-only pair anchors the earlier approximately 3x mixer claim. The
accumulated-update pair checks the same component under backward and optimizer
semantics. The key end-to-end outputs are mixer-only end-to-end speedup,
full-model end-to-end speedup, and the ratios of those speedups to the
corresponding isolated references.

## Frozen conditions

- Configuration:
  configs/braidnano_a1_recurrent_state_space_proposal.json.
- Seed 17; token batches are generated once from a fixed CPU generator and
  shared by every complete-model path.
- Hidden-state and target batches for the isolated mixer objective are also
  generated once and shared by both isolated paths.
- Micro-batch 2, context length 512, eight gradient-accumulation microbatches,
  FP16 autocast, AdamW, existing learning rate, weight decay, and gradient
  clipping.
- Each path performs one warm-up update and five measured updates.
- Each isolated forward path performs one warm-up forward and five measured
  forwards. The forward-only path is not treated as a training result.
- All batches are device-resident before CUDA-event timing. Timing covers
  zero_grad through optimizer.step and excludes host preparation, compiler
  warm-up, checkpoint I/O, and unrelated synchronization.
- The eager complete-model path is the reference. The isolated forward anchor
  uses torch.compile reduce-overhead mode to reproduce the earlier component
  measurement; it is inference-only and is not an approved training mode. The
  isolated training pair, mixer-only complete path, and full path use
  torch.compile default mode with CUDA graphs disabled. The mixer-only
  complete path compiles each state-space mixer forward and leaves embeddings,
  normalization, feed-forward layers, and language head eager. The full path
  compiles the complete forward module.
- The original complete-model module remains the parameter and state owner for
  every path. Optimizers use its parameters; no compiled wrapper is
  checkpointed.

## Measurements and tolerances

Record for every path:

- hardware and software versions, parameter counts, source/config hashes;
- compiler warm-up wall/CUDA time, measured CUDA time per update, throughput,
  and peak allocated/reserved VRAM;
- per-update loss and gradient norm, finite-value status, and final parameter
  relative-L2 difference to eager;
- model-state key digest and optimizer-state structure invariants.

The predeclared short-horizon numerical tolerances for compiled complete-model
paths are maximum relative loss difference 1e-4, maximum relative gradient-norm
difference 1e-2, and final parameter relative-L2 difference 0.02 over the five
measured updates. Values outside tolerance are a result requiring
adjudication, not an automatic rejection of the systems path.

The benchmark must write only its named aggregate and report. It must not
write checkpoints, modify model artifacts, launch a quality run, or change
the default eager runner.
