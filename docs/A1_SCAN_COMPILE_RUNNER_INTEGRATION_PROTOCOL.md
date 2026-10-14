# A1 scan compilation runner integration protocol

Status: frozen for default-off implementation; no quality training authorized.

This protocol defines the smallest runner integration after the completed
compile preflight and ten-update stability result. It adds an opt-in compiled
forward path for the existing A1 parallel-scan model while preserving eager
execution as the default.

## Frozen scope

- Add a boolean compile_model configuration field with default false.
- Accept compile_model=true only for the explicit A1 recurrent-state-space
  configuration using parallel_scan.
- When enabled, wrap the model only for forward/evaluation calls with
  torch.compile(mode="default").
- Keep the original model as the owner of parameters, optimizer state,
  parameter accounting, checkpoint state, and resume loading. This avoids
  compiled-wrapper state-dict key changes.
- Leave all equations, model parameters, optimizer settings, data order,
  precision, schedules, checkpoint schema, and quality budgets unchanged.
- Existing configurations and the normal runner remain compiled-off.
- Do not add CUDA-graph mode, compiler flags, A2 mechanisms, or quality
  training to this implementation slice.

## Required checks

1. Configuration validation accepts the explicit A1 opt-in and rejects
   compiled A0 or non-parallel paths.
2. Unit coverage verifies that the default path does not invoke the compiler
   and that the opt-in forwards mode default to torch.compile.
3. The full test suite, syntax compilation, and diff check pass.
4. A bounded host-CUDA one-update runner smoke uses a separate planned-only
   compiled A1 config, exercises checkpoint creation, and verifies that the
   checkpoint stores unwrapped model keys. It is an integration smoke, not a
   quality-training result.
5. The smoke checkpoint is loaded through the same compiled configuration to
   verify resume state compatibility; no checkpoint is retained as a result.

## Adoption boundary

Passing these checks only makes a default-off runner capability available.
The default A1 runner, existing proposal configuration, and any quality
training decision remain unchanged. A future quality run would require its
own explicit config hash, authorization, and result report.
