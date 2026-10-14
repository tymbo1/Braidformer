# A1 scan compilation runner integration result

Status: complete; default-off engineering capability only.

The frozen integration protocol was implemented without changing the A1
equations, parameters, optimizer, data order, precision, schedules, or
checkpoint schema. Existing configurations remain eager because
compile_model defaults to false. The separate
configs/braidnano_a1_recurrent_state_space_compiled_proposal.json enables the
opt-in path and remains planned_only.

## Implementation boundary

The runner keeps the original A1 module as the owner of parameters, optimizer
state, parameter accounting, checkpoint serialization, and resume loading.
Only forward and evaluation calls use torch.compile(mode="default") when the
explicit config field is true. This preserves ordinary model state-dict keys
and keeps the normal runner path unchanged.

## Bounded host-CUDA smoke

- Device: NVIDIA GeForce RTX 3060 Ti.
- Runtime: PyTorch 2.12.0+cu130.
- Config SHA-256:
  62f2002c4cacd8707882526eda10e253718d76cf74a8ea7441493de54b4da7d9.
- Scope: one optimizer update with max_steps=1; planned-only config passed
  through the explicit execute-approved smoke boundary.
- Outcome: completed with finite values; no failure or quality-training result.
- Parameter count: 41,870,336.
- Train loss: 10.540326; validation loss: 9.603049.
- Checkpoint sample keys remained unwrapped, including
  token_embedding.weight and blocks.0.state_space_norm.weight; no key began
  with _orig_mod.
- A second invocation loaded the checkpoint with the same compiled config,
  matched the config hash, and completed at step 1 without a new update.

The temporary smoke run and checkpoint were written outside the repository
under /tmp and are not retained as project artifacts. No quality-training
checkpoint or result was produced.

## Verification

- Full test suite: 80 passed.
- Python syntax compilation: passed.
- git diff --check: passed.

This result makes the compiled path available for a separately authorized
quality run. It does not authorize one, enable compilation by default, or
claim a quality improvement.
