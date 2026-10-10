# A0 seed-17 training result

Status: **completed**. This is the first empirical A0 pilot; it does not
establish architectural superiority or general reasoning capability.

## Reproducibility record

| Item | Value |
| --- | --- |
| Git commit | `9e43883` |
| Configuration | `configs/braidnano_a0_baseline_proposal.json` |
| Configuration SHA-256 | `6dd6b619be65c50437121235a0a9ae0a0493e5d210cc2f512a45d5079818d7e4` |
| Dataset | TinyStories revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64` |
| Tokenizer | 32,000-token byte-level BPE; SHA-256 `9d2b84ebf358729cf548a2962f10fe15eebc512a0b6f42b3f90a010d44b589f3` |
| Seed | 17 |
| Architecture | A0 conventional decoder-only Transformer |
| Parameters | 41,866,240 |
| Context / micro-batch / effective batch | 512 / 2 / 16 |
| Optimizer | AdamW, learning rate `3e-4`, weight decay `0.1` |
| Schedule | cosine with 10 warmup steps |
| Precision | FP16 CUDA autocast with FP32 parameters |
| Gradient checkpointing | **Not implemented; not enabled** |

## Measurements

- Optimizer steps: **1,000**
- Tokens processed: **8,192,000**
- Training wall time excluding validation: **208.55 s**
- Total elapsed wall time: **231.62 s**
- Mean training throughput: **39,320.9 tokens/s**
- Peak allocated/reserved VRAM: **1,329,410,560 / 1,591,738,368 bytes**
- Training loss: `10.5826` at the first step; `3.6161` at step 1,000
- Validation loss: `4.9929` at step 100; `3.7512` at step 1,000
- Validation ran every 100 optimizer steps over the fixed 131,072-token validation prefix.

The raw step log, run metadata, summary, and checkpoint are preserved locally
under `results/runs/a0_seed17_pilot/`. The final checkpoint is
`checkpoint_step_001000.pt`; its SHA-256 is
`23314261cd0a3efd5eeea2d461268d43908d51c62bcce553dd59a2ad4be38ca3`.

All 1,000 step records are contiguous and reach the exact token budget. The
checkpoint sidecar verifies successfully. No non-finite loss, gradient, CUDA
failure, or out-of-memory event occurred. PyTorch did emit a flash-attention
nondeterminism warning under the existing warn-only deterministic policy; this
is recorded as a reproducibility note, not as numerical instability.

This result is a TinyStories pilot measurement. It is not a claim of broad
model quality, and no A1 run has begun.
