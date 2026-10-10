# A0/A1 longer matched-CUDA-time result

Status: **completed for fresh seed 17** under frozen protocol commit `ff16dde`.
No additional seeds, scaling, A2 implementation, SSRE, or persistent-memory
work was started.

The full concise record, including all 100-step validation/training curve
points, is [`results/a0_a1_longer_equal_cuda_seed17.json`](../results/a0_a1_longer_equal_cuda_seed17.json).
Raw `steps.jsonl` logs and checkpoints remain local and excluded from Git.

## Decision summary

| Metric | A0 | A1 |
| --- | ---: | ---: |
| Parameters | 41,866,240 | 41,870,336 |
| Optimizer steps | 4,000 | 2,010 |
| Training tokens | 32,768,000 | 16,465,920 |
| Warmup steps | 40 | 21 |
| Final training loss | 2.921024 | 3.341156 |
| Final validation loss | 3.114907 | 3.003067 |
| CUDA elapsed time | 839.620 s | 817.642 s |
| Deviation from 808.428 s target | +3.858% | +1.140% |
| Training wall time | 843.858 s | 819.794 s |
| Total elapsed wall time | 933.405 s | 885.736 s |
| Training throughput | 38,831 tok/s | 20,085 tok/s |
| Peak allocated VRAM | 1,329,467,904 B | 1,657,272,320 B |
| Peak reserved VRAM | 1,593,835,520 B | 1,931,476,992 B |
| Numerical instability | None | None |

Both arms completed within the predeclared ±5% CUDA-time tolerance. The final
validation difference is:

```text
validation_loss_A1 - validation_loss_A0 = -0.1118402
```

That meets the exploratory `<= -0.05` rule for this one fresh seed under the
longer matched-CUDA protocol. It is not a replicated result or a general claim
of architectural superiority.

## Matched-CUDA validation anchors

The existing 100-step validation interval was retained. The nearest validation
records to the quarter-budget points were:

| Approx. budget | A0 step / validation loss | A1 step / validation loss |
| ---: | ---: | ---: |
| 25% | 1,000 / 3.615861 | 500 / 3.495068 |
| 50% | 2,000 / 3.307817 | 1,000 / 3.209486 |
| 75% | 3,000 / 3.182362 | 1,500 / 3.034014 |
| 100% | 4,000 / 3.114907 | 2,010 / 3.003067 |

The A1 anchors correspond to approximately 24.88%, 49.75%, 74.63%, and 100%
of its declared optimizer steps. They are observations from the same frozen
training schedules, not comparisons to intermediate checkpoints from a
differently scheduled run.

## Integrity and reproducibility

- Dataset revision: `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`.
- Dataset manifest SHA-256: `e67a3c36c2403f0eb4437def2b72aee5f30b508767cb3ea401efe155b3bfdd55`.
- Tokenizer SHA-256: `9d2b84ebf358729cf548a2962f10fe15eebc512a0b6f42b3f90a010d44b589f3`.
- A0 final checkpoint SHA-256: `70b81bc90c5ada413528f76b6d955caa4f5bb93a48d2ea87a64241ae2ac1b43c`.
- A1 final checkpoint SHA-256: `34619fa20f2d773c4a1efe387dc9be86d4c1c1157cf156285e564f5a73082f9d`.
- Both checkpoint sidecars verified `complete: true` with matching config and
  data-manifest hashes.
- Gradient checkpointing was configured in the inherited config shape but is
  not implemented and was not enabled.
- The known Flash Attention nondeterminism warning remains applicable; this
  run records empirical results but does not claim bit-for-bit replay.

The original A0/A1 pilot checkpoints, equal-CUDA results, and replication
records were not modified or overwritten.
