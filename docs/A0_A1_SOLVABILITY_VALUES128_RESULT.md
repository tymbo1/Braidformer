# BraidFormer A0/A1 128-value solvability-rung result

Status: **complete**. This is the bounded seed-17 next rung after the 64-value one-association calibration.

Frozen protocol commit: `68fbfd37b1c93a6941086a6e88b368243b48382f`.

## Final measurements

| Metric | A0 | A1 |
| --- | ---: | ---: |
| Parameters | 41,866,240 | 41,870,336 |
| Reserved answer tokens | 128 | 128 |
| Optimizer steps | 2,000 | 2,000 |
| Token positions | 256,000 | 256,000 |
| Final training target loss | 0.000509 | 0.000498 |
| Training greedy exact match | 1.0000 | 1.0000 |
| Final held-out target loss | 3.488070 | 4.252668 |
| Held-out greedy exact match | 0.6484 | 0.5938 |
| CUDA training seconds | 182.390 | 288.644 |
| Elapsed wall seconds | 219.680 | 330.418 |
| CUDA tokens/second | 1403.6 | 886.9 |
| Peak allocated VRAM | 0.808 GiB | 0.808 GiB |
| Peak reserved VRAM | 0.934 GiB | 0.936 GiB |

## Interpretation

Both unchanged 42M models passed the primary calibration observation: they reached 100% autoregressive one-token retrieval on the repeated training set with the 128-token candidate answer pool.

On the diagnostic held-out set, A0 reached `0.6484` and A1 `0.5938`. The paired A1 − A0 target-loss delta was `+0.764598` and the A1 − A0 exact-match delta was `-0.0547`. A0 was better on this diagnostic at this seed and budget. Because the held-out keys and mappings are disjoint while the earlier record value remains in the input, this measures transfer of the simple copy/retrieve operation rather than unseen fact learning.

A1 used `1.583×` A0's CUDA time and achieved `0.633×` A0's average token throughput. This was not a matched-compute comparison.

The result strengthens the evidence for an A0 advantage on this explicit retrieval diagnostic, but it is still one seed and one budget. It does not isolate model capacity from recurrent-state representation or optimization, and it does not authorize higher value pools, additional seeds, scaling, A2, persistent memory, or SNAIL/SSRE integration.

## Selected validation checkpoints

The machine-readable aggregate preserves all 20 logged validation checkpoints.

| Step | A0 held-out loss | A1 held-out loss | A1 − A0 | A0 exact | A1 exact |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 3.937495 | 4.909516 | +0.972022 | 0.6484 | 0.5625 |
| 500 | 3.475846 | 4.278783 | +0.802936 | 0.6484 | 0.5938 |
| 1,000 | 3.483192 | 4.255300 | +0.772108 | 0.6484 | 0.5938 |
| 1,500 | 3.487337 | 4.252874 | +0.765538 | 0.6484 | 0.5938 |
| 2,000 | 3.488070 | 4.252668 | +0.764598 | 0.6484 | 0.5938 |

## Integrity and scope

Both runs completed with finite losses and gradients, no CUDA/OOM failure, and SHA-256-verified final checkpoints containing optimizer and RNG state. Gradient checkpointing was configured in the protocol but remains unimplemented and disabled.

- A0 checkpoint: `af4dc212c3712fa99da9ac48cfda369e396a2a668b4cc996eb6b0b7e95507300`
- A1 checkpoint: `7da12d71482d622230145c39ada6204eba8f0008dea3d4b472dba747fb4406fa`
- Synthetic data hash: `937a210c99d23c494ae296f28305df2816a48ec827a3a2ac5fa92d323040b9b9`
- Raw logs, checkpoints, and sidecars remain in ignored `results/runs/` directories.
- No additional rung or architecture experiment was started.
