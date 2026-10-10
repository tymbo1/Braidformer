# A0/A1 4× matched-CUDA-time replications

Status: **complete for seeds 17, 23, and 29**.

The seed-23 and seed-29 runs replicate the frozen 4× seed-17 protocol in
`docs/A0_A1_LONGER_EQUAL_CUDA_PROTOCOL.md`. Each arm started from fresh
initialization and used the same approved TinyStories revision, tokenizer,
architecture, optimizer, normalized cosine schedule, validation policy, and
CUDA timing scope. Existing runs and checkpoints were not reused or
overwritten.

## Final comparison

Lower validation loss is better. `Delta` is `A1 - A0`.

| Seed | A0 validation | A1 validation | Delta | A0 CUDA s | A1 CUDA s | A0 tok/s | A1 tok/s | Threshold |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 17 | 3.114907 | 3.003067 | −0.111840 | 839.620 | 817.642 | 38,831 | 20,085 | met |
| 23 | 3.154583 | 3.028929 | −0.125653 | 842.139 | 826.106 | 38,716 | 19,881 | met |
| 29 | 3.139107 | 3.014810 | −0.124297 | 845.951 | 822.691 | 38,542 | 19,963 | met |

The target was `808.427581787` CUDA seconds per arm with ±5% acceptance. All
six arm totals were valid; the largest deviation was A0 seed 29 at +4.642%.
Each A0 arm processed 32,768,000 token positions in 4,000 steps, while each
A1 arm processed 16,465,920 positions in 2,010 steps.

Across the three paired replications, the mean validation-loss difference was
`−0.120597` with sample standard deviation `0.007614` and range
`[−0.125653, −0.111840]`. A1 met the predeclared exploratory criterion
`A1 − A0 <= −0.05` in all three seeds (3/3).

## Resource and integrity observations

A0 peak VRAM was `1,329,467,904` allocated and `1,593,835,520` reserved bytes
in all three runs. A1 peak VRAM was `1,657,272,320` allocated and
`1,931,476,992` reserved bytes in all three runs. A1 remained slower per token
but reached lower final validation loss under this matched CUDA-time protocol.

All six runs completed with finite losses and gradients, no OOM or CUDA
failures, and SHA-256-verified final checkpoints containing optimizer and RNG
state. The full per-arm measurements, checkpoint hashes, and provenance are in
[`results/a0_a1_longer_equal_cuda_replications.json`](../results/a0_a1_longer_equal_cuda_replications.json).
Raw step logs and checkpoints remain ignored and local.

## Interpretation boundary

This is replicated evidence for the specified 42M-parameter TinyStories
experiment under the measured CUDA-time protocol. It does not establish
general Transformer superiority, energy efficiency, true FLOP efficiency,
performance on other tasks or datasets, or architectural novelty. Energy was
not measured, and analytical FLOP labels were not used as the acceptance
metric. No additional training, 8× experiment, A2 implementation, scaling, or
persistent-memory integration was started.
