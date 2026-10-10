# A0/A1 equal-CUDA-time replications

Status: **completed for seeds 23 and 29** under the accepted protocol
`de41ef2`. Seed 17 is included as the accepted reference run. All runs used
fresh initialization in independent ignored run directories; no prior
checkpoint was reused or overwritten.

## Per-seed results

| Seed | A0 validation | A1 validation | A1 − A0 | A0 CUDA time | A1 CUDA time | CUDA budget valid |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 17 | 3.7396816 | 3.5777884 | −0.1618933 | 209.3490 s | 205.3696 s | Yes |
| 23 | 3.7953693 | 3.5946133 | −0.2007560 | 209.2909 s | 206.2034 s | Yes |
| 29 | 3.7232520 | 3.5728926 | −0.1503594 | 210.3746 s | 202.8638 s | Yes |

The target was `202.1068954` CUDA seconds with a ±5% validity tolerance. Every
arm in every seed fell within that tolerance. Each A0 run processed 8,192,000
tokens in 1,000 steps; each A1 run processed 4,112,384 tokens in 502 steps.

| Seed | A0 training tokens/s | A1 training tokens/s | A0 peak allocated/reserved | A1 peak allocated/reserved |
| ---: | ---: | ---: | ---: | ---: |
| 17 | 38,934.20 | 19,972.44 | 1,329,467,904 / 1,593,835,520 B | 1,657,272,320 / 1,931,476,992 B |
| 23 | 38,944.97 | 19,891.54 | 1,329,467,904 / 1,593,835,520 B | 1,657,272,320 / 1,931,476,992 B |
| 29 | 38,746.51 | 20,217.82 | 1,329,467,904 / 1,593,835,520 B | 1,657,272,320 / 1,931,476,992 B |

All six runs completed with finite losses and gradients, no CUDA/OOM failures,
and verified final checkpoints. The complete machine-readable record is
[`results/a0_a1_equal_cuda_replications.json`](../results/a0_a1_equal_cuda_replications.json).

## Aggregate decision

Across seeds 17, 23, and 29:

- Mean A1-minus-A0 validation-loss difference: **−0.1710029**.
- Sample standard deviation: **0.0264044**.
- Range: **0.0503966** (from −0.2007560 to −0.1503594).
- Advantage threshold: **−0.05**.
- Seeds meeting the threshold: **3/3**.
- CUDA-budget-valid pairs: **3/3**.

A1 therefore meets the predeclared advantage criterion consistently across the
three measured seeds. This supports a reproducible result for this specific
TinyStories, model-size, seed, schedule, and equal-CUDA-time protocol. It does
not establish general architectural superiority or replace broader datasets
and evaluations.

The previously accepted commits `de41ef2` and `59a71ee` were pushed to
`origin/main` after an audit found no tracked checkpoints, raw logs, or
secrets. The new replication checkpoints and raw logs remain excluded from
Git. No A2, scaling, architecture modification, or persistent-memory work
was started.
