# A0/A1 schedule-disentanglement result

Status: **complete for seeds 17, 23, and 29** under frozen protocol commit
`eb1cb6a`. The experiment used fresh initialization for all twelve arms. A0,
A1, the TinyStories data/tokenizer, batch geometry, optimizer, and model
implementations were unchanged.

The machine-readable record is
[`results/a0_a1_schedule_disentanglement_results.json`](../results/a0_a1_schedule_disentanglement_results.json).
Raw step logs and checkpoints remain in ignored local run directories.

## Conditions

The CUDA-fraction control used A0 at 8,000 steps / 65,536,000 tokens and A1
at 4,020 steps / 32,931,840 tokens, with the same cosine learning-rate
function indexed by cumulative CUDA progress. The target was `1,616.855164`
CUDA seconds per arm with the frozen ±5% tolerance.

The equal-token/equal-step control used 4,020 steps / 32,931,840 tokens for
both architectures and the identical 41-step warmup plus step-indexed cosine
schedule. CUDA time was measured but deliberately not matched; A1 therefore
used approximately twice the CUDA time in this condition.

`Delta` means final `A1 validation loss − A0 validation loss`; negative values
favour A1.

## CUDA-fraction schedule control

| Seed | A0 validation | A1 validation | Delta | A0 CUDA s (deviation) | A1 CUDA s (deviation) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 17 | 3.030860 | 2.795733 | −0.235127 | 1,641.599 (+1.530%) | 1,601.213 (−0.967%) |
| 23 | 3.035996 | 2.814932 | −0.221064 | 1,644.919 (+1.736%) | 1,606.956 (−0.612%) |
| 29 | 3.058668 | 2.817199 | −0.241468 | 1,651.907 (+2.168%) | 1,619.263 (+0.149%) |
| **Mean** | **3.041841** | **2.809288** | **−0.232553** | **1,646.142** | **1,609.144** |

The paired-delta sample SD was `0.010443`; mean relative loss advantage was
`7.645%`; A1 met the predeclared `Delta <= −0.05` criterion in `3/3` seeds.
All six CUDA totals passed the ±5% tolerance. Mean throughput was `39,608`
tokens/s for A0 and `20,411` tokens/s for A1. A0 processed 65,536,000 token
positions per run; A1 processed 32,931,840.

## Equal-token/equal-step schedule control

| Seed | A0 validation | A1 validation | Delta | A0 CUDA s | A1 CUDA s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 17 | 3.075444 | 2.801542 | −0.273902 | 837.367 | 1,646.738 |
| 23 | 3.118390 | 2.821473 | −0.296917 | 840.808 | 1,675.630 |
| 29 | 3.140847 | 2.818943 | −0.321903 | 831.268 | 1,605.329 |
| **Mean** | **3.111561** | **2.813986** | **−0.297574** | **836.481** | **1,642.566** |

The paired-delta sample SD was `0.024007`; mean relative loss advantage was
`9.564%`; A1 met the same criterion in `3/3` seeds. Both architectures
processed exactly 32,931,840 token positions per run. Mean throughput was
`39,170` tokens/s for A0 and `20,003` tokens/s for A1. This is an equal-token
and equal-update result, not an equal-CUDA-time or equal-energy result.

## Resource and integrity record

Peak memory was constant across these arms: A0 allocated/reserved
`1,329,467,904 / 1,593,835,520` bytes; A1 allocated/reserved
`1,657,272,320 / 1,931,476,992` bytes. Mean training wall times were:

| Condition | A0 training wall | A1 training wall |
| --- | ---: | ---: |
| CUDA-fraction | 1,654.633 s | 1,613.458 s |
| Equal-token/equal-step | 840.763 s | 1,646.880 s |

All twelve runs completed with finite losses and gradients, no CUDA/OOM
failure, and `failure: null`. All twelve final optimizer/RNG checkpoints
passed the existing `verify_checkpoint` SHA-256, byte-count, completeness,
and metadata checks. Gradient checkpointing remains unimplemented and was not
enabled.

## Validation curves and AUC

Validation curves use only observed validation records. Values are
piecewise-linearly interpolated within each seed's common observed interval;
no point is extrapolated beyond that interval. The following are mean paired
A1-minus-A0 gaps at fractions of that interval:

| Condition | 25% | 50% | 75% | 100% | Mean normalized AUC gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| CUDA-fraction, cumulative CUDA time | −0.044795 | −0.184407 | −0.190760 | −0.232576 | −0.149221 |
| Equal-token/equal-step, normalized optimizer steps | −0.330595 | −0.289621 | −0.283364 | −0.297574 | −0.325555 |

The CUDA-fraction curve shows the residual A1 advantage expanding over the
common observed interval rather than disappearing under the shared CUDA-based
learning-rate trajectory. The equal-token curve remains negative throughout
the observed normalized step interval. These curve summaries are descriptive;
they do not establish an asymptotic law or crossover point.

## Comparison with native 8×

The completed native 8× reference in `b6632cb` had mean final delta
`−0.245285` (sample SD `0.002728`). The CUDA-fraction control had mean delta
`−0.232553`, a difference of `+0.012732` relative to native 8×, and still
won in all three seeds. The equal-token/equal-step control had mean delta
`−0.297574`, `−0.052289` lower than native 8×, but it spent approximately
`1,642.566 / 836.481 = 1.96×` as much CUDA time on A1 as on A0. It therefore
cannot be interpreted as a matched-compute advantage.

The schedule controls support a residual A1 advantage under this specific
42M-parameter TinyStories setup after controlling either CUDA-progress
learning-rate alignment or token/update budget. They do not separate every
architecture/optimizer interaction, do not establish matched FLOPs or energy,
and do not establish general architectural superiority, transfer to another
corpus, or novelty.

No A2, 16× run, scaling, post-training, persistent memory, WNF, or unrelated
architecture work was started.
