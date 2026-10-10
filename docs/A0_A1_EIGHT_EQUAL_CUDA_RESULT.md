# A0/A1 8× matched-CUDA-time replication

Status: **complete for seeds 17, 23, and 29**.

The experiment used the frozen protocol in
[`A0_A1_EIGHT_EQUAL_CUDA_PROTOCOL.md`](A0_A1_EIGHT_EQUAL_CUDA_PROTOCOL.md),
committed as `188327e`. Each arm used fresh initialization in an isolated run
directory. A0 and A1 source, architectures, data, tokenizer, and prior
checkpoints were not modified or reused.

## Frozen protocol

The planned target was `1,616.855164` CUDA seconds per arm, derived as eight
times the original 1× target. A0 used 8,000 optimizer steps and 65,536,000
token positions. A1 used 4,020 steps and 32,931,840 positions, derived from
the established A1 CUDA-time calibration rather than equalizing token counts.
Both used context 512, micro-batch 2, gradient accumulation 8, effective
batch 16, AdamW at `3e-4`, normalized cosine decay, FP16 autocast, and the
existing validation policy. Warmup was 80 steps for A0 and 41 for A1; A1 used
the verified `parallel_scan` path. The acceptance tolerance remained ±5%.

## Final measurements

`Delta` is `A1 validation loss − A0 validation loss`; lower loss and more
negative deltas favour A1.

| Seed | A0 validation | A1 validation | Delta | A0 CUDA s (deviation) | A1 CUDA s (deviation) | A0 tok/s | A1 tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 17 | 3.043765 | 2.801542 | −0.242223 | 1,652.623 (+2.212%) | 1,606.473 (−0.642%) | 39,453 | 20,445 |
| 23 | 3.068930 | 2.821473 | −0.247457 | 1,676.955 (+3.717%) | 1,601.213 (−0.967%) | 38,884 | 20,512 |
| 29 | 3.065118 | 2.818943 | −0.246174 | 1,660.276 (+2.685%) | 1,611.768 (−0.315%) | 39,273 | 20,378 |
| **Mean** | — | — | **−0.245285** | — | — | — | — |

The sample standard deviation of the final paired deltas was `0.002728`.
A1 met the established `Delta <= −0.05` criterion in all three seeds. All
six CUDA totals passed the ±5% budget tolerance. A1 processed half as many
token positions and remained slower per token, as expected from its measured
throughput; this is a matched-CUDA result, not an equal-token result.

Peak memory was constant across these runs: A0 allocated/reserved
`1,329,467,904 / 1,593,835,520` bytes and A1 allocated/reserved
`1,657,272,320 / 1,931,476,992` bytes. No run reported a non-finite value,
CUDA failure, OOM, or numerical-instability stop. Each final checkpoint was
complete and SHA-256 verified, including optimizer and RNG state. The exact
hashes, configuration hashes, timings, token counts, wall times, and failure
fields are in
[`results/a0_a1_eight_equal_cuda_replications.json`](../results/a0_a1_eight_equal_cuda_replications.json).

## Validation loss versus CUDA time

Validation curves were aligned by cumulative CUDA-event time. Values below are
means of the three seed-paired A1-minus-A0 interpolations at the declared
fraction of each planned target. Interpolation was piecewise linear between
observed validation records; no extrapolation was used.

| Budget | 25% | 50% | 75% | 100% | Availability |
| --- | ---: | ---: | ---: | ---: | --- |
| 1× | −0.127417 | −0.220685 | −0.177913 | −0.170264 | 3/3 at every anchor |
| 4× | −0.118679 | −0.107389 | −0.137541 | −0.124303 | 3/3 at every anchor |
| 8× | −0.066196 | −0.157610 | −0.215533 | unavailable | 3/3 through 75%; A1 ended below the nominal target at 100% |

The 8× final endpoint remains directly comparable under the protocol's ±5%
acceptance rule, but it is not presented as an exact 100%-target interpolated
point because all three A1 runs ended slightly below `1,616.855164` CUDA
seconds. Their actual endpoint deltas are the final values in the table above.

The curve analysis also computed a paired AUC over each seed's common observed
interval, using the protocol's
`max(first validation times)` to `min(planned target, both actual totals)`
rule. All nine budget/seed AUCs were valid:

| Budget | Mean raw A1−A0 AUC (loss·s) | Mean normalized A1−A0 AUC | Sample SD of normalized delta |
| --- | ---: | ---: | ---: |
| 1× | −30.371 | −0.186775 | 0.022271 |
| 4× | −88.907 | −0.115493 | 0.005635 |
| 8× | −264.546 | −0.168758 | 0.010581 |

The complete per-seed anchors, intervals, AUCs, and source-run metadata are
machine-readable in the result JSON. The analysis helper is
[`scripts/analyze_a0_a1_cuda_curves.py`](../scripts/analyze_a0_a1_cuda_curves.py).

## Comparison across training budgets

The mean final delta was `−0.171003` at 1×, `−0.120597` at 4×, and
`−0.245285` at 8×. Thus the mean advantage narrowed from 1× to 4×, then
widened from 4× to 8×. It did not change sign or become inconsistent across
seeds: A1 won 3/3 at every budget level. The three-budget sequence is
non-monotonic, so it does not support a smooth convergence or crossover claim.
No asymptotic fit or crossover extrapolation was performed.

These measurements support the specific 42M-parameter TinyStories result
under the declared measured-CUDA protocol. They do not establish general
Transformer superiority, energy efficiency, analytical FLOP efficiency,
performance on other tasks or datasets, or architectural novelty.

No A2, scaling, persistent-memory, or unrelated architecture work was
started.
