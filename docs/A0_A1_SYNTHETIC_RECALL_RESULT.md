# A0/A1 synthetic recall-battery result

Status: complete. This report covers only the six fresh runs frozen in protocol commit `a0f2a95a65c696053d9b2d98b4f6106d421c0daa`.

The experiment used unchanged 41.87M-parameter A0/A1 implementations, fresh seeds 17, 23, and 29, the deterministic synthetic generator v1, 1,000 optimizer steps, and 8,192,000 token positions per arm. It is an equal-token/equal-update comparison; CUDA time was measured, not matched.

## Primary autoregressive result

Greedy generation was evaluated without exposing answer tokens. Both architectures produced zero exact whole-answer matches on every task and seed. Mean answer-token accuracy was 0.000109 for A0 and 0.000000 for A1 (macro-average over nine seed/task pairs). The primary exact-recall test therefore does not show a winner at this training budget.

| Task | A0 greedy token accuracy | A1 greedy token accuracy | A0 exact match | A1 exact match |
| --- | ---: | ---: | ---: | ---: |
| copy | 0.000326 | 0.000000 | 0.000000 | 0.000000 |
| associative | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| selective | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

## Secondary target-loss result

Target-only cross-entropy is secondary because it does not establish successful exact recall. Lower is better.

| Task | A0 test loss mean | A1 test loss mean | A1 − A0 | A1 lower-loss seeds |
| --- | ---: | ---: | ---: | ---: |
| copy | 9.241826 | 9.722888 | 0.481061 | 0/3 |
| associative | 9.386119 | 9.664218 | 0.278099 | 0/3 |
| selective | 9.396965 | 9.671885 | 0.274920 | 0/3 |

Across all nine seed/task pairs, A0 mean target loss was `9.341637` and A1 was `9.686330`. The paired mean delta was `0.344694` with sample SD `0.129459`; A1 had lower target loss in 0/9 pairs.

| Seed | A0 mean task loss | A1 mean task loss | A1 − A0 |
| ---: | ---: | ---: | ---: |
| 17 | 9.338866 | 9.656535 | 0.317669 |
| 23 | 9.324136 | 9.687004 | 0.362868 |
| 29 | 9.361908 | 9.715452 | 0.353544 |

## Resource measurements

| Architecture | Mean CUDA training s | Mean elapsed wall s | Mean CUDA tok/s | Peak allocated VRAM | Peak reserved VRAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| A0 | 195.319 | 227.427 | 41941.9 | 2.180 GiB | 3.191 GiB |
| A1 | 385.549 | 431.717 | 21249.0 | 2.180 GiB | 3.508 GiB |

A1 used `1.974×` the mean A0 CUDA training time for the same token/update budget. This is not a matched-compute, matched-FLOP, or energy comparison.

## Validation and integrity

Validation target loss was recorded every 100 steps and at step 1,000 for each task. The complete validation curves and per-run records are in the machine-readable aggregate. All six runs completed with finite values; all six final checkpoints passed the existing SHA-256 sidecar verification. Gradient checkpointing remains unimplemented and was not enabled.

## Interpretation

At this 1,000-step budget, the synthetic battery does not support an A1 advantage claim. The primary autoregressive exact-recall metric is zero for both models, while the secondary target loss favors A0 in all nine seed/task pairs. This result is specific to the declared generator, context, budget, optimizer, and 42M-parameter implementations; it does not overturn or generalize the prior TinyStories results. No further training, architecture change, long-context, second-corpus, A2, memory, or WNF work was started.

Raw logs, task metrics, checkpoints, and checkpoint sidecars remain in ignored `results/runs/` directories and are not tracked by Git.
