# BraidFormer A0/A1 solvability calibration result

Status: **complete**. This is the bounded seed-17 first-rung calibration for the synthetic recall ladder; no harder rung or architecture change was started.

Frozen protocol commit: `8a34cda33502da7a338203b1cd684c4a42af9269`.

## Final measurements

| Metric | A0 | A1 |
| --- | ---: | ---: |
| Parameters | 41,866,240 | 41,870,336 |
| Optimizer steps | 2,000 | 2,000 |
| Token positions | 256,000 | 256,000 |
| Final training target loss | 0.000435 | 0.000404 |
| Training greedy exact match | 1.0000 | 1.0000 |
| Final held-out target loss | 1.222888 | 1.509632 |
| Held-out greedy exact match | 0.8828 | 0.8516 |
| CUDA training seconds | 186.852 | 283.746 |
| Elapsed wall seconds | 201.242 | 306.325 |
| CUDA tokens/second | 1370.1 | 902.2 |
| Peak allocated VRAM | 0.808 GiB | 0.808 GiB |
| Peak reserved VRAM | 0.934 GiB | 0.936 GiB |

## Interpretation

Both unchanged 42M models passed the primary calibration observation: they reached 100% autoregressive one-token retrieval on the repeated training set. The earlier zero exact-match result on the harder synthetic battery therefore is not evidence that either model cannot learn any one-token association.

On the diagnostic held-out set, A0 reached `0.8828` and A1 `0.8516`. The paired A1 − A0 target-loss delta was `+0.286744` and the exact-match delta was `-0.0312`. Held-out keys and mappings are disjoint, but the earlier record value is present in each input, so this is evidence about transfer of the simple copy/retrieve operation, not unseen fact learning. A0 was better on this diagnostic at this seed and budget.

A1 used `1.519×` A0's CUDA time and achieved `0.659×` A0's average token throughput. This calibration was not a matched-compute comparison.

No A0-versus-A1 winner threshold was declared for this calibration. These results do not distinguish model capacity from state representation or optimization on the harder battery, and do not authorize scaling, architecture changes, A2, persistent memory, or SNAIL/SSRE integration.

## Selected validation checkpoints

The machine-readable aggregate preserves all 20 logged validation checkpoints. The table below samples the common optimizer-step checkpoints.

| Step | A0 held-out loss | A1 held-out loss | A1 − A0 | A0 exact | A1 exact |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 1.329138 | 1.835942 | +0.506804 | 0.8828 | 0.8047 |
| 500 | 1.207007 | 1.507076 | +0.300069 | 0.8828 | 0.8516 |
| 1,000 | 1.216534 | 1.505723 | +0.289189 | 0.8828 | 0.8516 |
| 1,500 | 1.221759 | 1.508913 | +0.287154 | 0.8828 | 0.8516 |
| 2,000 | 1.222888 | 1.509632 | +0.286744 | 0.8828 | 0.8516 |

## Integrity and scope

Both runs completed with finite losses and gradients, no CUDA/OOM failure, and SHA-256-verified final checkpoints containing optimizer and RNG state. Gradient checkpointing was configured in the protocol but remains unimplemented and disabled.

- A0 checkpoint: `4237ca73c0c3835559f1d816b02513e47f566551018323a6273515752928e0e0`
- A1 checkpoint: `473b03e6130daa942d43e81948c4661ece6df02a5bed9af8587e32a7b560dd72`
- Synthetic data hash: `c970d76186b767a11aee40f3e16d1c09541358da9ada34ce616acfac6d49a7bb`
- Raw logs, checkpoints, and sidecars remain in ignored `results/runs/` directories.
- No seeds 23/29, difficulty ladder, scaling, A2, persistent memory, WNF, or SNAIL/SSRE work was started.
