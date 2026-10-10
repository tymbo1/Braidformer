# A0 reproducible training readiness

Date of audit: 2026-10-10 (Australia/Brisbane)

Status: **not ready to train**. The A0 control and TinyStories preparation are
verified, but the CUDA hardware gate fails. A training configuration remains a
`planned_only` proposal. No checkpoint has been loaded and no training step has
run.

## Gate matrix

| Gate | Status | Evidence or remaining work |
| --- | --- | --- |
| Architecture construction | PASS | A0-only model guard, validated dimensions, tied-embedding accounting, and 41,866,240 unique trainable parameters. |
| Causal attention | PASS | Synthetic test confirms that changing future tokens does not change earlier logits. |
| Next-token loss | PASS | Synthetic test recomputes the shifted cross entropy and confirms that the first label is ignored. |
| Model-level leakage control | PASS | The model uses causal scaled dot-product attention and predicts label `t + 1` from positions through `t`. |
| Dataset provenance and split integrity | PASS | TinyStories is pinned to revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`; source, tokenizer, split manifest, and token-stream hashes pass `results/a0_data_verification.json`. No duplicate text hashes cross final splits. |
| Tokenizer and packing | PASS | 32,000-token byte-level BPE trained on final train only; 512-token non-overlapping streams are verified. |
| Automated tests | PASS | 20 tests pass; source line coverage is 461/461 (100%) for `src/braidformer` under the documented standard-library trace command. This is not branch coverage. |
| Training integrity safeguards | RECOMMENDED | The bounded monitor records losses, gradient norms, token counts, throughput, VRAM, hashes, and fail-fast events; atomic checkpoint sidecars are tested. A future runner must wire it before authorization. |
| Training proposal | PASS — proposal only | The JSON proposal fixes data, tokenizer, optimizer, schedule, context, batch arithmetic, token budgets, evaluation cadence, and seeds, but remains `planned_only: true`. |
| Hardware feasibility | BLOCKER | PCI metadata identifies an RTX 3060 Ti and the kernel module is loaded, but `nvidia-smi` fails, `/dev/nvidia0`/`nvidiactl` are absent, and PyTorch reports no CUDA device. VRAM and GPU throughput are unmeasured. |
| Empirical model performance | NOT MEASURED | No loss curve, validation result, tokens/second, memory profile, or task benchmark exists. No performance claim is made. |

## Required approval gate before training

Training remains paused until all of the following are recorded in the final
experiment configuration:

1. A passing device and memory smoke test for the proposed context and
   effective batch, or an explicit decision to run under a documented CPU
   fallback.
2. An approved final configuration hash and seed list. The current proposal
   uses seeds 17, 23, and 29 and is still not executable while the hardware
   gate fails.

The proposed experiment has already been presented in
[`configs/braidnano_a0_baseline_proposal.json`](../configs/braidnano_a0_baseline_proposal.json).
This readiness record does not authorize training; it identifies the hardware
evidence still required before that proposal can become an executed run.

## Recheck commands

From the repository root:

```bash
PYTHONPATH=src python3 scripts/verify_a0.py
PYTHONPATH=src python3 scripts/verify_tinystories.py
python3 -m pytest -q
python3 -m compileall -q src tests scripts
python3 scripts/hardware_report.py
```

The verification and test commands are construction/synthetic checks only.
They do not download data, access a checkpoint, or start training.
