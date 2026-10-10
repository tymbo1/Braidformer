# A0 reproducible training readiness

Date of audit: 2026-10-10 (Australia/Brisbane)

Status: **A0 seed-17 pilot completed**. The A0 control, TinyStories
preparation, host CUDA path, bounded training runner, and final checkpoint are
verified. Replication seeds 23 and 29 and A1 remain unstarted.

## Gate matrix

| Gate | Status | Evidence or remaining work |
| --- | --- | --- |
| Architecture construction | PASS | A0-only model guard, validated dimensions, tied-embedding accounting, and 41,866,240 unique trainable parameters. |
| Causal attention | PASS | Synthetic test confirms that changing future tokens does not change earlier logits. |
| Next-token loss | PASS | Synthetic test recomputes the shifted cross entropy and confirms that the first label is ignored. |
| Model-level leakage control | PASS | The model uses causal scaled dot-product attention and predicts label `t + 1` from positions through `t`. |
| Dataset provenance and split integrity | PASS | TinyStories is pinned to revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`; source, tokenizer, split manifest, and token-stream hashes pass `results/a0_data_verification.json`. No duplicate text hashes cross final splits. |
| Tokenizer and packing | PASS | 32,000-token byte-level BPE trained on final train only; 512-token non-overlapping streams are verified. |
| Automated tests | PASS | 23 tests pass. The earlier 461/461 source-line figure records pre-runner A0 verification and is not a coverage claim for the training runner. |
| Training integrity safeguards | PASS | The bounded runner records losses, gradient norms, token counts, throughput, VRAM, hashes, validation metrics, and fail-fast events; atomic checkpoint sidecars and resume/interruption paths are implemented. |
| Training proposal | EXECUTED — seed 17 | The approved proposal was executed once at 1,000 optimizer steps. Replication seeds 23 and 29 remain unstarted. |
| Hardware feasibility | PASS | Host-level CUDA smoke test and the full pilot passed on the RTX 3060 Ti. |
| Empirical pilot measurements | RECORDED | The pilot reached 8,192,000 tokens; final train loss was 3.6161 and final validation loss was 3.7512. These are TinyStories pilot measurements, not broad capability claims. |

## Completed training gate

The following evidence was recorded before the pilot:

1. A passing host-level device and memory smoke test for the proposed context
   and effective batch.
2. An approved configuration hash, seed 17, pinned data/tokenizer hashes, and
   the bounded runner capability check.

The proposed experiment has already been presented in
[`configs/braidnano_a0_baseline_proposal.json`](../configs/braidnano_a0_baseline_proposal.json).
The executed result is recorded in
[`docs/A0_TRAINING_RESULT.md`](A0_TRAINING_RESULT.md) and
[`results/a0_seed17_pilot_summary.json`](../results/a0_seed17_pilot_summary.json).

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
