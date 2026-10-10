# Bounded A0 training-integrity review

Date: 2026-10-10 (Australia/Brisbane)

Scope: the proposed 1,000-step A0 pilot only. This review does not start a
model run, alter the Transformer, change the approved data/tokenizer, or expand
the token budget.

## Findings

| Classification | Finding | Disposition |
| --- | --- | --- |
| BLOCKER | CUDA is not usable: `nvidia-smi` returns code 9, PyTorch reports zero CUDA devices, and the audited environment has no `/dev/nvidia0` or `/dev/nvidiactl` character devices. | No training or GPU smoke test is permitted until the existing installation is repaired and re-audited. No driver changes were made. |
| PASS | Dataset provenance and licensing metadata are pinned to TinyStories revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64` and recorded as CDLA-Sharing-1.0 with attribution/redistribution conditions. | Verified by the local artifact report and source metadata record. |
| PASS | Duplicate and leakage control is deterministic: first-seen normalized text hashes are unique across final train/validation/test; source train remains train and source validation is hash-partitioned. | Verified by `scripts/verify_tinystories.py`. |
| PASS | Tokenizer and packed streams are reproducible and hashed. The tokenizer is byte-level BPE with vocabulary 32,000 and is trained only on final train documents. | Verified source, tokenizer, split-manifest, and stream hashes. |
| RECOMMENDED | The integrity monitor records step-level losses, gradient norms, learning rate, token count, throughput, VRAM, and provenance metadata; it stops on non-finite values, loss ceiling violations, skipped steps, and counter regressions. | Implemented and unit-tested in `braidformer.training.integrity`; a future training entry point must wire it before authorization. |
| RECOMMENDED | Checkpoints need atomic writes and recovery metadata. | Implemented and unit-tested with SHA-256 sidecars containing step, tokens, seed, config hash, and data-manifest hash. |
| DEFERRED | Quality-at-equal-cost, cost-at-equal-quality, task metrics, uncertainty, and A1 comparisons. | These require empirical runs and remain outside preparation. |
| DEFERRED | Distributed training, hyperparameter search, security review, and long-run fault injection. | Not proportionate to the local 1,000-step pilot. |

## Evidence

- Tests: 20/20 passed.
- Reported source line coverage: 461/461 (100%) across `src/braidformer` under
  the documented standard-library trace command; this is not branch coverage.
- A0 parameters: 41,866,240 unique trainable parameters.
- Prepared train/validation/test documents: 1,799,248 / 7,567 / 7,822.
- Prepared token counts: 391,534,494 / 1,538,229 / 1,588,146.
- Data verification report: [`results/a0_data_verification.json`](../results/a0_data_verification.json).
- Integrity policy: [`configs/braidnano_a0_integrity_policy.json`](../configs/braidnano_a0_integrity_policy.json).

The only current BLOCKER is hardware. This record is not training
authorization. After the hardware gate passes, the final configuration and
readiness report must be presented for explicit authorization, then work stops
until that authorization is given.
