# A0 reproducible training readiness

Date of audit: 2026-10-10 (Australia/Brisbane)

Status: **not ready to train**. The A0 control is verified as an
implementation, and a training configuration has been presented as a
`planned_only` proposal. No dataset has been read, no tokenizer has been
selected, no checkpoint has been loaded, and no training step has run.

## Gate matrix

| Gate | Status | Evidence or remaining work |
| --- | --- | --- |
| Architecture construction | PASS | A0-only model guard, validated dimensions, tied-embedding accounting, and 41,866,240 unique trainable parameters. |
| Causal attention | PASS | Synthetic test confirms that changing future tokens does not change earlier logits. |
| Next-token loss | PASS | Synthetic test recomputes the shifted cross entropy and confirms that the first label is ignored. |
| Model-level leakage control | PASS | The model uses causal scaled dot-product attention and predicts label `t + 1` from positions through `t`. |
| Split/data leakage control | PENDING | No dataset or preprocessing pipeline exists yet. The eventual manifest must pin source revision, tokenizer, preprocessing, packing, and deterministic split assignments. |
| Automated tests | PASS | 14 tests pass; source line coverage remains 272/272 (100%) for `src/braidformer` under the documented standard-library trace command. This is not branch coverage. |
| Training proposal | PASS — proposal only | The JSON proposal fixes the optimizer, schedule, context, batch arithmetic, token budgets, evaluation cadence, and seeds, but retains `PENDING_APPROVAL` for data and tokenizer. |
| Hardware feasibility | BLOCKED | PCI metadata identifies an RTX 3060 Ti, but `nvidia-smi` fails to communicate and PyTorch reports no CUDA device. VRAM and GPU throughput are therefore unmeasured. |
| Empirical model performance | NOT MEASURED | No loss curve, validation result, tokens/second, memory profile, or task benchmark exists. No performance claim is made. |

## Required approval gate before training

Training remains paused until all of the following are recorded in the final
experiment configuration:

1. A legally usable dataset, immutable source revision, and deterministic
   train/validation/test split procedure.
2. A fixed tokenizer, vocabulary, tokenizer revision or checksum, special
   token policy, and preprocessing/packing rules.
3. A passing device and memory smoke test for the proposed context and
   effective batch, or an explicit decision to run under a documented CPU
   fallback.
4. An approved final configuration hash and seed list. The current proposal
   uses seeds 17, 23, and 29, but is not executable while its data/tokenizer
   field is unresolved.

The proposed experiment has already been presented in
[`configs/braidnano_a0_baseline_proposal.json`](../configs/braidnano_a0_baseline_proposal.json).
This readiness record does not authorize training; it identifies the evidence
still required before that proposal can become an executed run.

## Recheck commands

From the repository root:

```bash
PYTHONPATH=src python3 scripts/verify_a0.py
python3 -m pytest -q
python3 -m compileall -q src tests scripts
python3 scripts/hardware_report.py
```

The verification and test commands are construction/synthetic checks only.
They do not download data, access a checkpoint, or start training.
