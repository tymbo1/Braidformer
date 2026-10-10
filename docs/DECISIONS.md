# Decisions and environment findings

## 2026-10-10 — D000: isolate the research workspace

All BraidFormer source, environments, datasets, checkpoints, logs, and results stay under `/home/timbushnell/braidformer_lab`. The unrelated `projectbrain_dev` repository is out of scope and must not be inspected or used as a dependency.

## 2026-10-10 — D001: use falsification-first comparisons

The project will not assume BraidFormer-H1 is better. The A0 control and later stages will be compared with matched data, tokenizer, seed policy, optimizer, token budget, batch, context, evaluation, and approximate FLOPs wherever applicable. Claims must stay within measured evidence.

## 2026-10-10 — D002: start with BraidNano

The initial target is approximately 40M–60M parameters. This keeps architecture falsification affordable on the local machine and avoids premature scaling to 1B or 14B parameters.

## 2026-10-10 — D003: defer Qwen distillation

Qwen is excluded from Phase 0. It may later be used as a clearly labeled teacher or baseline only after the core architecture demonstrates basic viability.

## 2026-10-10 — D004: avoid heavyweight Phase 0 setup

Phase 0 uses the already available Python, PyTorch, and pytest installation. It adds no large dependencies, datasets, model checkpoints, or training run.

## 2026-10-10 — D005: audited machine differs from expectation

Read-only PCI metadata identifies an NVIDIA GeForce RTX 3060 Ti GDDR6X (GA104, ASUS subsystem `1043:88ae`, PCI device `10de:24c9`), not an RTX 3070. The loaded kernel driver reports version 580.126.18. `nvidia-smi` fails to communicate with the driver, and PyTorch 2.12.0+cu130 reports `torch.cuda.is_available() == False` with zero visible CUDA devices. The GPU SKU is commonly an 8-GiB device, but VRAM was not treated as measured because the runtime query failed.

## 2026-10-10 — D006: use JSON configs in the bootstrap

The bootstrap uses standard-library JSON configuration loading and validation so Phase 0 does not require a YAML dependency. Later experiment tooling may add a dependency only when its need is recorded and tested.

## 2026-10-10 — D007: implement the A0 control before architectural variants

The next implementation slice is a conventional decoder-only Transformer using the existing BraidNano dimensions. Its model type, dropout, normalization epsilon, bias policy, and tied-embedding policy are explicit configuration fields. The model rejects non-A0 stages, and validation is limited to synthetic CPU smoke tests; no dataset, checkpoint, or training run is authorized by this change.

## 2026-10-10 — D008: keep training gated on data and hardware evidence

The A0 control now has explicit synthetic checks for causal attention and exact next-token loss shifting. Training remains gated: a dataset/tokenizer and deterministic split manifest must be approved, and the device/memory smoke test must pass. The audited machine currently has no usable CUDA device, so no training-time performance or model-quality claim is permitted.
