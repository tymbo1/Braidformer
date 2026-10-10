# BraidFormer

BraidFormer is an independent research project for experimentally testing a small recurrent/state-space-majority architecture with selective exact retrieval, sparse latent experts, and conditional compute. The project is deliberately falsification-first: the proposed architecture is a hypothesis, not an assumed improvement.

The first target is **BraidNano**, approximately 40M–60M parameters. The
initial Phase 0 bootstrap was repository and hardware setup only. The later,
explicitly approved A0 readiness step prepared TinyStories data, and the
seed-17 A0 pilot has now completed. No Qwen distillation is used.

## Current status

- Phase 0 scaffold completed.
- A0 conventional decoder-only Transformer control implemented and covered by
  CPU-only synthetic shape, loss, backward, and causality smoke tests.
- TinyStories data, tokenizer, deterministic splits, and packed token streams
  were prepared and independently verified.
- The bounded seed-17 A0 pilot completed 1,000 optimizer steps and preserved a
  checkpoint; see [`docs/A0_TRAINING_RESULT.md`](docs/A0_TRAINING_RESULT.md).
- The host GPU is an NVIDIA GeForce RTX 3060 Ti with working CUDA access. A1
  architecture implementation and CPU verification are complete; A1 training
  has not started.

## Repository guide

- [`AGENTS.md`](AGENTS.md): persistent research and workspace rules.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): hypothesis and planned components.
- [`docs/A0_BASELINE.md`](docs/A0_BASELINE.md): verified A0 accounting, test coverage, and proposed experiment protocol.
- [`docs/A0_READINESS.md`](docs/A0_READINESS.md): pre-training gate matrix and current hardware blocker.
- [`docs/A1_PROPOSAL.md`](docs/A1_PROPOSAL.md): bounded recurrent/state-space-majority A1 proposal and training blockers.
- [`docs/DATASET_TOKENIZER.md`](docs/DATASET_TOKENIZER.md): pinned TinyStories provenance, tokenizer, split, and packing procedure.
- [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md): controlled experimental ladder and measurements.
- [`docs/DECISIONS.md`](docs/DECISIONS.md): dated decisions and environment findings.
- [`configs/braidnano_phase0_planning.json`](configs/braidnano_phase0_planning.json): conservative planning-only configuration.
- [`configs/braidnano_a0_baseline_proposal.json`](configs/braidnano_a0_baseline_proposal.json): approved A0 pilot configuration; source remains `planned_only`.
- [`configs/braidnano_a0_data.json`](configs/braidnano_a0_data.json): approved data/tokenizer preparation specification.
- [`configs/braidnano_a0_integrity_policy.json`](configs/braidnano_a0_integrity_policy.json): bounded pilot safeguards.
- [`configs/braidnano_a1_recurrent_state_space_proposal.json`](configs/braidnano_a1_recurrent_state_space_proposal.json): planned-only A1 comparison configuration.
- [`src/braidformer/models/transformer.py`](src/braidformer/models/transformer.py): A0 control model.
- [`src/braidformer/models/recurrent.py`](src/braidformer/models/recurrent.py): A1 recurrent/state-space-majority model.
- [`scripts/verify_a0.py`](scripts/verify_a0.py): construction and parameter-accounting verification.
- [`scripts/verify_a1.py`](scripts/verify_a1.py): A1 construction and parameter-accounting verification; it does not train.
- [`scripts/hardware_report.py`](scripts/hardware_report.py): read-only hardware/environment audit.
- [`scripts/prepare_tinystories.py`](scripts/prepare_tinystories.py): deterministic data/tokenizer preparation; never trains.
- [`scripts/verify_tinystories.py`](scripts/verify_tinystories.py): local artifact, hash, split, and token-stream verification.
- [`scripts/train_a0.py`](scripts/train_a0.py): explicitly authorized bounded A0 pilot runner with validation and recovery.
- [`scripts/evaluate_a0_prompts.py`](scripts/evaluate_a0_prompts.py): fixed-prompt greedy checkpoint inspection; it does not train.

## Cheap checks

From the repository root:

```bash
python3 scripts/hardware_report.py
PYTHONPATH=src python3 scripts/verify_a0.py
PYTHONPATH=src python3 scripts/verify_a1.py
python3 -m pytest
```

These commands do not launch training.
