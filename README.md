# BraidFormer

BraidFormer is an independent research project for experimentally testing a small recurrent/state-space-majority architecture with selective exact retrieval, sparse latent experts, and conditional compute. The project is deliberately falsification-first: the proposed architecture is a hypothesis, not an assumed improvement.

The first target is **BraidNano**, approximately 40M–60M parameters. Phase 0 is repository and hardware bootstrap only. It does not train, download a dataset, download a checkpoint, or use Qwen distillation.

## Current status

- Phase 0 scaffold completed.
- A0 conventional decoder-only Transformer control implemented and covered by
  CPU-only synthetic shape, loss, backward, and causality smoke tests.
- No dataset, checkpoint, or training run has been started.
- Hardware report generated at [`results/hardware_report.json`](results/hardware_report.json).
- The audited machine exposes an NVIDIA GeForce RTX 3060 Ti GDDR6X through PCI metadata, but the NVIDIA driver currently fails `nvidia-smi` and PyTorch cannot see CUDA. See the report and [`docs/DECISIONS.md`](docs/DECISIONS.md).

## Repository guide

- [`AGENTS.md`](AGENTS.md): persistent research and workspace rules.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): hypothesis and planned components.
- [`docs/A0_BASELINE.md`](docs/A0_BASELINE.md): verified A0 accounting, test coverage, and proposed experiment protocol.
- [`docs/A0_READINESS.md`](docs/A0_READINESS.md): pre-training gate matrix and unresolved data/hardware requirements.
- [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md): controlled experimental ladder and measurements.
- [`docs/DECISIONS.md`](docs/DECISIONS.md): dated decisions and environment findings.
- [`configs/braidnano_phase0_planning.json`](configs/braidnano_phase0_planning.json): conservative planning-only configuration.
- [`configs/braidnano_a0_baseline_proposal.json`](configs/braidnano_a0_baseline_proposal.json): reviewed but not executed training proposal.
- [`src/braidformer/models/transformer.py`](src/braidformer/models/transformer.py): A0 control model.
- [`scripts/verify_a0.py`](scripts/verify_a0.py): construction and parameter-accounting verification.
- [`scripts/hardware_report.py`](scripts/hardware_report.py): read-only hardware/environment audit.

## Cheap checks

From the repository root:

```bash
python3 scripts/hardware_report.py
PYTHONPATH=src python3 scripts/verify_a0.py
python3 -m pytest
```

These commands do not launch training.
