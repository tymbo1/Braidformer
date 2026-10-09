# BraidFormer

BraidFormer is an independent research project for experimentally testing a small recurrent/state-space-majority architecture with selective exact retrieval, sparse latent experts, and conditional compute. The project is deliberately falsification-first: the proposed architecture is a hypothesis, not an assumed improvement.

The first target is **BraidNano**, approximately 40M–60M parameters. Phase 0 is repository and hardware bootstrap only. It does not train, download a dataset, download a checkpoint, or use Qwen distillation.

## Current status

- Phase 0 scaffold created.
- Hardware report generated at [`results/hardware_report.json`](results/hardware_report.json).
- No model implementation or training run has been started.
- The audited machine exposes an NVIDIA GeForce RTX 3060 Ti GDDR6X through PCI metadata, but the NVIDIA driver currently fails `nvidia-smi` and PyTorch cannot see CUDA. See the report and [`docs/DECISIONS.md`](docs/DECISIONS.md).

## Repository guide

- [`AGENTS.md`](AGENTS.md): persistent research and workspace rules.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): hypothesis and planned components.
- [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md): controlled experimental ladder and measurements.
- [`docs/DECISIONS.md`](docs/DECISIONS.md): dated decisions and environment findings.
- [`configs/braidnano_phase0_planning.json`](configs/braidnano_phase0_planning.json): conservative planning-only configuration.
- [`scripts/hardware_report.py`](scripts/hardware_report.py): read-only hardware/environment audit.

## Cheap checks

From the repository root:

```bash
python3 scripts/hardware_report.py
python3 -m pytest
```

These commands do not launch training.
