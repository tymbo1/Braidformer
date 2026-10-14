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
  architecture implementation, CPU verification, one-update CUDA smoke test,
  bounded ten-update scan stability check, and the authorized seed-17 pilot
  are complete; see [`docs/A1_TRAINING_RESULT.md`](docs/A1_TRAINING_RESULT.md).
  The measured result is an equal-token pilot comparison, not an equal-compute
  claim.

## Repository guide

- [`AGENTS.md`](AGENTS.md): persistent research and workspace rules.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): hypothesis and planned components.
- [`docs/A0_BASELINE.md`](docs/A0_BASELINE.md): verified A0 accounting, test coverage, and proposed experiment protocol.
- [`docs/A0_READINESS.md`](docs/A0_READINESS.md): pre-training gate matrix and current hardware blocker.
- [`docs/A1_PROPOSAL.md`](docs/A1_PROPOSAL.md): bounded recurrent/state-space-majority A1 proposal and training blockers.
- [`docs/A1_TRAINING_RESULT.md`](docs/A1_TRAINING_RESULT.md): measured A1 seed-17 pilot and explicit equal-token/equal-compute comparison.
- [`docs/A1_EQUAL_COMPUTE_PROPOSAL.md`](docs/A1_EQUAL_COMPUTE_PROPOSAL.md): bounded proposal separating equal-token, equal-wall-clock, and equal-GPU-compute tests.
- [`docs/A0_A1_EQUAL_CUDA_RESULT.md`](docs/A0_A1_EQUAL_CUDA_RESULT.md): authorized seed-17 equal-CUDA-time result and decision.
- [`docs/A0_A1_EQUAL_CUDA_REPLICATIONS.md`](docs/A0_A1_EQUAL_CUDA_REPLICATIONS.md): seeds 23/29 replications and three-seed aggregate decision.
- [`docs/A2_OPTIONS_ASSESSMENT.md`](docs/A2_OPTIONS_ASSESSMENT.md): bounded prior-art mapping and next-step options; no A2 implementation or training authorized.
- [`docs/A0_A1_LONGER_EQUAL_CUDA_PROTOCOL.md`](docs/A0_A1_LONGER_EQUAL_CUDA_PROTOCOL.md): frozen 4× seed-17 matched-CUDA-time protocol.
- [`docs/A0_A1_LONGER_EQUAL_CUDA_RESULT.md`](docs/A0_A1_LONGER_EQUAL_CUDA_RESULT.md): measured seed-17 longer matched-CUDA-time result.
- [`docs/A0_A1_LONGER_EQUAL_CUDA_REPLICATION_PROTOCOL.md`](docs/A0_A1_LONGER_EQUAL_CUDA_REPLICATION_PROTOCOL.md): frozen 4× seed-23/29 replication protocol.
- [`docs/A0_A1_LONGER_EQUAL_CUDA_REPLICATION_RESULT.md`](docs/A0_A1_LONGER_EQUAL_CUDA_REPLICATION_RESULT.md): measured 4× seeds 17/23/29 replication result.
- [`docs/A0_A1_EIGHT_EQUAL_CUDA_PROTOCOL.md`](docs/A0_A1_EIGHT_EQUAL_CUDA_PROTOCOL.md): frozen 8× matched-CUDA-time protocol for seeds 17/23/29.
- [`docs/A0_A1_EIGHT_EQUAL_CUDA_RESULT.md`](docs/A0_A1_EIGHT_EQUAL_CUDA_RESULT.md): measured 8× seeds 17/23/29 replication and curve/AUC analysis.
- [`docs/A0_A1_SCHEDULE_DISENTANGLEMENT_PROTOCOL.md`](docs/A0_A1_SCHEDULE_DISENTANGLEMENT_PROTOCOL.md): frozen schedule-control protocol; see the completed [`docs/A0_A1_SCHEDULE_DISENTANGLEMENT_RESULT.md`](docs/A0_A1_SCHEDULE_DISENTANGLEMENT_RESULT.md).
- [`docs/A1_SCAN_COMPILE_PREFLIGHT_PROTOCOL.md`](docs/A1_SCAN_COMPILE_PREFLIGHT_PROTOCOL.md) and [`docs/A1_SCAN_COMPILE_PREFLIGHT_RESULT.md`](docs/A1_SCAN_COMPILE_PREFLIGHT_RESULT.md): engineering-only A1 compiler preflight; runner integration remains pending stability validation.
- [`docs/A1_SCAN_COMPILE_STABILITY_PROTOCOL.md`](docs/A1_SCAN_COMPILE_STABILITY_PROTOCOL.md) and [`docs/A1_SCAN_COMPILE_STABILITY_RESULT.md`](docs/A1_SCAN_COMPILE_STABILITY_RESULT.md): ten-update eager-versus-compiled stability result; normal-runner adoption remains separately gated.
- [`docs/A0_A1_SOLVABILITY_VALUES128_PROTOCOL.md`](docs/A0_A1_SOLVABILITY_VALUES128_PROTOCOL.md): frozen protocol for the completed seed-17 128-value solvability-ladder rung.
- [`docs/A0_A1_SOLVABILITY_VALUES128_RESULT.md`](docs/A0_A1_SOLVABILITY_VALUES128_RESULT.md): measured seed-17 128-value solvability-ladder result.
- [`docs/A0_A1_SOLVABILITY_VALUES256_PROTOCOL.md`](docs/A0_A1_SOLVABILITY_VALUES256_PROTOCOL.md): frozen protocol for the completed seed-17 256-value solvability-ladder rung.
- [`docs/A0_A1_SOLVABILITY_VALUES256_RESULT.md`](docs/A0_A1_SOLVABILITY_VALUES256_RESULT.md): measured seed-17 256-value solvability-ladder result.
- [`docs/A0_A1_SOLVABILITY_VALUES1024_PROTOCOL.md`](docs/A0_A1_SOLVABILITY_VALUES1024_PROTOCOL.md): frozen protocol for the completed seed-17 1024-value solvability-ladder rung.
- [`docs/A0_A1_SOLVABILITY_VALUES1024_RESULT.md`](docs/A0_A1_SOLVABILITY_VALUES1024_RESULT.md): measured seed-17 1024-value solvability-ladder result.
- [`docs/A0_A1_SOLVABILITY_VALUES8000_PROTOCOL.md`](docs/A0_A1_SOLVABILITY_VALUES8000_PROTOCOL.md): frozen seed-17 8000-value solvability-ladder protocol.
- [`docs/A0_A1_SOLVABILITY_VALUES8000_RESULT.md`](docs/A0_A1_SOLVABILITY_VALUES8000_RESULT.md): measured seed-17 8000-value solvability-ladder result.
- [`docs/A0_A1_SOLVABILITY_ASSOCIATIONS2_VALUES8000_PROTOCOL.md`](docs/A0_A1_SOLVABILITY_ASSOCIATIONS2_VALUES8000_PROTOCOL.md): frozen seed-17 two-association protocol.
- [`docs/A0_A1_SOLVABILITY_ASSOCIATIONS2_VALUES8000_RESULT.md`](docs/A0_A1_SOLVABILITY_ASSOCIATIONS2_VALUES8000_RESULT.md): measured seed-17 two-association solvability result.
- [`docs/A0_A1_SOLVABILITY_ASSOCIATIONS4_VALUES8000_PROTOCOL.md`](docs/A0_A1_SOLVABILITY_ASSOCIATIONS4_VALUES8000_PROTOCOL.md): frozen seed-17 four-association protocol.
- [`docs/A0_A1_SOLVABILITY_ASSOCIATIONS4_VALUES8000_RESULT.md`](docs/A0_A1_SOLVABILITY_ASSOCIATIONS4_VALUES8000_RESULT.md): measured seed-17 four-association solvability result.
- [`docs/A0_A1_SOLVABILITY_ASSOCIATIONS8_VALUES8000_PROTOCOL.md`](docs/A0_A1_SOLVABILITY_ASSOCIATIONS8_VALUES8000_PROTOCOL.md): frozen seed-17 eight-association protocol.
- [`docs/A0_A1_SOLVABILITY_ASSOCIATIONS8_VALUES8000_RESULT.md`](docs/A0_A1_SOLVABILITY_ASSOCIATIONS8_VALUES8000_RESULT.md): measured seed-17 eight-association solvability result.
- [`docs/A0_A1_SOLVABILITY_DISTANCE4_VALUES8000_PROTOCOL.md`](docs/A0_A1_SOLVABILITY_DISTANCE4_VALUES8000_PROTOCOL.md): frozen seed-17 distance-4 retrieval protocol.
- [`docs/A0_A1_SOLVABILITY_DISTANCE16_VALUES8000_PROTOCOL.md`](docs/A0_A1_SOLVABILITY_DISTANCE16_VALUES8000_PROTOCOL.md): frozen seed-17 distance-16 retrieval protocol.
- [`docs/A0_A1_SOLVABILITY_DISTANCE64_VALUES8000_PROTOCOL.md`](docs/A0_A1_SOLVABILITY_DISTANCE64_VALUES8000_PROTOCOL.md): frozen seed-17 distance-64 retrieval protocol.
- [`docs/A0_A1_SOLVABILITY_DISTANCES_VALUES8000_RESULT.md`](docs/A0_A1_SOLVABILITY_DISTANCES_VALUES8000_RESULT.md): verified seed-17 distractor-distance curve and integrity summary.
- [`docs/A1_STATE_DECAY_FORENSICS_VALUES8000.md`](docs/A1_STATE_DECAY_FORENSICS_VALUES8000.md): read-only A1 time-scale, state-drift, and frozen-probe analysis.
- [`docs/TRAJECTORY_AWARE_TRAINING_PROPOSAL.md`](docs/TRAJECTORY_AWARE_TRAINING_PROPOSAL.md): deferred predictive training-policy experiment; no implementation or training authorized.
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
- [`scripts/train_a0.py`](scripts/train_a0.py): explicitly authorized bounded A0/A1 pilot runner with validation and recovery; model selection is config-driven.
- [`scripts/evaluate_a0_prompts.py`](scripts/evaluate_a0_prompts.py): fixed-prompt greedy A0/A1 checkpoint inspection; it does not train.
- [`scripts/smoke_test_a1.py`](scripts/smoke_test_a1.py): one-update A1 CUDA readiness test; it writes no checkpoint and is not a pilot.
- [`scripts/benchmark_a1_recurrence.py`](scripts/benchmark_a1_recurrence.py): matched sequential-versus-associative-scan A1 benchmark; it does not train a pilot.
- [`scripts/compare_a1_scan_stability.py`](scripts/compare_a1_scan_stability.py): bounded ten-update sequential-versus-scan numerical-stability check; it does not train a pilot.
- [`scripts/measure_a0_a1_cuda_timing.py`](scripts/measure_a0_a1_cuda_timing.py): bounded CUDA-event calibration for the proposed equal-compute protocol; it discards in-memory updates and does not train a pilot.
- [`results/a0_a1_cuda_timing_calibration.json`](results/a0_a1_cuda_timing_calibration.json): measured CUDA timing, analytical FLOP labels, and the proposal-only 1,000/502-step budget.
- [`results/a0_a1_equal_cuda_comparison.json`](results/a0_a1_equal_cuda_comparison.json): concise measured equal-CUDA-time curves, budgets, and decision.
- [`results/a0_a1_eight_equal_cuda_replications.json`](results/a0_a1_eight_equal_cuda_replications.json): machine-readable 1×/4×/8× curve, AUC, and replication measurements.
- [`results/a0_a1_equal_cuda_replications.json`](results/a0_a1_equal_cuda_replications.json): per-seed replication metrics and aggregate variation.
- [`results/a0_a1_longer_equal_cuda_replications.json`](results/a0_a1_longer_equal_cuda_replications.json): per-seed 4× replication metrics, resource records, and checkpoint hashes.
- [`scripts/analyze_synthetic_solvability_distances_values8000.py`](scripts/analyze_synthetic_solvability_distances_values8000.py): read-only verifier and aggregator for the completed distractor-distance axis.
- [`scripts/analyze_a1_state_decay_forensics.py`](scripts/analyze_a1_state_decay_forensics.py): read-only A1 decay and recurrent-state forensic analyzer.

## Cheap checks

From the repository root:

```bash
python3 scripts/hardware_report.py
PYTHONPATH=src python3 scripts/verify_a0.py
PYTHONPATH=src python3 scripts/verify_a1.py
python3 -m pytest
```

These commands do not launch training.
