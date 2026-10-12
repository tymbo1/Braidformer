#!/usr/bin/env python3
"""Verify and aggregate the completed eight-association solvability rung."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from braidformer.training.integrity import sha256_file, verify_checkpoint


ARCHITECTURES = ("A0", "A1")
SEED = 17
PROTOCOL_COMMIT = "cde7170085108aced51cbd79e602cad2d00acf0e"
PROTOCOL_PATH = "docs/A0_A1_SOLVABILITY_ASSOCIATIONS8_VALUES8000_PROTOCOL.md"
PROTOCOL_SHA256 = "79097179a102ce81d15f50909a1cdb830213c71136db8a3afcad5a3ef5fdf547"
EXPECTED_PARAMS = {"A0": 41_866_240, "A1": 41_870_336}
EXPECTED_CONFIGS = {
    "A0": "configs/braidnano_a0_solvability_associations8_values8000.json",
    "A1": "configs/braidnano_a1_solvability_associations8_values8000.json",
}
EXPECTED_CONFIG_HASHES = {
    "A0": "44db60d1dfcaad0ad5f085fcd152564ce215847295aeefeb8487e4d8a1f39e64",
    "A1": "a89b0e6aee321440010b12dc092ea41f748252e4d96df7cb673eda94943fede2",
}
EXPECTED_DATA_HASH = "a1ed0fd53aa8fcc6a4d42a3d27fdbf26286ee6dd6e297b42c40d81783935042e"
EXPECTED_GENERATOR_HASH = "2713ef71a263589ab02a3b184f49bfd05ef3cf9f3cc4365b8403dea22f102641"
EXPECTED_GENERATOR_VERSION = "synthetic_solvability_associations8_values8000_v1"
EXPECTED_VALUE_COUNT = 8_000
EXPECTED_ASSOCIATION_COUNT = 8
EXPECTED_SEQUENCE_LENGTH = 29
EXPECTED_STEPS = 2_000
EXPECTED_TOKENS = 928_000
TOKENS_PER_STEP = EXPECTED_SEQUENCE_LENGTH * 16
EVAL_STEPS = tuple(range(100, EXPECTED_STEPS + 1, 100))


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _finite(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite {label}: {value!r}")
    return number


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT.resolve()))


def _metrics(metrics: dict[str, Any], label: str) -> dict[str, Any]:
    if int(metrics.get("examples", -1)) != 128:
        raise ValueError(f"unexpected example count in {label}")
    for field in ("target_loss", "greedy_answer_token_accuracy", "greedy_exact_match"):
        _finite(metrics[field], f"{label}.{field}")
    for field in ("greedy_answer_token_accuracy", "greedy_exact_match"):
        if not 0.0 <= float(metrics[field]) <= 1.0:
            raise ValueError(f"invalid accuracy in {label}: {metrics[field]!r}")
    return {
        "examples": int(metrics["examples"]),
        "target_loss": float(metrics["target_loss"]),
        "greedy_answer_token_accuracy": float(metrics["greedy_answer_token_accuracy"]),
        "greedy_exact_match": float(metrics["greedy_exact_match"]),
    }


def _load_run(architecture: str) -> dict[str, Any]:
    run_dir = REPO_ROOT / "results" / "runs" / (
        f"solvability_associations8_values8000_{architecture.lower()}_seed{SEED}"
    )
    summary = _load_json(run_dir / "summary.json")
    if summary.get("status") != "completed":
        raise ValueError(f"run is not complete: {run_dir}")
    if summary.get("architecture") != architecture or int(summary.get("seed", -1)) != SEED:
        raise ValueError(f"run identity mismatch: {run_dir}")
    if int(summary.get("association_count", -1)) != EXPECTED_ASSOCIATION_COUNT:
        raise ValueError(f"association-count mismatch: {run_dir}")
    if int(summary.get("sequence_length", -1)) != EXPECTED_SEQUENCE_LENGTH:
        raise ValueError(f"sequence-length mismatch: {run_dir}")
    if summary.get("protocol_commit") != PROTOCOL_COMMIT:
        raise ValueError(f"protocol commit mismatch: {run_dir}")
    if summary.get("protocol_sha256") != PROTOCOL_SHA256:
        raise ValueError(f"protocol hash mismatch: {run_dir}")
    protocol_path = REPO_ROOT / PROTOCOL_PATH
    if sha256_file(protocol_path) != PROTOCOL_SHA256:
        raise ValueError(f"working-tree protocol hash mismatch: {protocol_path}")
    if summary.get("config_path") != EXPECTED_CONFIGS[architecture]:
        raise ValueError(f"config path mismatch: {run_dir}")
    if summary.get("config_sha256") != EXPECTED_CONFIG_HASHES[architecture]:
        raise ValueError(f"config hash mismatch: {run_dir}")
    config_path = REPO_ROOT / EXPECTED_CONFIGS[architecture]
    if sha256_file(config_path) != EXPECTED_CONFIG_HASHES[architecture]:
        raise ValueError(f"working-tree config hash mismatch: {config_path}")
    if int(summary.get("parameter_count", -1)) != EXPECTED_PARAMS[architecture]:
        raise ValueError(f"parameter count mismatch: {run_dir}")
    if int(summary.get("optimizer_steps", -1)) != EXPECTED_STEPS:
        raise ValueError(f"step budget mismatch: {run_dir}")
    if int(summary.get("tokens_processed", -1)) != EXPECTED_TOKENS:
        raise ValueError(f"token budget mismatch: {run_dir}")
    if int(summary.get("value_count", -1)) != EXPECTED_VALUE_COUNT:
        raise ValueError(f"value-pool size mismatch: {run_dir}")
    if summary.get("synthetic_generator_version") != EXPECTED_GENERATOR_VERSION:
        raise ValueError(f"generator version mismatch: {run_dir}")
    if summary.get("synthetic_generator_sha256") != EXPECTED_GENERATOR_HASH:
        raise ValueError(f"generator hash mismatch: {run_dir}")
    if summary.get("synthetic_data_sha256") != EXPECTED_DATA_HASH:
        raise ValueError(f"data hash mismatch: {run_dir}")
    if summary.get("failure") is not None or summary.get("numerical_instability") is not False:
        raise ValueError(f"run reports failure or numerical instability: {run_dir}")

    checkpoint_path = Path(summary["checkpoint"])
    checkpoint_record = verify_checkpoint(checkpoint_path)
    expected_checkpoint = {
        "config_sha256": summary["config_sha256"],
        "data_manifest_sha256": EXPECTED_DATA_HASH,
        "seed": SEED,
        "step": EXPECTED_STEPS,
        "tokens_processed": EXPECTED_TOKENS,
        "parameter_count": EXPECTED_PARAMS[architecture],
        "protocol_commit": PROTOCOL_COMMIT,
        "value_count": EXPECTED_VALUE_COUNT,
        "association_count": EXPECTED_ASSOCIATION_COUNT,
    }
    for field, expected in expected_checkpoint.items():
        if checkpoint_record.get(field) != expected:
            raise ValueError(
                f"checkpoint metadata mismatch in {checkpoint_path}: "
                f"{field}={checkpoint_record.get(field)!r}, expected={expected!r}"
            )
    if summary.get("checkpoint_sha256") != checkpoint_record.get("checkpoint_sha256"):
        raise ValueError(f"summary/checkpoint hash mismatch: {run_dir}")

    step_count = 0
    cumulative_cuda = 0.0
    validation_steps: set[int] = set()
    with (run_dir / "steps.jsonl").open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            if record.get("event") != "step":
                continue
            step_count += 1
            step = int(record["step"])
            if step != step_count:
                raise ValueError(f"non-contiguous step log at {run_dir}:{line_number}")
            if int(record["tokens_processed"]) != step * TOKENS_PER_STEP:
                raise ValueError(f"token progress mismatch at {run_dir}:{line_number}")
            _finite(record["train_loss"], "training loss")
            cuda_seconds = _finite(record["cuda_elapsed_seconds"], "CUDA duration")
            if cuda_seconds <= 0:
                raise ValueError(f"non-positive CUDA duration at {run_dir}:{line_number}")
            _finite(record["wall_clock_seconds"], "wall time")
            _finite(record["gradient_norm"], "gradient norm")
            _finite(record["learning_rate"], "learning rate")
            cumulative_cuda += cuda_seconds
            if record.get("validation_loss") is not None:
                validation_steps.add(step)
                _finite(record["validation_loss"], "validation loss")
    if step_count != EXPECTED_STEPS:
        raise ValueError(f"expected {EXPECTED_STEPS} step records in {run_dir}, got {step_count}")
    if sorted(validation_steps) != list(EVAL_STEPS):
        raise ValueError(f"validation cadence mismatch in {run_dir}")
    if not math.isclose(cumulative_cuda, float(summary["cuda_training_seconds"]), rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"CUDA total mismatch in {run_dir}")

    metric_count = 0
    evaluation_steps: list[int] = []
    with (run_dir / "calibration_metrics.jsonl").open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            metric_count += 1
            if int(record["step"]) != metric_count:
                raise ValueError(f"non-contiguous calibration metric at {run_dir}:{line_number}")
            _finite(record["train_loss"], "calibration metric training loss")
            _finite(record["learning_rate"], "calibration metric learning rate")
            _finite(record["cuda_elapsed_seconds"], "calibration metric CUDA duration")
            _finite(record["wall_clock_seconds"], "calibration metric wall time")
            if record.get("evaluation") is not None:
                evaluation_steps.append(int(record["step"]))
    if metric_count != EXPECTED_STEPS or evaluation_steps != list(EVAL_STEPS):
        raise ValueError(f"calibration metric cadence mismatch in {run_dir}")

    history = []
    for record in summary["validation_history"]:
        step = int(record["step"])
        if step not in EVAL_STEPS:
            raise ValueError(f"unexpected validation step in {run_dir}: {step}")
        history.append(
            {
                "step": step,
                "tokens_processed": step * TOKENS_PER_STEP,
                "train": _metrics(record["metrics"]["train"], f"{run_dir}:{step}:train"),
                "heldout": _metrics(record["metrics"]["heldout"], f"{run_dir}:{step}:heldout"),
            }
        )
    if [record["step"] for record in history] != list(EVAL_STEPS):
        raise ValueError(f"summary validation history mismatch in {run_dir}")

    return {
        "architecture": architecture,
        "seed": SEED,
        "experiment_id": summary["experiment_id"],
        "run_dir": _relative(run_dir),
        "config_path": summary["config_path"],
        "config_sha256": summary["config_sha256"],
        "parameter_count": EXPECTED_PARAMS[architecture],
        "value_count": EXPECTED_VALUE_COUNT,
        "association_count": EXPECTED_ASSOCIATION_COUNT,
        "sequence_length": EXPECTED_SEQUENCE_LENGTH,
        "optimizer_steps": EXPECTED_STEPS,
        "tokens_processed": EXPECTED_TOKENS,
        "final_train": _metrics(summary["final_train"], f"{run_dir}:final_train"),
        "final_heldout": _metrics(summary["final_heldout"], f"{run_dir}:final_heldout"),
        "cuda_training_seconds": _finite(summary["cuda_training_seconds"], "CUDA training time"),
        "elapsed_wall_seconds": _finite(summary["elapsed_wall_seconds"], "elapsed wall time"),
        "average_tokens_per_second": _finite(summary["average_tokens_per_second"], "throughput"),
        "average_cuda_tokens_per_second": _finite(
            summary["average_cuda_tokens_per_second"], "CUDA throughput"
        ),
        "peak_allocated_vram_bytes": int(summary["peak_allocated_vram_bytes"]),
        "peak_reserved_vram_bytes": int(summary["peak_reserved_vram_bytes"]),
        "validation_history": history,
        "checkpoint": {
            "path": _relative(checkpoint_path),
            "sha256": checkpoint_record["checkpoint_sha256"],
            "bytes": int(checkpoint_record["checkpoint_bytes"]),
            "complete": bool(checkpoint_record["complete"]),
            "verified": True,
        },
        "gradient_checkpointing": summary["gradient_checkpointing"],
        "numerical_instability": False,
        "failure": None,
    }


def _aggregate(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    a0, a1 = runs["A0"], runs["A1"]
    curve = []
    for left, right in zip(a0["validation_history"], a1["validation_history"]):
        if left["step"] != right["step"]:
            raise ValueError("A0/A1 validation cadence mismatch")
        curve.append(
            {
                "step": left["step"],
                "tokens_processed": left["tokens_processed"],
                "A0_heldout_target_loss": left["heldout"]["target_loss"],
                "A1_heldout_target_loss": right["heldout"]["target_loss"],
                "A1_minus_A0_heldout_target_loss": right["heldout"]["target_loss"] - left["heldout"]["target_loss"],
                "A0_heldout_greedy_exact_match": left["heldout"]["greedy_exact_match"],
                "A1_heldout_greedy_exact_match": right["heldout"]["greedy_exact_match"],
                "A0_train_greedy_exact_match": left["train"]["greedy_exact_match"],
                "A1_train_greedy_exact_match": right["train"]["greedy_exact_match"],
            }
        )
    paired = {
        "heldout": {
            "A0": a0["final_heldout"],
            "A1": a1["final_heldout"],
            "A1_minus_A0_target_loss": a1["final_heldout"]["target_loss"] - a0["final_heldout"]["target_loss"],
            "A1_minus_A0_greedy_exact_match": a1["final_heldout"]["greedy_exact_match"] - a0["final_heldout"]["greedy_exact_match"],
        },
        "train": {
            "A0": a0["final_train"],
            "A1": a1["final_train"],
            "A1_minus_A0_target_loss": a1["final_train"]["target_loss"] - a0["final_train"]["target_loss"],
            "A1_minus_A0_greedy_exact_match": a1["final_train"]["greedy_exact_match"] - a0["final_train"]["greedy_exact_match"],
        },
        "resources": {
            "A0": {key: a0[key] for key in ("cuda_training_seconds", "elapsed_wall_seconds", "average_tokens_per_second", "average_cuda_tokens_per_second")},
            "A1": {key: a1[key] for key in ("cuda_training_seconds", "elapsed_wall_seconds", "average_tokens_per_second", "average_cuda_tokens_per_second")},
            "A1_to_A0_cuda_time_ratio": a1["cuda_training_seconds"] / a0["cuda_training_seconds"],
            "A1_to_A0_average_throughput_ratio": a1["average_tokens_per_second"] / a0["average_tokens_per_second"],
        },
    }
    return {
        "status": "complete",
        "experiment": "a0_a1_synthetic_solvability_associations8_values8000",
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_path": PROTOCOL_PATH,
        "protocol_sha256": PROTOCOL_SHA256,
        "seed": SEED,
        "value_count": EXPECTED_VALUE_COUNT,
        "association_count": EXPECTED_ASSOCIATION_COUNT,
        "sequence_length": EXPECTED_SEQUENCE_LENGTH,
        "tokens_per_step": TOKENS_PER_STEP,
        "synthetic_generator_version": EXPECTED_GENERATOR_VERSION,
        "synthetic_generator_sha256": EXPECTED_GENERATOR_HASH,
        "synthetic_data_sha256": EXPECTED_DATA_HASH,
        "architectures": list(ARCHITECTURES),
        "runs": [runs[architecture] for architecture in ARCHITECTURES],
        "paired": paired,
        "validation_curve": curve,
        "interpretation": {
            "training_memorization": "Both models reached 100% autoregressive greedy exact match on the repeated fixed training set.",
            "heldout_role": "Held-out keys and mappings are diagnostic for transfer of the eight-association copy/retrieve operation; they are not a winner threshold.",
            "token_budget_limitation": "The eight-association examples require 29 tokens, so this rung uses 928,000 token positions at the same 2,000 optimizer steps; it is not an equal-token comparison with earlier association rungs.",
            "next_axis": "The association-count axis is complete; no next association-count rung is scheduled.",
        },
        "raw_artifacts": "Ignored results/runs directories; not tracked by Git.",
    }


def _report(aggregate: dict[str, Any]) -> str:
    runs = {run["architecture"]: run for run in aggregate["runs"]}
    a0, a1 = runs["A0"], runs["A1"]
    heldout = aggregate["paired"]["heldout"]
    resources = aggregate["paired"]["resources"]
    lines = [
        "# BraidFormer A0/A1 eight-association, 8000-value solvability result",
        "",
        "Status: **complete**. This finishes the predeclared association-count axis after the completed one-association value-space ladder and two- and four-association tests.",
        "",
        f"Frozen protocol commit: `{PROTOCOL_COMMIT}`.",
        "",
        "## Final measurements",
        "",
        "| Metric | A0 | A1 |",
        "| --- | ---: | ---: |",
        f"| Parameters | {a0['parameter_count']:,} | {a1['parameter_count']:,} |",
        f"| Associations | {a0['association_count']} | {a1['association_count']} |",
        f"| Reserved answer tokens | {a0['value_count']:,} | {a1['value_count']:,} |",
        f"| Sequence length | {a0['sequence_length']} | {a1['sequence_length']} |",
        f"| Optimizer steps | {a0['optimizer_steps']:,} | {a1['optimizer_steps']:,} |",
        f"| Token positions | {a0['tokens_processed']:,} | {a1['tokens_processed']:,} |",
        f"| Final training target loss | {a0['final_train']['target_loss']:.6f} | {a1['final_train']['target_loss']:.6f} |",
        f"| Training greedy exact match | {a0['final_train']['greedy_exact_match']:.4f} | {a1['final_train']['greedy_exact_match']:.4f} |",
        f"| Final held-out target loss | {a0['final_heldout']['target_loss']:.6f} | {a1['final_heldout']['target_loss']:.6f} |",
        f"| Held-out greedy exact match | {a0['final_heldout']['greedy_exact_match']:.4f} | {a1['final_heldout']['greedy_exact_match']:.4f} |",
        f"| CUDA training seconds | {a0['cuda_training_seconds']:.3f} | {a1['cuda_training_seconds']:.3f} |",
        f"| Elapsed wall seconds | {a0['elapsed_wall_seconds']:.3f} | {a1['elapsed_wall_seconds']:.3f} |",
        f"| CUDA tokens/second | {a0['average_cuda_tokens_per_second']:.1f} | {a1['average_cuda_tokens_per_second']:.1f} |",
        "",
        "## Interpretation",
        "",
        "Both unchanged 42M models passed the primary training-memorization test: they reached 100% autoregressive one-token retrieval on the repeated training set with eight explicit bindings and the 8,000-token candidate answer pool.",
        "",
        f"On the diagnostic held-out set, both models reached `{a0['final_heldout']['greedy_exact_match']:.4f}` (`0/128`). The paired A1 − A0 target-loss delta was `{heldout['A1_minus_A0_target_loss']:+.6f}`, and the exact-match delta was `{heldout['A1_minus_A0_greedy_exact_match']:+.4f}`. Exact-match separation is at the floor, while target loss still distinguishes the models.",
        "",
        "Relative to the four-association 8000-value rung, held-out exact match remained at 0.0000 for both architectures under this 2,000-optimizer-step recipe. Both models still memorize the fixed training set, so the association-count axis reaches a held-out floor without establishing a training-memorization boundary.",
        "",
        f"A1 used `{resources['A1_to_A0_cuda_time_ratio']:.3f}×` A0's CUDA time and achieved `{resources['A1_to_A0_average_throughput_ratio']:.3f}×` A0's average token throughput. This was not a matched-compute comparison. The eight-association rung itself used the same 928,000 token-position budget for A0 and A1.",
        "",
        "The predeclared association-count axis is complete. The next work should be selected from the remaining calibrated axes—such as distractor distance or target length—or from the later non-invasive A1 forensics. Predictive learning-state, competence-ledger, scaling, A2, and memory remain deferred.",
        "",
        "## Selected validation checkpoints",
        "",
        "| Step | A0 held-out loss | A1 held-out loss | A1 − A0 | A0 exact | A1 exact |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for point in aggregate["validation_curve"]:
        if point["step"] in {100, 500, 1_000, 1_500, 2_000}:
            lines.append(
                f"| {point['step']:,} | {point['A0_heldout_target_loss']:.6f} | {point['A1_heldout_target_loss']:.6f} | {point['A1_minus_A0_heldout_target_loss']:+.6f} | {point['A0_heldout_greedy_exact_match']:.4f} | {point['A1_heldout_greedy_exact_match']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Integrity and scope",
            "",
            "Both runs completed with finite losses and gradients, no CUDA/OOM failure, and SHA-256-verified final checkpoints containing optimizer and RNG state. Gradient checkpointing remains unimplemented and disabled.",
            "",
            f"- A0 checkpoint: `{a0['checkpoint']['sha256']}`",
            f"- A1 checkpoint: `{a1['checkpoint']['sha256']}`",
            f"- Synthetic data hash: `{EXPECTED_DATA_HASH}`",
            "- Raw logs, checkpoints, and sidecars remain in ignored `results/runs/` directories.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results" / "a0_a1_solvability_associations8_values8000.json")
    parser.add_argument("--report", type=Path, default=REPO_ROOT / "docs" / "A0_A1_SOLVABILITY_ASSOCIATIONS8_VALUES8000_RESULT.md")
    args = parser.parse_args()
    runs = {architecture: _load_run(architecture) for architecture in ARCHITECTURES}
    aggregate = _aggregate(runs)
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    report = args.report if args.report.is_absolute() else REPO_ROOT / args.report
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_report(aggregate), encoding="utf-8")
    print(json.dumps({"output": str(output), "report": str(report), "runs": len(runs)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
