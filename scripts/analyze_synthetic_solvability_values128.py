#!/usr/bin/env python3
"""Verify and aggregate the completed 128-value solvability-rung runs.

The training outputs live in ignored ``results/runs`` directories.  This
read-only analyzer validates the durable summaries, step logs, validation
cadence, and final checkpoint sidecars before writing a concise tracked
aggregate and report.
"""

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
PROTOCOL_COMMIT = "68fbfd37b1c93a6941086a6e88b368243b48382f"
PROTOCOL_PATH = "docs/A0_A1_SOLVABILITY_VALUES128_PROTOCOL.md"
PROTOCOL_SHA256 = "79c50d431efc30097d9fdfa0668bf5e000c4285a08ba6ef5c52ed60dd5ed4c05"
EXPECTED_PARAMS = {"A0": 41_866_240, "A1": 41_870_336}
EXPECTED_CONFIGS = {
    "A0": "configs/braidnano_a0_solvability_values128.json",
    "A1": "configs/braidnano_a1_solvability_values128.json",
}
EXPECTED_CONFIG_HASHES = {
    "A0": "1fdf5066bee2218a8a922c1bb816bcdc94bc8990616136f0d620e166c0f31f79",
    "A1": "6321474743d438e0029a59210d2735ae7ffca98e86f09abc05a9a6dbbb2d9b9f",
}
EXPECTED_DATA_HASH = "937a210c99d23c494ae296f28305df2816a48ec827a3a2ac5fa92d323040b9b9"
EXPECTED_GENERATOR_VERSION = "synthetic_solvability_values128_v1"
EXPECTED_VALUE_COUNT = 128
EXPECTED_STEPS = 2_000
EXPECTED_TOKENS = 256_000
TOKENS_PER_STEP = 128
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


def _validate_metrics(metrics: dict[str, Any], label: str) -> dict[str, Any]:
    if int(metrics.get("examples", -1)) != EXPECTED_VALUE_COUNT:
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
    run_dir = REPO_ROOT / "results" / "runs" / f"solvability_values128_{architecture.lower()}_seed{SEED}"
    summary = _load_json(run_dir / "summary.json")
    if summary.get("status") != "completed":
        raise ValueError(f"run is not complete: {run_dir}")
    if summary.get("architecture") != architecture or int(summary.get("seed", -1)) != SEED:
        raise ValueError(f"run identity mismatch: {run_dir}")
    if summary.get("protocol_commit") != PROTOCOL_COMMIT:
        raise ValueError(f"protocol commit mismatch: {run_dir}")
    if summary.get("protocol_sha256") != PROTOCOL_SHA256:
        raise ValueError(f"protocol hash mismatch in summary: {run_dir}")
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
    }
    for field, expected in expected_checkpoint.items():
        if checkpoint_record.get(field) != expected:
            raise ValueError(
                f"checkpoint metadata mismatch in {checkpoint_path}: "
                f"{field}={checkpoint_record.get(field)!r}, expected={expected!r}"
            )
    if summary.get("checkpoint_sha256") != checkpoint_record.get("checkpoint_sha256"):
        raise ValueError(f"summary/checkpoint hash mismatch: {run_dir}")

    step_records: list[dict[str, Any]] = []
    cumulative_cuda = 0.0
    validation_by_step: dict[int, dict[str, Any]] = {}
    with (run_dir / "steps.jsonl").open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            if record.get("event") != "step":
                continue
            step = int(record["step"])
            if step != len(step_records) + 1:
                raise ValueError(f"non-contiguous step log at {run_dir}:{line_number}")
            if int(record["tokens_processed"]) != step * TOKENS_PER_STEP:
                raise ValueError(f"token progress mismatch at {run_dir}:{line_number}")
            train_loss = _finite(record["train_loss"], "training loss")
            cuda_seconds = _finite(record["cuda_elapsed_seconds"], "CUDA duration")
            if cuda_seconds <= 0:
                raise ValueError(f"non-positive CUDA duration at {run_dir}:{line_number}")
            _finite(record["wall_clock_seconds"], "wall time")
            _finite(record["gradient_norm"], "gradient norm")
            learning_rate = _finite(record["learning_rate"], "learning rate")
            cumulative_cuda += cuda_seconds
            if record.get("validation_loss") is not None:
                validation_by_step[step] = {
                    "step": step,
                    "tokens_processed": int(record["tokens_processed"]),
                    "cumulative_cuda_seconds": cumulative_cuda,
                    "validation_loss": _finite(record["validation_loss"], "validation loss"),
                    "learning_rate": learning_rate,
                }
            step_records.append(
                {
                    "step": step,
                    "tokens_processed": int(record["tokens_processed"]),
                    "train_loss": train_loss,
                    "cuda_elapsed_seconds": cuda_seconds,
                }
            )
    if len(step_records) != EXPECTED_STEPS:
        raise ValueError(f"expected {EXPECTED_STEPS} step records in {run_dir}, got {len(step_records)}")
    if sorted(validation_by_step) != list(EVAL_STEPS):
        raise ValueError(f"validation cadence mismatch in {run_dir}")
    if not math.isclose(cumulative_cuda, float(summary["cuda_training_seconds"]), rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"CUDA total mismatch in {run_dir}")

    metric_records: list[dict[str, Any]] = []
    with (run_dir / "calibration_metrics.jsonl").open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            step = int(record["step"])
            if step != len(metric_records) + 1:
                raise ValueError(f"non-contiguous calibration metric step at {run_dir}:{line_number}")
            _finite(record["train_loss"], "calibration metric training loss")
            _finite(record["learning_rate"], "calibration metric learning rate")
            _finite(record["cuda_elapsed_seconds"], "calibration metric CUDA duration")
            _finite(record["wall_clock_seconds"], "calibration metric wall time")
            metric_records.append(record)
    if len(metric_records) != EXPECTED_STEPS:
        raise ValueError(f"expected {EXPECTED_STEPS} calibration metric records in {run_dir}")
    evaluation_records = [record for record in metric_records if record.get("evaluation") is not None]
    if [int(record["step"]) for record in evaluation_records] != list(EVAL_STEPS):
        raise ValueError(f"calibration metric evaluation cadence mismatch in {run_dir}")

    validation_history: list[dict[str, Any]] = []
    for record in summary["validation_history"]:
        step = int(record["step"])
        if step not in EVAL_STEPS:
            raise ValueError(f"unexpected summary validation step in {run_dir}: {step}")
        validation_history.append(
            {
                "step": step,
                "tokens_processed": step * TOKENS_PER_STEP,
                "cumulative_cuda_seconds": validation_by_step[step]["cumulative_cuda_seconds"],
                "learning_rate": validation_by_step[step]["learning_rate"],
                "train": _validate_metrics(record["metrics"]["train"], f"{run_dir}:{step}:train"),
                "heldout": _validate_metrics(record["metrics"]["heldout"], f"{run_dir}:{step}:heldout"),
            }
        )
    if [record["step"] for record in validation_history] != list(EVAL_STEPS):
        raise ValueError(f"summary validation history mismatch in {run_dir}")

    final_train = _validate_metrics(summary["final_train"], f"{run_dir}:final_train")
    final_heldout = _validate_metrics(summary["final_heldout"], f"{run_dir}:final_heldout")
    return {
        "architecture": architecture,
        "seed": SEED,
        "experiment_id": summary["experiment_id"],
        "run_dir": _relative(run_dir),
        "config_path": summary["config_path"],
        "config_sha256": summary["config_sha256"],
        "parameter_count": int(summary["parameter_count"]),
        "value_count": int(summary["value_count"]),
        "optimizer_steps": int(summary["optimizer_steps"]),
        "tokens_processed": int(summary["tokens_processed"]),
        "sequence_length": int(summary["sequence_length"]),
        "warmup_steps": int(summary["warmup_steps"]),
        "learning_rate": float(summary["learning_rate"]),
        "lr_schedule": summary["lr_schedule"],
        "training_wall_seconds": _finite(summary["training_wall_seconds"], "training wall time"),
        "elapsed_wall_seconds": _finite(summary["elapsed_wall_seconds"], "elapsed wall time"),
        "cuda_training_seconds": _finite(summary["cuda_training_seconds"], "CUDA training time"),
        "average_tokens_per_second": _finite(summary["average_tokens_per_second"], "throughput"),
        "average_cuda_tokens_per_second": _finite(
            summary["average_cuda_tokens_per_second"], "CUDA throughput"
        ),
        "peak_allocated_vram_bytes": int(summary["peak_allocated_vram_bytes"]),
        "peak_reserved_vram_bytes": int(summary["peak_reserved_vram_bytes"]),
        "final_train": final_train,
        "final_heldout": final_heldout,
        "validation_history": validation_history,
        "checkpoint": {
            "path": _relative(checkpoint_path),
            "sha256": checkpoint_record["checkpoint_sha256"],
            "bytes": int(checkpoint_record["checkpoint_bytes"]),
            "complete": bool(checkpoint_record["complete"]),
            "verified": True,
        },
        "gradient_checkpointing": summary["gradient_checkpointing"],
        "numerical_instability": bool(summary["numerical_instability"]),
        "failure": summary["failure"],
    }


def _paired(a0: dict[str, Any], a1: dict[str, Any]) -> dict[str, Any]:
    return {
        "seed": SEED,
        "train": {
            "A0": a0["final_train"],
            "A1": a1["final_train"],
            "A1_minus_A0_target_loss": a1["final_train"]["target_loss"] - a0["final_train"]["target_loss"],
            "A1_minus_A0_greedy_exact_match": (
                a1["final_train"]["greedy_exact_match"] - a0["final_train"]["greedy_exact_match"]
            ),
        },
        "heldout": {
            "A0": a0["final_heldout"],
            "A1": a1["final_heldout"],
            "A1_minus_A0_target_loss": a1["final_heldout"]["target_loss"] - a0["final_heldout"]["target_loss"],
            "A1_minus_A0_greedy_exact_match": (
                a1["final_heldout"]["greedy_exact_match"] - a0["final_heldout"]["greedy_exact_match"]
            ),
        },
        "resources": {
            "A0": {
                "cuda_training_seconds": a0["cuda_training_seconds"],
                "elapsed_wall_seconds": a0["elapsed_wall_seconds"],
                "average_tokens_per_second": a0["average_tokens_per_second"],
                "average_cuda_tokens_per_second": a0["average_cuda_tokens_per_second"],
            },
            "A1": {
                "cuda_training_seconds": a1["cuda_training_seconds"],
                "elapsed_wall_seconds": a1["elapsed_wall_seconds"],
                "average_tokens_per_second": a1["average_tokens_per_second"],
                "average_cuda_tokens_per_second": a1["average_cuda_tokens_per_second"],
            },
            "A1_to_A0_cuda_time_ratio": a1["cuda_training_seconds"] / a0["cuda_training_seconds"],
            "A1_to_A0_average_throughput_ratio": (
                a1["average_tokens_per_second"] / a0["average_tokens_per_second"]
            ),
        },
    }


def _aggregate(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    a0 = runs["A0"]
    a1 = runs["A1"]
    curve: list[dict[str, Any]] = []
    for a0_point, a1_point in zip(a0["validation_history"], a1["validation_history"]):
        if a0_point["step"] != a1_point["step"]:
            raise ValueError("A0/A1 validation cadence mismatch")
        curve.append(
            {
                "step": a0_point["step"],
                "tokens_processed": a0_point["tokens_processed"],
                "A0_cumulative_cuda_seconds": a0_point["cumulative_cuda_seconds"],
                "A1_cumulative_cuda_seconds": a1_point["cumulative_cuda_seconds"],
                "A0_heldout_target_loss": a0_point["heldout"]["target_loss"],
                "A1_heldout_target_loss": a1_point["heldout"]["target_loss"],
                "A1_minus_A0_heldout_target_loss": (
                    a1_point["heldout"]["target_loss"] - a0_point["heldout"]["target_loss"]
                ),
                "A0_heldout_greedy_exact_match": a0_point["heldout"]["greedy_exact_match"],
                "A1_heldout_greedy_exact_match": a1_point["heldout"]["greedy_exact_match"],
                "A0_train_greedy_exact_match": a0_point["train"]["greedy_exact_match"],
                "A1_train_greedy_exact_match": a1_point["train"]["greedy_exact_match"],
            }
        )
    return {
        "status": "complete",
        "experiment": "a0_a1_synthetic_solvability_values128",
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_path": PROTOCOL_PATH,
        "protocol_sha256": PROTOCOL_SHA256,
        "seed": SEED,
        "value_count": EXPECTED_VALUE_COUNT,
        "synthetic_generator_version": EXPECTED_GENERATOR_VERSION,
        "synthetic_data_sha256": EXPECTED_DATA_HASH,
        "architectures": list(ARCHITECTURES),
        "runs": [runs[architecture] for architecture in ARCHITECTURES],
        "paired": _paired(a0, a1),
        "validation_curve": curve,
        "interpretation": {
            "training_memorization": "Both models reached 100% autoregressive greedy exact match on the repeated fixed training set.",
            "heldout_role": "Held-out keys and mappings are diagnostic for transfer of the one-association copy/retrieve operation; they are not a winner threshold.",
            "next_ladder_rung_started": False,
        },
        "raw_artifacts": "Ignored results/runs directories; not tracked by Git.",
    }


def _report(aggregate: dict[str, Any]) -> str:
    runs = {run["architecture"]: run for run in aggregate["runs"]}
    paired = aggregate["paired"]
    a0 = runs["A0"]
    a1 = runs["A1"]
    heldout = paired["heldout"]
    resources = paired["resources"]
    key_steps = {100, 500, 1_000, 1_500, 2_000}
    lines = [
        "# BraidFormer A0/A1 128-value solvability-rung result",
        "",
        "Status: **complete**. This is the bounded seed-17 next rung after the "
        "64-value one-association calibration.",
        "",
        f"Frozen protocol commit: `{PROTOCOL_COMMIT}`.",
        "",
        "## Final measurements",
        "",
        "| Metric | A0 | A1 |",
        "| --- | ---: | ---: |",
        f"| Parameters | {a0['parameter_count']:,} | {a1['parameter_count']:,} |",
        f"| Reserved answer tokens | {a0['value_count']} | {a1['value_count']} |",
        f"| Optimizer steps | {a0['optimizer_steps']:,} | {a1['optimizer_steps']:,} |",
        f"| Token positions | {a0['tokens_processed']:,} | {a1['tokens_processed']:,} |",
        f"| Final training target loss | {a0['final_train']['target_loss']:.6f} | {a1['final_train']['target_loss']:.6f} |",
        f"| Training greedy exact match | {a0['final_train']['greedy_exact_match']:.4f} | {a1['final_train']['greedy_exact_match']:.4f} |",
        f"| Final held-out target loss | {a0['final_heldout']['target_loss']:.6f} | {a1['final_heldout']['target_loss']:.6f} |",
        f"| Held-out greedy exact match | {a0['final_heldout']['greedy_exact_match']:.4f} | {a1['final_heldout']['greedy_exact_match']:.4f} |",
        f"| CUDA training seconds | {a0['cuda_training_seconds']:.3f} | {a1['cuda_training_seconds']:.3f} |",
        f"| Elapsed wall seconds | {a0['elapsed_wall_seconds']:.3f} | {a1['elapsed_wall_seconds']:.3f} |",
        f"| CUDA tokens/second | {a0['average_cuda_tokens_per_second']:.1f} | {a1['average_cuda_tokens_per_second']:.1f} |",
        f"| Peak allocated VRAM | {a0['peak_allocated_vram_bytes'] / (1024 ** 3):.3f} GiB | {a1['peak_allocated_vram_bytes'] / (1024 ** 3):.3f} GiB |",
        f"| Peak reserved VRAM | {a0['peak_reserved_vram_bytes'] / (1024 ** 3):.3f} GiB | {a1['peak_reserved_vram_bytes'] / (1024 ** 3):.3f} GiB |",
        "",
        "## Interpretation",
        "",
        "Both unchanged 42M models passed the primary calibration observation: "
        "they reached 100% autoregressive one-token retrieval on the repeated "
        "training set with the 128-token candidate answer pool.",
        "",
        f"On the diagnostic held-out set, A0 reached `{a0['final_heldout']['greedy_exact_match']:.4f}` "
        f"and A1 `{a1['final_heldout']['greedy_exact_match']:.4f}`. The paired A1 − A0 "
        f"target-loss delta was `{heldout['A1_minus_A0_target_loss']:+.6f}` and the A1 − A0 "
        f"exact-match delta was `{heldout['A1_minus_A0_greedy_exact_match']:+.4f}`. "
        "A0 was better on this diagnostic at this seed and budget. Because the "
        "held-out keys and mappings are disjoint while the earlier record value "
        "remains in the input, this measures transfer of the simple copy/retrieve "
        "operation rather than unseen fact learning.",
        "",
        f"A1 used `{resources['A1_to_A0_cuda_time_ratio']:.3f}×` A0's CUDA time and "
        f"achieved `{resources['A1_to_A0_average_throughput_ratio']:.3f}×` A0's average "
        "token throughput. This was not a matched-compute comparison.",
        "",
        "The result strengthens the evidence for an A0 advantage on this explicit "
        "retrieval diagnostic, but it is still one seed and one budget. It does "
        "not isolate model capacity from recurrent-state representation or "
        "optimization, and it does not authorize higher value pools, additional "
        "seeds, scaling, A2, persistent memory, or SNAIL/SSRE integration.",
        "",
        "## Selected validation checkpoints",
        "",
        "The machine-readable aggregate preserves all 20 logged validation checkpoints.",
        "",
        "| Step | A0 held-out loss | A1 held-out loss | A1 − A0 | A0 exact | A1 exact |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for point in aggregate["validation_curve"]:
        if point["step"] in key_steps:
            lines.append(
                f"| {point['step']:,} | {point['A0_heldout_target_loss']:.6f} | "
                f"{point['A1_heldout_target_loss']:.6f} | "
                f"{point['A1_minus_A0_heldout_target_loss']:+.6f} | "
                f"{point['A0_heldout_greedy_exact_match']:.4f} | "
                f"{point['A1_heldout_greedy_exact_match']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Integrity and scope",
            "",
            "Both runs completed with finite losses and gradients, no CUDA/OOM "
            "failure, and SHA-256-verified final checkpoints containing optimizer "
            "and RNG state. Gradient checkpointing was configured in the protocol "
            "but remains unimplemented and disabled.",
            "",
            f"- A0 checkpoint: `{a0['checkpoint']['sha256']}`",
            f"- A1 checkpoint: `{a1['checkpoint']['sha256']}`",
            f"- Synthetic data hash: `{EXPECTED_DATA_HASH}`",
            "- Raw logs, checkpoints, and sidecars remain in ignored `results/runs/` directories.",
            "- No additional rung or architecture experiment was started.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "a0_a1_solvability_values128.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "docs" / "A0_A1_SOLVABILITY_VALUES128_RESULT.md",
    )
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
