#!/usr/bin/env python3
"""Analyze the frozen A0/A1 schedule-disentanglement runs.

The script reads completed summaries and ignored step logs, verifies every
final checkpoint with the existing integrity utility, and writes only derived
machine-readable evidence.  It never loads model weights or changes runtime
artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from braidformer.training.integrity import verify_checkpoint


SEEDS = (17, 23, 29)
ADVANTAGE_THRESHOLD = -0.05
ANCHOR_FRACTIONS = (0.25, 0.50, 0.75, 1.00)
CUDA_TARGET_SECONDS = 1616.8551635742188
CONDITIONS = {
    "cuda_fraction": {
        "run_patterns": {
            "A0": "a0_schedule_cuda_seed{seed}",
            "A1": "a1_schedule_cuda_seed{seed}",
        },
        "curve_axis": "cumulative_cuda_seconds",
        "description": "shared cosine schedule indexed by cumulative CUDA progress",
        "cuda_time_tolerance_fraction": 0.05,
    },
    "equal_token_step": {
        "run_patterns": {
            "A0": "a0_equal_token_schedule_seed{seed}",
            "A1": "a1_equal_token_schedule_seed{seed}",
        },
        "curve_axis": "normalized_optimizer_step",
        "description": "identical optimizer-step/token budget and cosine schedule",
        "cuda_time_tolerance_fraction": None,
    },
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _finite(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


def _load_run(run_dir: Path, architecture: str) -> dict[str, Any]:
    summary = _load_json(run_dir / "summary.json")
    cumulative_cuda = 0.0
    validation_points: list[dict[str, Any]] = []
    with (run_dir / "steps.jsonl").open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            duration = record.get("cuda_elapsed_seconds")
            if not _finite(float(duration)) or float(duration) < 0:
                raise ValueError(f"invalid CUDA duration at {run_dir}:{line_number}")
            cumulative_cuda += float(duration)
            validation_loss = record.get("validation_loss")
            if validation_loss is None:
                continue
            validation_loss = float(validation_loss)
            if not _finite(validation_loss):
                raise ValueError(f"invalid validation loss at {run_dir}:{line_number}")
            validation_points.append(
                {
                    "step": int(record["step"]),
                    "tokens_processed": int(record["tokens_processed"]),
                    "cumulative_cuda_seconds": cumulative_cuda,
                    "wall_clock_seconds": float(record["wall_clock_seconds"]),
                    "learning_rate": float(record["learning_rate"]),
                    "validation_loss": validation_loss,
                }
            )

    reported_cuda = float(summary["cuda_timing"]["total_cuda_elapsed_seconds"])
    if not math.isclose(cumulative_cuda, reported_cuda, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(
            f"CUDA total mismatch in {run_dir}: log={cumulative_cuda}, summary={reported_cuda}"
        )
    if not validation_points:
        raise ValueError(f"no validation points in {run_dir}")

    checkpoint_path = Path(summary["checkpoint"])
    checkpoint_record = verify_checkpoint(checkpoint_path)
    expected_checkpoint_fields = {
        "config_sha256": summary["config_sha256"],
        "data_manifest_sha256": summary["data_manifest_sha256"],
        "seed": int(summary["seed"]),
        "step": int(summary["final_step"]),
        "tokens_processed": int(summary["tokens_processed"]),
    }
    for field, expected in expected_checkpoint_fields.items():
        if checkpoint_record.get(field) != expected:
            raise ValueError(
                f"checkpoint metadata mismatch in {checkpoint_path}: "
                f"{field}={checkpoint_record.get(field)!r}, expected {expected!r}"
            )

    target = summary.get("schedule_target_cuda_seconds")
    target = None if target in (None, 0, 0.0) else float(target)
    deviation = None if target is None else (reported_cuda - target) / target * 100.0
    return {
        "architecture": architecture,
        "run_dir": str(run_dir),
        "experiment_id": summary["experiment_id"],
        "seed": int(summary["seed"]),
        "parameter_count": int(summary["parameter_count"]),
        "optimizer_steps": int(summary["optimizer_steps"]),
        "tokens_processed": int(summary["tokens_processed"]),
        "validation_loss": float(summary["validation_loss"]),
        "train_loss": float(summary["train_loss"]),
        "cuda_seconds": reported_cuda,
        "cuda_target_seconds": target,
        "cuda_deviation_percent": deviation,
        "training_wall_seconds": float(summary["training_wall_seconds"]),
        "elapsed_wall_seconds": float(summary["elapsed_wall_seconds"]),
        "tokens_per_second": float(summary["average_tokens_per_second"]),
        "peak_allocated_vram_bytes": int(summary["peak_allocated_vram_bytes"]),
        "peak_reserved_vram_bytes": int(summary["peak_reserved_vram_bytes"]),
        "config_sha256": summary["config_sha256"],
        "data_manifest_sha256": summary["data_manifest_sha256"],
        "tokenizer_sha256": summary["tokenizer_sha256"],
        "warmup_steps": int(summary["warmup_steps"]),
        "lr_progress_reference": summary["lr_progress_reference"],
        "failure": summary.get("failure"),
        "numerical_instability": bool(summary["numerical_instability"]),
        "status": summary["status"],
        "gradient_checkpointing": summary["gradient_checkpointing"],
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_record["checkpoint_sha256"],
            "bytes": int(checkpoint_record["checkpoint_bytes"]),
            "complete": bool(checkpoint_record["complete"]),
            "verified": True,
        },
        "validation_points": validation_points,
    }


def _axis_value(point: dict[str, Any], arm: dict[str, Any], axis: str) -> float:
    if axis == "cumulative_cuda_seconds":
        return float(point["cumulative_cuda_seconds"])
    if axis == "normalized_optimizer_step":
        return float(point["step"]) / float(arm["optimizer_steps"])
    raise ValueError(f"unsupported curve axis: {axis}")


def _interpolate(
    points: list[dict[str, Any]], arm: dict[str, Any], axis: str, x: float
) -> float | None:
    values = [(_axis_value(point, arm, axis), float(point["validation_loss"])) for point in points]
    if x < values[0][0] or x > values[-1][0]:
        return None
    for (left_x, left_y), (right_x, right_y) in zip(values, values[1:]):
        if x == left_x:
            return left_y
        if x <= right_x:
            if right_x == left_x:
                return right_y
            fraction = (x - left_x) / (right_x - left_x)
            return left_y + fraction * (right_y - left_y)
    return values[-1][1] if x == values[-1][0] else None


def _integral(
    points: list[dict[str, Any]], arm: dict[str, Any], axis: str, start: float, end: float
) -> float:
    if end <= start:
        raise ValueError("invalid integration interval")
    interior = [
        _axis_value(point, arm, axis)
        for point in points
        if start < _axis_value(point, arm, axis) < end
    ]
    xs = [start, *interior, end]
    ys = [_interpolate(points, arm, axis, x) for x in xs]
    if any(value is None for value in ys):
        raise ValueError("integration interval is outside observed curve")
    return sum(
        (xs[index + 1] - xs[index]) * (ys[index] + ys[index + 1]) / 2.0
        for index in range(len(xs) - 1)
    )


def _curve_comparison(
    arms: dict[str, dict[int, dict[str, Any]]], axis: str
) -> dict[str, Any]:
    per_seed: list[dict[str, Any]] = []
    for seed in SEEDS:
        a0 = arms["A0"][seed]
        a1 = arms["A1"][seed]
        a0_points = a0["validation_points"]
        a1_points = a1["validation_points"]
        a0_start = _axis_value(a0_points[0], a0, axis)
        a1_start = _axis_value(a1_points[0], a1, axis)
        a0_end = _axis_value(a0_points[-1], a0, axis)
        a1_end = _axis_value(a1_points[-1], a1, axis)
        common_start = max(a0_start, a1_start)
        common_end = min(a0_end, a1_end)
        if common_end <= common_start:
            raise ValueError(f"no common observed curve interval for seed {seed}")

        anchors: list[dict[str, Any]] = []
        for fraction in ANCHOR_FRACTIONS:
            x = common_start + fraction * (common_end - common_start)
            a0_loss = _interpolate(a0_points, a0, axis, x)
            a1_loss = _interpolate(a1_points, a1, axis, x)
            if a0_loss is None or a1_loss is None:
                raise ValueError(f"anchor outside observed curve for seed {seed}")
            anchors.append(
                {
                    "fraction_of_common_observed_interval": fraction,
                    "axis_value": x,
                    "A0_validation_loss": a0_loss,
                    "A1_validation_loss": a1_loss,
                    "A1_minus_A0": a1_loss - a0_loss,
                }
            )

        a0_auc = _integral(a0_points, a0, axis, common_start, common_end)
        a1_auc = _integral(a1_points, a1, axis, common_start, common_end)
        interval = common_end - common_start
        per_seed.append(
            {
                "seed": seed,
                "common_start": common_start,
                "common_end": common_end,
                "interval": interval,
                "anchors": anchors,
                "A0_auc": a0_auc,
                "A1_auc": a1_auc,
                "A1_minus_A0_auc": a1_auc - a0_auc,
                "A0_normalized_auc": a0_auc / interval,
                "A1_normalized_auc": a1_auc / interval,
                "A1_minus_A0_normalized_auc": (a1_auc - a0_auc) / interval,
            }
        )

    normalized_auc_deltas = [row["A1_minus_A0_normalized_auc"] for row in per_seed]
    anchor_means = []
    for index, fraction in enumerate(ANCHOR_FRACTIONS):
        deltas = [row["anchors"][index]["A1_minus_A0"] for row in per_seed]
        anchor_means.append(
            {
                "fraction_of_common_observed_interval": fraction,
                "mean_A1_minus_A0": mean(deltas),
                "sample_sd_A1_minus_A0": stdev(deltas),
            }
        )
    return {
        "axis": axis,
        "alignment": "piecewise-linear interpolation within the common observed interval; no extrapolation",
        "per_seed": per_seed,
        "mean_anchor_deltas": anchor_means,
        "mean_A1_minus_A0_normalized_auc": mean(normalized_auc_deltas),
        "sample_sd_A1_minus_A0_normalized_auc": stdev(normalized_auc_deltas),
    }


def _aggregate(arms: dict[str, dict[int, dict[str, Any]]], tolerance: float | None) -> dict[str, Any]:
    a0 = [arms["A0"][seed] for seed in SEEDS]
    a1 = [arms["A1"][seed] for seed in SEEDS]
    deltas = [right["validation_loss"] - left["validation_loss"] for left, right in zip(a0, a1)]
    mean_a0 = mean(record["validation_loss"] for record in a0)
    mean_a1 = mean(record["validation_loss"] for record in a1)
    all_arms = [*a0, *a1]
    cuda_totals = {architecture: [record["cuda_seconds"] for record in arms[architecture].values()] for architecture in ("A0", "A1")}
    cuda_deviations = {
        architecture: [record["cuda_deviation_percent"] for record in arms[architecture].values()]
        for architecture in ("A0", "A1")
    }
    return {
        "mean_A0_validation_loss": mean_a0,
        "mean_A1_validation_loss": mean_a1,
        "mean_paired_A1_minus_A0": mean(deltas),
        "sample_sd_paired_A1_minus_A0": stdev(deltas),
        "relative_loss_advantage_percent": (mean_a0 - mean_a1) / mean_a0 * 100.0,
        "per_seed_A1_minus_A0": {str(seed): deltas[index] for index, seed in enumerate(SEEDS)},
        "seeds_meeting_threshold": sum(delta <= ADVANTAGE_THRESHOLD for delta in deltas),
        "seeds_tested": len(deltas),
        "advantage_threshold": ADVANTAGE_THRESHOLD,
        "average_throughput_tokens_per_second": {
            architecture: mean(record["tokens_per_second"] for record in arms[architecture].values())
            for architecture in ("A0", "A1")
        },
        "throughput_by_seed_tokens_per_second": {
            architecture: {str(seed): arms[architecture][seed]["tokens_per_second"] for seed in SEEDS}
            for architecture in ("A0", "A1")
        },
        "cuda_seconds_by_architecture": cuda_totals,
        "mean_cuda_seconds_by_architecture": {
            architecture: mean(values) for architecture, values in cuda_totals.items()
        },
        "cuda_deviation_percent_by_architecture": cuda_deviations,
        "mean_cuda_deviation_percent_by_architecture": {
            architecture: None if values[0] is None else mean(values)
            for architecture, values in cuda_deviations.items()
        },
        "cuda_time_tolerance_fraction": tolerance,
        "all_cuda_times_within_tolerance": (
            None
            if tolerance is None
            else all(
                abs(record["cuda_deviation_percent"]) <= tolerance * 100.0
                for record in all_arms
            )
        ),
        "tokens_processed_by_architecture": {
            architecture: {str(seed): arms[architecture][seed]["tokens_processed"] for seed in SEEDS}
            for architecture in ("A0", "A1")
        },
        "peak_allocated_vram_bytes_by_architecture": {
            architecture: {str(seed): arms[architecture][seed]["peak_allocated_vram_bytes"] for seed in SEEDS}
            for architecture in ("A0", "A1")
        },
        "peak_reserved_vram_bytes_by_architecture": {
            architecture: {str(seed): arms[architecture][seed]["peak_reserved_vram_bytes"] for seed in SEEDS}
            for architecture in ("A0", "A1")
        },
        "checkpoint_integrity_failures": sum(
            not record["checkpoint"]["verified"] for record in all_arms
        ),
        "checkpoint_integrity_all_verified": all(
            record["checkpoint"]["verified"] and record["checkpoint"]["complete"]
            for record in all_arms
        ),
        "numerical_instability_reported": any(record["numerical_instability"] for record in all_arms),
        "failures": [record["failure"] for record in all_arms if record["failure"] is not None],
        "status_values": sorted({record["status"] for record in all_arms}),
    }


def _native_8x_summary(result_path: Path) -> dict[str, Any]:
    source = _load_json(result_path)
    budget = source["budgets"]["8x"]
    per_seed = []
    for seed in SEEDS:
        a0 = budget["arms"]["A0"][str(seed)]
        a1 = budget["arms"]["A1"][str(seed)]
        per_seed.append(
            {
                "seed": seed,
                "A0_validation_loss": a0["validation_loss"],
                "A1_validation_loss": a1["validation_loss"],
                "A1_minus_A0": a1["validation_loss"] - a0["validation_loss"],
                "A0_cuda_seconds": a0["cuda_seconds"],
                "A1_cuda_seconds": a1["cuda_seconds"],
            }
        )
    deltas = [row["A1_minus_A0"] for row in per_seed]
    mean_a0 = mean(row["A0_validation_loss"] for row in per_seed)
    mean_a1 = mean(row["A1_validation_loss"] for row in per_seed)
    return {
        "source": str(result_path),
        "protocol_commit": "b6632cb",
        "per_seed": per_seed,
        "mean_A0_validation_loss": mean_a0,
        "mean_A1_validation_loss": mean_a1,
        "mean_paired_A1_minus_A0": mean(deltas),
        "sample_sd_paired_A1_minus_A0": stdev(deltas),
        "seeds_meeting_threshold": sum(delta <= ADVANTAGE_THRESHOLD for delta in deltas),
    }


def analyze(root: Path, native_result: Path) -> dict[str, Any]:
    conditions: dict[str, Any] = {}
    for condition, metadata in CONDITIONS.items():
        arms: dict[str, dict[int, dict[str, Any]]] = {"A0": {}, "A1": {}}
        for architecture, pattern in metadata["run_patterns"].items():
            for seed in SEEDS:
                run_dir = root / pattern.format(seed=seed)
                arms[architecture][seed] = _load_run(run_dir, architecture)
        conditions[condition] = {
            "description": metadata["description"],
            "curve_axis": metadata["curve_axis"],
            "target_cuda_seconds_per_arm": (
                CUDA_TARGET_SECONDS if condition == "cuda_fraction" else None
            ),
            "aggregate": _aggregate(arms, metadata["cuda_time_tolerance_fraction"]),
            "curve_analysis": _curve_comparison(arms, metadata["curve_axis"]),
            "arms": arms,
        }

    native = _native_8x_summary(native_result)
    return {
        "status": "completed",
        "analysis_method": {
            "final_checkpoint_verification": "existing braidformer.training.integrity.verify_checkpoint, including SHA-256 sidecar and metadata checks",
            "validation_curves": "piecewise-linear interpolation between observed validation points within each seed's common observed interval; no extrapolation",
            "curve_auc": "trapezoidal integration over the same common observed interval",
            "final_delta": "final validation loss A1 minus A0 at each completed arm endpoint",
            "relative_loss_advantage": "(mean A0 validation loss - mean A1 validation loss) / mean A0 validation loss",
        },
        "protocol": {
            "commit": "eb1cb6a",
            "seeds": list(SEEDS),
            "advantage_threshold": ADVANTAGE_THRESHOLD,
            "cuda_time_tolerance_fraction": 0.05,
            "native_8x_reference_commit": "b6632cb",
        },
        "conditions": conditions,
        "native_8x_reference": native,
        "schedule_control_comparison": {
            condition: {
                "mean_delta_minus_native_8x_mean_delta": (
                    conditions[condition]["aggregate"]["mean_paired_A1_minus_A0"]
                    - native["mean_paired_A1_minus_A0"]
                ),
                "mean_delta": conditions[condition]["aggregate"]["mean_paired_A1_minus_A0"],
                "native_8x_mean_delta": native["mean_paired_A1_minus_A0"],
            }
            for condition in CONDITIONS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/runs"))
    parser.add_argument(
        "--native-result",
        type=Path,
        default=Path("results/a0_a1_eight_equal_cuda_replications.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root, args.native_result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
