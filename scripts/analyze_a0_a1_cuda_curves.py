#!/usr/bin/env python3
"""Analyze the frozen A0/A1 measured-CUDA learning curves.

This is a read-only analysis of completed run summaries and ignored step logs.
It does not load model checkpoints or alter training artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any


TARGETS = {
    "1x": 202.10689544677734,
    "4x": 808.4275817871094,
    "8x": 1616.8551635742188,
}
SEEDS = (17, 23, 29)
ANCHOR_FRACTIONS = (0.25, 0.50, 0.75, 1.00)
RUN_SUFFIXES = {
    "1x": "equal_cuda",
    "4x": "longer_equal_cuda",
    "8x": "eight_equal_cuda",
}


def _finite(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_curve(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, float]]]:
    summary = _load_json(run_dir / "summary.json")
    cumulative = 0.0
    points: list[dict[str, float]] = []
    with (run_dir / "steps.jsonl").open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            duration = float(record["cuda_elapsed_seconds"])
            if not _finite(duration) or duration < 0:
                raise ValueError(f"invalid CUDA duration at {run_dir}:{line_number}")
            cumulative += duration
            validation_loss = record.get("validation_loss")
            if validation_loss is not None:
                value = float(validation_loss)
                if not _finite(value):
                    raise ValueError(f"invalid validation loss at {run_dir}:{line_number}")
                points.append(
                    {
                        "step": int(record["step"]),
                        "cuda_seconds": cumulative,
                        "validation_loss": value,
                    }
                )
    if not points:
        raise ValueError(f"no validation points in {run_dir}")
    reported_total = float(summary["cuda_timing"]["total_cuda_elapsed_seconds"])
    if abs(cumulative - reported_total) > 1e-6:
        raise ValueError(
            f"CUDA total mismatch in {run_dir}: log={cumulative}, summary={reported_total}"
        )
    return summary, points


def _interpolate(points: list[dict[str, float]], x: float) -> float | None:
    if x < points[0]["cuda_seconds"] or x > points[-1]["cuda_seconds"]:
        return None
    for left, right in zip(points, points[1:]):
        x_left = left["cuda_seconds"]
        x_right = right["cuda_seconds"]
        if x == x_left:
            return left["validation_loss"]
        if x <= x_right:
            if x_right == x_left:
                return right["validation_loss"]
            fraction = (x - x_left) / (x_right - x_left)
            return left["validation_loss"] + fraction * (
                right["validation_loss"] - left["validation_loss"]
            )
    return points[-1]["validation_loss"] if x == points[-1]["cuda_seconds"] else None


def _integral(points: list[dict[str, float]], start: float, end: float) -> float:
    if end <= start:
        raise ValueError("invalid integration interval")
    interior = [
        point["cuda_seconds"]
        for point in points
        if start < point["cuda_seconds"] < end
    ]
    xs = [start, *interior, end]
    ys = [_interpolate(points, x) for x in xs]
    if any(y is None for y in ys):
        raise ValueError("integration interval is outside observed curve")
    return sum(
        (xs[index + 1] - xs[index]) * (ys[index] + ys[index + 1]) / 2.0
        for index in range(len(xs) - 1)
    )


def _checkpoint_record(run_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    checkpoint = Path(summary["checkpoint"])
    sidecar = _load_json(checkpoint.with_suffix(checkpoint.suffix + ".json"))
    return {
        "sha256": sidecar["checkpoint_sha256"],
        "complete": bool(sidecar["complete"]),
        "step": int(sidecar["step"]),
        "tokens_processed": int(sidecar["tokens_processed"]),
        "config_sha256": sidecar["config_sha256"],
    }


def _arm_record(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, float]]]:
    summary, points = _load_curve(run_dir)
    total_cuda = float(summary["cuda_timing"]["total_cuda_elapsed_seconds"])
    checkpoint = _checkpoint_record(run_dir, summary)
    record = {
        "run_dir": str(run_dir),
        "experiment_id": summary["experiment_id"],
        "seed": int(summary["seed"]),
        "parameter_count": int(summary["parameter_count"]),
        "optimizer_steps": int(summary["optimizer_steps"]),
        "tokens_processed": int(summary["tokens_processed"]),
        "validation_loss": float(summary["validation_loss"]),
        "train_loss": float(summary["train_loss"]),
        "cuda_seconds": total_cuda,
        "training_wall_seconds": float(summary["training_wall_seconds"]),
        "elapsed_wall_seconds": float(summary["elapsed_wall_seconds"]),
        "tokens_per_second": float(summary["average_tokens_per_second"]),
        "peak_allocated_vram_bytes": int(summary["peak_allocated_vram_bytes"]),
        "peak_reserved_vram_bytes": int(summary["peak_reserved_vram_bytes"]),
        "config_sha256": summary["config_sha256"],
        "data_manifest_sha256": summary["data_manifest_sha256"],
        "tokenizer_sha256": summary["tokenizer_sha256"],
        "warmup_steps": int(summary["warmup_steps"]),
        "failure": summary.get("failure"),
        "numerical_instability": bool(summary["numerical_instability"]),
        "status": summary["status"],
        "checkpoint": checkpoint,
        "validation_point_count": len(points),
        "first_validation_cuda_seconds": points[0]["cuda_seconds"],
        "last_validation_cuda_seconds": points[-1]["cuda_seconds"],
    }
    return record, points


def _round_or_none(value: float | None) -> float | None:
    return None if value is None else float(value)


def analyze_budget(root: Path, budget: str) -> dict[str, Any]:
    target = TARGETS[budget]
    suffix = RUN_SUFFIXES[budget]
    arms: dict[str, dict[int, dict[str, Any]]] = {"A0": {}, "A1": {}}
    curves: dict[str, dict[int, list[dict[str, float]]]] = {"A0": {}, "A1": {}}
    for architecture in ("A0", "A1"):
        for seed in SEEDS:
            run_dir = root / f"{architecture.lower()}_seed{seed}_{suffix}"
            arms[architecture][seed], curves[architecture][seed] = _arm_record(run_dir)
            arms[architecture][seed]["cuda_target_seconds"] = target
            arms[architecture][seed]["cuda_deviation_percent"] = (
                (arms[architecture][seed]["cuda_seconds"] - target) / target * 100.0
            )

    anchors: list[dict[str, Any]] = []
    for fraction in ANCHOR_FRACTIONS:
        target_time = target * fraction
        per_seed: list[dict[str, Any]] = []
        for seed in SEEDS:
            a0 = _interpolate(curves["A0"][seed], target_time)
            a1 = _interpolate(curves["A1"][seed], target_time)
            per_seed.append(
                {
                    "seed": seed,
                    "target_cuda_seconds": target_time,
                    "A0_validation_loss": _round_or_none(a0),
                    "A1_validation_loss": _round_or_none(a1),
                    "A1_minus_A0": _round_or_none(None if a0 is None or a1 is None else a1 - a0),
                    "available": a0 is not None and a1 is not None,
                }
            )
        deltas = [row["A1_minus_A0"] for row in per_seed if row["A1_minus_A0"] is not None]
        anchors.append(
            {
                "fraction_of_target": fraction,
                "target_cuda_seconds": target_time,
                "per_seed": per_seed,
                "mean_A1_minus_A0": mean(deltas) if deltas else None,
                "available_seed_count": len(deltas),
            }
        )

    aucs: list[dict[str, Any]] = []
    for seed in SEEDS:
        a0_points = curves["A0"][seed]
        a1_points = curves["A1"][seed]
        common_start = max(a0_points[0]["cuda_seconds"], a1_points[0]["cuda_seconds"])
        common_end = min(
            target,
            arms["A0"][seed]["cuda_seconds"],
            arms["A1"][seed]["cuda_seconds"],
        )
        if common_end <= common_start:
            aucs.append(
                {
                    "seed": seed,
                    "valid": False,
                    "common_start_cuda_seconds": common_start,
                    "common_end_cuda_seconds": common_end,
                }
            )
            continue
        a0_auc = _integral(a0_points, common_start, common_end)
        a1_auc = _integral(a1_points, common_start, common_end)
        interval = common_end - common_start
        aucs.append(
            {
                "seed": seed,
                "valid": True,
                "common_start_cuda_seconds": common_start,
                "common_end_cuda_seconds": common_end,
                "interval_cuda_seconds": interval,
                "A0_auc_loss_seconds": a0_auc,
                "A1_auc_loss_seconds": a1_auc,
                "A1_minus_A0_auc_loss_seconds": a1_auc - a0_auc,
                "A0_normalized_auc": a0_auc / interval,
                "A1_normalized_auc": a1_auc / interval,
                "A1_minus_A0_normalized_auc": (a1_auc - a0_auc) / interval,
            }
        )
    valid_aucs = [row for row in aucs if row["valid"]]
    normalized_deltas = [row["A1_minus_A0_normalized_auc"] for row in valid_aucs]
    raw_deltas = [row["A1_minus_A0_auc_loss_seconds"] for row in valid_aucs]

    final_deltas = [
        arms["A1"][seed]["validation_loss"] - arms["A0"][seed]["validation_loss"]
        for seed in SEEDS
    ]
    return {
        "target_cuda_seconds_per_arm": target,
        "cuda_tolerance_fraction": 0.05,
        "arms": arms,
        "anchors": anchors,
        "auc": {
            "per_seed": aucs,
            "valid_seed_count": len(valid_aucs),
            "mean_A1_minus_A0_auc_loss_seconds": mean(raw_deltas) if raw_deltas else None,
            "sample_sd_A1_minus_A0_auc_loss_seconds": stdev(raw_deltas) if len(raw_deltas) > 1 else None,
            "mean_A1_minus_A0_normalized_auc": mean(normalized_deltas) if normalized_deltas else None,
            "sample_sd_A1_minus_A0_normalized_auc": stdev(normalized_deltas) if len(normalized_deltas) > 1 else None,
        },
        "aggregate": {
            "final_deltas": final_deltas,
            "mean_final_A1_minus_A0": mean(final_deltas),
            "sample_sd_final_A1_minus_A0": stdev(final_deltas),
            "min_final_A1_minus_A0": min(final_deltas),
            "max_final_A1_minus_A0": max(final_deltas),
            "seeds_meeting_advantage_threshold": sum(delta <= -0.05 for delta in final_deltas),
            "seeds_tested": len(final_deltas),
            "all_cuda_budgets_within_tolerance": all(
                abs(arms[architecture][seed]["cuda_deviation_percent"]) <= 5.0
                for architecture in ("A0", "A1")
                for seed in SEEDS
            ),
            "numerical_instability_reported": any(
                arms[architecture][seed]["numerical_instability"]
                for architecture in ("A0", "A1")
                for seed in SEEDS
            ),
            "failures": [
                arms[architecture][seed]["failure"]
                for architecture in ("A0", "A1")
                for seed in SEEDS
                if arms[architecture][seed]["failure"] is not None
            ],
            "checkpoint_integrity_failures": sum(
                not arms[architecture][seed]["checkpoint"]["complete"]
                for architecture in ("A0", "A1")
                for seed in SEEDS
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/runs"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    budgets = {budget: analyze_budget(args.root, budget) for budget in ("1x", "4x", "8x")}
    mean_deltas = {budget: budgets[budget]["aggregate"]["mean_final_A1_minus_A0"] for budget in budgets}
    previous = mean_deltas["4x"]
    current = mean_deltas["8x"]
    if current < previous:
        trend = "widens"
    elif current > previous:
        trend = "narrows"
    else:
        trend = "approximately stable"
    result = {
        "status": "completed",
        "analysis_method": {
            "validation_curve": "piecewise-linear interpolation between observed validation points indexed by cumulative CUDA-event seconds",
            "anchor_fractions": list(ANCHOR_FRACTIONS),
            "auc": "trapezoidal integration of the same piecewise-linear curves over the common observed interval; no extrapolation",
            "common_interval": "max(first validation times) to min(planned target, both actual totals)",
            "final_delta": "reported final validation loss A1 minus A0 at each completed arm endpoint",
        },
        "budgets": budgets,
        "cross_budget_summary": {
            "mean_final_A1_minus_A0": mean_deltas,
            "mean_delta_change_1x_to_4x": mean_deltas["4x"] - mean_deltas["1x"],
            "mean_delta_change_4x_to_8x": mean_deltas["8x"] - mean_deltas["4x"],
            "classification_4x_to_8x": trend,
            "asymptotic_crossover_fit": "not performed",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
