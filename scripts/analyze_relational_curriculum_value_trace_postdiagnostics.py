#!/usr/bin/env python3
"""Rerun frozen layerwise and query/readout diagnostics on value-trace A1."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from braidformer.training.integrity import sha256_file
from scripts import analyze_layerwise_representation_forensics as layerwise
from scripts import analyze_query_readout_localization as query
from scripts import analyze_relational_curriculum_query_readout_localization as query_wrapper
from scripts import run_relational_curriculum_value_trace_intervention as value_trace


PLAN_PATH = "docs/A0_A1_RELATIONAL_CURRICULUM_VALUE_TRACE_POSTDIAGNOSTIC_PLAN.md"
VALUE_TRACE_RESULT_PATH = "results/a0_a1_relational_curriculum_value_trace_intervention.json"
VALUE_TRACE_RESULT_SHA256 = "13dbace59f46a16ff6fc30e00532500b582f5a1ad5a8df8210297604e2532112"
LAYERWISE_CONTROL_PATH = "results/a0_a1_relational_curriculum_representation_probes.json"
LAYERWISE_CONTROL_SHA256 = "b36749b40024b1f84b21fddbe825f60105f28cfe4bf99b21673d649f347268b3"
QUERY_CONTROL_PATH = "results/a0_a1_relational_curriculum_query_readout_localization.json"
QUERY_CONTROL_SHA256 = "73dc7df64d9d57099d3f31ede2628ba673ed8cf716249fbf17b83164a8d77e50"
TRAINING_COMMIT = "fdcd032d640053084b84c204b8e737fe72242464"
TARGET_LENGTHS = (2, 4, 8)
ARMS = ("staged", "shuffled")
OUTPUT_VERSION = "relational_curriculum_value_trace_postdiagnostics_v1"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _assert_finite(value: Any, label: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"non-finite value at {label}: {value!r}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite(item, f"{label}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite(item, f"{label}.{key}")
        return
    raise TypeError(f"unsupported value at {label}: {type(value).__name__}")


def _current_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def _rows_with_arm(run_result: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in rows:
        row["arm"] = run_result["arm"]
        row["checkpoint_sha256"] = run_result["source_checkpoint_sha256"]
    return rows


def _delta_rows(
    intervention_rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
    dimensions: tuple[str, ...],
    metrics: tuple[str, ...],
) -> list[dict[str, Any]]:
    intervention = {
        (row["arm"], *tuple(row[key] for key in dimensions)): row
        for row in intervention_rows
    }
    control = {
        (row["arm"], *tuple(row[key] for key in dimensions)): row
        for row in control_rows
        if row.get("architecture") == "A1"
    }
    deltas = []
    for key in sorted(intervention):
        if key not in control:
            raise ValueError(f"missing matched control row: {key}")
        arm = key[0]
        dimension_values = key[1:]
        for metric in metrics:
            actual = float(intervention[key][metric])
            baseline = float(control[key][metric])
            deltas.append(
                {
                    **{
                        dimension: value
                        for dimension, value in zip(dimensions, dimension_values)
                    },
                    "arm": arm,
                    "metric": metric,
                    "intervention": actual,
                    "control": baseline,
                    "intervention_minus_control": actual - baseline,
                }
            )
    return deltas


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--analysis-commit", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results/a0_a1_relational_curriculum_value_trace_postdiagnostics.json",
    )
    args = parser.parse_args()
    current_commit = _current_commit()
    if current_commit != args.analysis_commit:
        raise RuntimeError(
            f"analysis commit mismatch: current {current_commit}, expected {args.analysis_commit}"
        )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    value_trace_path = REPO_ROOT / VALUE_TRACE_RESULT_PATH
    if sha256_file(value_trace_path) != VALUE_TRACE_RESULT_SHA256:
        raise ValueError("value-trace result hash mismatch")
    value_trace_result = _load_json(value_trace_path)
    if value_trace_result.get("status") != "complete":
        raise ValueError("value-trace result is not complete")
    if value_trace_result.get("analysis_commit") != TRAINING_COMMIT:
        raise ValueError("value-trace training commit mismatch")

    layerwise_control_path = REPO_ROOT / LAYERWISE_CONTROL_PATH
    if sha256_file(layerwise_control_path) != LAYERWISE_CONTROL_SHA256:
        raise ValueError("layerwise control aggregate hash mismatch")
    layerwise_control = _load_json(layerwise_control_path)
    query_control_path = REPO_ROOT / QUERY_CONTROL_PATH
    if sha256_file(query_control_path) != QUERY_CONTROL_SHA256:
        raise ValueError("query control aggregate hash mismatch")
    query_control = _load_json(query_control_path)
    plan_path = REPO_ROOT / PLAN_PATH
    if not plan_path.is_file():
        raise FileNotFoundError(plan_path)
    device = torch.device(args.device)

    source_runs: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        raw_summary = _load_json(
            REPO_ROOT
            / "results"
            / "runs"
            / f"relational_curriculum_v1_a1_value_trace_{arm}"
            / "summary.json"
        )
        verified = value_trace._verify_intervention_summary(raw_summary, arm, TRAINING_COMMIT)
        source_runs[arm] = {
            "arm": arm,
            "run_dir": verified["run_dir"],
            "summary_path": verified["summary_path"],
            "summary_sha256": sha256_file(Path(verified["summary_path"])),
            "checkpoint": {
                "path": verified["checkpoint"],
                "sha256": verified["checkpoint_sha256"],
            },
            "source_checkpoint_sha256": verified["checkpoint_sha256"],
        }

    original_layerwise_configs = layerwise.CONFIGS
    original_query_configs = query.CONFIGS
    original_query_examples = query.make_evaluation_examples
    layerwise.CONFIGS = {
        (target_length, "A1"): value_trace.CONFIG_PATH
        for target_length in TARGET_LENGTHS
    }
    query.CONFIGS = {
        (target_length, "A1"): value_trace.CONFIG_PATH
        for target_length in TARGET_LENGTHS
    }
    query.make_evaluation_examples = query_wrapper._compatibility_make_examples(
        original_query_examples
    )
    try:
        layerwise_runs: list[dict[str, Any]] = []
        query_runs: list[dict[str, Any]] = []
        for arm in ARMS:
            source_run = source_runs[arm]
            for target_length in TARGET_LENGTHS:
                layerwise_result = layerwise._analyze_model("A1", target_length, source_run, device)
                layerwise_result.update(
                    {
                        "arm": arm,
                        "target_length": target_length,
                        "source_summary_path": source_run["summary_path"],
                        "source_summary_sha256": source_run["summary_sha256"],
                        "source_checkpoint_sha256": source_run["source_checkpoint_sha256"],
                    }
                )
                layerwise_runs.append(layerwise_result)

                query_result = query._run_analysis("A1", target_length, source_run, device)
                query_result.update(
                    {
                        "arm": arm,
                        "target_length": target_length,
                        "source_summary_path": source_run["summary_path"],
                        "source_summary_sha256": source_run["summary_sha256"],
                        "source_checkpoint_sha256": source_run["source_checkpoint_sha256"],
                    }
                )
                query_runs.append(query_result)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    finally:
        layerwise.CONFIGS = original_layerwise_configs
        query.CONFIGS = original_query_configs
        query.make_evaluation_examples = original_query_examples

    layerwise_rows: list[dict[str, Any]] = []
    for run in layerwise_runs:
        layerwise_rows.extend(
            _rows_with_arm(run, layerwise._summary_rows({"runs": [run]}))
        )
    query_rows: list[dict[str, Any]] = []
    for run in query_runs:
        query_rows.extend(_rows_with_arm(run, query._summary_rows({"runs": [run]})))

    layerwise_control_rows = layerwise_control["summary_rows"]
    query_control_rows = query_control["summary_rows"]
    layerwise_deltas = _delta_rows(
        layerwise_rows,
        layerwise_control_rows,
        ("target_length", "path"),
        (
            "mean_key_heldout_fit_top1",
            "mean_value_heldout_fit_top1",
            "mean_value_fixed_pre_query_transfer_top1",
            "target_position_train_fit_heldout_accuracy",
        ),
    )
    query_deltas = _delta_rows(
        query_rows,
        query_control_rows,
        ("target_length", "path", "target_position"),
        (
            "native_value_top1",
            "fixed_pre_query_pre_top1",
            "fixed_pre_query_post_top1",
            "fixed_pre_query_target_top1",
            "site_specific_target_top1",
        ),
    )

    result: dict[str, Any] = {
        "status": "complete",
        "version": OUTPUT_VERSION,
        "read_only": True,
        "analysis_commit": args.analysis_commit,
        "analysis_device": str(device),
        "analysis_plan_path": PLAN_PATH,
        "analysis_plan_sha256": sha256_file(plan_path),
        "value_trace_result_path": VALUE_TRACE_RESULT_PATH,
        "value_trace_result_sha256": VALUE_TRACE_RESULT_SHA256,
        "layerwise_control_path": LAYERWISE_CONTROL_PATH,
        "layerwise_control_sha256": LAYERWISE_CONTROL_SHA256,
        "query_control_path": QUERY_CONTROL_PATH,
        "query_control_sha256": QUERY_CONTROL_SHA256,
        "training_commit": TRAINING_COMMIT,
        "target_lengths": list(TARGET_LENGTHS),
        "arms": list(ARMS),
        "layerwise": {
            "runs": layerwise_runs,
            "summary_rows": layerwise_rows,
            "intervention_minus_control": layerwise_deltas,
        },
        "query_readout": {
            "runs": query_runs,
            "summary_rows": query_rows,
            "intervention_minus_control": query_deltas,
        },
    }
    _assert_finite(result, "result")
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "layerwise_runs": len(layerwise_runs), "query_runs": len(query_runs), "status": result["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
