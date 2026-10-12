#!/usr/bin/env python3
"""Run frozen representation probes on the relational-curriculum checkpoints.

This is a read-only wrapper around the existing layerwise forensic analysis.
It verifies the curriculum source artifacts, patches only the forensic
module's configuration lookup to the curriculum config, and writes a
machine-readable aggregate. No model or checkpoint is modified.
"""

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

from braidformer.training.integrity import sha256_file, verify_checkpoint
from scripts import analyze_layerwise_representation_forensics as layerwise


PLAN_PATH = "docs/A0_A1_RELATIONAL_CURRICULUM_REPRESENTATION_PROBE_PLAN.md"
SOURCE_RESULT_PATH = "results/a0_a1_relational_curriculum_v1.json"
SOURCE_RESULT_SHA256 = "66f68cfb3a2f84f5e2193dfcdd606c9c5ef9e8e111aee3e99254ce5c975d8d71"
SOURCE_PROTOCOL_COMMIT = "30488e4f6b95994f02e15de9fab5710ce9af2aa4"
SOURCE_PROTOCOL_SHA256 = "eb3e172d6057d070ad4d5c6a64529246e9831c688f7a7d609db6abc5942696e5"
SOURCE_RUNNER_SHA256 = "56e0d0f7361b86c1d0fb3deef031cf8203c75fabd4fd764e943f131f9b7062dd"
EXPOSURE_SHA256 = "fbbc2ade9f57566ecc18a1752b828880d2218b7ff42692638c6ecf74ea4e4d78"
CURRICULUM_CONFIGS = {
    "A0": "configs/braidnano_a0_relational_curriculum_v1.json",
    "A1": "configs/braidnano_a1_relational_curriculum_v1.json",
}
CURRICULUM_CONFIG_SHA256 = {
    "A0": "39ed06dc15a4d8260dd9074e3816f0d08ac12d81edf89a7453ea942e511ded01",
    "A1": "cf4c8b387c9c9d9bdc01df66382514131ac5953e85c8bd8776ff66e1b988d1fa",
}
EXPECTED_PARAMS = {"A0": 41_866_240, "A1": 41_870_336}
TARGET_LENGTHS = (2, 4, 8)
EXPECTED_TOKENS = 928_000
EXPECTED_STEPS = 2_000
EXPECTED_TOKENS_PER_STEP = 464
EXPECTED_SCRIPT_SHA256 = SOURCE_RUNNER_SHA256
EXPECTED_CHECKPOINTS = {
    ("A1", "staged"): "0b6ba5eff90ff477a0dc8dc810e4bcfa1b396f2b31f5b2a72c100c5645eb5e8b",
    ("A1", "shuffled"): "be7a71017ea6ea88dbbb99476611e4bfef98c398b3788e74d2a57fd40a5fec8b",
    ("A0", "staged"): "8c15b235dd5329ef349c64e96f24476dcfdaa904cdf4e654f3ad92ca6427c105",
    ("A0", "shuffled"): "e6c998e3d90acb5b102ddf4f1f3dd4cfb892400f90543e798001fe814126ebff",
}
EXPECTED_RUN_DIRS = {
    (architecture, arm): REPO_ROOT
    / "results"
    / "runs"
    / f"relational_curriculum_v1_{architecture.lower()}_{arm}"
    for architecture in ("A0", "A1")
    for arm in ("staged", "shuffled")
}
OUTPUT_VERSION = "relational_curriculum_representation_probes_v1"


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
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def _verify_summary(
    architecture: str,
    arm: str,
    source: dict[str, Any],
) -> dict[str, Any]:
    summary_path = REPO_ROOT / "results" / "runs" / (
        f"relational_curriculum_v1_{architecture.lower()}_{arm}"
    ) / "summary.json"
    summary = _load_json(summary_path)
    expected = {
        "status": "completed",
        "architecture": architecture,
        "arm": arm,
        "seed": 17,
        "parameter_count": EXPECTED_PARAMS[architecture],
        "config_path": CURRICULUM_CONFIGS[architecture],
        "config_sha256": CURRICULUM_CONFIG_SHA256[architecture],
        "protocol_commit": SOURCE_PROTOCOL_COMMIT,
        "protocol_sha256": SOURCE_PROTOCOL_SHA256,
        "script_sha256": EXPECTED_SCRIPT_SHA256,
        "tokens_processed": EXPECTED_TOKENS,
        "optimizer_steps": EXPECTED_STEPS,
        "tokens_per_step": EXPECTED_TOKENS_PER_STEP,
        "order_independent_exposure_sha256": EXPOSURE_SHA256,
    }
    for field, expected_value in expected.items():
        if summary.get(field) != expected_value:
            raise ValueError(
                f"{summary_path}: {field}={summary.get(field)!r}, "
                f"expected {expected_value!r}"
            )

    config_path = REPO_ROOT / CURRICULUM_CONFIGS[architecture]
    if sha256_file(config_path) != CURRICULUM_CONFIG_SHA256[architecture]:
        raise ValueError(f"configuration hash mismatch: {config_path}")

    expected_checkpoint = EXPECTED_CHECKPOINTS[(architecture, arm)]
    checkpoint_path = Path(summary.get("checkpoint", ""))
    if checkpoint_path.resolve() != (EXPECTED_RUN_DIRS[(architecture, arm)] / checkpoint_path.name).resolve():
        raise ValueError(f"checkpoint path is outside the expected run: {checkpoint_path}")
    if summary.get("checkpoint_sha256") != expected_checkpoint:
        raise ValueError(f"summary checkpoint hash mismatch: {summary_path}")

    checkpoint_record = verify_checkpoint(checkpoint_path)
    if checkpoint_record.get("checkpoint_sha256") != expected_checkpoint:
        raise ValueError(f"checkpoint sidecar hash mismatch: {checkpoint_path}")
    for field, expected_value in {
        "config_sha256": CURRICULUM_CONFIG_SHA256[architecture],
        "seed": 17,
        "step": EXPECTED_STEPS,
        "tokens_processed": EXPECTED_TOKENS,
        "parameter_count": EXPECTED_PARAMS[architecture],
        "protocol_commit": SOURCE_PROTOCOL_COMMIT,
    }.items():
        if checkpoint_record.get(field) != expected_value:
            raise ValueError(
                f"checkpoint sidecar mismatch {checkpoint_path}: "
                f"{field}={checkpoint_record.get(field)!r}"
            )

    aggregate_arm = source["arms"].get(f"{architecture.lower()}_{arm}")
    if not isinstance(aggregate_arm, dict):
        raise ValueError(f"missing aggregate arm: {architecture.lower()}_{arm}")
    for field in (
        "architecture",
        "arm",
        "checkpoint_sha256",
        "config_path",
        "config_sha256",
        "protocol_commit",
        "seed",
        "parameter_count",
        "tokens_processed",
        "optimizer_steps",
    ):
        if aggregate_arm.get(field) != summary.get(field):
            raise ValueError(f"aggregate/summary mismatch for {architecture} {arm}: {field}")

    _assert_finite(summary, f"summary[{architecture},{arm}]")
    return {
        "architecture": architecture,
        "arm": arm,
        "run_dir": str(EXPECTED_RUN_DIRS[(architecture, arm)]),
        "summary_path": str(summary_path.relative_to(REPO_ROOT)),
        "summary_sha256": sha256_file(summary_path),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": expected_checkpoint,
        },
        "checkpoint_sha256": expected_checkpoint,
        "config_path": CURRICULUM_CONFIGS[architecture],
        "config_sha256": CURRICULUM_CONFIG_SHA256[architecture],
        "seed": 17,
        "parameter_count": EXPECTED_PARAMS[architecture],
        "source_summary": summary,
    }


def _summary_rows_with_arm(run_result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = layerwise._summary_rows({"runs": [run_result]})
    for row in rows:
        row["arm"] = run_result["arm"]
        row["checkpoint_sha256"] = run_result["source_checkpoint_sha256"]
    return rows


def _comparisons(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (row["architecture"], row["target_length"], row["path"], row["arm"]): row
        for row in summary_rows
    }
    metrics = (
        "mean_key_heldout_fit_top1",
        "mean_value_heldout_fit_top1",
        "mean_value_fixed_pre_query_transfer_top1",
        "target_position_train_fit_heldout_accuracy",
    )
    comparisons = []
    for architecture in ("A1", "A0"):
        for target_length in TARGET_LENGTHS:
            paths = sorted(
                row["path"]
                for row in summary_rows
                if row["architecture"] == architecture
                and row["target_length"] == target_length
            )
            for path in paths:
                staged = by_key[(architecture, target_length, path, "staged")]
                shuffled = by_key[(architecture, target_length, path, "shuffled")]
                for metric in metrics:
                    staged_value = float(staged[metric])
                    shuffled_value = float(shuffled[metric])
                    comparisons.append(
                        {
                            "architecture": architecture,
                            "target_length": target_length,
                            "path": path,
                            "metric": metric,
                            "staged": staged_value,
                            "shuffled": shuffled_value,
                            "staged_minus_shuffled": staged_value - shuffled_value,
                        }
                    )
    return comparisons


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--analysis-commit", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results/a0_a1_relational_curriculum_representation_probes.json",
    )
    args = parser.parse_args()

    if _current_commit() != args.analysis_commit:
        raise RuntimeError(
            f"analysis commit mismatch: current {_current_commit()}, "
            f"expected {args.analysis_commit}"
        )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    source_path = REPO_ROOT / SOURCE_RESULT_PATH
    if sha256_file(source_path) != SOURCE_RESULT_SHA256:
        raise ValueError(f"source aggregate hash mismatch: {source_path}")
    source = _load_json(source_path)
    if source.get("status") != "PASS":
        raise ValueError(f"source aggregate is not PASS: {source.get('status')!r}")
    if source.get("protocol_commit") != SOURCE_PROTOCOL_COMMIT:
        raise ValueError("source protocol commit mismatch")
    if source.get("order_independent_exposure_sha256") != EXPOSURE_SHA256:
        raise ValueError("source exposure hash mismatch")

    plan_path = REPO_ROOT / PLAN_PATH
    if not plan_path.is_file():
        raise FileNotFoundError(plan_path)
    device = torch.device(args.device)

    source_runs: dict[tuple[str, str], dict[str, Any]] = {}
    for architecture, arm in (
        ("A1", "staged"),
        ("A1", "shuffled"),
        ("A0", "staged"),
        ("A0", "shuffled"),
    ):
        source_runs[(architecture, arm)] = _verify_summary(architecture, arm, source)

    original_configs = layerwise.CONFIGS
    layerwise.CONFIGS = {
        (target_length, architecture): CURRICULUM_CONFIGS[architecture]
        for target_length in TARGET_LENGTHS
        for architecture in ("A0", "A1")
    }
    try:
        runs: list[dict[str, Any]] = []
        for architecture, arm in (
            ("A1", "staged"),
            ("A1", "shuffled"),
            ("A0", "staged"),
            ("A0", "shuffled"),
        ):
            source_run = source_runs[(architecture, arm)]
            for target_length in TARGET_LENGTHS:
                analysis_run = layerwise._analyze_model(
                    architecture,
                    target_length,
                    source_run,
                    device,
                )
                analysis_run.update(
                    {
                        "architecture": architecture,
                        "arm": arm,
                        "target_length": target_length,
                        "source_summary_path": source_run["summary_path"],
                        "source_summary_sha256": source_run["summary_sha256"],
                        "source_checkpoint_sha256": source_run["checkpoint_sha256"],
                        "config_path": source_run["config_path"],
                        "config_sha256": source_run["config_sha256"],
                    }
                )
                runs.append(analysis_run)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    finally:
        layerwise.CONFIGS = original_configs

    summary_rows: list[dict[str, Any]] = []
    for run in runs:
        summary_rows.extend(_summary_rows_with_arm(run))

    result: dict[str, Any] = {
        "status": "complete",
        "version": OUTPUT_VERSION,
        "read_only": True,
        "analysis_commit": args.analysis_commit,
        "analysis_device": str(device),
        "analysis_plan_path": PLAN_PATH,
        "analysis_plan_sha256": sha256_file(plan_path),
        "forensic_script_path": "scripts/analyze_layerwise_representation_forensics.py",
        "forensic_script_sha256": sha256_file(
            REPO_ROOT / "scripts/analyze_layerwise_representation_forensics.py"
        ),
        "source_result_path": SOURCE_RESULT_PATH,
        "source_result_sha256": SOURCE_RESULT_SHA256,
        "source_protocol_commit": SOURCE_PROTOCOL_COMMIT,
        "source_protocol_sha256": SOURCE_PROTOCOL_SHA256,
        "source_runner_sha256": SOURCE_RUNNER_SHA256,
        "order_independent_exposure_sha256": EXPOSURE_SHA256,
        "seed": 17,
        "value_count": layerwise.VALUE_COUNT,
        "association_count": layerwise.ASSOCIATION_COUNT,
        "distractor_distance": layerwise.DISTRACTOR_DISTANCE,
        "target_lengths": list(TARGET_LENGTHS),
        "architectures": ["A1", "A0"],
        "arms": ["staged", "shuffled"],
        "runs": runs,
        "summary_rows": summary_rows,
        "comparisons": _comparisons(summary_rows),
        "source_checkpoints": {
            f"{architecture.lower()}_{arm}": digest
            for (architecture, arm), digest in EXPECTED_CHECKPOINTS.items()
        },
    }
    _assert_finite(result, "result")
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "runs": len(runs), "status": result["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
