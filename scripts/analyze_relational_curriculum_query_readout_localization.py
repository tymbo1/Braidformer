#!/usr/bin/env python3
"""Run frozen query/readout localization on relational-curriculum checkpoints."""

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
from scripts import analyze_query_readout_localization as query
from scripts import analyze_relational_curriculum_representation_probes as curriculum


PLAN_PATH = "docs/A0_A1_RELATIONAL_CURRICULUM_QUERY_READOUT_LOCALIZATION_PLAN.md"
SOURCE_RESULT_PATH = "results/a0_a1_relational_curriculum_v1.json"
SOURCE_RESULT_SHA256 = "66f68cfb3a2f84f5e2193dfcdd606c9c5ef9e8e111aee3e99254ce5c975d8d71"
REPRESENTATION_RESULT_PATH = "results/a0_a1_relational_curriculum_representation_probes.json"
REPRESENTATION_RESULT_SHA256 = "b36749b40024b1f84b21fddbe825f60105f28cfe4bf99b21673d649f347268b3"
REPRESENTATION_ANALYSIS_COMMIT = "19a2ef68db8ab6f43b725167dc4b63b961091ddc"
QUERY_SCRIPT_PATH = "scripts/analyze_query_readout_localization.py"
QUERY_SCRIPT_SHA256 = "4e9788c1f2598a278d37e36c93df35ce86d03f6ada17094c63c75b7a2e6699d4"
TARGET_LENGTHS = (2, 4, 8)
OUTPUT_VERSION = "relational_curriculum_query_readout_localization_v1"


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


def _compatibility_make_examples(
    original: Any,
) -> Any:
    """Normalize the old query script's duplicated distance argument."""

    def make_examples(
        base_seed: int,
        split: str,
        value_count: int,
        association_count: int,
        distractor_distance: int,
        *remaining: int,
    ) -> list[Any]:
        if len(remaining) == 1:
            target_length = remaining[0]
        elif len(remaining) == 2 and remaining[0] == distractor_distance:
            target_length = remaining[1]
        else:
            raise ValueError(f"unexpected evaluation-example arguments: {remaining!r}")
        return original(
            base_seed,
            split,
            value_count,
            association_count,
            distractor_distance,
            target_length,
        )

    return make_examples


def _summary_rows_with_arm(run_result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = query._summary_rows({"runs": [run_result]})
    for row in rows:
        row["arm"] = run_result["arm"]
        row["checkpoint_sha256"] = run_result["source_checkpoint_sha256"]
    return rows


def _comparisons(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (row["architecture"], row["target_length"], row["path"], row["target_position"], row["arm"]): row
        for row in summary_rows
    }
    metrics = (
        "native_value_top1",
        "fixed_pre_query_pre_top1",
        "fixed_pre_query_post_top1",
        "fixed_pre_query_target_top1",
        "site_specific_target_top1",
    )
    comparisons = []
    for architecture in ("A1", "A0"):
        for target_length in TARGET_LENGTHS:
            paths = sorted(
                {
                    row["path"]
                    for row in summary_rows
                    if row["architecture"] == architecture
                    and row["target_length"] == target_length
                }
            )
            positions = range(1, target_length + 1)
            for path in paths:
                for position in positions:
                    staged = by_key[(architecture, target_length, path, position, "staged")]
                    shuffled = by_key[(architecture, target_length, path, position, "shuffled")]
                    for metric in metrics:
                        staged_value = float(staged[metric])
                        shuffled_value = float(shuffled[metric])
                        comparisons.append(
                            {
                                "architecture": architecture,
                                "target_length": target_length,
                                "path": path,
                                "target_position": position,
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
        default=REPO_ROOT / "results/a0_a1_relational_curriculum_query_readout_localization.json",
    )
    args = parser.parse_args()
    current_commit = _current_commit()
    if current_commit != args.analysis_commit:
        raise RuntimeError(
            f"analysis commit mismatch: current {current_commit}, expected {args.analysis_commit}"
        )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    source_path = REPO_ROOT / SOURCE_RESULT_PATH
    if sha256_file(source_path) != SOURCE_RESULT_SHA256:
        raise ValueError(f"source aggregate hash mismatch: {source_path}")
    source = _load_json(source_path)
    if source.get("status") != "PASS":
        raise ValueError(f"source aggregate is not PASS: {source.get('status')!r}")

    representation_path = REPO_ROOT / REPRESENTATION_RESULT_PATH
    if sha256_file(representation_path) != REPRESENTATION_RESULT_SHA256:
        raise ValueError(f"representation aggregate hash mismatch: {representation_path}")
    representation = _load_json(representation_path)
    if representation.get("status") != "complete" or representation.get("read_only") is not True:
        raise ValueError("preceding representation aggregate is not a complete read-only result")
    if representation.get("analysis_commit") != REPRESENTATION_ANALYSIS_COMMIT:
        raise ValueError("preceding representation analysis commit mismatch")

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
        source_runs[(architecture, arm)] = curriculum._verify_summary(architecture, arm, source)

    original_configs = query.CONFIGS
    original_examples = query.make_evaluation_examples
    query.CONFIGS = {
        (target_length, architecture): curriculum.CURRICULUM_CONFIGS[architecture]
        for target_length in TARGET_LENGTHS
        for architecture in ("A0", "A1")
    }
    query.make_evaluation_examples = _compatibility_make_examples(original_examples)
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
                analysis_run = query._run_analysis(
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
        query.CONFIGS = original_configs
        query.make_evaluation_examples = original_examples

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
        "query_script_path": QUERY_SCRIPT_PATH,
        "query_script_sha256": sha256_file(REPO_ROOT / QUERY_SCRIPT_PATH),
        "source_result_path": SOURCE_RESULT_PATH,
        "source_result_sha256": SOURCE_RESULT_SHA256,
        "preceding_representation_result_path": REPRESENTATION_RESULT_PATH,
        "preceding_representation_result_sha256": REPRESENTATION_RESULT_SHA256,
        "preceding_representation_analysis_commit": REPRESENTATION_ANALYSIS_COMMIT,
        "source_protocol_commit": curriculum.SOURCE_PROTOCOL_COMMIT,
        "source_protocol_sha256": curriculum.SOURCE_PROTOCOL_SHA256,
        "source_runner_sha256": curriculum.SOURCE_RUNNER_SHA256,
        "order_independent_exposure_sha256": curriculum.EXPOSURE_SHA256,
        "compatibility_shim": "normalize duplicated distractor-distance argument in frozen query script call",
        "seed": 17,
        "value_count": query.VALUE_COUNT,
        "association_count": query.ASSOCIATION_COUNT,
        "distractor_distance": query.DISTRACTOR_DISTANCE,
        "target_lengths": list(TARGET_LENGTHS),
        "architectures": ["A1", "A0"],
        "arms": ["staged", "shuffled"],
        "heldout_examples": query.FIXED_EXAMPLES,
        "batch_size": query.BATCH_SIZE,
        "fold_count": query.FOLD_COUNT,
        "fold_assignment": "heldout_index_modulo_fold_count",
        "runs": runs,
        "summary_rows": summary_rows,
        "comparisons": _comparisons(summary_rows),
        "source_checkpoints": {
            f"{architecture.lower()}_{arm}": digest
            for (architecture, arm), digest in curriculum.EXPECTED_CHECKPOINTS.items()
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
