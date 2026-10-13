#!/usr/bin/env python3
"""Run a reversible layer-7 state-replay intervention on curriculum A1."""

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
from scripts import analyze_relational_curriculum_representation_probes as curriculum
from scripts import analyze_state_replay_intervention as replay


PLAN_PATH = "docs/A0_A1_RELATIONAL_CURRICULUM_STATE_REPLAY_INTERVENTION_PLAN.md"
SOURCE_RESULT_PATH = "results/a0_a1_relational_curriculum_v1.json"
SOURCE_RESULT_SHA256 = "66f68cfb3a2f84f5e2193dfcdd606c9c5ef9e8e111aee3e99254ce5c975d8d71"
REPRESENTATION_RESULT_PATH = "results/a0_a1_relational_curriculum_representation_probes.json"
REPRESENTATION_RESULT_SHA256 = "b36749b40024b1f84b21fddbe825f60105f28cfe4bf99b21673d649f347268b3"
QUERY_RESULT_PATH = "results/a0_a1_relational_curriculum_query_readout_localization.json"
QUERY_RESULT_SHA256 = "73dc7df64d9d57099d3f31ede2628ba673ed8cf716249fbf17b83164a8d77e50"
REPLAY_SCRIPT_PATH = "scripts/analyze_state_replay_intervention.py"
REPLAY_SCRIPT_SHA256 = "d364b06f88338e92c2c600700079f0656d90108672e21e9205339fdb6cc43d3e"
TARGET_LENGTHS = (2, 4, 8)
OUTPUT_VERSION = "relational_curriculum_state_replay_intervention_v1"


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


def _verify_preceding_result(path: str, digest: str, required: dict[str, Any]) -> dict[str, Any]:
    result_path = REPO_ROOT / path
    if sha256_file(result_path) != digest:
        raise ValueError(f"preceding result hash mismatch: {result_path}")
    result = _load_json(result_path)
    for field, expected in required.items():
        if result.get(field) != expected:
            raise ValueError(f"preceding result mismatch {path}: {field}")
    _assert_finite(result, path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--analysis-commit", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results/a0_a1_relational_curriculum_state_replay_intervention.json",
    )
    args = parser.parse_args()
    current_commit = _current_commit()
    if current_commit != args.analysis_commit:
        raise RuntimeError(
            f"analysis commit mismatch: current {current_commit}, expected {args.analysis_commit}"
        )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    source = _verify_preceding_result(
        SOURCE_RESULT_PATH,
        SOURCE_RESULT_SHA256,
        {"status": "PASS", "protocol_commit": curriculum.SOURCE_PROTOCOL_COMMIT},
    )
    representation = _verify_preceding_result(
        REPRESENTATION_RESULT_PATH,
        REPRESENTATION_RESULT_SHA256,
        {"status": "complete", "read_only": True},
    )
    query_result = _verify_preceding_result(
        QUERY_RESULT_PATH,
        QUERY_RESULT_SHA256,
        {"status": "complete", "read_only": True},
    )
    plan_path = REPO_ROOT / PLAN_PATH
    if not plan_path.is_file():
        raise FileNotFoundError(plan_path)
    if sha256_file(REPO_ROOT / REPLAY_SCRIPT_PATH) != REPLAY_SCRIPT_SHA256:
        raise ValueError("frozen replay script hash mismatch")
    device = torch.device(args.device)

    source_runs: dict[tuple[str, str], dict[str, Any]] = {}
    for architecture, arm in (
        ("A1", "staged"),
        ("A1", "shuffled"),
        ("A0", "staged"),
        ("A0", "shuffled"),
    ):
        source_runs[(architecture, arm)] = curriculum._verify_summary(architecture, arm, source)

    original_configs = replay.CONFIGS
    replay.CONFIGS = {
        (target_length, "A1"): curriculum.CURRICULUM_CONFIGS["A1"]
        for target_length in TARGET_LENGTHS
    }
    try:
        runs: list[dict[str, Any]] = []
        for arm in ("staged", "shuffled"):
            for target_length in TARGET_LENGTHS:
                source_run = source_runs[("A1", arm)]
                model = replay._load_model(target_length, source_run, device)
                before_digest = replay._parameter_digest(model)
                examples = replay.make_evaluation_examples(
                    replay.SEED,
                    "heldout",
                    replay.VALUE_COUNT,
                    replay.ASSOCIATION_COUNT,
                    replay.DISTRACTOR_DISTANCE,
                    target_length,
                )
                run_result = replay._collect_run(model, examples, target_length, device)
                after_digest = replay._parameter_digest(model)
                if before_digest != after_digest:
                    raise RuntimeError("functional intervention changed model parameters")
                run_result.update(
                    {
                        "architecture": "A1",
                        "arm": arm,
                        "target_length": target_length,
                        "checkpoint": source_run["checkpoint"],
                        "source_run": source_run["run_dir"],
                        "source_summary_path": source_run["summary_path"],
                        "source_summary_sha256": source_run["summary_sha256"],
                        "source_checkpoint_sha256": source_run["checkpoint_sha256"],
                        "config_path": source_run["config_path"],
                        "config_sha256": source_run["config_sha256"],
                        "parameter_digest_before": before_digest,
                        "parameter_digest_after": after_digest,
                    }
                )
                runs.append(run_result)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    finally:
        replay.CONFIGS = original_configs

    result: dict[str, Any] = {
        "status": "complete",
        "version": OUTPUT_VERSION,
        "read_only": True,
        "analysis_commit": args.analysis_commit,
        "analysis_device": str(device),
        "analysis_plan_path": PLAN_PATH,
        "analysis_plan_sha256": sha256_file(plan_path),
        "replay_script_path": REPLAY_SCRIPT_PATH,
        "replay_script_sha256": REPLAY_SCRIPT_SHA256,
        "source_result_path": SOURCE_RESULT_PATH,
        "source_result_sha256": SOURCE_RESULT_SHA256,
        "preceding_representation_result_path": REPRESENTATION_RESULT_PATH,
        "preceding_representation_result_sha256": REPRESENTATION_RESULT_SHA256,
        "preceding_query_result_path": QUERY_RESULT_PATH,
        "preceding_query_result_sha256": QUERY_RESULT_SHA256,
        "source_protocol_commit": curriculum.SOURCE_PROTOCOL_COMMIT,
        "source_protocol_sha256": curriculum.SOURCE_PROTOCOL_SHA256,
        "source_runner_sha256": curriculum.SOURCE_RUNNER_SHA256,
        "order_independent_exposure_sha256": curriculum.EXPOSURE_SHA256,
        "seed": replay.SEED,
        "value_count": replay.VALUE_COUNT,
        "association_count": replay.ASSOCIATION_COUNT,
        "distractor_distance": replay.DISTRACTOR_DISTANCE,
        "target_lengths": list(TARGET_LENGTHS),
        "verified_architectures": ["A1", "A0"],
        "intervened_architecture": "A1",
        "arms": ["staged", "shuffled"],
        "heldout_examples": replay.FIXED_EXAMPLES,
        "patch_layer": replay.PATCH_LAYER,
        "intervention": "copy_layer7_recurrent_state_from_each_stored_value_to_matching_target_output",
        "verified_source_runs": [
            {
                "architecture": architecture,
                "arm": arm,
                "checkpoint": source_runs[(architecture, arm)]["checkpoint"],
                "checkpoint_sha256": source_runs[(architecture, arm)]["checkpoint_sha256"],
            }
            for architecture, arm in (
                ("A1", "staged"),
                ("A1", "shuffled"),
                ("A0", "staged"),
                ("A0", "shuffled"),
            )
        ],
        "preceding_result_status": {
            "representation": representation["status"],
            "query_readout": query_result["status"],
        },
        "runs": runs,
    }
    _assert_finite(result, "result")
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "runs": len(runs), "status": result["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
