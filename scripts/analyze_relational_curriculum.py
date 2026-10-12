#!/usr/bin/env python3
"""Aggregate and integrity-check the four relational curriculum arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from braidformer.training.integrity import sha256_file, verify_checkpoint


EXPECTED_KEYS = (
    "values128",
    "values256",
    "values1024",
    "values8000",
    "associations2_values8000",
    "associations4_values8000",
    "associations8_values8000",
    "distance4_values8000",
    "distance16_values8000",
    "distance64_values8000",
    "targetlength2_values8000",
    "targetlength4_values8000",
    "targetlength8_values8000",
)


def _load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"summary is not an object: {path}")
    if payload.get("status") != "completed":
        raise ValueError(f"run is not completed: {path}: {payload.get('status')!r}")
    evaluation = payload.get("final_evaluation")
    if not isinstance(evaluation, dict) or set(evaluation) != set(EXPECTED_KEYS):
        raise ValueError(f"final evaluation keys are incomplete: {path}")
    for key, metrics in evaluation.items():
        if not isinstance(metrics, dict):
            raise ValueError(f"metrics are not an object for {key}: {path}")
        for field in ("target_loss", "greedy_answer_token_accuracy", "greedy_exact_match"):
            value = metrics.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"non-finite {field} for {key}: {path}")
    checkpoint = payload.get("checkpoint")
    if not checkpoint:
        raise ValueError(f"completed run has no checkpoint: {path}")
    verify_checkpoint(REPO_ROOT / checkpoint if not Path(checkpoint).is_absolute() else checkpoint)
    return payload


def _relative(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def aggregate(paths: dict[str, Path], output: Path) -> dict[str, Any]:
    summaries = {label: _load_summary(path) for label, path in paths.items()}
    if set(summaries) != {"a0_staged", "a0_shuffled", "a1_staged", "a1_shuffled"}:
        raise ValueError("exactly four labelled arms are required")
    expected_exposure = {summary["order_independent_exposure_sha256"] for summary in summaries.values()}
    if len(expected_exposure) != 1:
        raise ValueError("order-independent exposure hash differs across arms")
    expected_seed = {summary["seed"] for summary in summaries.values()}
    if expected_seed != {17}:
        raise ValueError(f"seed mismatch: {expected_seed}")
    expected_budget = {
        (summary["optimizer_steps"], summary["tokens_processed"], summary["tokens_per_step"])
        for summary in summaries.values()
    }
    if expected_budget != {(2000, 928000, 464)}:
        raise ValueError(f"training budget mismatch: {expected_budget}")
    expected_protocol = {summary["protocol_commit"] for summary in summaries.values()}
    if len(expected_protocol) != 1:
        raise ValueError("protocol commit differs across arms")

    arms: dict[str, Any] = {}
    for label, summary in summaries.items():
        arms[label] = {
            "summary_path": _relative(paths[label]),
            "architecture": summary["architecture"],
            "arm": summary["arm"],
            "config_path": summary["config_path"],
            "config_sha256": summary["config_sha256"],
            "protocol_commit": summary["protocol_commit"],
            "protocol_sha256": summary["protocol_sha256"],
            "script_sha256": summary["script_sha256"],
            "seed": summary["seed"],
            "parameter_count": summary["parameter_count"],
            "optimizer_steps": summary["optimizer_steps"],
            "tokens_processed": summary["tokens_processed"],
            "tokens_per_step": summary["tokens_per_step"],
            "cuda_training_seconds": summary["cuda_training_seconds"],
            "training_wall_seconds": summary["training_wall_seconds"],
            "checkpoint": summary["checkpoint"],
            "checkpoint_sha256": summary["checkpoint_sha256"],
            "final_evaluation": summary["final_evaluation"],
        }

    comparisons: dict[str, Any] = {}
    for architecture in ("A0", "A1"):
        staged = summaries[f"a{architecture[1]}_staged"]["final_evaluation"]
        shuffled = summaries[f"a{architecture[1]}_shuffled"]["final_evaluation"]
        comparisons[architecture] = {
            key: {
                "staged_exact_match": staged[key]["greedy_exact_match"],
                "shuffled_exact_match": shuffled[key]["greedy_exact_match"],
                "delta_exact_match_staged_minus_shuffled": (
                    staged[key]["greedy_exact_match"] - shuffled[key]["greedy_exact_match"]
                ),
                "staged_target_loss": staged[key]["target_loss"],
                "shuffled_target_loss": shuffled[key]["target_loss"],
                "delta_target_loss_staged_minus_shuffled": (
                    staged[key]["target_loss"] - shuffled[key]["target_loss"]
                ),
            }
            for key in EXPECTED_KEYS
        }

    result = {
        "status": "PASS",
        "campaign": "relational_curriculum_v1",
        "protocol": "docs/A0_A1_RELATIONAL_CURRICULUM_V1_PROTOCOL.md",
        "protocol_commit": next(iter(expected_protocol)),
        "order_independent_exposure_sha256": next(iter(expected_exposure)),
        "arms": arms,
        "comparisons": comparisons,
        "interpretation_guardrail": (
            "Staged-vs-shuffled deltas test curriculum order under this fixed synthetic exposure. "
            "They do not by themselves prove a unique A1 mechanism or authorize an architecture change."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a0-staged", type=Path, required=True)
    parser.add_argument("--a0-shuffled", type=Path, required=True)
    parser.add_argument("--a1-staged", type=Path, required=True)
    parser.add_argument("--a1-shuffled", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else REPO_ROOT / path

    paths = {
        "a0_staged": resolve(args.a0_staged),
        "a0_shuffled": resolve(args.a0_shuffled),
        "a1_staged": resolve(args.a1_staged),
        "a1_shuffled": resolve(args.a1_shuffled),
    }
    output = resolve(args.output)
    result = aggregate(paths, output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
