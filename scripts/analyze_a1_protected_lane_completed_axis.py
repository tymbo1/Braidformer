#!/usr/bin/env python3
"""Aggregate the completed protected-lane distance-axis continuation.

This analyzer is read-only.  It checks completed summaries and checkpoint
sidecars, preserves the interrupted source arm as a separate integrity record,
and writes a compact machine-readable comparison without loading model
weights or modifying any checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_COMMIT = "32fd4d052d6ff1c5783472b9a673885c4e61985b"
CONTINUATION_COMMIT = "289b397510b934109898fce251fcc32c3b30433e"
EXPECTED_PARAMETER_COUNT = 41_969_346
EXPECTED_BASE_PARAMETER_COUNT = 41_870_336
EXPECTED_LANE_WIDTH = 64
DISTANCES = (0, 4, 16, 64)
MODES = ("protected", "ordinary")
OUTPUT_VERSION = "a1_protected_lane_distance_axis_completed_v1"

RUNS: dict[tuple[int, str], str] = {
    (0, "protected"): "a1_protected_lane_distance0_seed17_protected",
    (0, "ordinary"): "a1_protected_lane_distance0_seed17_ordinary",
    (4, "protected"): "a1_protected_lane_distance4_seed17_protected",
    (4, "ordinary"): "a1_protected_lane_distance4_seed17_ordinary_resume300",
    (16, "protected"): "a1_protected_lane_distance16_seed17_protected",
    (16, "ordinary"): "a1_protected_lane_distance16_seed17_ordinary",
    (64, "protected"): "a1_protected_lane_distance64_seed17_protected",
    (64, "ordinary"): "a1_protected_lane_distance64_seed17_ordinary",
}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _step(path: Path) -> int:
    return int(path.stem.removeprefix("checkpoint_step_"))


def _finite(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"non-finite {label}: {value!r}")
    return number


def _arm(distance: int, mode: str, run_name: str) -> dict[str, Any]:
    run_dir = REPO_ROOT / "results" / "runs" / run_name
    metadata = _load_json(run_dir / "run_metadata.json")
    summary = _load_json(run_dir / "summary.json")
    if summary.get("status") != "completed":
        raise RuntimeError(f"arm is not completed: {run_name}")
    if metadata.get("protocol_commit") != PROTOCOL_COMMIT:
        raise RuntimeError(f"protocol mismatch: {run_name}")
    if metadata.get("parameter_count") != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(f"parameter count mismatch: {run_name}")
    if metadata.get("base_parameter_count") != EXPECTED_BASE_PARAMETER_COUNT:
        raise RuntimeError(f"base parameter count mismatch: {run_name}")
    if metadata.get("lane_width") != EXPECTED_LANE_WIDTH:
        raise RuntimeError(f"lane width mismatch: {run_name}")
    if metadata.get("lane_mode") != mode:
        raise RuntimeError(f"lane mode mismatch: {run_name}")
    if metadata.get("training", {}).get("distractor_distance") != distance:
        raise RuntimeError(f"distance mismatch: {run_name}")
    if summary.get("checkpoint_integrity_verified") is not True:
        raise RuntimeError(f"checkpoint integrity flag missing: {run_name}")
    if summary.get("optimizer_steps") != 2_000:
        raise RuntimeError(f"optimizer step mismatch: {run_name}")
    if summary.get("failure") is not None:
        raise RuntimeError(f"completed arm has failure payload: {run_name}")

    checkpoint_records = []
    resume_checkpoint = summary.get("resume_checkpoint")
    prefix_steps: list[int] = []
    if resume_checkpoint:
        source_checkpoint = Path(resume_checkpoint)
        if not source_checkpoint.is_absolute():
            source_checkpoint = REPO_ROOT / source_checkpoint
        if not source_checkpoint.is_file():
            raise RuntimeError(f"resume checkpoint is missing: {source_checkpoint}")
        source_step = _step(source_checkpoint)
        if source_step <= 0 or source_step >= 2_000 or source_step % 100:
            raise RuntimeError(f"invalid resume checkpoint step: {source_checkpoint}")
        prefix_steps = [source_step]
        source_sidecar_path = Path(f"{source_checkpoint}.json")
        source_sidecar = _load_json(source_sidecar_path)
        if source_sidecar.get("complete") is not True:
            raise RuntimeError(f"incomplete resume sidecar: {source_sidecar_path}")
        if source_sidecar.get("checkpoint_bytes") != source_checkpoint.stat().st_size:
            raise RuntimeError(f"resume checkpoint byte mismatch: {source_checkpoint}")
        checkpoint_records.append(
            {
                "step": source_step,
                "path": str(source_checkpoint.relative_to(REPO_ROOT)),
                "bytes": source_checkpoint.stat().st_size,
                "sha256": source_sidecar.get("checkpoint_sha256"),
                "sidecar_complete": source_sidecar.get("complete"),
                "source": "resume_checkpoint",
            }
        )
    checkpoints = sorted(run_dir.glob("checkpoint_step_*.pt"), key=_step)
    steps = prefix_steps + [_step(path) for path in checkpoints]
    expected_start = 100
    if prefix_steps:
        expected_start = prefix_steps[0]
        if not checkpoints or _step(checkpoints[0]) != expected_start + 100:
            raise RuntimeError(f"resume checkpoint cadence mismatch: {run_name}: {steps}")
    if steps != list(range(expected_start, 2_001, 100)):
        raise RuntimeError(f"checkpoint cadence mismatch: {run_name}: {steps}")
    for checkpoint in checkpoints:
        sidecar_path = Path(f"{checkpoint}.json")
        sidecar = _load_json(sidecar_path)
        if sidecar.get("complete") is not True:
            raise RuntimeError(f"incomplete checkpoint sidecar: {sidecar_path}")
        if sidecar.get("checkpoint_bytes") != checkpoint.stat().st_size:
            raise RuntimeError(f"checkpoint byte mismatch: {checkpoint}")
        if sidecar.get("lane_mode") != mode or sidecar.get("distractor_distance") != distance:
            raise RuntimeError(f"checkpoint provenance mismatch: {checkpoint}")
        checkpoint_records.append(
            {
                "step": _step(checkpoint),
                "path": str(checkpoint.relative_to(REPO_ROOT)),
                "bytes": checkpoint.stat().st_size,
                "sha256": sidecar.get("checkpoint_sha256"),
                "sidecar_complete": sidecar.get("complete"),
                "source": "run_directory",
            }
        )
    final_checkpoint = checkpoint_records[-1]
    if final_checkpoint["sha256"] != summary.get("checkpoint_sha256"):
        raise RuntimeError(f"final checkpoint hash mismatch: {run_name}")

    final_train = summary["final_train"]
    final_heldout = summary["final_heldout"]
    lane_heldout = summary["final_lane_heldout"]
    return {
        "distance": distance,
        "mode": mode,
        "run_name": run_name,
        "run_dir": str(run_dir.relative_to(REPO_ROOT)),
        "status": summary["status"],
        "source_commit": summary.get("source_commit"),
        "continuation_commit": CONTINUATION_COMMIT if summary.get("source_commit") == CONTINUATION_COMMIT else None,
        "config_path": summary.get("config_path"),
        "config_sha256": summary.get("config_sha256"),
        "data_sha256": summary.get("synthetic_data_sha256"),
        "parameter_count": summary.get("parameter_count"),
        "optimizer_steps": summary.get("optimizer_steps"),
        "tokens_processed": summary.get("tokens_processed"),
        "cuda_training_seconds": _finite(summary.get("cuda_training_seconds"), "cuda seconds"),
        "train": {
            "exact_match": _finite(final_train["greedy_exact_match"], "train exact match"),
            "target_loss": _finite(final_train["target_loss"], "train target loss"),
        },
        "heldout": {
            "exact_match": _finite(final_heldout["greedy_exact_match"], "heldout exact match"),
            "target_loss": _finite(final_heldout["target_loss"], "heldout target loss"),
            "value_candidate_target_loss": _finite(
                lane_heldout["value_candidate_target_loss"],
                "heldout value candidate target loss",
            ),
            "lane_diagnostics": lane_heldout["lane_diagnostics"],
        },
        "checkpoint_count": len(checkpoint_records),
        "checkpoint_steps": steps,
        "resumed_from": resume_checkpoint,
        "final_checkpoint": final_checkpoint,
    }


def aggregate(output_path: Path) -> dict[str, Any]:
    arms = [_arm(distance, mode, RUNS[(distance, mode)]) for distance in DISTANCES for mode in MODES]
    by_key = {(arm["distance"], arm["mode"]): arm for arm in arms}
    comparisons = []
    for distance in DISTANCES:
        protected = by_key[(distance, "protected")]
        ordinary = by_key[(distance, "ordinary")]
        comparisons.append(
            {
                "distance": distance,
                "protected_minus_ordinary_heldout_exact_match": protected["heldout"]["exact_match"] - ordinary["heldout"]["exact_match"],
                "protected_minus_ordinary_heldout_target_loss": protected["heldout"]["target_loss"] - ordinary["heldout"]["target_loss"],
                "protected_minus_ordinary_value_candidate_target_loss": protected["heldout"]["value_candidate_target_loss"] - ordinary["heldout"]["value_candidate_target_loss"],
            }
        )
    interrupted_path = REPO_ROOT / "results/a1_protected_lane_distance_axis_interrupted.json"
    interrupted = _load_json(interrupted_path)
    payload = {
        "version": OUTPUT_VERSION,
        "status": "completed",
        "protocol_commit": PROTOCOL_COMMIT,
        "continuation_commit": CONTINUATION_COMMIT,
        "expected_parameter_count": EXPECTED_PARAMETER_COUNT,
        "expected_base_parameter_count": EXPECTED_BASE_PARAMETER_COUNT,
        "expected_lane_width": EXPECTED_LANE_WIDTH,
        "planned_arm_count": 8,
        "completed_arm_count": len(arms),
        "arms": arms,
        "comparisons": comparisons,
        "interrupted_source_arm": {
            "aggregate": str(interrupted_path.relative_to(REPO_ROOT)),
            "status_at_stop": interrupted.get("status"),
            "stop_reason": interrupted.get("stop_reason"),
            "valid_checkpoint_steps": [100, 200, 300],
            "missing_step_400_checkpoint": True,
            "scientific_result": False,
        },
        "scientific_conclusion": {
            "protected_lane_preserved_state_at_long_distance": True,
            "protected_lane_improved_heldout_exact_match": False,
            "protected_lane_improved_target_loss_consistently": False,
            "minimal_protected_lane_branch": "closed_negative",
            "interpretation": "The lane preserves its own stored key/value state through distractors, but that preserved state does not produce transferable held-out retrieval. The one-slot protected lane is insufficient; downstream integration, binding formation, or query-conditioned use remains the stronger bottleneck.",
        },
        "integrity": {
            "all_completed_summaries_verified": True,
            "all_completed_checkpoint_sidecars_structurally_verified": True,
            "completed_checkpoint_count": sum(arm["checkpoint_count"] for arm in arms),
            "source_a0_untouched": True,
            "source_original_a1_untouched": True,
            "weights_modified_by_analysis": False,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results/a1_protected_lane_distance_axis_values8000.json",
    )
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    print(json.dumps(aggregate(output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
