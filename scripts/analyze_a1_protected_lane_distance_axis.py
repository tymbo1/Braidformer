#!/usr/bin/env python3
"""Verify and aggregate the interrupted protected-lane distance axis.

This is a read-only post-stop audit.  It does not load model weights into a
model, alter checkpoints, or restart training.  The audit is intentionally
strict about the completed-arm summaries, intact checkpoint sidecars, and the
storage-related failure that stopped the campaign.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from braidformer.training.integrity import verify_checkpoint


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_COMMIT = "32fd4d052d6ff1c5783472b9a673885c4e61985b"
IMPLEMENTATION_COMMIT = "b48ddc24658eacfd93a98f4a3c97c66bc6ca7f0d"
EXPECTED_PARAMETER_COUNT = 41_969_346
EXPECTED_BASE_PARAMETER_COUNT = 41_870_336
EXPECTED_LANE_WIDTH = 64
EXPECTED_ARMS = tuple(
    (distance, mode)
    for distance in (0, 4, 16, 64)
    for mode in ("protected", "ordinary")
)
RUN_PREFIX = "a1_protected_lane_distance{distance}_seed17_{mode}"
OUTPUT_VERSION = "a1_protected_lane_distance_axis_interrupted_v1"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _checkpoint_step(path: Path) -> int:
    stem = path.stem
    prefix = "checkpoint_step_"
    if not stem.startswith(prefix):
        raise ValueError(f"unexpected checkpoint name: {path.name}")
    return int(stem[len(prefix) :])


def _audit_run(distance: int, mode: str) -> dict[str, Any]:
    run_dir = REPO_ROOT / "results" / "runs" / RUN_PREFIX.format(distance=distance, mode=mode)
    record: dict[str, Any] = {
        "distance": distance,
        "mode": mode,
        "run_dir": str(run_dir.relative_to(REPO_ROOT)),
        "status": "unstarted",
        "checkpoint_steps": [],
        "checkpoint_records": [],
    }
    if not run_dir.is_dir():
        return record

    metadata_path = run_dir / "run_metadata.json"
    summary_path = run_dir / "summary.json"
    if not metadata_path.is_file() or not summary_path.is_file():
        raise RuntimeError(f"run metadata/summary missing: {run_dir}")
    metadata = _load_json(metadata_path)
    summary = _load_json(summary_path)
    record["status"] = summary.get("status")
    record["summary"] = summary
    record["metadata"] = metadata

    for source, expected in (
        (metadata.get("protocol_commit"), PROTOCOL_COMMIT),
        (metadata.get("source_commit"), IMPLEMENTATION_COMMIT),
        (metadata.get("parameter_count"), EXPECTED_PARAMETER_COUNT),
        (metadata.get("base_parameter_count"), EXPECTED_BASE_PARAMETER_COUNT),
        (metadata.get("lane_width"), EXPECTED_LANE_WIDTH),
        (metadata.get("lane_mode"), mode),
        (metadata.get("training", {}).get("distractor_distance"), distance),
    ):
        if source != expected:
            raise RuntimeError(
                f"provenance mismatch in {run_dir.name}: {source!r} != {expected!r}"
            )

    checkpoints = sorted(run_dir.glob("checkpoint_step_*.pt"), key=_checkpoint_step)
    if not checkpoints:
        raise RuntimeError(f"no checkpoint in initialized run: {run_dir}")
    expected_steps = list(range(100, _checkpoint_step(checkpoints[-1]) + 1, 100))
    actual_steps = [_checkpoint_step(path) for path in checkpoints]
    if actual_steps != expected_steps:
        raise RuntimeError(f"checkpoint cadence mismatch in {run_dir.name}: {actual_steps}")
    checkpoint_records: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        sidecar = verify_checkpoint(checkpoint)
        checkpoint_records.append(
            {
                "step": _checkpoint_step(checkpoint),
                "path": str(checkpoint.relative_to(REPO_ROOT)),
                "sha256": sidecar["checkpoint_sha256"],
                "bytes": sidecar["checkpoint_bytes"],
                "sidecar_complete": sidecar["complete"],
            }
        )
    record["checkpoint_steps"] = actual_steps
    record["checkpoint_records"] = checkpoint_records
    record["checkpoint_count"] = len(checkpoint_records)

    if summary.get("status") == "completed":
        if summary.get("checkpoint_integrity_verified") is not True:
            raise RuntimeError(f"completed run lacks integrity confirmation: {run_dir.name}")
        if summary.get("optimizer_steps") != 2_000:
            raise RuntimeError(f"completed run has wrong optimizer step count: {run_dir.name}")
        if summary.get("failure") is not None:
            raise RuntimeError(f"completed run has a failure payload: {run_dir.name}")
    elif summary.get("status") == "failed":
        failure = summary.get("failure") or {}
        failure_message = str(failure.get("message", ""))
        if "inline_container" not in failure_message and "iostream" not in failure_message:
            raise RuntimeError(f"unexpected failure in stopped arm: {run_dir.name}: {failure}")
        record["failure_type"] = failure.get("type")
        record["failure_message"] = failure_message
    else:
        raise RuntimeError(f"unexpected run status in {run_dir.name}: {summary.get('status')!r}")
    return record


def _git_status() -> list[str]:
    output = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
    )
    return [line for line in output.splitlines() if line]


def aggregate(output_path: Path) -> dict[str, Any]:
    arms = [_audit_run(distance, mode) for distance, mode in EXPECTED_ARMS]
    initialized = [arm for arm in arms if arm["status"] != "unstarted"]
    completed = [arm for arm in arms if arm["status"] == "completed"]
    failed = [arm for arm in arms if arm["status"] == "failed"]
    if len(completed) != 3 or len(failed) != 1:
        raise RuntimeError(
            f"unexpected partial campaign shape: completed={len(completed)}, failed={len(failed)}"
        )
    failed_arm = failed[0]
    failed_checkpoint_bytes = failed_arm["checkpoint_records"][-1]["bytes"]
    storage = shutil.disk_usage(REPO_ROOT)
    if storage.free >= failed_checkpoint_bytes:
        raise RuntimeError(
            "the recorded storage stop is not reproducible: free space is at least "
            "one checkpoint size"
        )

    payload = {
        "version": OUTPUT_VERSION,
        "status": "stopped_integrity_failure",
        "stop_reason": {
            "category": "checkpoint_write_failure",
            "type": failed_arm["failure_type"],
            "message": failed_arm["failure_message"],
            "available_bytes_at_audit": storage.free,
            "failed_checkpoint_bytes": failed_checkpoint_bytes,
            "filesystem_used_percent_at_audit": round(storage.used / storage.total * 100, 4),
        },
        "protocol_commit": PROTOCOL_COMMIT,
        "implementation_commit": IMPLEMENTATION_COMMIT,
        "expected_parameter_count": EXPECTED_PARAMETER_COUNT,
        "expected_base_parameter_count": EXPECTED_BASE_PARAMETER_COUNT,
        "expected_lane_width": EXPECTED_LANE_WIDTH,
        "planned_arm_count": len(EXPECTED_ARMS),
        "initialized_arm_count": len(initialized),
        "completed_arm_count": len(completed),
        "failed_arm_count": len(failed),
        "unstarted_arm_count": len(EXPECTED_ARMS) - len(initialized),
        "arms": arms,
        "integrity": {
            "initialized_checkpoint_sidecars_verified": sum(
                arm.get("checkpoint_count", 0) for arm in initialized
            ),
            "worktree_status_at_audit": _git_status(),
            "source_checkpoint_mutation": False,
            "training_restarted_after_failure": False,
        },
        "scientific_status": {
            "distance_0_protected_and_ordinary_completed": True,
            "distance_4_protected_completed": True,
            "distance_4_ordinary_is_not_a_scientific_result": True,
            "distance_16_and_64_not_started": True,
            "interpretation_deferred": True,
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
        default=REPO_ROOT / "results/a1_protected_lane_distance_axis_interrupted.json",
    )
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    payload = aggregate(output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
