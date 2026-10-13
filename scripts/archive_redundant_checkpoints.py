#!/usr/bin/env python3
"""Archive redundant training checkpoints without removing provenance.

The root worktree keeps the newest valid checkpoint for every run and every
checkpoint referenced by a tracked report/result.  Other checkpoint files and
their sidecars are copied, hash-verified, and removed from the root only
after the copy is complete.  The archive lives on the separately mounted SSD.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from braidformer.training.integrity import verify_checkpoint


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "results" / "runs"
ARCHIVE_ROOT = Path("/mnt/ssd2/braidformer_checkpoint_archive_20261014")
MANIFEST_PATH = REPO_ROOT / "results" / "checkpoint_retention_manifest.json"
CHECKPOINT_PATTERN = re.compile(
    r"results/runs/([^/\s\"'`]+)/(checkpoint_step_[0-9]{6}\.pt)"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _checkpoint_step(path: Path) -> int:
    return int(path.stem.removeprefix("checkpoint_step_"))


def _tracked_checkpoint_references() -> set[str]:
    tracked = subprocess.check_output(
        ["git", "ls-files"], cwd=REPO_ROOT, text=True
    ).splitlines()
    references: set[str] = set()
    for relative in tracked:
        path = REPO_ROOT / relative
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for run_name, checkpoint_name in CHECKPOINT_PATTERN.findall(text):
            references.add(f"results/runs/{run_name}/{checkpoint_name}")
    return references


def _structural_sidecar(path: Path) -> dict[str, Any]:
    sidecar = Path(f"{path}.json")
    if not sidecar.is_file():
        raise RuntimeError(f"checkpoint sidecar missing: {path}")
    record = _load_json(sidecar)
    if record.get("complete") is not True:
        raise RuntimeError(f"checkpoint sidecar is not complete: {sidecar}")
    if record.get("checkpoint_bytes") != path.stat().st_size:
        raise RuntimeError(f"checkpoint byte count disagrees with sidecar: {path}")
    if not isinstance(record.get("checkpoint_sha256"), str):
        raise RuntimeError(f"checkpoint hash missing from sidecar: {sidecar}")
    return record


def _build_plan() -> dict[str, Any]:
    if not RUNS_ROOT.is_dir():
        raise RuntimeError(f"run root is missing: {RUNS_ROOT}")
    references = _tracked_checkpoint_references()
    entries: list[dict[str, Any]] = []
    run_count = 0
    for run_dir in sorted(path for path in RUNS_ROOT.iterdir() if path.is_dir()):
        run_count += 1
        checkpoints = sorted(run_dir.glob("checkpoint_step_*.pt"), key=_checkpoint_step)
        if not checkpoints:
            continue
        newest = checkpoints[-1]
        for checkpoint in checkpoints:
            relative = checkpoint.relative_to(REPO_ROOT).as_posix()
            sidecar = _structural_sidecar(checkpoint)
            reasons: list[str] = []
            if checkpoint == newest:
                reasons.append("newest_checkpoint_in_run")
            if relative in references:
                reasons.append("referenced_by_tracked_report_or_result")
            keep = bool(reasons)
            entries.append(
                {
                    "source": relative,
                    "sidecar": f"{relative}.json",
                    "run": run_dir.name,
                    "step": _checkpoint_step(checkpoint),
                    "bytes": checkpoint.stat().st_size,
                    "sidecar_bytes": Path(f"{checkpoint}.json").stat().st_size,
                    "sidecar_sha256": sidecar["checkpoint_sha256"],
                    "action": "keep_on_root" if keep else "archive_on_ssd2",
                    "keep_reasons": reasons,
                    "hash_verified_before_archive": False,
                    "archive_path": None,
                }
            )
    archive_entries = [entry for entry in entries if entry["action"] == "archive_on_ssd2"]
    keep_entries = [entry for entry in entries if entry["action"] == "keep_on_root"]
    return {
        "version": "checkpoint_retention_manifest_v1",
        "status": "planned",
        "root": str(REPO_ROOT),
        "runs_root": str(RUNS_ROOT.relative_to(REPO_ROOT)),
        "archive_root": str(ARCHIVE_ROOT),
        "run_count": run_count,
        "checkpoint_count": len(entries),
        "keep_count": len(keep_entries),
        "archive_count": len(archive_entries),
        "keep_bytes": sum(entry["bytes"] + entry["sidecar_bytes"] for entry in keep_entries),
        "archive_bytes": sum(entry["bytes"] + entry["sidecar_bytes"] for entry in archive_entries),
        "tracked_reference_count": len(references),
        "tracked_references": sorted(references),
        "entries": entries,
    }


def _write_manifest(payload: dict[str, Any]) -> None:
    temporary = Path(f"{MANIFEST_PATH}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, MANIFEST_PATH)


def _copy_and_verify(source: Path, destination: Path, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{destination}.partial")
    if temporary.exists():
        temporary.unlink()
    shutil.copy2(source, temporary)
    actual = _sha256(temporary)
    if actual != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"archive hash mismatch for {source}: expected {expected_sha256}, got {actual}"
        )
    os.replace(temporary, destination)


def execute(payload: dict[str, Any]) -> dict[str, Any]:
    archive = ARCHIVE_ROOT.resolve()
    if not archive.is_dir():
        raise RuntimeError(f"archive mount point is missing: {archive}")
    if os.stat(RUNS_ROOT).st_dev == os.stat(archive).st_dev:
        raise RuntimeError("archive is on the same filesystem as the root run directory")

    # Verify every checkpoint that will remain in the root before moving any
    # redundant file.  This protects the retention point if the operation is
    # interrupted halfway through.
    for entry in payload["entries"]:
        if entry["action"] != "keep_on_root":
            continue
        checkpoint = REPO_ROOT / entry["source"]
        verified = verify_checkpoint(checkpoint)
        if verified["checkpoint_sha256"] != entry["sidecar_sha256"]:
            raise RuntimeError(f"kept checkpoint hash changed: {checkpoint}")
        entry["hash_verified_before_archive"] = True

    payload["status"] = "in_progress"
    _write_manifest(payload)
    moved_bytes = 0
    moved_count = 0
    for entry in payload["entries"]:
        if entry["action"] != "archive_on_ssd2":
            continue
        source = REPO_ROOT / entry["source"]
        source_sidecar = REPO_ROOT / entry["sidecar"]
        destination = archive / entry["source"]
        destination_sidecar = archive / entry["sidecar"]
        if source.exists():
            _copy_and_verify(source, destination, entry["sidecar_sha256"])
            if destination_sidecar.exists():
                existing = _load_json(destination_sidecar)
                if existing != _load_json(source_sidecar):
                    raise RuntimeError(f"archive sidecar conflict: {destination_sidecar}")
            else:
                shutil.copy2(source_sidecar, destination_sidecar)
            source.unlink()
            source_sidecar.unlink()
        else:
            if not destination.is_file() or _sha256(destination) != entry["sidecar_sha256"]:
                raise RuntimeError(f"source missing and archive copy is invalid: {source}")
            if not destination_sidecar.is_file():
                raise RuntimeError(f"archive sidecar missing for existing copy: {destination}")
        entry["archive_path"] = str(destination.relative_to(archive.parent))
        entry["hash_verified_before_archive"] = True
        moved_count += 1
        moved_bytes += entry["bytes"] + entry["sidecar_bytes"]
        payload["moved_count"] = moved_count
        payload["moved_bytes"] = moved_bytes
        _write_manifest(payload)

    payload["status"] = "completed"
    payload["moved_count"] = moved_count
    payload["moved_bytes"] = moved_bytes
    _write_manifest(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute-approved", action="store_true")
    args = parser.parse_args()
    plan = _build_plan()
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if not args.execute_approved:
        raise SystemExit("pass --dry-run or --execute-approved")
    print(json.dumps(execute(plan), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
