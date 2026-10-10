#!/usr/bin/env python3
"""Verify the local pinned TinyStories preparation without training."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from array import array
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from braidformer.training.integrity import sha256_file
from scripts.prepare_tinystories import assign_validation_split


DEFAULT_MANIFEST = REPO_ROOT / "data" / "tinystories_a0" / "manifest.json"


def _error(errors: list[str], message: str) -> None:
    errors.append(message)


def _report_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _verify_stream(
    path: Path,
    expected: dict[str, Any],
    vocab_size: int,
    errors: list[str],
) -> dict[str, Any]:
    actual_tokens = 0
    maximum = -1
    if path.stat().st_size % 2:
        _error(errors, f"token stream is not uint16-aligned: {path}")
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            values = array("H")
            values.frombytes(chunk[: len(chunk) - (len(chunk) % 2)])
            if sys.byteorder != "little":
                values.byteswap()
            if values:
                maximum = max(maximum, max(values))
            actual_tokens += len(values)
    if maximum >= vocab_size:
        _error(errors, f"token id {maximum} exceeds vocabulary {vocab_size}: {path}")
    if actual_tokens != expected["tokens"]:
        _error(errors, f"token count mismatch for {path}: {actual_tokens} != {expected['tokens']}")
    if path.stat().st_size != expected["bytes"]:
        _error(errors, f"byte count mismatch for {path}: {path.stat().st_size} != {expected['bytes']}")
    complete, remainder = divmod(actual_tokens, expected.get("context_length", 512))
    if complete != expected["complete_context_sequences"] or remainder != expected["discarded_tail_tokens"]:
        _error(errors, f"packing arithmetic mismatch for {path}")
    actual_hash = sha256_file(path)
    if actual_hash != expected["sha256"]:
        _error(errors, f"SHA-256 mismatch for {path}: {actual_hash} != {expected['sha256']}")
    return {"path": _report_path(path), "tokens": actual_tokens, "max_token_id": maximum, "sha256": actual_hash}


def verify(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    data_root = manifest_path.parent
    vocab_size = int(manifest["tokenizer"]["vocab_size"])
    if vocab_size != 32_000:
        _error(errors, f"unexpected tokenizer vocabulary: {vocab_size}")
    if manifest["dataset"]["license"] != "CDLA-Sharing-1.0":
        _error(errors, "dataset license is not the approved CDLA-Sharing-1.0 record")
    if not manifest["partition"]["no_cross_split_duplicate_text_hashes"]:
        _error(errors, "manifest does not assert duplicate exclusion")

    source_checks: list[dict[str, Any]] = []
    for split_records in manifest["dataset"]["source_files"].values():
        for record in split_records:
            path = data_root / "raw" / record["path"]
            if not path.is_file():
                _error(errors, f"missing source shard: {path}")
                continue
            actual_hash = sha256_file(path)
            source_checks.append({"path": _report_path(path), "sha256": actual_hash})
            if actual_hash != record["sha256"]:
                _error(errors, f"source SHA-256 mismatch for {path}")
            if path.stat().st_size != record["bytes"]:
                _error(errors, f"source byte-count mismatch for {path}")

    tokenizer_path = REPO_ROOT / manifest["tokenizer"]["path"]
    if not tokenizer_path.is_file():
        _error(errors, f"missing tokenizer: {tokenizer_path}")
    else:
        actual_hash = sha256_file(tokenizer_path)
        if actual_hash != manifest["tokenizer"]["sha256"]:
            _error(errors, "tokenizer SHA-256 mismatch")
        try:
            from tokenizers import Tokenizer

            tokenizer = Tokenizer.from_file(str(tokenizer_path))
            if tokenizer.get_vocab_size(with_added_tokens=True) != vocab_size:
                _error(errors, "tokenizer vocabulary size does not match manifest")
        except Exception as exc:  # pragma: no cover - dependency/runtime failure
            _error(errors, f"tokenizer load failed: {type(exc).__name__}: {exc}")

    split_manifest = REPO_ROOT / manifest["partition"]["manifest"]["path"]
    seen_hashes: set[str] = set()
    split_counts = {"train": 0, "validation": 0, "test": 0}
    if not split_manifest.is_file():
        _error(errors, f"missing split manifest: {split_manifest}")
    else:
        actual_hash = sha256_file(split_manifest)
        if actual_hash != manifest["partition"]["manifest"]["sha256"]:
            _error(errors, "split manifest SHA-256 mismatch")
        with split_manifest.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                record = json.loads(line)
                digest = record["text_sha256"]
                if digest in seen_hashes:
                    _error(errors, f"duplicate text hash at split manifest line {line_number}")
                seen_hashes.add(digest)
                final_split = record["final_split"]
                if final_split not in split_counts:
                    _error(errors, f"unknown final split at line {line_number}: {final_split}")
                    continue
                split_counts[final_split] += 1
                expected_split = "train" if record["source_split"] == "train" else assign_validation_split(digest)
                if final_split != expected_split:
                    _error(errors, f"non-deterministic split assignment at line {line_number}")
        if split_counts != manifest["partition"]["assigned_documents"]:
            _error(errors, f"split document counts mismatch: {split_counts}")

    stream_checks: dict[str, Any] = {}
    for split, expected in manifest["packing"]["streams"].items():
        expected = dict(expected)
        expected["context_length"] = manifest["packing"]["context_length"]
        path = REPO_ROOT / expected["path"]
        if not path.is_file():
            _error(errors, f"missing token stream: {path}")
            continue
        stream_checks[split] = _verify_stream(path, expected, vocab_size, errors)

    return {
        "schema_version": 1,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": _report_path(manifest_path),
        "dataset_revision": manifest["dataset"]["revision"],
        "tokenizer_sha256": manifest["tokenizer"]["sha256"],
        "split_counts": split_counts,
        "source_files_checked": source_checks,
        "streams_checked": stream_checks,
        "duplicate_text_hashes_found": max(0, sum(split_counts.values()) - len(seen_hashes)),
        "errors": errors,
        "status": "PASS" if not errors else "FAIL",
        "runtime": {"python": platform.python_version()},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results" / "a0_data_verification.json")
    args = parser.parse_args(argv)
    manifest_path = args.manifest if args.manifest.is_absolute() else REPO_ROOT / args.manifest
    report = verify(manifest_path)
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
