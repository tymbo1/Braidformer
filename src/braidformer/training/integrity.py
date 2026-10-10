"""Low-risk integrity controls for the bounded A0 pilot.

This module does not define a model or training policy. The bounded A0 runner
uses it to make provenance, progress, numerical safety, and checkpoint
recovery observable without changing the model architecture.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


class IntegrityError(RuntimeError):
    """Raised when a run violates an integrity or numerical safety condition."""


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it all at once."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Mapping[str, Any]) -> str:
    """Hash a JSON-compatible mapping using canonical key ordering."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def set_reproducible_seed(seed: int, deterministic: bool = True) -> dict[str, Any]:
    """Seed available runtimes and report which deterministic controls applied."""

    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    report: dict[str, Any] = {"seed": seed, "python": True, "numpy": False, "torch": False}

    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        np.random.seed(seed)
        report["numpy"] = True

    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False
        report["torch"] = True
    report["deterministic_algorithms_requested"] = deterministic
    return report


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _json_safe(value: Any) -> Any:
    """Keep failure records valid JSON even when recording NaN/Inf evidence."""

    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


@dataclass
class IntegrityMonitor:
    """Write synchronized step records and stop on unsafe pilot state."""

    run_dir: Path
    metadata: Mapping[str, Any]
    tokens_per_step: int
    loss_ceiling: float = 100.0
    initial_step: int = 0
    initial_tokens: int = 0
    _last_step: int = field(init=False)
    _last_tokens: int = field(init=False)
    _started_at: float = field(init=False)
    _steps_path: Path = field(init=False)

    def __post_init__(self) -> None:
        if self.tokens_per_step <= 0:
            raise ValueError("tokens_per_step must be positive")
        if self.loss_ceiling <= 0 or not math.isfinite(self.loss_ceiling):
            raise ValueError("loss_ceiling must be finite and positive")
        if self.initial_step < 0 or self.initial_tokens < 0:
            raise ValueError("initial progress must be non-negative")
        self.run_dir = Path(self.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._steps_path = self.run_dir / "steps.jsonl"
        if self._steps_path.exists():
            raise FileExistsError(f"refusing to append to an existing run log: {self._steps_path}")
        self._last_step = self.initial_step
        self._last_tokens = self.initial_tokens
        self._started_at = time.monotonic()
        run_metadata = dict(self.metadata)
        run_metadata.update(
            {
                "integrity_schema_version": 1,
                "tokens_per_step": self.tokens_per_step,
                "loss_ceiling": self.loss_ceiling,
                "initial_step": self.initial_step,
                "initial_tokens": self.initial_tokens,
            }
        )
        _atomic_write_json(self.run_dir / "run_metadata.json", run_metadata)

    def _append(self, record: Mapping[str, Any]) -> None:
        with self._steps_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(record), sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def warning(self, name: str, details: Mapping[str, Any] | None = None) -> None:
        """Record a non-fatal warning without changing progress state."""

        self._append(
            {
                "event": "warning",
                "name": name,
                "details": _json_safe(details or {}),
                "wall_clock_seconds": time.monotonic() - self._started_at,
            }
        )

    def fail(self, reason: str, details: Mapping[str, Any] | None = None) -> None:
        """Record a fatal integrity event and raise :class:`IntegrityError`.

        The runner uses this public boundary for failures detected before a
        complete optimizer step can be recorded, such as a non-finite loss or
        gradient.  Keeping the event in the durable JSONL stream preserves the
        same fail-fast behavior as :meth:`record_step`.
        """

        self._failure(reason, details or {})

    def _failure(self, reason: str, details: Mapping[str, Any]) -> None:
        self._append(
            {
                "event": "integrity_failure",
                "reason": reason,
                "details": _json_safe(details),
                "step": self._last_step,
                "tokens_processed": self._last_tokens,
                "wall_clock_seconds": time.monotonic() - self._started_at,
            }
        )
        raise IntegrityError(f"{reason}: {dict(details)}")

    def record_step(
        self,
        *,
        step: int,
        tokens_processed: int,
        train_loss: float,
        validation_loss: float | None = None,
        gradient_norm: float | None = None,
        learning_rate: float | None = None,
        tokens_per_second: float | None = None,
        peak_allocated_vram_bytes: int | None = None,
        peak_reserved_vram_bytes: int | None = None,
    ) -> dict[str, Any]:
        """Validate and durably record one optimizer step.

        The monitor treats non-finite losses/gradient norms, implausibly large
        loss, skipped steps, and non-monotonic token counters as hard failures.
        """

        if step != self._last_step + 1:
            self._failure("non_monotonic_step", {"expected": self._last_step + 1, "received": step})
        expected_tokens = self._last_tokens + self.tokens_per_step
        if tokens_processed != expected_tokens:
            self._failure(
                "unexpected_token_counter",
                {"expected": expected_tokens, "received": tokens_processed},
            )
        if not isinstance(train_loss, (int, float)) or not math.isfinite(train_loss):
            self._failure("non_finite_train_loss", {"value": train_loss})
        if train_loss > self.loss_ceiling:
            self._failure("loss_ceiling_exceeded", {"value": train_loss, "ceiling": self.loss_ceiling})
        for name, value in (
            ("validation_loss", validation_loss),
            ("gradient_norm", gradient_norm),
            ("learning_rate", learning_rate),
            ("tokens_per_second", tokens_per_second),
        ):
            if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value)):
                self._failure(f"non_finite_{name}", {"value": value})
        if validation_loss is not None and validation_loss > self.loss_ceiling:
            self._failure(
                "validation_loss_ceiling_exceeded",
                {"value": validation_loss, "ceiling": self.loss_ceiling},
            )
        for name, value in (
            ("peak_allocated_vram_bytes", peak_allocated_vram_bytes),
            ("peak_reserved_vram_bytes", peak_reserved_vram_bytes),
        ):
            if value is not None and (not isinstance(value, int) or value < 0):
                self._failure(f"invalid_{name}", {"value": value})

        record = {
            "event": "step",
            "step": step,
            "tokens_processed": tokens_processed,
            "train_loss": float(train_loss),
            "validation_loss": None if validation_loss is None else float(validation_loss),
            "gradient_norm": None if gradient_norm is None else float(gradient_norm),
            "learning_rate": None if learning_rate is None else float(learning_rate),
            "tokens_per_second": None if tokens_per_second is None else float(tokens_per_second),
            "peak_allocated_vram_bytes": peak_allocated_vram_bytes,
            "peak_reserved_vram_bytes": peak_reserved_vram_bytes,
            "wall_clock_seconds": time.monotonic() - self._started_at,
        }
        self._append(record)
        self._last_step = step
        self._last_tokens = tokens_processed
        return record


def save_atomic_checkpoint(
    payload: Any,
    checkpoint_path: str | Path,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Save a checkpoint and its recovery metadata without exposing partial files."""

    required = {"step", "tokens_processed", "seed", "config_sha256", "data_manifest_sha256"}
    missing = sorted(required.difference(metadata))
    if missing:
        raise ValueError(f"checkpoint metadata missing required fields: {', '.join(missing)}")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - project requires torch for models
        raise RuntimeError("checkpoint saving requires torch") from exc

    destination = Path(checkpoint_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{destination}.tmp")
    try:
        torch.save(payload, temporary)
        digest = sha256_file(temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    record = {
        "checkpoint_schema_version": 1,
        "checkpoint_path": str(destination),
        "checkpoint_sha256": digest,
        "checkpoint_bytes": destination.stat().st_size,
        "complete": True,
        **dict(metadata),
    }
    _atomic_write_json(Path(f"{destination}.json"), record)
    return record


def verify_checkpoint(checkpoint_path: str | Path) -> dict[str, Any]:
    """Verify a checkpoint against its atomic sidecar metadata."""

    destination = Path(checkpoint_path)
    sidecar = Path(f"{destination}.json")
    if not destination.is_file() or not sidecar.is_file():
        raise IntegrityError(f"checkpoint or sidecar is missing: {destination}")
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    if record.get("complete") is not True:
        raise IntegrityError(f"checkpoint is not marked complete: {destination}")
    actual_digest = sha256_file(destination)
    if actual_digest != record.get("checkpoint_sha256"):
        raise IntegrityError(
            f"checkpoint hash mismatch: expected {record.get('checkpoint_sha256')}, got {actual_digest}"
        )
    if destination.stat().st_size != record.get("checkpoint_bytes"):
        raise IntegrityError(f"checkpoint byte-count mismatch: {destination}")
    return record


__all__ = [
    "IntegrityError",
    "IntegrityMonitor",
    "save_atomic_checkpoint",
    "set_reproducible_seed",
    "sha256_file",
    "sha256_json",
    "verify_checkpoint",
]
