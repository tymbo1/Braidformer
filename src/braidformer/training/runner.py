"""Bounded, reproducible BraidNano training runner.

This runner implements the approved BraidNano pilot mechanics:
deterministic packed-token consumption, mixed-precision AdamW, cosine warmup,
fixed validation evaluation, durable step logging, and atomic checkpoint
recovery.  It supports only explicit A0 and A1 model configurations and does
not implement gradient checkpointing or later BraidFormer architectures.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import sys
import time
from array import array
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

from braidformer.config import ExperimentConfig, load_config
from braidformer.models import A0TransformerLM, A1RecurrentStateSpaceLM
from braidformer.training.integrity import (
    IntegrityError,
    IntegrityMonitor,
    save_atomic_checkpoint,
    set_reproducible_seed,
    sha256_file,
    verify_checkpoint,
)


class TrainingConfigurationError(ValueError):
    """Raised when an approved BraidNano configuration or artifact is unsuitable."""


LanguageModel = A0TransformerLM | A1RecurrentStateSpaceLM


def _model_class(config: ExperimentConfig) -> type[LanguageModel]:
    """Return the model class named by an explicit config without construction."""

    if config.stage == "A0" and config.model_type == "transformer":
        return A0TransformerLM
    if config.stage == "A1" and config.model_type == "recurrent_state_space":
        return A1RecurrentStateSpaceLM
    raise TrainingConfigurationError(
        "unsupported model selection; expected A0/transformer or "
        "A1/recurrent_state_space, received "
        f"{config.stage!r}/{config.model_type!r}"
    )


def build_model(config: ExperimentConfig) -> LanguageModel:
    """Construct exactly the architecture named by an explicit config."""

    return _model_class(config)(config)


def cosine_warmup_learning_rate(
    base_learning_rate: float,
    step: int,
    warmup_steps: int,
    max_steps: int,
) -> float:
    """Return the configured one-indexed warmup-plus-cosine learning rate."""

    if base_learning_rate <= 0 or step <= 0 or warmup_steps <= 0 or max_steps <= 0:
        raise ValueError("learning-rate schedule arguments must be positive")
    if step <= warmup_steps:
        return base_learning_rate * step / warmup_steps
    decay_steps = max(1, max_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
    return base_learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))


class PackedTokenStream:
    """Read little-endian uint16 token windows without loading a full corpus."""

    def __init__(self, path: str | Path, expected_tokens: int | None = None) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        byte_count = self.path.stat().st_size
        if byte_count % 2:
            raise TrainingConfigurationError(f"token stream is not uint16-aligned: {self.path}")
        self.token_count = byte_count // 2
        if expected_tokens is not None and self.token_count != expected_tokens:
            raise TrainingConfigurationError(
                f"token count mismatch for {self.path}: {self.token_count} != {expected_tokens}"
            )
        self._handle = self.path.open("rb")

    def read_tokens(self, start: int, count: int) -> torch.Tensor:
        """Return a CPU long tensor for a contiguous token window."""

        if start < 0 or count <= 0 or start + count > self.token_count:
            raise IndexError(
                f"token window [{start}, {start + count}) exceeds {self.token_count} tokens"
            )
        self._handle.seek(start * 2)
        raw = self._handle.read(count * 2)
        if len(raw) != count * 2:
            raise IOError(f"short read from token stream: {self.path}")
        values = array("H")
        values.frombytes(raw)
        if sys.byteorder != "little":
            values.byteswap()
        return torch.tensor(values, dtype=torch.long)

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "PackedTokenStream":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


@dataclass(frozen=True)
class ApprovedArtifacts:
    """Pinned local artifacts used by the A0 pilot."""

    data_config_path: Path
    data_config_sha256: str
    manifest_path: Path
    manifest_sha256: str
    verification_report_path: Path
    verification_report_sha256: str
    tokenizer_path: Path
    tokenizer_sha256: str
    train_stream_path: Path
    train_stream_sha256: str
    validation_stream_path: Path
    validation_stream_sha256: str
    train_tokens: int
    validation_tokens: int
    dataset_revision: str

    def metadata(self, repo_root: Path) -> dict[str, Any]:
        """Return JSON-safe provenance metadata for a run record."""

        relative = lambda path: str(path.relative_to(repo_root))
        return {
            "data_config_path": relative(self.data_config_path),
            "data_config_sha256": self.data_config_sha256,
            "data_manifest_path": relative(self.manifest_path),
            "data_manifest_sha256": self.manifest_sha256,
            "verification_report_path": relative(self.verification_report_path),
            "verification_report_sha256": self.verification_report_sha256,
            "tokenizer_path": relative(self.tokenizer_path),
            "tokenizer_sha256": self.tokenizer_sha256,
            "train_stream_path": relative(self.train_stream_path),
            "train_stream_sha256": self.train_stream_sha256,
            "validation_stream_path": relative(self.validation_stream_path),
            "validation_stream_sha256": self.validation_stream_sha256,
            "train_tokens_available": self.train_tokens,
            "validation_tokens_available": self.validation_tokens,
            "dataset_revision": self.dataset_revision,
        }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a small JSON record atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _cpu_copy(value: Any) -> Any:
    """Move checkpoint tensors to CPU while preserving nested state structure."""

    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return value


def _rng_state() -> dict[str, Any]:
    """Capture Python, CPU, and CUDA RNG state for interruption recovery."""

    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state().cpu(),
        "torch_cuda": [state.cpu() for state in torch.cuda.get_rng_state_all()],
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore RNG state saved in an A0 checkpoint."""

    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    cuda_states = state.get("torch_cuda")
    if cuda_states:
        torch.cuda.set_rng_state_all([item.cpu() for item in cuda_states])


def _set_optimizer_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def _finite_gradients(model: torch.nn.Module) -> tuple[bool, int, int]:
    finite = True
    tensor_count = 0
    element_count = 0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        tensor_count += 1
        element_count += parameter.grad.numel()
        if not bool(torch.isfinite(parameter.grad).all().item()):
            finite = False
    return finite, tensor_count, element_count


def _load_approved_artifacts(repo_root: Path, config: ExperimentConfig) -> ApprovedArtifacts:
    """Verify the locally prepared tokenizer and packed streams before training."""

    data_config_path = repo_root / config.data_and_tokenizer
    if not data_config_path.is_file():
        raise TrainingConfigurationError(f"missing data configuration: {data_config_path}")
    data_config = json.loads(data_config_path.read_text(encoding="utf-8"))
    artifact_root = repo_root / data_config["artifact_root"]
    prepared = data_config["prepared_artifacts"]
    manifest_path = repo_root / prepared["manifest"]
    verification_report_path = repo_root / prepared["verification_report"]
    tokenizer_path = repo_root / data_config["tokenizer"]["artifact"]
    if not manifest_path.is_file() or not verification_report_path.is_file() or not tokenizer_path.is_file():
        raise TrainingConfigurationError("approved data, verification, or tokenizer artifact is missing")

    data_config_sha256 = sha256_file(data_config_path)
    manifest_sha256 = sha256_file(manifest_path)
    if manifest_sha256 != prepared["manifest_sha256"]:
        raise TrainingConfigurationError("data manifest SHA-256 does not match approved record")
    verification_report_sha256 = sha256_file(verification_report_path)
    if verification_report_sha256 != prepared["verification_report_sha256"]:
        raise TrainingConfigurationError("data verification report SHA-256 does not match approved record")
    verification_report = json.loads(verification_report_path.read_text(encoding="utf-8"))
    if verification_report.get("status") != "PASS":
        raise TrainingConfigurationError("approved data verification report is not PASS")

    tokenizer_sha256 = sha256_file(tokenizer_path)
    if tokenizer_sha256 != prepared["tokenizer_sha256"]:
        raise TrainingConfigurationError("tokenizer SHA-256 does not match approved record")
    try:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    except Exception as exc:  # pragma: no cover - environment dependency failure
        raise TrainingConfigurationError(f"approved tokenizer could not be loaded: {exc}") from exc
    if tokenizer.get_vocab_size(with_added_tokens=True) != config.vocab_size:
        raise TrainingConfigurationError("approved tokenizer vocabulary does not match A0 config")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["tokenizer"]["sha256"] != tokenizer_sha256:
        raise TrainingConfigurationError("manifest tokenizer hash does not match tokenizer artifact")
    if manifest["packing"]["context_length"] != config.context_length:
        raise TrainingConfigurationError("packed-stream context does not match A0 config")

    stream_records = manifest["packing"]["streams"]
    stream_paths: dict[str, Path] = {}
    stream_hashes: dict[str, str] = {}
    for split in ("train", "validation"):
        record = stream_records[split]
        path = repo_root / record["path"]
        if not path.is_file() or path.stat().st_size != record["bytes"]:
            raise TrainingConfigurationError(f"approved {split} token stream is missing or has wrong size")
        actual_hash = sha256_file(path)
        if actual_hash != record["sha256"]:
            raise TrainingConfigurationError(f"approved {split} token stream SHA-256 mismatch")
        stream_paths[split] = path
        stream_hashes[split] = actual_hash

    tokens_per_step = config.micro_batch_size * config.gradient_accumulation_steps * config.context_length
    if config.max_steps * tokens_per_step != config.training_token_budget:
        raise TrainingConfigurationError("A0 training budget does not equal steps times tokens per step")
    if config.evaluation_token_budget > stream_records["validation"]["tokens"]:
        raise TrainingConfigurationError("A0 evaluation budget exceeds validation stream")
    eval_batch_tokens = config.micro_batch_size * config.context_length
    if config.evaluation_token_budget % eval_batch_tokens:
        raise TrainingConfigurationError("A0 evaluation budget is not divisible by an evaluation batch")

    return ApprovedArtifacts(
        data_config_path=data_config_path,
        data_config_sha256=data_config_sha256,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        verification_report_path=verification_report_path,
        verification_report_sha256=verification_report_sha256,
        tokenizer_path=tokenizer_path,
        tokenizer_sha256=tokenizer_sha256,
        train_stream_path=stream_paths["train"],
        train_stream_sha256=stream_hashes["train"],
        validation_stream_path=stream_paths["validation"],
        validation_stream_sha256=stream_hashes["validation"],
        train_tokens=stream_records["train"]["tokens"],
        validation_tokens=stream_records["validation"]["tokens"],
        dataset_revision=manifest["dataset"]["revision"],
    )


class TrainingRunner:
    """Execute or validate an explicitly approved bounded A0/A1 pilot."""

    def __init__(
        self,
        *,
        repo_root: str | Path,
        config_path: str | Path,
        run_dir: str | Path,
        execute_approved: bool = False,
        resume: str | Path | None = None,
        device: str = "cuda",
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.config_path = Path(config_path)
        if not self.config_path.is_absolute():
            self.config_path = self.repo_root / self.config_path
        self.run_dir = Path(run_dir)
        if not self.run_dir.is_absolute():
            self.run_dir = self.repo_root / self.run_dir
        self.execute_approved = execute_approved
        self.resume_path = None if resume is None else Path(resume)
        if self.resume_path is not None and not self.resume_path.is_absolute():
            self.resume_path = self.repo_root / self.resume_path
        self.device_name = device
        self.config = load_config(self.config_path)
        self.config_hash = sha256_file(self.config_path)
        self.artifacts = _load_approved_artifacts(self.repo_root, self.config)
        self.policy = json.loads(
            (self.repo_root / "configs" / "braidnano_a0_integrity_policy.json").read_text(
                encoding="utf-8"
            )
        )
        self.loss_ceiling = float(self.policy["loss_ceiling"])
        self.tokens_per_step = (
            self.config.micro_batch_size
            * self.config.gradient_accumulation_steps
            * self.config.context_length
        )
        self.warmup_steps = max(1, math.ceil(self.config.warmup_ratio * self.config.max_steps))

        _model_class(self.config)
        if self.config.planned_only and not self.execute_approved:
            raise TrainingConfigurationError(
                "the proposal remains planned_only; pass --execute-approved after explicit authorization"
            )
        if self.device_name != "cuda":
            raise TrainingConfigurationError("the approved pilot must run on the host CUDA device")

    def dry_run_report(self) -> dict[str, Any]:
        """Return the verified execution plan without constructing or training."""

        return {
            "status": "READY_TO_EXECUTE" if self.execute_approved else "PLAN_VALID",
            "entrypoint_capabilities": {
                "atomic_checkpoints": True,
                "checkpoint_sha256_sidecars": True,
                "resume_from_checkpoint": True,
                "interrupt_recovery_checkpoint": True,
                "validation_evaluation": True,
                "durable_step_metric_logging": True,
                "bounded_integrity_circuit_breakers": True,
            },
            "gradient_checkpointing": {
                "configured": self.config.gradient_checkpointing,
                "implemented": False,
                "enabled": False,
            },
            "config_path": str(self.config_path.relative_to(self.repo_root)),
            "config_sha256": self.config_hash,
            "data": self.artifacts.metadata(self.repo_root),
            "seed": self.config.seed,
            "tokens_per_step": self.tokens_per_step,
            "warmup_steps": self.warmup_steps,
            "max_steps": self.config.max_steps,
        }

    def _metadata(self, model: LanguageModel, device: torch.device, initial_step: int) -> dict[str, Any]:
        metadata = {
            "experiment_id": self.run_dir.name,
            "status": "running",
            "config_path": str(self.config_path.relative_to(self.repo_root)),
            "config_sha256": self.config_hash,
            "config": self.config.to_dict(),
            "integrity_policy": self.policy,
            "seed": self.config.seed,
            "seed_report": {
                "seed": self.config.seed,
                "deterministic_algorithms_requested": True,
            },
            "parameter_count": model.parameter_count(),
            "device": torch.cuda.get_device_name(device),
            "device_capability": list(torch.cuda.get_device_capability(device)),
            "torch_version": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "gradient_checkpointing": {
                "configured": self.config.gradient_checkpointing,
                "implemented": False,
                "enabled": False,
            },
            "precision_execution": {
                "config": self.config.precision,
                "autocast_dtype": "float16" if self.config.precision == "fp16" else self.config.precision,
                "parameter_dtype": "float32",
            },
            "data_order": "deterministic sequential non-overlapping packed windows",
            "evaluation_policy": {
                "tokens": self.config.evaluation_token_budget,
                "interval_steps": self.config.eval_interval_steps,
                "split": "validation",
                "window_start_token": 0,
            },
            "initial_step": initial_step,
            "checkpoint_policy": {
                "interval_steps": self.config.checkpoint_interval_steps,
                "atomic": True,
                "sha256_sidecar": True,
                "resume_supported": True,
                "interrupt_recovery_supported": True,
            },
        }
        metadata.update(self.artifacts.metadata(self.repo_root))
        if self.resume_path is not None:
            metadata["resumed_from"] = str(self.resume_path)
        return metadata

    def _checkpoint(
        self,
        *,
        model: LanguageModel,
        optimizer: torch.optim.Optimizer,
        step: int,
        tokens_processed: int,
        checkpoint_path: Path,
        kind: str,
    ) -> dict[str, Any]:
        payload = {
            "checkpoint_schema_version": 1,
            "model_state_dict": _cpu_copy(model.state_dict()),
            "optimizer_state_dict": _cpu_copy(optimizer.state_dict()),
            "step": step,
            "tokens_processed": tokens_processed,
            "seed": self.config.seed,
            "scheduler_state": {"last_step": step},
            "rng_state": _rng_state(),
            "gradient_checkpointing": False,
        }
        metadata = {
            "step": step,
            "tokens_processed": tokens_processed,
            "seed": self.config.seed,
            "config_sha256": self.config_hash,
            "data_manifest_sha256": self.artifacts.manifest_sha256,
            "checkpoint_kind": kind,
            "parameter_count": model.parameter_count(),
            "gradient_checkpointing": False,
        }
        return save_atomic_checkpoint(payload, checkpoint_path, metadata)

    def _load_resume(
        self,
        *,
        model: LanguageModel,
        optimizer: torch.optim.Optimizer,
    ) -> tuple[int, int, dict[str, Any]]:
        if self.resume_path is None:
            return 0, 0, {}
        record = verify_checkpoint(self.resume_path)
        if record.get("config_sha256") != self.config_hash:
            raise TrainingConfigurationError("resume checkpoint config hash does not match current config")
        if record.get("data_manifest_sha256") != self.artifacts.manifest_sha256:
            raise TrainingConfigurationError("resume checkpoint data manifest hash does not match")
        if record.get("seed") != self.config.seed:
            raise TrainingConfigurationError("resume checkpoint seed does not match current config")
        payload = torch.load(self.resume_path, map_location="cpu", weights_only=False)
        step = int(payload["step"])
        tokens_processed = int(payload["tokens_processed"])
        if step < 0 or tokens_processed != step * self.tokens_per_step:
            raise TrainingConfigurationError("resume checkpoint progress is inconsistent")
        if step > self.config.max_steps:
            raise TrainingConfigurationError("resume checkpoint is beyond configured max_steps")
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(next(model.parameters()).device)
        _restore_rng_state(payload["rng_state"])
        return step, tokens_processed, record

    def _evaluate(
        self,
        *,
        model: LanguageModel,
        stream: PackedTokenStream,
        device: torch.device,
    ) -> float:
        model.eval()
        batch_tokens = self.config.micro_batch_size * self.config.context_length
        total_loss = 0.0
        total_predictions = 0
        amp_enabled = self.config.precision != "fp32"
        amp_dtype = torch.float16 if self.config.precision == "fp16" else torch.bfloat16
        try:
            with torch.inference_mode():
                for start in range(0, self.config.evaluation_token_budget, batch_tokens):
                    batch = stream.read_tokens(start, batch_tokens).view(
                        self.config.micro_batch_size,
                        self.config.context_length,
                    ).to(device)
                    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                        output = model(batch, labels=batch)
                    if output.loss is None or not bool(torch.isfinite(output.loss).item()):
                        raise IntegrityError("non_finite_validation_loss: validation loss is not finite")
                    predictions = self.config.micro_batch_size * (self.config.context_length - 1)
                    total_loss += float(output.loss.item()) * predictions
                    total_predictions += predictions
        finally:
            model.train()
        if total_predictions == 0:
            raise TrainingConfigurationError("validation evaluation produced no predictions")
        return total_loss / total_predictions

    def run(self) -> dict[str, Any]:
        """Run the bounded pilot and return its durable summary."""

        if not torch.cuda.is_available():
            raise TrainingConfigurationError("CUDA is unavailable in the host training process")
        if self.run_dir.exists() and any(self.run_dir.iterdir()):
            raise TrainingConfigurationError(f"run directory is not empty: {self.run_dir}")
        self.run_dir.mkdir(parents=True, exist_ok=True)

        run_started = time.perf_counter()
        seed_report = set_reproducible_seed(self.config.seed, deterministic=True)
        device = torch.device("cuda")
        model = build_model(self.config).to(device)
        parameter_count = model.parameter_count()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.optimizer_beta1, self.config.optimizer_beta2),
            eps=self.config.optimizer_epsilon,
            weight_decay=self.config.weight_decay,
        )
        initial_step, tokens_processed, resume_record = self._load_resume(
            model=model,
            optimizer=optimizer,
        )
        _set_optimizer_learning_rate(
            optimizer,
            cosine_warmup_learning_rate(
                self.config.learning_rate,
                max(1, initial_step + 1),
                self.warmup_steps,
                self.config.max_steps,
            )
            if initial_step < self.config.max_steps
            else 0.0,
        )
        torch.cuda.reset_peak_memory_stats(device)

        monitor = IntegrityMonitor(
            run_dir=self.run_dir,
            metadata=self._metadata(model, device, initial_step),
            tokens_per_step=self.tokens_per_step,
            loss_ceiling=self.loss_ceiling,
            initial_step=initial_step,
            initial_tokens=tokens_processed,
        )
        run_metadata = json.loads((self.run_dir / "run_metadata.json").read_text(encoding="utf-8"))
        run_metadata["seed_report"] = seed_report
        run_metadata["resume_checkpoint"] = resume_record
        _write_json(self.run_dir / "run_metadata.json", run_metadata)

        stop_requested = False
        previous_handlers: dict[int, Any] = {}

        def request_stop(signum: int, _frame: Any) -> None:
            nonlocal stop_requested
            stop_requested = True
            monitor.warning("interrupt_requested", {"signal": signum})

        for signal_number in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signal_number] = signal.getsignal(signal_number)
            signal.signal(signal_number, request_stop)

        status = "completed"
        failure: dict[str, Any] | None = None
        last_completed_step = initial_step
        last_train_loss: float | None = None
        last_validation_loss: float | None = None
        total_training_wall = 0.0
        total_step_tokens = tokens_processed
        try:
            with (
                PackedTokenStream(
                    self.artifacts.train_stream_path,
                    expected_tokens=self.artifacts.train_tokens,
                ) as train_stream,
                PackedTokenStream(
                    self.artifacts.validation_stream_path,
                    expected_tokens=self.artifacts.validation_tokens,
                ) as validation_stream,
            ):
                for step in range(initial_step + 1, self.config.max_steps + 1):
                    step_start = time.perf_counter()
                    learning_rate = cosine_warmup_learning_rate(
                        self.config.learning_rate,
                        step,
                        self.warmup_steps,
                        self.config.max_steps,
                    )
                    _set_optimizer_learning_rate(optimizer, learning_rate)
                    optimizer.zero_grad(set_to_none=True)
                    losses: list[torch.Tensor] = []
                    amp_enabled = self.config.precision != "fp32"
                    amp_dtype = torch.float16 if self.config.precision == "fp16" else torch.bfloat16
                    for accumulation_index in range(self.config.gradient_accumulation_steps):
                        batch_start = tokens_processed + accumulation_index * (
                            self.config.micro_batch_size * self.config.context_length
                        )
                        batch = train_stream.read_tokens(
                            batch_start,
                            self.config.micro_batch_size * self.config.context_length,
                        ).view(self.config.micro_batch_size, self.config.context_length).to(device)
                        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                            output = model(batch, labels=batch)
                            loss = output.loss
                        if loss is None or not bool(torch.isfinite(loss).item()):
                            monitor.fail(
                                "non_finite_train_loss",
                                {"step": step, "value": None if loss is None else float(loss.item())},
                            )
                        losses.append(loss.detach())
                        (loss / self.config.gradient_accumulation_steps).backward()

                    finite_gradients, gradient_tensor_count, gradient_element_count = _finite_gradients(model)
                    if not finite_gradients:
                        monitor.fail(
                            "non_finite_gradient",
                            {"step": step, "gradient_tensors": gradient_tensor_count},
                        )
                    try:
                        gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            max_norm=self.config.gradient_clip_norm,
                            error_if_nonfinite=True,
                        )
                    except RuntimeError as exc:
                        monitor.fail("non_finite_gradient_norm", {"step": step, "error": str(exc)})
                    gradient_norm = float(gradient_norm_tensor.item())
                    if not math.isfinite(gradient_norm):
                        monitor.fail("non_finite_gradient_norm", {"step": step, "value": gradient_norm})

                    optimizer.step()
                    torch.cuda.synchronize(device)
                    step_elapsed = time.perf_counter() - step_start
                    total_training_wall += step_elapsed
                    train_loss = sum(float(loss.item()) for loss in losses) / len(losses)
                    next_tokens = tokens_processed + self.tokens_per_step
                    validation_loss = None
                    if step % self.config.eval_interval_steps == 0 or step == self.config.max_steps:
                        validation_loss = self._evaluate(
                            model=model,
                            stream=validation_stream,
                            device=device,
                        )
                    if not math.isfinite(train_loss):
                        monitor.fail("non_finite_train_loss", {"step": step, "value": train_loss})
                    if validation_loss is not None and not math.isfinite(validation_loss):
                        monitor.fail("non_finite_validation_loss", {"step": step, "value": validation_loss})
                    record = monitor.record_step(
                        step=step,
                        tokens_processed=next_tokens,
                        train_loss=train_loss,
                        validation_loss=validation_loss,
                        gradient_norm=gradient_norm,
                        learning_rate=learning_rate,
                        tokens_per_second=self.tokens_per_step / step_elapsed,
                        peak_allocated_vram_bytes=torch.cuda.max_memory_allocated(device),
                        peak_reserved_vram_bytes=torch.cuda.max_memory_reserved(device),
                    )
                    tokens_processed = next_tokens
                    total_step_tokens = next_tokens
                    last_completed_step = step
                    last_train_loss = train_loss
                    if validation_loss is not None:
                        last_validation_loss = validation_loss

                    if step % self.config.checkpoint_interval_steps == 0 or step == self.config.max_steps:
                        self._checkpoint(
                            model=model,
                            optimizer=optimizer,
                            step=step,
                            tokens_processed=tokens_processed,
                            checkpoint_path=self.run_dir / f"checkpoint_step_{step:06d}.pt",
                            kind="interval_or_final",
                        )
                    if stop_requested:
                        status = "interrupted"
                        self._checkpoint(
                            model=model,
                            optimizer=optimizer,
                            step=last_completed_step,
                            tokens_processed=tokens_processed,
                            checkpoint_path=self.run_dir / "checkpoint_interrupt.pt",
                            kind="interrupt_recovery",
                        )
                        break
        except KeyboardInterrupt:
            status = "interrupted"
            try:
                self._checkpoint(
                    model=model,
                    optimizer=optimizer,
                    step=last_completed_step,
                    tokens_processed=total_step_tokens,
                    checkpoint_path=self.run_dir / "checkpoint_interrupt.pt",
                    kind="interrupt_recovery",
                )
            except Exception as checkpoint_error:  # pragma: no cover - failure path
                failure = {"type": type(checkpoint_error).__name__, "message": str(checkpoint_error)}
        except Exception as exc:
            status = "failed"
            failure = {"type": type(exc).__name__, "message": str(exc)}
            try:
                monitor.warning("run_failed", failure)
                self._checkpoint(
                    model=model,
                    optimizer=optimizer,
                    step=last_completed_step,
                    tokens_processed=total_step_tokens,
                    checkpoint_path=self.run_dir / "checkpoint_failure.pt",
                    kind="failure_recovery",
                )
            except Exception as checkpoint_error:  # pragma: no cover - failure path
                failure["checkpoint_error"] = str(checkpoint_error)
        finally:
            for signal_number, handler in previous_handlers.items():
                signal.signal(signal_number, handler)

        elapsed_wall = time.perf_counter() - run_started
        summary: dict[str, Any] = {
            "status": status,
            "experiment_id": self.run_dir.name,
            "run_dir": str(self.run_dir),
            "config_path": str(self.config_path.relative_to(self.repo_root)),
            "config_sha256": self.config_hash,
            "data_manifest_sha256": self.artifacts.manifest_sha256,
            "dataset_revision": self.artifacts.dataset_revision,
            "tokenizer_sha256": self.artifacts.tokenizer_sha256,
            "seed": self.config.seed,
            "parameter_count": parameter_count,
            "gradient_checkpointing": {
                "configured": self.config.gradient_checkpointing,
                "implemented": False,
                "enabled": False,
            },
            "optimizer_steps": last_completed_step - initial_step,
            "final_step": last_completed_step,
            "tokens_processed": total_step_tokens,
            "training_token_budget": self.config.training_token_budget,
            "effective_batch_size": self.config.micro_batch_size * self.config.gradient_accumulation_steps,
            "micro_batch_size": self.config.micro_batch_size,
            "context_length": self.config.context_length,
            "train_loss": last_train_loss,
            "validation_loss": last_validation_loss,
            "elapsed_wall_seconds": elapsed_wall,
            "training_wall_seconds": total_training_wall,
            "average_tokens_per_second": (
                total_step_tokens - initial_step * self.tokens_per_step
            )
            / total_training_wall
            if total_training_wall > 0
            else 0.0,
            "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device),
            "torch_version": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "device_capability": list(torch.cuda.get_device_capability(device)),
            "checkpoint": str(self.run_dir / f"checkpoint_step_{last_completed_step:06d}.pt")
            if status == "completed"
            else str(self.run_dir / "checkpoint_interrupt.pt"),
            "failure": failure,
            "numerical_instability": failure is not None and "finite" in failure.get("message", ""),
        }
        _write_json(self.run_dir / "summary.json", summary)
        if status == "failed":
            raise RuntimeError(json.dumps(summary, sort_keys=True))
        return summary


def _default_run_dir(repo_root: Path, seed: int, stage: str = "A0") -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = "a0" if stage == "A0" else stage.lower()
    return repo_root / "results" / "runs" / f"{prefix}_seed{seed}_{stamp}"


def main(argv: list[str] | None = None) -> int:
    """CLI for an explicitly approved A0 or A1 pilot."""

    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=repo_root / "configs" / "braidnano_a0_baseline_proposal.json",
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--execute-approved", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config if args.config.is_absolute() else repo_root / args.config)
    run_dir = args.run_dir or _default_run_dir(repo_root, config.seed, config.stage)
    runner = TrainingRunner(
        repo_root=repo_root,
        config_path=args.config,
        run_dir=run_dir,
        execute_approved=args.execute_approved,
        resume=args.resume,
    )
    if args.dry_run:
        print(json.dumps(runner.dry_run_report(), indent=2, sort_keys=True))
        return 0
    if not args.execute_approved:
        parser.error("training requires --execute-approved after explicit experiment authorization")
    summary = runner.run()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


__all__ = [
    "A0TrainingRunner",
    "LanguageModel",
    "PackedTokenStream",
    "TrainingRunner",
    "TrainingConfigurationError",
    "build_model",
    "cosine_warmup_learning_rate",
    "main",
]


# Backward-compatible public name used by the original A0 entrypoint.  The
# implementation now selects A0 or A1 from the validated configuration.
A0TrainingRunner = TrainingRunner
