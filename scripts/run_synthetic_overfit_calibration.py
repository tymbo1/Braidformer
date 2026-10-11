#!/usr/bin/env python3
"""Run one approved fixed-set A0/A1 solvability calibration arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import torch

from braidformer.config import ExperimentConfig, load_config
from braidformer.models import A0TransformerLM, A1RecurrentStateSpaceLM
from braidformer.training.integrity import (
    IntegrityError,
    IntegrityMonitor,
    save_atomic_checkpoint,
    set_reproducible_seed,
    sha256_file,
    sha256_json,
    verify_checkpoint,
)


GENERATOR_VERSION = "synthetic_overfit_calibration_v1"
PROTOCOL_COMMIT_REQUIRED = "pending"
SEQUENCE_LENGTH = 8
FIXED_EXAMPLES = 128
MICRO_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
EFFECTIVE_BATCH_SIZE = MICRO_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
MAX_STEPS = 2_000
WARMUP_STEPS = 20
EVAL_INTERVAL = 100
KEY_TRAIN_BASE = 1_000
KEY_HELDOUT_BASE = 2_000
VALUE_BASE = 8_000
VALUE_COUNT = 64
MARKERS = {"task": 30_000, "record": 30_001, "query": 30_002, "answer": 30_003}
SPLIT_OFFSETS = {"train": 0x13579BDF, "heldout": 0x2468ACE0}


@dataclass(frozen=True)
class CalibrationExample:
    split: str
    index: int
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    target: int

    @property
    def prefix_length(self) -> int:
        return SEQUENCE_LENGTH - 1


def _derived_seed(base_seed: int, split: str, index: int) -> int:
    payload = f"{GENERATOR_VERSION}:{base_seed}:{split}:{index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def make_example(base_seed: int, split: str, index: int) -> CalibrationExample:
    if split not in SPLIT_OFFSETS:
        raise ValueError(f"unknown split: {split}")
    if not 0 <= index < FIXED_EXAMPLES:
        raise ValueError(f"index out of range: {index}")
    rng = random.Random(_derived_seed(base_seed + SPLIT_OFFSETS[split], split, index))
    key_base = KEY_TRAIN_BASE if split == "train" else KEY_HELDOUT_BASE
    key = key_base + index
    value = VALUE_BASE + rng.randrange(VALUE_COUNT)
    input_ids = (
        MARKERS["task"],
        MARKERS["record"],
        key,
        value,
        MARKERS["query"],
        key,
        MARKERS["answer"],
        value,
    )
    labels = (-100, -100, -100, -100, -100, -100, -100, value)
    return CalibrationExample(split, index, input_ids, labels, value)


def make_training_batch(base_seed: int, optimizer_step: int) -> tuple[torch.Tensor, torch.Tensor]:
    examples = [
        make_example(base_seed, "train", ((optimizer_step - 1) * EFFECTIVE_BATCH_SIZE + offset) % FIXED_EXAMPLES)
        for offset in range(EFFECTIVE_BATCH_SIZE)
    ]
    return (
        torch.tensor([example.input_ids for example in examples], dtype=torch.long),
        torch.tensor([example.labels for example in examples], dtype=torch.long),
    )


def make_evaluation_examples(base_seed: int, split: str) -> list[CalibrationExample]:
    return [make_example(base_seed, split, index) for index in range(FIXED_EXAMPLES)]


def generator_spec(base_seed: int) -> dict[str, Any]:
    return {
        "version": GENERATOR_VERSION,
        "base_seed": base_seed,
        "sequence_length": SEQUENCE_LENGTH,
        "fixed_examples": FIXED_EXAMPLES,
        "key_train_base": KEY_TRAIN_BASE,
        "key_heldout_base": KEY_HELDOUT_BASE,
        "value_base": VALUE_BASE,
        "value_count": VALUE_COUNT,
        "markers": MARKERS,
        "split_offsets": SPLIT_OFFSETS,
    }


def _cpu_copy(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return value


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"python": random.getstate(), "torch": torch.get_rng_state()}
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:
        state["numpy"] = None
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return _cpu_copy(state)


def _set_learning_rate(optimizer: torch.optim.Optimizer, value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = value


def _learning_rate(base: float, step: int) -> float:
    if step <= WARMUP_STEPS:
        return base * step / WARMUP_STEPS
    progress = min(1.0, max(0.0, (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)))
    return base * 0.5 * (1.0 + math.cos(math.pi * progress))


def _build_model(config: ExperimentConfig) -> torch.nn.Module:
    if config.stage == "A0":
        return A0TransformerLM(config)
    return A1RecurrentStateSpaceLM(config, use_parallel_scan=True)


def _batched(items: list[CalibrationExample], batch_size: int) -> Iterable[list[CalibrationExample]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _metrics(
    model: torch.nn.Module,
    examples: list[CalibrationExample],
    device: torch.device,
    batch_size: int = 16,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_exact = 0
    total_examples = 0
    with torch.inference_mode():
        for batch_examples in _batched(examples, batch_size):
            inputs = torch.tensor([example.input_ids for example in batch_examples], dtype=torch.long, device=device)
            labels = torch.tensor([example.labels for example in batch_examples], dtype=torch.long, device=device)
            targets = torch.tensor([example.target for example in batch_examples], dtype=torch.long, device=device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model(inputs, labels=labels)
            if output.loss is None or not bool(torch.isfinite(output.loss).item()):
                raise IntegrityError("non_finite_calibration_loss")
            predictions = output.logits[:, -2, :].argmax(dim=-1)
            total_loss += float(output.loss.item()) * len(batch_examples)
            total_correct += int((predictions == targets).sum().item())
            total_exact += int((predictions == targets).sum().item())
            total_examples += len(batch_examples)
    model.train()
    return {
        "target_loss": total_loss / total_examples,
        "greedy_answer_token_accuracy": total_correct / total_examples,
        "greedy_exact_match": total_exact / total_examples,
        "examples": total_examples,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _data_hash(base_seed: int) -> str:
    digest = hashlib.sha256()
    for split in ("train", "heldout"):
        for example in make_evaluation_examples(base_seed, split):
            digest.update(split.encode("ascii"))
            digest.update(
                b"".join(int(token).to_bytes(2, "little") for token in example.input_ids)
            )
            digest.update(
                b"".join(int(label).to_bytes(2, "little", signed=True) for label in example.labels)
            )
    return digest.hexdigest()


def run(config_path: Path, run_dir: Path, seed: int, protocol_commit: str) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the host training process")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"run directory is not empty: {run_dir}")
    current_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    if current_commit != protocol_commit:
        raise RuntimeError(f"protocol commit mismatch: current={current_commit}, expected={protocol_commit}")

    config = load_config(config_path)
    expected_tokens = MAX_STEPS * SEQUENCE_LENGTH * EFFECTIVE_BATCH_SIZE
    if config.context_length != 512 or config.micro_batch_size != MICRO_BATCH_SIZE:
        raise RuntimeError("calibration model geometry does not match the frozen protocol")
    if config.gradient_accumulation_steps != GRADIENT_ACCUMULATION_STEPS:
        raise RuntimeError("calibration accumulation does not match the frozen protocol")
    if config.max_steps != MAX_STEPS or config.training_token_budget != expected_tokens:
        raise RuntimeError("calibration budget does not match the frozen protocol")
    if config.lr_progress_reference != "optimizer_steps" or config.warmup_ratio != 0.01:
        raise RuntimeError("calibration schedule does not match the frozen protocol")

    run_dir.mkdir(parents=True, exist_ok=True)
    config_hash = sha256_file(config_path)
    script_hash = sha256_file(Path(__file__))
    protocol_path = REPO_ROOT / "docs" / "A0_A1_SOLVABILITY_CALIBRATION_PROTOCOL.md"
    protocol_hash = sha256_file(protocol_path)
    spec = generator_spec(seed)
    generator_hash = sha256_json(spec)
    data_hash = _data_hash(seed)
    seed_report = set_reproducible_seed(seed, deterministic=True)
    device = torch.device("cuda")
    model = _build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.optimizer_beta1, config.optimizer_beta2),
        eps=config.optimizer_epsilon,
        weight_decay=config.weight_decay,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    expected_parameters = 41_866_240 if config.stage == "A0" else 41_870_336
    if parameter_count != expected_parameters:
        raise RuntimeError(f"unexpected parameter count: {parameter_count} != {expected_parameters}")

    train_examples = make_evaluation_examples(seed, "train")
    heldout_examples = make_evaluation_examples(seed, "heldout")
    _write_json(
        run_dir / "run_metadata.json",
        {
            "experiment_id": run_dir.name,
            "status": "running",
            "config_path": str(config_path.relative_to(REPO_ROOT)),
            "config_sha256": config_hash,
            "protocol_commit": protocol_commit,
            "protocol_sha256": protocol_hash,
            "script_sha256": script_hash,
            "synthetic_generator_version": GENERATOR_VERSION,
            "synthetic_generator_sha256": generator_hash,
            "synthetic_data_sha256": data_hash,
            "seed": seed,
            "seed_report": seed_report,
            "parameter_count": parameter_count,
            "device": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "gradient_checkpointing": {"configured": True, "implemented": False, "enabled": False},
            "training": {
                "max_steps": MAX_STEPS,
                "sequence_length": SEQUENCE_LENGTH,
                "tokens_per_step": SEQUENCE_LENGTH * EFFECTIVE_BATCH_SIZE,
                "tokens": expected_tokens,
                "fixed_training_examples": FIXED_EXAMPLES,
                "warmup_steps": WARMUP_STEPS,
                "validation_interval_steps": EVAL_INTERVAL,
            },
            "cuda_timing_scope": "CUDA event from optimizer.zero_grad through optimizer.step; evaluation excluded",
        },
    )
    monitor = IntegrityMonitor(
        run_dir=run_dir,
        metadata={
            "config_sha256": config_hash,
            "data_manifest_sha256": data_hash,
            "seed": seed,
            "parameter_count": parameter_count,
            "protocol_commit": protocol_commit,
        },
        tokens_per_step=SEQUENCE_LENGTH * EFFECTIVE_BATCH_SIZE,
    )
    metrics_path = run_dir / "calibration_metrics.jsonl"
    run_started = time.perf_counter()
    training_wall_seconds = 0.0
    total_cuda_seconds = 0.0
    validation_history: list[dict[str, Any]] = []
    last_train_loss: float | None = None
    last_step = 0
    failure: dict[str, Any] | None = None
    try:
        model.train()
        torch.cuda.reset_peak_memory_stats(device)
        for step in range(1, MAX_STEPS + 1):
            step_started = time.perf_counter()
            learning_rate = _learning_rate(config.learning_rate, step)
            _set_learning_rate(optimizer, learning_rate)
            cpu_inputs, cpu_labels = make_training_batch(seed, step)
            device_inputs = cpu_inputs.to(device)
            device_labels = cpu_labels.to(device)
            torch.cuda.synchronize(device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            optimizer.zero_grad(set_to_none=True)
            losses: list[torch.Tensor] = []
            for accumulation_index in range(GRADIENT_ACCUMULATION_STEPS):
                start = accumulation_index * MICRO_BATCH_SIZE
                end = start + MICRO_BATCH_SIZE
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = model(device_inputs[start:end], labels=device_labels[start:end])
                    loss = output.loss
                if loss is None or not bool(torch.isfinite(loss).item()):
                    monitor.fail("non_finite_train_loss", {"step": step})
                losses.append(loss.detach())
                (loss / GRADIENT_ACCUMULATION_STEPS).backward()
            gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=config.gradient_clip_norm, error_if_nonfinite=True
            )
            gradient_norm = float(gradient_norm_tensor.item())
            if not math.isfinite(gradient_norm):
                monitor.fail("non_finite_gradient_norm", {"step": step})
            optimizer.step()
            end_event.record()
            end_event.synchronize()
            cuda_seconds = start_event.elapsed_time(end_event) / 1000.0
            total_cuda_seconds += cuda_seconds
            step_wall = time.perf_counter() - step_started
            training_wall_seconds += step_wall
            train_loss = sum(float(loss.item()) for loss in losses) / len(losses)
            if not math.isfinite(train_loss):
                monitor.fail("non_finite_train_loss", {"step": step, "value": train_loss})
            evaluation = None
            if step % EVAL_INTERVAL == 0 or step == MAX_STEPS:
                train_metrics = _metrics(model, train_examples, device)
                heldout_metrics = _metrics(model, heldout_examples, device)
                evaluation = {"train": train_metrics, "heldout": heldout_metrics}
                validation_history.append({"step": step, "metrics": evaluation})
            monitor.record_step(
                step=step,
                tokens_processed=step * SEQUENCE_LENGTH * EFFECTIVE_BATCH_SIZE,
                train_loss=train_loss,
                validation_loss=None if evaluation is None else evaluation["heldout"]["target_loss"],
                gradient_norm=gradient_norm,
                learning_rate=learning_rate,
                tokens_per_second=(SEQUENCE_LENGTH * EFFECTIVE_BATCH_SIZE) / step_wall,
                cuda_elapsed_seconds=cuda_seconds,
                peak_allocated_vram_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_vram_bytes=torch.cuda.max_memory_reserved(device),
            )
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "step": step,
                            "train_loss": train_loss,
                            "learning_rate": learning_rate,
                            "cuda_elapsed_seconds": cuda_seconds,
                            "wall_clock_seconds": time.perf_counter() - run_started,
                            "evaluation": evaluation,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            last_step = step
            last_train_loss = train_loss

        final_train = _metrics(model, train_examples, device)
        final_heldout = _metrics(model, heldout_examples, device)
        checkpoint_path = run_dir / f"checkpoint_step_{MAX_STEPS:06d}.pt"
        checkpoint_record = save_atomic_checkpoint(
            {
                "checkpoint_schema_version": 1,
                "model_state_dict": _cpu_copy(model.state_dict()),
                "optimizer_state_dict": _cpu_copy(optimizer.state_dict()),
                "step": MAX_STEPS,
                "tokens_processed": expected_tokens,
                "seed": seed,
                "rng_state": _rng_state(),
                "synthetic_generator": spec,
                "gradient_checkpointing": False,
            },
            checkpoint_path,
            {
                "step": MAX_STEPS,
                "tokens_processed": expected_tokens,
                "seed": seed,
                "config_sha256": config_hash,
                "data_manifest_sha256": data_hash,
                "checkpoint_kind": "final",
                "parameter_count": parameter_count,
                "protocol_commit": protocol_commit,
                "gradient_checkpointing": False,
            },
        )
        verify_checkpoint(checkpoint_path)
    except Exception as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        elapsed_wall_seconds = time.perf_counter() - run_started
        if last_step == MAX_STEPS and failure is None:
            summary = {
                "status": "completed",
                "experiment_id": run_dir.name,
                "run_dir": str(run_dir),
                "architecture": config.stage,
                "config_path": str(config_path.relative_to(REPO_ROOT)),
                "config_sha256": config_hash,
                "protocol_commit": protocol_commit,
                "protocol_sha256": protocol_hash,
                "script_sha256": script_hash,
                "synthetic_generator_version": GENERATOR_VERSION,
                "synthetic_generator_sha256": generator_hash,
                "synthetic_data_sha256": data_hash,
                "seed": seed,
                "parameter_count": parameter_count,
                "optimizer_steps": MAX_STEPS,
                "sequence_length": SEQUENCE_LENGTH,
                "tokens_processed": expected_tokens,
                "warmup_steps": WARMUP_STEPS,
                "learning_rate": config.learning_rate,
                "lr_schedule": "cosine",
                "training_wall_seconds": training_wall_seconds,
                "elapsed_wall_seconds": elapsed_wall_seconds,
                "cuda_training_seconds": total_cuda_seconds,
                "average_tokens_per_second": expected_tokens / training_wall_seconds,
                "average_cuda_tokens_per_second": expected_tokens / total_cuda_seconds,
                "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device),
                "train_loss": last_train_loss,
                "validation_history": validation_history,
                "final_train": final_train,
                "final_heldout": final_heldout,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_record["checkpoint_sha256"],
                "checkpoint_integrity_verified": True,
                "gradient_checkpointing": {"configured": True, "implemented": False, "enabled": False},
                "numerical_instability": False,
                "failure": None,
            }
        else:
            summary = {
                "status": "failed",
                "architecture": config.stage,
                "seed": seed,
                "failure": failure,
                "final_step": last_step,
                "cuda_training_seconds": total_cuda_seconds,
            }
        _write_json(run_dir / "summary.json", summary)
    return _load_json(run_dir / "summary.json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--protocol-commit", required=True)
    parser.add_argument("--execute-approved", action="store_true")
    args = parser.parse_args()
    if not args.execute_approved:
        raise SystemExit("training is gated; pass --execute-approved after protocol authorization")
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    run_dir = args.run_dir if args.run_dir.is_absolute() else REPO_ROOT / args.run_dir
    print(json.dumps(run(config_path, run_dir, args.seed, args.protocol_commit), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
