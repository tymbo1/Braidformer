#!/usr/bin/env python3
"""Run one approved A0/A1 synthetic recall-battery arm.

The task generator is deterministic and intentionally independent of
TinyStories.  Each run trains one frozen architecture from fresh
initialization, records target-only losses and exact-answer metrics, and saves
one integrity-checked optimizer/RNG checkpoint in an ignored run directory.
"""

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


GENERATOR_VERSION = "synthetic_recall_battery_v1"
TASKS = ("copy", "associative", "selective")
SPLIT_OFFSETS = {"train": 0x13579BDF, "validation": 0x2468ACE0, "test": 0x31415926}
MARKERS = {
    "copy_task": 30000,
    "copy_source": 30001,
    "copy_query": 30002,
    "associative_task": 30003,
    "associative_record": 30004,
    "associative_query": 30005,
    "selective_task": 30006,
    "selective_record": 30007,
    "selective_query": 30008,
    "answer": 30009,
}
NOISE_RANGE = (24000, 29900)
KEY_RANGE = (1000, 7999)
VALUE_RANGE = (8000, 15999)
COPY_RANGE = (16000, 23999)
CONTEXT_LENGTH = 512
MICRO_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
EFFECTIVE_BATCH_SIZE = MICRO_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
MAX_STEPS = 1000
WARMUP_STEPS = 10
VALIDATION_EXAMPLES_PER_TASK = 64
TEST_EXAMPLES_PER_TASK = 128


@dataclass(frozen=True)
class SyntheticExample:
    task: str
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    target: tuple[int, ...]

    @property
    def prefix_length(self) -> int:
        return len(self.input_ids) - len(self.target)


def _derived_seed(base_seed: int, split: str, task: str, index: int) -> int:
    payload = f"{GENERATOR_VERSION}:{base_seed}:{split}:{task}:{index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _random_tokens(rng: random.Random, bounds: tuple[int, int], count: int) -> list[int]:
    return [rng.randrange(bounds[0], bounds[1]) for _ in range(count)]


def _unique_tokens(rng: random.Random, bounds: tuple[int, int], count: int) -> list[int]:
    return rng.sample(range(bounds[0], bounds[1]), count)


def _finish_example(
    *,
    task: str,
    rng: random.Random,
    prefix: list[int],
    query_suffix: list[int],
    target: list[int],
) -> SyntheticExample:
    prefix_length = CONTEXT_LENGTH - len(target)
    if len(prefix) + len(query_suffix) > prefix_length:
        raise ValueError(f"{task} prefix exceeds context budget")
    filler_count = prefix_length - len(prefix) - len(query_suffix)
    prefix = prefix + _random_tokens(rng, NOISE_RANGE, filler_count) + query_suffix
    input_ids = tuple(prefix + target)
    labels = [-100] * CONTEXT_LENGTH
    labels[prefix_length:] = target
    if len(input_ids) != CONTEXT_LENGTH or len(labels) != CONTEXT_LENGTH:
        raise AssertionError("synthetic sequence length invariant failed")
    return SyntheticExample(task, input_ids, tuple(labels), tuple(target))


def make_example(base_seed: int, split: str, task: str, index: int) -> SyntheticExample:
    if split not in SPLIT_OFFSETS:
        raise ValueError(f"unknown split: {split}")
    if task not in TASKS:
        raise ValueError(f"unknown task: {task}")
    seed = _derived_seed(base_seed + SPLIT_OFFSETS[split], split, task, index)
    rng = random.Random(seed)

    if task == "copy":
        target = _random_tokens(rng, COPY_RANGE, 8)
        prefix = [MARKERS["copy_task"], MARKERS["copy_source"], *target]
        query_suffix = [MARKERS["copy_query"], MARKERS["answer"]]
        return _finish_example(
            task=task,
            rng=rng,
            prefix=prefix,
            query_suffix=query_suffix,
            target=target,
        )

    if task == "associative":
        keys = _unique_tokens(rng, KEY_RANGE, 8)
        values = [_random_tokens(rng, VALUE_RANGE, 4) for _ in keys]
        target_index = rng.randrange(len(keys))
        records: list[int] = [MARKERS["associative_task"]]
        for key, value in zip(keys, values):
            records.extend([MARKERS["associative_record"], key, *value])
        query_suffix = [MARKERS["associative_query"], keys[target_index], MARKERS["answer"]]
        return _finish_example(
            task=task,
            rng=rng,
            prefix=records,
            query_suffix=query_suffix,
            target=values[target_index],
        )

    keys = _unique_tokens(rng, KEY_RANGE, 32)
    values = [_random_tokens(rng, VALUE_RANGE, 4) for _ in keys]
    target_index = rng.randrange(len(keys))
    records = [MARKERS["selective_task"]]
    for key, value in zip(keys, values):
        records.extend([MARKERS["selective_record"], key, *value])
    query_suffix = [MARKERS["selective_query"], keys[target_index], MARKERS["answer"]]
    return _finish_example(
        task=task,
        rng=rng,
        prefix=records,
        query_suffix=query_suffix,
        target=values[target_index],
    )


def make_training_batch(base_seed: int, optimizer_step: int) -> tuple[torch.Tensor, torch.Tensor]:
    examples: list[SyntheticExample] = []
    start_index = (optimizer_step - 1) * EFFECTIVE_BATCH_SIZE
    for offset in range(EFFECTIVE_BATCH_SIZE):
        index = start_index + offset
        task = TASKS[index % len(TASKS)]
        examples.append(make_example(base_seed, "train", task, index))
    return (
        torch.tensor([example.input_ids for example in examples], dtype=torch.long),
        torch.tensor([example.labels for example in examples], dtype=torch.long),
    )


def make_evaluation_examples(
    base_seed: int, split: str, examples_per_task: int
) -> dict[str, list[SyntheticExample]]:
    return {
        task: [make_example(base_seed, split, task, index) for index in range(examples_per_task)]
        for task in TASKS
    }


def synthetic_data_fingerprint(base_seed: int) -> str:
    digest = hashlib.sha256()
    for split, count in (("validation", VALIDATION_EXAMPLES_PER_TASK), ("test", TEST_EXAMPLES_PER_TASK)):
        for task in TASKS:
            for example in make_evaluation_examples(base_seed, split, count)[task]:
                digest.update(task.encode("ascii"))
                digest.update(bytes().join(int(token).to_bytes(2, "little") for token in example.input_ids))
                digest.update(bytes().join(int(token).to_bytes(2, "little", signed=True) for token in example.labels))
    return digest.hexdigest()


def generator_spec(base_seed: int) -> dict[str, Any]:
    return {
        "version": GENERATOR_VERSION,
        "base_seed": base_seed,
        "tasks": list(TASKS),
        "split_offsets": SPLIT_OFFSETS,
        "context_length": CONTEXT_LENGTH,
        "training_examples": MAX_STEPS * EFFECTIVE_BATCH_SIZE,
        "validation_examples_per_task": VALIDATION_EXAMPLES_PER_TASK,
        "test_examples_per_task": TEST_EXAMPLES_PER_TASK,
        "markers": MARKERS,
        "noise_range": NOISE_RANGE,
        "key_range": KEY_RANGE,
        "value_range": VALUE_RANGE,
        "copy_range": COPY_RANGE,
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


def _batched(items: list[SyntheticExample], batch_size: int) -> Iterable[list[SyntheticExample]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _target_metrics(
    model: torch.nn.Module,
    examples: dict[str, list[SyntheticExample]],
    device: torch.device,
    batch_size: int = 8,
) -> dict[str, Any]:
    model.eval()
    result: dict[str, Any] = {}
    amp_dtype = torch.float16
    with torch.inference_mode():
        for task in TASKS:
            task_examples = examples[task]
            total_loss = 0.0
            total_tokens = 0
            total_correct = 0
            total_exact = 0
            total_answers = 0
            for batch_examples in _batched(task_examples, batch_size):
                inputs = torch.tensor([example.input_ids for example in batch_examples], dtype=torch.long, device=device)
                labels = torch.tensor([example.labels for example in batch_examples], dtype=torch.long, device=device)
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    output = model(inputs, labels=labels)
                if output.loss is None or not bool(torch.isfinite(output.loss).item()):
                    raise IntegrityError(f"non_finite_{task}_target_loss")
                target_length = len(batch_examples[0].target)
                prefix_length = batch_examples[0].prefix_length
                positions = torch.arange(
                    prefix_length - 1,
                    prefix_length - 1 + target_length,
                    device=device,
                )
                predictions = output.logits[:, positions, :].argmax(dim=-1)
                targets = labels[:, prefix_length:prefix_length + target_length]
                total_loss += float(output.loss.item()) * len(batch_examples)
                total_tokens += len(batch_examples) * target_length
                total_correct += int((predictions == targets).sum().item())
                total_exact += int((predictions == targets).all(dim=1).sum().item())
                total_answers += len(batch_examples)
            result[task] = {
                "target_loss": total_loss / total_answers,
                "teacher_forced_answer_token_accuracy": total_correct / total_tokens,
                "teacher_forced_exact_match": total_exact / total_answers,
                "examples": total_answers,
            }
    model.train()
    return result


def _greedy_metrics(
    model: torch.nn.Module,
    examples: dict[str, list[SyntheticExample]],
    device: torch.device,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Measure autoregressive answer generation without exposing target tokens."""

    model.eval()
    result: dict[str, Any] = {}
    with torch.inference_mode():
        for task in TASKS:
            task_examples = examples[task]
            total_correct = 0
            total_exact = 0
            total_tokens = 0
            total_answers = 0
            for batch_examples in _batched(task_examples, batch_size):
                prefix_length = batch_examples[0].prefix_length
                target_length = len(batch_examples[0].target)
                generated = torch.tensor(
                    [example.input_ids[:prefix_length] for example in batch_examples],
                    dtype=torch.long,
                    device=device,
                )
                targets = torch.tensor(
                    [example.target for example in batch_examples],
                    dtype=torch.long,
                    device=device,
                )
                for _ in range(target_length):
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        output = model(generated)
                    next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    generated = torch.cat((generated, next_token), dim=1)
                predictions = generated[:, prefix_length:prefix_length + target_length]
                total_correct += int((predictions == targets).sum().item())
                total_exact += int((predictions == targets).all(dim=1).sum().item())
                total_tokens += len(batch_examples) * target_length
                total_answers += len(batch_examples)
            result[task] = {
                "greedy_answer_token_accuracy": total_correct / total_tokens,
                "greedy_exact_match": total_exact / total_answers,
                "examples": total_answers,
            }
    model.train()
    return result


def _mean_metric(metrics: dict[str, Any], field: str) -> float:
    return sum(float(metrics[task][field]) for task in TASKS) / len(TASKS)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def run(config_path: Path, run_dir: Path, seed: int, protocol_commit: str) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the host training process")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"run directory is not empty: {run_dir}")

    current_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    if current_commit != protocol_commit:
        raise RuntimeError(f"protocol commit mismatch: current={current_commit}, expected={protocol_commit}")

    config = load_config(config_path)
    if config.context_length != CONTEXT_LENGTH:
        raise RuntimeError("synthetic battery requires context_length=512")
    if config.micro_batch_size != MICRO_BATCH_SIZE or config.gradient_accumulation_steps != GRADIENT_ACCUMULATION_STEPS:
        raise RuntimeError("synthetic battery batch geometry does not match the frozen protocol")
    if config.max_steps != MAX_STEPS or config.training_token_budget != MAX_STEPS * CONTEXT_LENGTH * EFFECTIVE_BATCH_SIZE:
        raise RuntimeError("synthetic battery step/token budget does not match the frozen protocol")
    if config.lr_progress_reference != "optimizer_steps" or config.warmup_ratio != 0.01:
        raise RuntimeError("synthetic battery schedule does not match the frozen protocol")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    run_dir.mkdir(parents=True, exist_ok=True)
    config_hash = sha256_file(config_path)
    script_hash = sha256_file(Path(__file__))
    protocol_path = REPO_ROOT / "docs" / "A0_A1_SYNTHETIC_RECALL_PROTOCOL.md"
    protocol_hash = sha256_file(protocol_path)
    spec = generator_spec(seed)
    generator_hash = sha256_json(spec)
    data_hash = synthetic_data_fingerprint(seed)
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

    validation_examples = make_evaluation_examples(seed, "validation", VALIDATION_EXAMPLES_PER_TASK)
    test_examples = make_evaluation_examples(seed, "test", TEST_EXAMPLES_PER_TASK)
    metadata = {
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
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "gradient_checkpointing": {"configured": True, "implemented": False, "enabled": False},
        "training": {
            "max_steps": MAX_STEPS,
            "tokens_per_step": CONTEXT_LENGTH * EFFECTIVE_BATCH_SIZE,
            "tokens": MAX_STEPS * CONTEXT_LENGTH * EFFECTIVE_BATCH_SIZE,
            "warmup_steps": WARMUP_STEPS,
            "validation_interval_steps": 100,
            "task_balance": "round_robin_copy_associative_selective",
        },
        "checkpoint_policy": {"atomic": True, "sha256_sidecar": True, "optimizer_and_rng_state": True},
        "cuda_timing_scope": "CUDA event from optimizer.zero_grad through optimizer.step; batch preparation and validation excluded",
    }
    _write_json(run_dir / "run_metadata.json", metadata)
    monitor = IntegrityMonitor(
        run_dir=run_dir,
        metadata={
            "config_sha256": config_hash,
            "data_manifest_sha256": data_hash,
            "seed": seed,
            "parameter_count": parameter_count,
            "protocol_commit": protocol_commit,
        },
        tokens_per_step=CONTEXT_LENGTH * EFFECTIVE_BATCH_SIZE,
        initial_step=0,
        initial_tokens=0,
    )
    metrics_path = run_dir / "task_metrics.jsonl"
    run_started = time.perf_counter()
    training_wall_seconds = 0.0
    total_cuda_seconds = 0.0
    last_step = 0
    last_train_loss = None
    last_validation = None
    validation_history: list[dict[str, Any]] = []
    status = "completed"
    failure: dict[str, Any] | None = None
    torch.cuda.reset_peak_memory_stats(device)
    try:
        model.train()
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

            validation = None
            if step % 100 == 0 or step == MAX_STEPS:
                validation = _target_metrics(model, validation_examples, device)
                validation_history.append({"step": step, "metrics": validation})
                last_validation = validation
            monitor.record_step(
                step=step,
                tokens_processed=step * CONTEXT_LENGTH * EFFECTIVE_BATCH_SIZE,
                train_loss=train_loss,
                validation_loss=None if validation is None else _mean_metric(validation, "target_loss"),
                gradient_norm=gradient_norm,
                learning_rate=learning_rate,
                tokens_per_second=(CONTEXT_LENGTH * EFFECTIVE_BATCH_SIZE) / step_wall,
                cuda_elapsed_seconds=cuda_seconds,
                peak_allocated_vram_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_vram_bytes=torch.cuda.max_memory_reserved(device),
            )
            metric_record = {
                "step": step,
                "train_loss": train_loss,
                "learning_rate": learning_rate,
                "cuda_elapsed_seconds": cuda_seconds,
                "wall_clock_seconds": time.perf_counter() - run_started,
                "validation": validation,
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metric_record, sort_keys=True) + "\n")
            last_step = step
            last_train_loss = train_loss

        test_metrics = _target_metrics(model, test_examples, device)
        greedy_test_metrics = _greedy_metrics(model, test_examples, device)
        for task in TASKS:
            test_metrics[task].update(greedy_test_metrics[task])
        checkpoint_path = run_dir / f"checkpoint_step_{MAX_STEPS:06d}.pt"
        checkpoint_payload = {
            "checkpoint_schema_version": 1,
            "model_state_dict": _cpu_copy(model.state_dict()),
            "optimizer_state_dict": _cpu_copy(optimizer.state_dict()),
            "step": MAX_STEPS,
            "tokens_processed": MAX_STEPS * CONTEXT_LENGTH * EFFECTIVE_BATCH_SIZE,
            "seed": seed,
            "rng_state": _rng_state(),
            "synthetic_generator": spec,
            "gradient_checkpointing": False,
        }
        checkpoint_metadata = {
            "step": MAX_STEPS,
            "tokens_processed": MAX_STEPS * CONTEXT_LENGTH * EFFECTIVE_BATCH_SIZE,
            "seed": seed,
            "config_sha256": config_hash,
            "data_manifest_sha256": data_hash,
            "checkpoint_kind": "final",
            "parameter_count": parameter_count,
            "protocol_commit": protocol_commit,
            "gradient_checkpointing": False,
        }
        checkpoint_record = save_atomic_checkpoint(checkpoint_payload, checkpoint_path, checkpoint_metadata)
        verify_checkpoint(checkpoint_path)
    except Exception as exc:
        status = "failed"
        failure = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        elapsed_wall_seconds = time.perf_counter() - run_started
        if last_step == MAX_STEPS and failure is None:
            summary = {
                "status": status,
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
                "tokens_processed": MAX_STEPS * CONTEXT_LENGTH * EFFECTIVE_BATCH_SIZE,
                "training_token_budget": MAX_STEPS * CONTEXT_LENGTH * EFFECTIVE_BATCH_SIZE,
                "warmup_steps": WARMUP_STEPS,
                "learning_rate": config.learning_rate,
                "lr_schedule": "cosine",
                "lr_progress_reference": "optimizer_steps",
                "validation_interval_steps": 100,
                "training_wall_seconds": training_wall_seconds,
                "elapsed_wall_seconds": elapsed_wall_seconds,
                "cuda_training_seconds": total_cuda_seconds,
                "average_tokens_per_second": (MAX_STEPS * CONTEXT_LENGTH * EFFECTIVE_BATCH_SIZE) / training_wall_seconds,
                "average_cuda_tokens_per_second": (MAX_STEPS * CONTEXT_LENGTH * EFFECTIVE_BATCH_SIZE) / total_cuda_seconds,
                "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device),
                "train_loss": last_train_loss,
                "validation_history": validation_history,
                "final_validation": last_validation,
                "test_metrics": test_metrics,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_record["checkpoint_sha256"],
                "checkpoint_integrity_verified": True,
                "gradient_checkpointing": {"configured": True, "implemented": False, "enabled": False},
                "numerical_instability": False,
                "failure": None,
            }
            _write_json(run_dir / "summary.json", summary)
        else:
            _write_json(
                run_dir / "summary.json",
                {
                    "status": status,
                    "architecture": config.stage,
                    "seed": seed,
                    "failure": failure,
                    "final_step": last_step,
                    "cuda_training_seconds": total_cuda_seconds,
                },
            )
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
    summary = run(config_path, run_dir, args.seed, args.protocol_commit)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
