#!/usr/bin/env python3
"""Run one frozen A0/A1 staged-relational curriculum arm.

This runner deliberately reuses the original model classes and optimizer
mechanics from the synthetic solvability work while changing only the
deterministic training-example schedule. It is a separate experiment runner so
the frozen A0/A1 generator and its historical artifacts remain untouched.
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
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import torch

from braidformer.config import load_config
from braidformer.training.integrity import (
    IntegrityError,
    IntegrityMonitor,
    save_atomic_checkpoint,
    set_reproducible_seed,
    sha256_file,
    sha256_json,
    verify_checkpoint,
)
from scripts.run_synthetic_overfit_calibration import (
    CalibrationExample,
    _build_model,
    _cpu_copy,
    _learning_rate,
    _metrics,
    _restore_rng_state,
    _set_learning_rate,
    make_evaluation_examples,
    make_example,
)


CURRICULUM_VERSION = "synthetic_relational_curriculum_v1"
PROTOCOL_PATH = "docs/A0_A1_RELATIONAL_CURRICULUM_V1_PROTOCOL.md"
MAX_STEPS = 2_000
MICRO_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
EFFECTIVE_BATCH_SIZE = MICRO_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
TRAIN_SEQUENCE_LENGTH = 29
TOKENS_PER_STEP = EFFECTIVE_BATCH_SIZE * TRAIN_SEQUENCE_LENGTH
TRAINING_TOKEN_BUDGET = MAX_STEPS * TOKENS_PER_STEP
WARMUP_STEPS = 20
EVAL_INTERVAL = 100
CHECKPOINT_INTERVAL = 100
FIXED_EXAMPLES = 128
PAD_MARKER = 30_005
RELATION_MARKER = 30_006
CHAIN_KEY_A_BASE = 4_000
CHAIN_KEY_B_BASE = 5_000
VALUE_BASE = 8_000


@dataclass(frozen=True)
class Stage:
    name: str
    steps: int


STAGES = (
    Stage("direct64", 250),
    Stage("chain64", 250),
    Stage("direct128", 250),
    Stage("direct1024", 250),
    Stage("associations2_1024", 250),
    Stage("associations4_1024", 250),
    Stage("associations8_8000", 250),
    Stage("stress8000", 250),
)


@dataclass(frozen=True)
class CurriculumExample:
    stage: str
    occurrence: int
    offset: int
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    target: int | tuple[int, ...]


def _stable_int(*parts: object) -> int:
    payload = ":".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _stage_index(name: str) -> int:
    for index, stage in enumerate(STAGES):
        if stage.name == name:
            return index
    raise ValueError(f"unknown curriculum stage: {name}")


def _schedule(seed: int, arm: str) -> list[int]:
    if arm not in {"staged", "shuffled"}:
        raise ValueError(f"unknown curriculum arm: {arm}")
    schedule: list[int] = []
    for index, stage in enumerate(STAGES):
        schedule.extend([index] * stage.steps)
    if len(schedule) != MAX_STEPS:
        raise AssertionError("stage steps must sum to MAX_STEPS")
    if arm == "shuffled":
        random.Random(_stable_int(CURRICULUM_VERSION, seed, "stage-order")).shuffle(schedule)
    return schedule


def _occurrences(schedule: list[int]) -> list[int]:
    seen = [0] * len(STAGES)
    result: list[int] = []
    for stage_index in schedule:
        result.append(seen[stage_index])
        seen[stage_index] += 1
    return result


def _pad_example(
    stage: str,
    occurrence: int,
    offset: int,
    example: CalibrationExample,
) -> CurriculumExample:
    if len(example.input_ids) > TRAIN_SEQUENCE_LENGTH:
        raise ValueError(
            f"curriculum example is longer than fixed training shape: "
            f"{stage} has {len(example.input_ids)} tokens"
        )
    padding = TRAIN_SEQUENCE_LENGTH - len(example.input_ids)
    return CurriculumExample(
        stage=stage,
        occurrence=occurrence,
        offset=offset,
        input_ids=(PAD_MARKER,) * padding + example.input_ids,
        labels=(-100,) * padding + example.labels,
        target=example.target,
    )


def _chain_example(seed: int, occurrence: int, offset: int) -> CalibrationExample:
    index = (occurrence * EFFECTIVE_BATCH_SIZE + offset) % FIXED_EXAMPLES
    rng = random.Random(_stable_int(CURRICULUM_VERSION, seed, "chain64", occurrence, offset))
    key_a = CHAIN_KEY_A_BASE + index
    key_b = CHAIN_KEY_B_BASE + index
    value = VALUE_BASE + rng.randrange(64)
    input_ids = (
        30_000,
        RELATION_MARKER,
        key_a,
        key_b,
        RELATION_MARKER,
        key_b,
        value,
        30_002,
        key_a,
        30_003,
        value,
    )
    labels = (-100,) * (len(input_ids) - 1) + (value,)
    return CalibrationExample("train", index, input_ids, labels, value)


def _direct(
    seed: int,
    occurrence: int,
    offset: int,
    *,
    value_count: int,
    association_count: int = 1,
    distractor_distance: int = 0,
    target_length: int = 1,
) -> CalibrationExample:
    index = (occurrence * EFFECTIVE_BATCH_SIZE + offset) % FIXED_EXAMPLES
    return make_example(
        seed,
        "train",
        index,
        value_count=value_count,
        association_count=association_count,
        distractor_distance=distractor_distance,
        target_length=target_length,
    )


def _associations(
    seed: int,
    occurrence: int,
    offset: int,
    *,
    value_count: int,
    association_count: int,
) -> CalibrationExample:
    """Generate a curriculum-only multi-binding example for any value pool."""

    if association_count not in {2, 4, 8}:
        raise ValueError(f"unsupported curriculum association count: {association_count}")
    index = (occurrence * EFFECTIVE_BATCH_SIZE + offset) % FIXED_EXAMPLES
    rng = random.Random(
        _stable_int(
            CURRICULUM_VERSION,
            seed,
            "associations",
            value_count,
            association_count,
            occurrence,
            offset,
        )
    )
    key_base = 1_000 + index * association_count
    keys = tuple(key_base + item for item in range(association_count))
    values = tuple(VALUE_BASE + item for item in rng.sample(range(value_count), association_count))
    query_index = rng.randrange(association_count)
    record_tokens: list[int] = []
    for key, value in zip(keys, values):
        record_tokens.extend((30_001, key, value))
    target = values[query_index]
    input_ids = tuple(
        [30_000, *record_tokens, 30_002, keys[query_index], 30_003, target]
    )
    labels = (-100,) * (len(input_ids) - 1) + (target,)
    return CalibrationExample("train", index, input_ids, labels, target)


def make_curriculum_example(
    seed: int,
    stage: str,
    occurrence: int,
    offset: int,
) -> CurriculumExample:
    """Create one deterministic fixed-shape curriculum example."""

    if stage == "direct64":
        example = _direct(seed, occurrence, offset, value_count=64)
    elif stage == "chain64":
        example = _chain_example(seed, occurrence, offset)
    elif stage == "direct128":
        example = _direct(seed, occurrence, offset, value_count=128)
    elif stage == "direct1024":
        example = _direct(seed, occurrence, offset, value_count=1024)
    elif stage == "associations2_1024":
        example = _associations(
            seed, occurrence, offset, value_count=1024, association_count=2
        )
    elif stage == "associations4_1024":
        example = _associations(
            seed, occurrence, offset, value_count=1024, association_count=4
        )
    elif stage == "associations8_8000":
        example = _associations(
            seed, occurrence, offset, value_count=8000, association_count=8
        )
    elif stage == "stress8000":
        variant = occurrence % 7
        if variant == 0:
            example = _direct(seed, occurrence, offset, value_count=8000)
        elif variant == 1:
            example = _associations(
                seed, occurrence, offset, value_count=8000, association_count=8
            )
        elif variant == 2:
            example = _direct(
                seed,
                occurrence,
                offset,
                value_count=8000,
                distractor_distance=4,
            )
        elif variant == 3:
            example = _direct(
                seed,
                occurrence,
                offset,
                value_count=8000,
                distractor_distance=16,
            )
        elif variant in {4, 5, 6}:
            target_lengths = {4: 2, 5: 4, 6: 8}
            example = _direct(
                seed,
                occurrence,
                offset,
                value_count=8000,
                target_length=target_lengths[variant],
            )
        else:  # pragma: no cover - variant is modulo seven
            raise AssertionError(f"unexpected stress variant {variant}")
    else:
        raise ValueError(f"unknown curriculum stage: {stage}")
    return _pad_example(stage, occurrence, offset, example)


def make_training_batch(
    seed: int,
    step: int,
    schedule: list[int],
    occurrence_by_step: list[int],
) -> tuple[torch.Tensor, torch.Tensor, str, int]:
    if not 1 <= step <= len(schedule):
        raise ValueError(f"step out of range: {step}")
    stage_index = schedule[step - 1]
    stage = STAGES[stage_index].name
    occurrence = occurrence_by_step[step - 1]
    examples = [
        make_curriculum_example(seed, stage, occurrence, offset)
        for offset in range(EFFECTIVE_BATCH_SIZE)
    ]
    return (
        torch.tensor([example.input_ids for example in examples], dtype=torch.long),
        torch.tensor([example.labels for example in examples], dtype=torch.long),
        stage,
        occurrence,
    )


def _exposure_hash(seed: int) -> str:
    digest = hashlib.sha256()
    for stage in STAGES:
        for occurrence in range(stage.steps):
            for offset in range(EFFECTIVE_BATCH_SIZE):
                example = make_curriculum_example(seed, stage.name, occurrence, offset)
                digest.update(stage.name.encode("ascii"))
                digest.update(occurrence.to_bytes(4, "little"))
                digest.update(offset.to_bytes(2, "little"))
                digest.update(b"".join(token.to_bytes(2, "little") for token in example.input_ids))
                digest.update(
                    b"".join(label.to_bytes(2, "little", signed=True) for label in example.labels)
                )
    return digest.hexdigest()


def curriculum_spec(seed: int, arm: str) -> dict[str, Any]:
    schedule = _schedule(seed, arm)
    occurrences = _occurrences(schedule)
    stage_counts = {stage.name: schedule.count(index) for index, stage in enumerate(STAGES)}
    exposure_hash = _exposure_hash(seed)
    schedule_hash = sha256_json({"schedule": schedule})
    return {
        "version": CURRICULUM_VERSION,
        "seed": seed,
        "arm": arm,
        "max_steps": MAX_STEPS,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "sequence_length": TRAIN_SEQUENCE_LENGTH,
        "tokens_per_step": TOKENS_PER_STEP,
        "training_token_budget": TRAINING_TOKEN_BUDGET,
        "stages": [{"name": stage.name, "steps": stage.steps} for stage in STAGES],
        "stage_counts": stage_counts,
        "schedule": schedule,
        "schedule_sha256": schedule_hash,
        "order_independent_exposure_sha256": exposure_hash,
        "pad_marker": PAD_MARKER,
        "relation_marker": RELATION_MARKER,
        "occurrence_digest": sha256_json({"occurrences": occurrences}),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _final_evaluation_specs() -> tuple[tuple[str, dict[str, int]], ...]:
    return (
        ("values128", {"value_count": 128}),
        ("values256", {"value_count": 256}),
        ("values1024", {"value_count": 1024}),
        ("values8000", {"value_count": 8000}),
        ("associations2_values8000", {"value_count": 8000, "association_count": 2}),
        ("associations4_values8000", {"value_count": 8000, "association_count": 4}),
        ("associations8_values8000", {"value_count": 8000, "association_count": 8}),
        (
            "distance4_values8000",
            {"value_count": 8000, "distractor_distance": 4},
        ),
        (
            "distance16_values8000",
            {"value_count": 8000, "distractor_distance": 16},
        ),
        (
            "distance64_values8000",
            {"value_count": 8000, "distractor_distance": 64},
        ),
        (
            "targetlength2_values8000",
            {"value_count": 8000, "target_length": 2},
        ),
        (
            "targetlength4_values8000",
            {"value_count": 8000, "target_length": 4},
        ),
        (
            "targetlength8_values8000",
            {"value_count": 8000, "target_length": 8},
        ),
    )


def evaluate_frozen_tests(
    model: torch.nn.Module,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, kwargs in _final_evaluation_specs():
        examples = make_evaluation_examples(seed, "heldout", **kwargs)
        result[name] = _metrics(model, examples, device)
    return result


def _build_optimizer(model: torch.nn.Module, config: Any) -> torch.optim.Optimizer:
    if config.optimizer != "AdamW":
        raise RuntimeError(f"protocol requires AdamW, received {config.optimizer!r}")
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.optimizer_beta1, config.optimizer_beta2),
        eps=config.optimizer_epsilon,
        weight_decay=config.weight_decay,
    )


def _validate_config(config: Any, expected_architecture: str) -> None:
    if config.stage != expected_architecture:
        raise RuntimeError(f"config architecture {config.stage!r} != {expected_architecture!r}")
    if config.state_transition_initialization != "uniform_0.9":
        raise RuntimeError("curriculum arm must use the untouched uniform_0.9 initialization")
    checks = {
        "context_length": 512,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "max_steps": MAX_STEPS,
        "training_token_budget": TRAINING_TOKEN_BUDGET,
        "eval_interval_steps": EVAL_INTERVAL,
        "checkpoint_interval_steps": CHECKPOINT_INTERVAL,
    }
    for field, expected in checks.items():
        if getattr(config, field) != expected:
            raise RuntimeError(f"{field}={getattr(config, field)!r} != {expected!r}")
    if config.data_and_tokenizer != CURRICULUM_VERSION:
        raise RuntimeError("config data identity does not match this curriculum")
    if config.lr_progress_reference != "optimizer_steps" or config.warmup_ratio != 0.01:
        raise RuntimeError("curriculum schedule controls do not match the frozen protocol")


def run(
    config_path: Path,
    run_dir: Path,
    seed: int,
    arm: str,
    protocol_commit: str,
    resume_path: Path | None = None,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the host training process")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"run directory is not empty: {run_dir}")
    config = load_config(config_path)
    _validate_config(config, config.stage)
    current_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    if current_commit != protocol_commit:
        raise RuntimeError(
            f"protocol commit mismatch: current={current_commit}, expected={protocol_commit}"
        )

    spec = curriculum_spec(seed, arm)
    schedule = spec["schedule"]
    occurrence_by_step = _occurrences(schedule)
    config_hash = sha256_file(config_path)
    protocol_path = REPO_ROOT / PROTOCOL_PATH
    protocol_hash = sha256_file(protocol_path)
    script_hash = sha256_file(Path(__file__))
    data_manifest_hash = sha256_json(
        {
            "version": CURRICULUM_VERSION,
            "seed": seed,
            "arm": arm,
            "schedule_sha256": spec["schedule_sha256"],
            "order_independent_exposure_sha256": spec["order_independent_exposure_sha256"],
        }
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    seed_report = set_reproducible_seed(seed, deterministic=True)
    device = torch.device("cuda")
    model = _build_model(config).to(device)
    optimizer = _build_optimizer(model, config)
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    expected_parameters = 41_866_240 if config.stage == "A0" else 41_870_336
    if parameter_count != expected_parameters:
        raise RuntimeError(f"unexpected parameter count: {parameter_count} != {expected_parameters}")

    initial_step = 0
    initial_tokens = 0
    total_cuda_seconds = 0.0
    training_wall_seconds = 0.0
    validation_history: list[dict[str, Any]] = []
    metric_history: list[dict[str, Any]] = []
    last_train_loss: float | None = None
    resume_record: dict[str, Any] | None = None
    if resume_path is not None:
        resume_record = verify_checkpoint(resume_path)
        expected_fields = {
            "config_sha256": config_hash,
            "data_manifest_sha256": data_manifest_hash,
            "seed": seed,
            "parameter_count": parameter_count,
            "protocol_commit": protocol_commit,
            "curriculum_schedule_sha256": spec["schedule_sha256"],
        }
        for field, expected in expected_fields.items():
            if resume_record.get(field) != expected:
                raise RuntimeError(
                    f"resume checkpoint metadata mismatch for {field}: "
                    f"{resume_record.get(field)!r} != {expected!r}"
                )
        payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        if payload.get("curriculum_spec") != spec:
            raise RuntimeError("resume checkpoint curriculum spec does not match this run")
        initial_step = int(payload["step"])
        initial_tokens = int(payload["tokens_processed"])
        if not 0 < initial_step < MAX_STEPS:
            raise RuntimeError("resume checkpoint must be an interior checkpoint")
        if initial_tokens != initial_step * TOKENS_PER_STEP:
            raise RuntimeError("resume checkpoint token progress is inconsistent")
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        _restore_rng_state(payload["rng_state"])
        total_cuda_seconds = float(payload.get("total_cuda_seconds", 0.0))
        training_wall_seconds = float(payload.get("training_wall_seconds", 0.0))
        validation_history = list(payload.get("validation_history", []))
        metric_history = list(payload.get("metric_history", []))
        last_train_loss = payload.get("last_train_loss")

    run_metadata = {
        "experiment_id": run_dir.name,
        "status": "running",
        "campaign": "relational_curriculum_v1",
        "arm": arm,
        "architecture": config.stage,
        "config_path": str(config_path.relative_to(REPO_ROOT)),
        "config_sha256": config_hash,
        "protocol_commit": protocol_commit,
        "protocol_sha256": protocol_hash,
        "script_sha256": script_hash,
        "curriculum_version": CURRICULUM_VERSION,
        "curriculum_spec_sha256": sha256_json(spec),
        "data_manifest_sha256": data_manifest_hash,
        "order_independent_exposure_sha256": spec["order_independent_exposure_sha256"],
        "schedule_sha256": spec["schedule_sha256"],
        "seed": seed,
        "seed_report": seed_report,
        "parameter_count": parameter_count,
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "gradient_checkpointing": {
            "configured": True,
            "implemented": False,
            "enabled": False,
        },
        "training": {
            "max_steps": MAX_STEPS,
            "sequence_length": TRAIN_SEQUENCE_LENGTH,
            "tokens_per_step": TOKENS_PER_STEP,
            "tokens": TRAINING_TOKEN_BUDGET,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "micro_batch_size": MICRO_BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "warmup_steps": WARMUP_STEPS,
            "validation_interval_steps": EVAL_INTERVAL,
            "checkpoint_interval_steps": CHECKPOINT_INTERVAL,
        },
        "initial_step": initial_step,
        "initial_tokens": initial_tokens,
        "resume_checkpoint": None if resume_path is None else str(resume_path),
        "cuda_timing_scope": "CUDA event from optimizer.zero_grad through optimizer.step; evaluation excluded",
    }
    monitor = IntegrityMonitor(
        run_dir=run_dir,
        metadata=run_metadata,
        tokens_per_step=TOKENS_PER_STEP,
        initial_step=initial_step,
        initial_tokens=initial_tokens,
    )
    metrics_path = run_dir / "curriculum_metrics.jsonl"
    if metric_history:
        metrics_path.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in metric_history),
            encoding="utf-8",
        )

    validation_examples = make_evaluation_examples(seed, "heldout", value_count=8000)
    run_started = time.perf_counter()
    checkpoint_path: Path | None = None
    checkpoint_record: dict[str, Any] | None = None
    last_step = initial_step
    status = "completed"
    failure: dict[str, Any] | None = None

    def save_progress_checkpoint(step: int, kind: str) -> None:
        nonlocal checkpoint_path, checkpoint_record
        checkpoint_path = run_dir / f"checkpoint_step_{step:06d}.pt"
        checkpoint_record = save_atomic_checkpoint(
            {
                "checkpoint_schema_version": 2,
                "model_state_dict": _cpu_copy(model.state_dict()),
                "optimizer_state_dict": _cpu_copy(optimizer.state_dict()),
                "step": step,
                "tokens_processed": step * TOKENS_PER_STEP,
                "seed": seed,
                "rng_state": _cpu_copy({
                    "python": random.getstate(),
                    "torch": torch.get_rng_state(),
                    "torch_cuda": torch.cuda.get_rng_state_all(),
                }),
                "curriculum_spec": spec,
                "total_cuda_seconds": total_cuda_seconds,
                "training_wall_seconds": training_wall_seconds,
                "validation_history": validation_history,
                "metric_history": metric_history,
                "last_train_loss": last_train_loss,
                "gradient_checkpointing": False,
            },
            checkpoint_path,
            {
                "step": step,
                "tokens_processed": step * TOKENS_PER_STEP,
                "seed": seed,
                "config_sha256": config_hash,
                "data_manifest_sha256": data_manifest_hash,
                "checkpoint_kind": kind,
                "parameter_count": parameter_count,
                "protocol_commit": protocol_commit,
                "curriculum_schedule_sha256": spec["schedule_sha256"],
                "curriculum_version": CURRICULUM_VERSION,
                "arm": arm,
                "gradient_checkpointing": False,
            },
        )
        verify_checkpoint(checkpoint_path)

    try:
        model.train()
        torch.cuda.reset_peak_memory_stats(device)
        for step in range(initial_step + 1, MAX_STEPS + 1):
            step_started = time.perf_counter()
            learning_rate = _learning_rate(config.learning_rate, step, MAX_STEPS, WARMUP_STEPS)
            _set_learning_rate(optimizer, learning_rate)
            cpu_inputs, cpu_labels, stage_name, stage_occurrence = make_training_batch(
                seed, step, schedule, occurrence_by_step
            )
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
                    monitor.fail("non_finite_train_loss", {"step": step, "stage": stage_name})
                losses.append(loss.detach())
                (loss / GRADIENT_ACCUMULATION_STEPS).backward()
            gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=config.gradient_clip_norm, error_if_nonfinite=True
            )
            gradient_norm = float(gradient_norm_tensor.item())
            if not math.isfinite(gradient_norm):
                monitor.fail("non_finite_gradient_norm", {"step": step, "stage": stage_name})
            optimizer.step()
            end_event.record()
            end_event.synchronize()
            cuda_seconds = start_event.elapsed_time(end_event) / 1000.0
            total_cuda_seconds += cuda_seconds
            step_wall = time.perf_counter() - step_started
            training_wall_seconds += step_wall
            train_loss = sum(float(loss.item()) for loss in losses) / len(losses)
            evaluation = None
            if step % EVAL_INTERVAL == 0 or step == MAX_STEPS:
                validation_metrics = _metrics(model, validation_examples, device)
                validation_history.append({"step": step, "metrics": validation_metrics})
                evaluation = validation_metrics
            monitor.record_step(
                step=step,
                tokens_processed=step * TOKENS_PER_STEP,
                train_loss=train_loss,
                validation_loss=None if evaluation is None else evaluation["target_loss"],
                gradient_norm=gradient_norm,
                learning_rate=learning_rate,
                tokens_per_second=TOKENS_PER_STEP / step_wall,
                cuda_elapsed_seconds=cuda_seconds,
                peak_allocated_vram_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_vram_bytes=torch.cuda.max_memory_reserved(device),
            )
            metric_record = {
                "step": step,
                "stage": stage_name,
                "stage_occurrence": stage_occurrence,
                "train_loss": train_loss,
                "learning_rate": learning_rate,
                "cuda_elapsed_seconds": cuda_seconds,
                "wall_clock_seconds": time.perf_counter() - run_started,
                "evaluation": evaluation,
            }
            metric_history.append(metric_record)
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metric_record, sort_keys=True) + "\n")
            last_step = step
            last_train_loss = train_loss
            if step % CHECKPOINT_INTERVAL == 0 or step == MAX_STEPS:
                save_progress_checkpoint(step, "interval_or_final")

        if last_step != MAX_STEPS or checkpoint_path is None or checkpoint_record is None:
            raise RuntimeError("completed run did not reach the final checkpoint")
        final_evaluation = evaluate_frozen_tests(model, seed, device)
    except KeyboardInterrupt as exc:
        status = "interrupted"
        failure = {"type": type(exc).__name__, "message": str(exc)}
        if last_step > initial_step:
            save_progress_checkpoint(last_step, "interrupt_recovery")
        final_evaluation = None
    except Exception as exc:
        status = "failed"
        failure = {"type": type(exc).__name__, "message": str(exc)}
        if last_step > initial_step:
            save_progress_checkpoint(last_step, "failure_recovery")
        final_evaluation = None
        raise

    elapsed_wall_seconds = time.perf_counter() - run_started
    summary = {
        "status": status,
        "experiment_id": run_dir.name,
        "run_dir": str(run_dir),
        "campaign": "relational_curriculum_v1",
        "arm": arm,
        "architecture": config.stage,
        "config_path": str(config_path.relative_to(REPO_ROOT)),
        "config_sha256": config_hash,
        "protocol_commit": protocol_commit,
        "protocol_sha256": protocol_hash,
        "script_sha256": script_hash,
        "curriculum_version": CURRICULUM_VERSION,
        "curriculum_spec_sha256": sha256_json(spec),
        "data_manifest_sha256": data_manifest_hash,
        "order_independent_exposure_sha256": spec["order_independent_exposure_sha256"],
        "schedule_sha256": spec["schedule_sha256"],
        "seed": seed,
        "parameter_count": parameter_count,
        "optimizer_steps": last_step - initial_step,
        "tokens_processed": last_step * TOKENS_PER_STEP,
        "training_token_budget": TRAINING_TOKEN_BUDGET,
        "sequence_length": TRAIN_SEQUENCE_LENGTH,
        "tokens_per_step": TOKENS_PER_STEP,
        "warmup_steps": WARMUP_STEPS,
        "learning_rate": config.learning_rate,
        "lr_schedule": "cosine",
        "cuda_training_seconds": total_cuda_seconds,
        "training_wall_seconds": training_wall_seconds,
        "elapsed_wall_seconds": elapsed_wall_seconds,
        "checkpoint": None if checkpoint_path is None else str(checkpoint_path),
        "checkpoint_sha256": None if checkpoint_record is None else checkpoint_record["checkpoint_sha256"],
        "checkpoint_integrity_verified": checkpoint_record is not None,
        "validation_history": validation_history,
        "final_evaluation": final_evaluation,
        "failure": failure,
        "gradient_checkpointing": {"configured": True, "implemented": False, "enabled": False},
    }
    _write_json(run_dir / "summary.json", summary)
    return summary


def _plan(config_path: Path, arm: str, seed: int) -> dict[str, Any]:
    config = load_config(config_path)
    _validate_config(config, config.stage)
    spec = curriculum_spec(seed, arm)
    return {
        "config": str(config_path.relative_to(REPO_ROOT)),
        "architecture": config.stage,
        "arm": arm,
        "curriculum_version": CURRICULUM_VERSION,
        "max_steps": MAX_STEPS,
        "tokens_per_step": TOKENS_PER_STEP,
        "training_token_budget": TRAINING_TOKEN_BUDGET,
        "parameter_count": 41_866_240 if config.stage == "A0" else 41_870_336,
        "schedule_sha256": spec["schedule_sha256"],
        "order_independent_exposure_sha256": spec["order_independent_exposure_sha256"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm", choices=("staged", "shuffled"), required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--protocol-commit", required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--execute-approved", action="store_true")
    args = parser.parse_args(argv)
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    if args.plan:
        print(json.dumps(_plan(config_path, args.arm, args.seed), indent=2, sort_keys=True))
        return 0
    if not args.execute_approved:
        raise SystemExit("training is gated; pass --execute-approved after protocol authorization")
    if args.run_dir is None:
        raise SystemExit("--run-dir is required for execution")
    run_dir = args.run_dir if args.run_dir.is_absolute() else REPO_ROOT / args.run_dir
    resume_path = None if args.resume is None else (
        args.resume if args.resume.is_absolute() else REPO_ROOT / args.resume
    )
    summary = run(
        config_path=config_path,
        run_dir=run_dir,
        seed=args.seed,
        arm=args.arm,
        protocol_commit=args.protocol_commit,
        resume_path=resume_path,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
