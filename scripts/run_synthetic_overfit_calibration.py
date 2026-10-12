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


GENERATOR_VERSIONS = {
    64: "synthetic_overfit_calibration_v1",
    128: "synthetic_solvability_values128_v1",
    256: "synthetic_solvability_values256_v1",
    1024: "synthetic_solvability_values1024_v1",
    8000: "synthetic_solvability_values8000_v1",
}
GENERATOR_VERSION = GENERATOR_VERSIONS[64]
PROTOCOL_PATHS = {
    64: "docs/A0_A1_SOLVABILITY_CALIBRATION_PROTOCOL.md",
    128: "docs/A0_A1_SOLVABILITY_VALUES128_PROTOCOL.md",
    256: "docs/A0_A1_SOLVABILITY_VALUES256_PROTOCOL.md",
    1024: "docs/A0_A1_SOLVABILITY_VALUES1024_PROTOCOL.md",
    8000: "docs/A0_A1_SOLVABILITY_VALUES8000_PROTOCOL.md",
}
ASSOCIATION_GENERATOR_VERSIONS = {
    (8000, 2): "synthetic_solvability_associations2_values8000_v1",
    (8000, 4): "synthetic_solvability_associations4_values8000_v1",
    (8000, 8): "synthetic_solvability_associations8_values8000_v1",
}
ASSOCIATION_PROTOCOL_PATHS = {
    (8000, 2): "docs/A0_A1_SOLVABILITY_ASSOCIATIONS2_VALUES8000_PROTOCOL.md",
    (8000, 4): "docs/A0_A1_SOLVABILITY_ASSOCIATIONS4_VALUES8000_PROTOCOL.md",
    (8000, 8): "docs/A0_A1_SOLVABILITY_ASSOCIATIONS8_VALUES8000_PROTOCOL.md",
}
DISTRACTOR_GENERATOR_VERSIONS = {
    (8000, 1, 4): "synthetic_solvability_distance4_values8000_v1",
    (8000, 1, 16): "synthetic_solvability_distance16_values8000_v1",
    (8000, 1, 64): "synthetic_solvability_distance64_values8000_v1",
}
DISTRACTOR_PROTOCOL_PATHS = {
    (8000, 1, 4): "docs/A0_A1_SOLVABILITY_DISTANCE4_VALUES8000_PROTOCOL.md",
    (8000, 1, 16): "docs/A0_A1_SOLVABILITY_DISTANCE16_VALUES8000_PROTOCOL.md",
    (8000, 1, 64): "docs/A0_A1_SOLVABILITY_DISTANCE64_VALUES8000_PROTOCOL.md",
}
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
DISTRACTOR_MARKER = 30_004
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
        return len(self.input_ids) - 1


def sequence_length(association_count: int = 1, distractor_distance: int = 0) -> int:
    if association_count < 1:
        raise ValueError(f"association_count must be positive: {association_count}")
    if distractor_distance < 0:
        raise ValueError(f"distractor_distance must be non-negative: {distractor_distance}")
    return SEQUENCE_LENGTH + 3 * (association_count - 1) + distractor_distance


def generator_version(
    value_count: int = VALUE_COUNT,
    association_count: int = 1,
    distractor_distance: int = 0,
) -> str:
    """Return the immutable generator identity for a ladder value-pool size."""

    if distractor_distance != 0:
        try:
            return DISTRACTOR_GENERATOR_VERSIONS[
                (value_count, association_count, distractor_distance)
            ]
        except KeyError as exc:
            supported = ", ".join(
                f"values={values}, associations={associations}, distance={distance}"
                for values, associations, distance in sorted(DISTRACTOR_GENERATOR_VERSIONS)
            )
            raise ValueError(
                "unsupported distractor-distance rung "
                f"value_count={value_count}, association_count={association_count}, "
                f"distractor_distance={distractor_distance}; expected one of {supported}"
            ) from exc
    if association_count != 1:
        try:
            return ASSOCIATION_GENERATOR_VERSIONS[(value_count, association_count)]
        except KeyError as exc:
            supported = ", ".join(
                f"values={values}, associations={associations}"
                for values, associations in sorted(ASSOCIATION_GENERATOR_VERSIONS)
            )
            raise ValueError(
                "unsupported association rung "
                f"value_count={value_count}, association_count={association_count}; "
                f"expected one of {supported}"
            ) from exc
    try:
        return GENERATOR_VERSIONS[value_count]
    except KeyError as exc:
        supported = ", ".join(str(value) for value in sorted(GENERATOR_VERSIONS))
        raise ValueError(f"unsupported value_count={value_count}; expected one of {supported}") from exc


def _derived_seed(
    base_seed: int,
    split: str,
    index: int,
    value_count: int = VALUE_COUNT,
    association_count: int = 1,
    distractor_distance: int = 0,
) -> int:
    version = generator_version(value_count, association_count, distractor_distance)
    if association_count == 1 and distractor_distance == 0:
        payload = f"{version}:{base_seed}:{split}:{index}".encode("utf-8")
    elif distractor_distance != 0:
        # Keep the distance curve paired with the existing distance-0 mappings.
        baseline_version = generator_version(value_count, 1, 0)
        payload = f"{baseline_version}:{base_seed}:{split}:{index}".encode("utf-8")
    else:
        payload = f"{version}:{base_seed}:{split}:{index}:{association_count}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def make_example(
    base_seed: int,
    split: str,
    index: int,
    value_count: int = VALUE_COUNT,
    association_count: int = 1,
    distractor_distance: int = 0,
) -> CalibrationExample:
    if split not in SPLIT_OFFSETS:
        raise ValueError(f"unknown split: {split}")
    if not 0 <= index < FIXED_EXAMPLES:
        raise ValueError(f"index out of range: {index}")
    generator_version(value_count, association_count, distractor_distance)
    rng = random.Random(
        _derived_seed(
            base_seed + SPLIT_OFFSETS[split],
            split,
            index,
            value_count,
            association_count,
            distractor_distance,
        )
    )
    key_base = KEY_TRAIN_BASE if split == "train" else KEY_HELDOUT_BASE
    if association_count == 1:
        key = key_base + index
        value = VALUE_BASE + rng.randrange(value_count)
        prefix = (
            MARKERS["task"],
            MARKERS["record"],
            key,
            value,
        )
        suffix = (
            MARKERS["query"],
            key,
            MARKERS["answer"],
            value,
        )
        input_ids = prefix + (DISTRACTOR_MARKER,) * distractor_distance + suffix
        labels = (-100,) * (len(input_ids) - 1) + (value,)
        return CalibrationExample(split, index, input_ids, labels, value)
    if association_count not in (2, 4, 8):
        raise ValueError(f"association generator does not implement count {association_count}")
    keys = tuple(key_base + index * association_count + offset for offset in range(association_count))
    values = tuple(VALUE_BASE + offset for offset in rng.sample(range(value_count), association_count))
    query_index = rng.randrange(association_count)
    query_key = keys[query_index]
    target = values[query_index]
    record_tokens: list[int] = []
    for key, value in zip(keys, values):
        record_tokens.extend((MARKERS["record"], key, value))
    input_ids = tuple(
        [MARKERS["task"], *record_tokens, MARKERS["query"], query_key, MARKERS["answer"], target]
    )
    labels = (-100,) * (len(input_ids) - 1) + (target,)
    return CalibrationExample(split, index, input_ids, labels, target)


def make_training_batch(
    base_seed: int,
    optimizer_step: int,
    value_count: int = VALUE_COUNT,
    association_count: int = 1,
    distractor_distance: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    examples = [
        make_example(
            base_seed,
            "train",
            ((optimizer_step - 1) * EFFECTIVE_BATCH_SIZE + offset) % FIXED_EXAMPLES,
            value_count,
            association_count,
            distractor_distance,
        )
        for offset in range(EFFECTIVE_BATCH_SIZE)
    ]
    return (
        torch.tensor([example.input_ids for example in examples], dtype=torch.long),
        torch.tensor([example.labels for example in examples], dtype=torch.long),
    )


def make_evaluation_examples(
    base_seed: int,
    split: str,
    value_count: int = VALUE_COUNT,
    association_count: int = 1,
    distractor_distance: int = 0,
) -> list[CalibrationExample]:
    return [
        make_example(
            base_seed,
            split,
            index,
            value_count,
            association_count,
            distractor_distance,
        )
        for index in range(FIXED_EXAMPLES)
    ]


def generator_spec(
    base_seed: int,
    value_count: int = VALUE_COUNT,
    association_count: int = 1,
    distractor_distance: int = 0,
) -> dict[str, Any]:
    spec = {
        "version": generator_version(value_count, association_count, distractor_distance),
        "base_seed": base_seed,
        "sequence_length": sequence_length(association_count, distractor_distance),
        "fixed_examples": FIXED_EXAMPLES,
        "key_train_base": KEY_TRAIN_BASE,
        "key_heldout_base": KEY_HELDOUT_BASE,
        "value_base": VALUE_BASE,
        "value_count": value_count,
        "markers": MARKERS,
        "split_offsets": SPLIT_OFFSETS,
    }
    if association_count != 1:
        spec["association_count"] = association_count
    if distractor_distance != 0:
        spec["distractor_distance"] = distractor_distance
        spec["distractor_marker"] = DISTRACTOR_MARKER
    return spec


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


def _restore_rng_state(state: dict[str, Any]) -> None:
    """Restore all RNG streams captured in a ladder checkpoint."""

    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    numpy_state = state.get("numpy")
    if numpy_state is not None:
        try:
            import numpy as np

            np.random.set_state(numpy_state)
        except ImportError:
            pass
    cuda_states = state.get("torch_cuda")
    if cuda_states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in cuda_states])


def _set_learning_rate(optimizer: torch.optim.Optimizer, value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = value


def _learning_rate(
    base: float,
    step: int,
    max_steps: int = MAX_STEPS,
    warmup_steps: int = WARMUP_STEPS,
) -> float:
    if step <= warmup_steps:
        return base * step / warmup_steps
    progress = min(1.0, max(0.0, (step - warmup_steps) / (max_steps - warmup_steps)))
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


def _data_hash(
    base_seed: int,
    value_count: int = VALUE_COUNT,
    association_count: int = 1,
    distractor_distance: int = 0,
) -> str:
    digest = hashlib.sha256()
    for split in ("train", "heldout"):
        for example in make_evaluation_examples(
            base_seed,
            split,
            value_count,
            association_count,
            distractor_distance,
        ):
            digest.update(split.encode("ascii"))
            digest.update(
                b"".join(int(token).to_bytes(2, "little") for token in example.input_ids)
            )
            digest.update(
                b"".join(int(label).to_bytes(2, "little", signed=True) for label in example.labels)
            )
    return digest.hexdigest()


def run(
    config_path: Path,
    run_dir: Path,
    seed: int,
    protocol_commit: str,
    *,
    value_count: int = VALUE_COUNT,
    association_count: int = 1,
    distractor_distance: int = 0,
    resume_path: Path | None = None,
) -> dict[str, Any]:
    """Run one fixed-set rung, checkpointing often enough for power-loss recovery.

    A resumed run always writes to a fresh output directory.  This preserves the
    interrupted directory as evidence and avoids appending a second trajectory to
    a possibly truncated JSONL log.
    """

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the host training process")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"run directory is not empty: {run_dir}")
    version = generator_version(value_count, association_count, distractor_distance)
    if distractor_distance != 0:
        protocol_relative_path = DISTRACTOR_PROTOCOL_PATHS[
            (value_count, association_count, distractor_distance)
        ]
    elif association_count == 1:
        protocol_relative_path = PROTOCOL_PATHS[value_count]
    else:
        protocol_relative_path = ASSOCIATION_PROTOCOL_PATHS[(value_count, association_count)]
    current_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    if current_commit != protocol_commit:
        raise RuntimeError(
            f"protocol commit mismatch: current={current_commit}, expected={protocol_commit}"
        )

    config = load_config(config_path)
    current_sequence_length = sequence_length(association_count, distractor_distance)
    tokens_per_step = current_sequence_length * EFFECTIVE_BATCH_SIZE
    expected_tokens = MAX_STEPS * tokens_per_step
    if config.context_length != 512 or config.micro_batch_size != MICRO_BATCH_SIZE:
        raise RuntimeError("calibration model geometry does not match the frozen protocol")
    if config.gradient_accumulation_steps != GRADIENT_ACCUMULATION_STEPS:
        raise RuntimeError("calibration accumulation does not match the frozen protocol")
    if config.max_steps != MAX_STEPS or config.training_token_budget != expected_tokens:
        raise RuntimeError("calibration budget does not match the frozen protocol")
    if config.lr_progress_reference != "optimizer_steps" or config.warmup_ratio != 0.01:
        raise RuntimeError("calibration schedule does not match the frozen protocol")
    if config.data_and_tokenizer != version:
        raise RuntimeError(
            f"config data identity does not match value_count={value_count}, "
            f"association_count={association_count}: "
            f"distractor_distance={distractor_distance}: "
            f"{config.data_and_tokenizer!r} != {version!r}"
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    config_hash = sha256_file(config_path)
    script_hash = sha256_file(Path(__file__))
    protocol_path = REPO_ROOT / protocol_relative_path
    protocol_hash = sha256_file(protocol_path)
    spec = generator_spec(seed, value_count, association_count, distractor_distance)
    generator_hash = sha256_json(spec)
    data_hash = _data_hash(seed, value_count, association_count, distractor_distance)
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
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    expected_parameters = 41_866_240 if config.stage == "A0" else 41_870_336
    if parameter_count != expected_parameters:
        raise RuntimeError(f"unexpected parameter count: {parameter_count} != {expected_parameters}")

    initial_step = 0
    initial_tokens = 0
    resume_record: dict[str, Any] = {}
    total_cuda_seconds = 0.0
    training_wall_seconds = 0.0
    validation_history: list[dict[str, Any]] = []
    metric_history: list[dict[str, Any]] = []
    last_train_loss: float | None = None
    if resume_path is not None:
        resume_record = verify_checkpoint(resume_path)
        expected_checkpoint_fields = {
            "config_sha256": config_hash,
            "data_manifest_sha256": data_hash,
            "seed": seed,
            "parameter_count": parameter_count,
            "protocol_commit": protocol_commit,
        }
        for field, expected in expected_checkpoint_fields.items():
            if resume_record.get(field) != expected:
                raise RuntimeError(
                    f"resume checkpoint metadata mismatch for {field}: "
                    f"{resume_record.get(field)!r} != {expected!r}"
                )
        payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        if payload.get("synthetic_generator") != spec:
            raise RuntimeError("resume checkpoint synthetic generator does not match the rung")
        initial_step = int(payload["step"])
        initial_tokens = int(payload["tokens_processed"])
        if initial_step <= 0 or initial_step >= MAX_STEPS:
            raise RuntimeError("resume checkpoint must be an interior checkpoint")
        if initial_tokens != initial_step * tokens_per_step:
            raise RuntimeError("resume checkpoint progress is inconsistent")
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

    train_examples = make_evaluation_examples(
        seed, "train", value_count, association_count, distractor_distance
    )
    heldout_examples = make_evaluation_examples(
        seed, "heldout", value_count, association_count, distractor_distance
    )
    run_metadata = {
        "experiment_id": run_dir.name,
        "status": "running",
        "config_path": str(config_path.relative_to(REPO_ROOT)),
        "config_sha256": config_hash,
        "protocol_commit": protocol_commit,
        "protocol_sha256": protocol_hash,
        "script_sha256": script_hash,
        "synthetic_generator_version": version,
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
            "sequence_length": current_sequence_length,
            "tokens_per_step": tokens_per_step,
            "tokens": expected_tokens,
            "fixed_training_examples": FIXED_EXAMPLES,
            "value_count": value_count,
            "association_count": association_count,
            "distractor_distance": distractor_distance,
            "warmup_steps": WARMUP_STEPS,
            "validation_interval_steps": EVAL_INTERVAL,
            "checkpoint_interval_steps": config.checkpoint_interval_steps,
        },
        "initial_step": initial_step,
        "initial_tokens": initial_tokens,
        "resume_checkpoint": str(resume_path) if resume_path is not None else None,
        "cuda_timing_scope": "CUDA event from optimizer.zero_grad through optimizer.step; evaluation excluded",
    }
    _write_json(run_dir / "run_metadata.json", run_metadata)
    monitor = IntegrityMonitor(
        run_dir=run_dir,
        metadata=run_metadata,
        tokens_per_step=tokens_per_step,
        initial_step=initial_step,
        initial_tokens=initial_tokens,
    )
    metrics_path = run_dir / "calibration_metrics.jsonl"
    if metric_history:
        metrics_path.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in metric_history),
            encoding="utf-8",
        )
    run_started = time.perf_counter()
    last_step = initial_step
    checkpoint_path: Path | None = None
    checkpoint_record: dict[str, Any] | None = None
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
                "tokens_processed": step * tokens_per_step,
                "seed": seed,
                "rng_state": _rng_state(),
                "synthetic_generator": spec,
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
                "tokens_processed": step * tokens_per_step,
                "seed": seed,
                "config_sha256": config_hash,
                "data_manifest_sha256": data_hash,
                "checkpoint_kind": kind,
                "parameter_count": parameter_count,
                "protocol_commit": protocol_commit,
                "synthetic_generator_version": version,
                "value_count": value_count,
                "association_count": association_count,
                "distractor_distance": distractor_distance,
                "gradient_checkpointing": False,
            },
        )
        verify_checkpoint(checkpoint_path)

    try:
        model.train()
        torch.cuda.reset_peak_memory_stats(device)
        for step in range(initial_step + 1, MAX_STEPS + 1):
            step_started = time.perf_counter()
            learning_rate = _learning_rate(config.learning_rate, step)
            _set_learning_rate(optimizer, learning_rate)
            cpu_inputs, cpu_labels = make_training_batch(
                seed, step, value_count, association_count, distractor_distance
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
                tokens_processed=step * tokens_per_step,
                train_loss=train_loss,
                validation_loss=None if evaluation is None else evaluation["heldout"]["target_loss"],
                gradient_norm=gradient_norm,
                learning_rate=learning_rate,
                tokens_per_second=tokens_per_step / step_wall,
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
                "evaluation": evaluation,
            }
            metric_history.append(metric_record)
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metric_record, sort_keys=True) + "\n")
            last_step = step
            last_train_loss = train_loss
            if step % config.checkpoint_interval_steps == 0 or step == MAX_STEPS:
                save_progress_checkpoint(step, "interval_or_final")

        final_train = _metrics(model, train_examples, device)
        final_heldout = _metrics(model, heldout_examples, device)
        if checkpoint_path is None or checkpoint_record is None:
            raise RuntimeError("completed run did not produce a final checkpoint")
    except KeyboardInterrupt as exc:
        status = "interrupted"
        failure = {"type": type(exc).__name__, "message": str(exc)}
        if last_step > initial_step:
            save_progress_checkpoint(last_step, "interrupt_recovery")
    except Exception as exc:
        status = "failed"
        failure = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        elapsed_wall_seconds = time.perf_counter() - run_started
        if status == "completed" and last_step == MAX_STEPS and checkpoint_record is not None:
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
                "synthetic_generator_version": version,
                "synthetic_generator_sha256": generator_hash,
                "synthetic_data_sha256": data_hash,
                "seed": seed,
                "parameter_count": parameter_count,
                "optimizer_steps": MAX_STEPS,
                "initial_step": initial_step,
                "sequence_length": current_sequence_length,
                "association_count": association_count,
                "distractor_distance": distractor_distance,
                "tokens_processed": expected_tokens,
                "warmup_steps": WARMUP_STEPS,
                "learning_rate": config.learning_rate,
                "lr_schedule": "cosine",
                "value_count": value_count,
                "checkpoint_interval_steps": config.checkpoint_interval_steps,
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
                "resume_checkpoint": str(resume_path) if resume_path is not None else None,
                "gradient_checkpointing": {"configured": True, "implemented": False, "enabled": False},
                "numerical_instability": False,
                "failure": None,
            }
        else:
            summary = {
                "status": status,
                "architecture": config.stage,
                "seed": seed,
                "failure": failure,
                "final_step": last_step,
                "cuda_training_seconds": total_cuda_seconds,
                "checkpoint": None if checkpoint_path is None else str(checkpoint_path),
            }
        _write_json(run_dir / "summary.json", summary)
    return _load_json(run_dir / "summary.json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--protocol-commit", required=True)
    parser.add_argument(
        "--value-count",
        type=int,
        default=VALUE_COUNT,
        choices=sorted(GENERATOR_VERSIONS),
        help="reserved answer-token pool for the selected solvability rung",
    )
    parser.add_argument(
        "--association-count",
        type=int,
        default=1,
        choices=(1, 2, 4, 8),
        help="number of key/value associations in the selected solvability rung",
    )
    parser.add_argument(
        "--distractor-distance",
        type=int,
        default=0,
        choices=(0, 4, 16, 64),
        help="number of neutral distractor markers between the binding and query",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="verified interior checkpoint; writes the resumed trajectory to the new run directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the rung configuration and print the bounded execution plan",
    )
    parser.add_argument("--execute-approved", action="store_true")
    args = parser.parse_args()
    if not args.execute_approved:
        if not args.dry_run:
            raise SystemExit("training is gated; pass --execute-approved after protocol authorization")
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    run_dir = args.run_dir if args.run_dir.is_absolute() else REPO_ROOT / args.run_dir
    resume_path = None if args.resume is None else (
        args.resume if args.resume.is_absolute() else REPO_ROOT / args.resume
    )
    if args.dry_run:
        config = load_config(config_path)
        version = generator_version(
            args.value_count, args.association_count, args.distractor_distance
        )
        current_sequence_length = sequence_length(
            args.association_count, args.distractor_distance
        )
        expected_tokens = MAX_STEPS * current_sequence_length * EFFECTIVE_BATCH_SIZE
        if config.data_and_tokenizer != version:
            raise SystemExit(
                f"config data identity does not match value_count={args.value_count}, "
                f"association_count={args.association_count}: "
                f"distractor_distance={args.distractor_distance}: "
                f"{config.data_and_tokenizer!r} != {version!r}"
            )
        if config.max_steps != MAX_STEPS or config.training_token_budget != expected_tokens:
            raise SystemExit("calibration budget does not match the frozen protocol")
        print(
            json.dumps(
                {
                    "status": "PLAN_VALID",
                    "config": str(config_path.relative_to(REPO_ROOT)),
                    "architecture": config.stage,
                    "seed": args.seed,
                    "value_count": args.value_count,
                    "association_count": args.association_count,
                    "distractor_distance": args.distractor_distance,
                    "sequence_length": current_sequence_length,
                    "generator_version": version,
                    "max_steps": MAX_STEPS,
                    "training_token_budget": expected_tokens,
                    "checkpoint_interval_steps": config.checkpoint_interval_steps,
                    "resume_checkpoint": None if resume_path is None else str(resume_path),
                    "run_dir": str(run_dir),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    print(
        json.dumps(
            run(
                config_path,
                run_dir,
                args.seed,
                args.protocol_commit,
                value_count=args.value_count,
                association_count=args.association_count,
                distractor_distance=args.distractor_distance,
                resume_path=resume_path,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
