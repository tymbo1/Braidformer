#!/usr/bin/env python3
"""Train one frozen protected/ordinary associative-lane arm.

This runner is deliberately separate from the original solvability runner so
the frozen A0/A1 training path is not modified.  It reuses the same generator,
optimizer schedule, integrity monitor, checkpoint format, and evaluation
helpers while constructing only the new A1 associative-lane model.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from braidformer.config import load_config
from braidformer.models import (
    A1AssociativeLaneLM,
    EXPECTED_LANE_PARAMETERS,
    LANE_WIDTH,
)
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
    EFFECTIVE_BATCH_SIZE,
    EVAL_INTERVAL,
    FIXED_EXAMPLES,
    GRADIENT_ACCUMULATION_STEPS,
    MAX_STEPS,
    MICRO_BATCH_SIZE,
    _cpu_copy,
    _data_hash,
    _learning_rate,
    _metrics,
    _restore_rng_state,
    _rng_state,
    _set_learning_rate,
    make_evaluation_examples,
    make_training_batch,
    generator_spec,
    generator_version,
    sequence_length,
)


PROTOCOL_PATH = "docs/A1_PROTECTED_ASSOCIATIVE_LANE_PROTOCOL.md"
VALUE_COUNT = 8_000
ASSOCIATION_COUNT = 1
TARGET_LENGTH = 1
EXPECTED_BASE_PARAMETERS = 41_870_336
EXPECTED_PARAMETERS = EXPECTED_BASE_PARAMETERS + EXPECTED_LANE_PARAMETERS
LANE_MODES = ("protected", "ordinary")
OUTPUT_VERSION = "a1_protected_lane_training_v1"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _finite(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite {label}: {value!r}")
    return number


def _lane_metrics(
    model: A1AssociativeLaneLM,
    examples: list[Any],
    device: torch.device,
    distance: int,
    batch_size: int = 16,
) -> dict[str, Any]:
    model.eval()
    full_loss = 0.0
    value_loss = 0.0
    correct = 0
    total = 0
    diagnostic_parts: dict[str, list[float]] = {
        "post_value_memory_key_norm": [],
        "post_value_memory_value_norm": [],
        "post_distractor_memory_key_norm": [],
        "post_distractor_memory_value_norm": [],
        "answer_memory_key_norm": [],
        "answer_memory_value_norm": [],
        "answer_key_query_cosine": [],
        "answer_read_gate": [],
        "answer_lane_delta_norm": [],
    }
    target_position = 6 + distance
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            inputs = torch.tensor(
                [example.input_ids for example in batch], dtype=torch.long, device=device
            )
            labels = torch.tensor(
                [example.labels for example in batch], dtype=torch.long, device=device
            )
            targets = torch.tensor(
                [example.target for example in batch], dtype=torch.long, device=device
            )
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output, trace = model.forward_with_lane_trace(inputs, labels=labels)
            logits = output.logits[:, target_position, :].float()
            target_logits = logits.gather(-1, targets[:, None]).squeeze(-1)
            full_loss += float((-target_logits + torch.logsumexp(logits, dim=-1)).mean()) * len(batch)
            value_logits = logits[:, 8_000:16_000]
            value_targets = targets - 8_000
            value_loss += float(F.cross_entropy(value_logits, value_targets).item()) * len(batch)
            correct += int(logits.argmax(dim=-1).eq(targets).sum().item())
            total += len(batch)

            answer_key = trace["memory_key"][:, target_position, :].float()
            answer_value = trace["memory_value"][:, target_position, :].float()
            post_value_key = trace["memory_key"][:, 3, :].float()
            post_value_value = trace["memory_value"][:, 3, :].float()
            post_distractor_position = 3 + distance
            post_distractor_key = trace["memory_key"][:, post_distractor_position, :].float()
            post_distractor_value = trace["memory_value"][:, post_distractor_position, :].float()
            # The lane trace already records this exact cosine at the answer
            # position.  Reusing it keeps the diagnostic read-only and avoids
            # a second full eight-layer A1 forward pass per evaluation batch.
            answer_cosine = trace["similarity"][:, target_position].float()
            answer_gate = trace["read_gate"][:, target_position].float()
            answer_delta = trace["lane_delta"][:, target_position, :].float()
            diagnostic_parts["post_value_memory_key_norm"].extend(
                post_value_key.norm(dim=-1).tolist()
            )
            diagnostic_parts["post_value_memory_value_norm"].extend(
                post_value_value.norm(dim=-1).tolist()
            )
            diagnostic_parts["post_distractor_memory_key_norm"].extend(
                post_distractor_key.norm(dim=-1).tolist()
            )
            diagnostic_parts["post_distractor_memory_value_norm"].extend(
                post_distractor_value.norm(dim=-1).tolist()
            )
            diagnostic_parts["answer_memory_key_norm"].extend(answer_key.norm(dim=-1).tolist())
            diagnostic_parts["answer_memory_value_norm"].extend(answer_value.norm(dim=-1).tolist())
            diagnostic_parts["answer_key_query_cosine"].extend(answer_cosine.tolist())
            diagnostic_parts["answer_read_gate"].extend(answer_gate.tolist())
            diagnostic_parts["answer_lane_delta_norm"].extend(answer_delta.norm(dim=-1).tolist())
    model.train()
    if total != len(examples):
        raise RuntimeError("lane metric example count mismatch")
    diagnostics = {
        key: _finite(sum(values) / len(values), key)
        for key, values in diagnostic_parts.items()
    }
    return {
        "examples": total,
        "target_loss": _finite(full_loss / total, "lane full target loss"),
        "value_candidate_target_loss": _finite(value_loss / total, "lane value target loss"),
        "greedy_answer_token_accuracy": correct / total,
        "greedy_exact_match": correct / total,
        "lane_diagnostics": diagnostics,
    }


def run(
    config_path: Path,
    run_dir: Path,
    seed: int,
    protocol_commit: str,
    mode: str,
    distance: int,
    resume_path: Path | None = None,
) -> dict[str, Any]:
    if mode not in LANE_MODES:
        raise ValueError(f"unknown lane mode: {mode}")
    if distance not in (0, 4, 16, 64):
        raise ValueError(f"unsupported distance: {distance}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the host training process")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"run directory is not empty: {run_dir}")
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    dirty_files = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    if dirty_files:
        raise RuntimeError(
            "training requires a clean tracked worktree; commit the frozen "
            f"implementation first (changes: {dirty_files!r})"
        )

    config = load_config(config_path)
    version = generator_version(VALUE_COUNT, ASSOCIATION_COUNT, distance, TARGET_LENGTH)
    if config.data_and_tokenizer != version:
        raise RuntimeError(f"config data identity mismatch: {config.data_and_tokenizer} != {version}")
    current_sequence_length = sequence_length(ASSOCIATION_COUNT, distance, TARGET_LENGTH)
    expected_tokens = MAX_STEPS * current_sequence_length * EFFECTIVE_BATCH_SIZE
    if config.max_steps != MAX_STEPS or config.training_token_budget != expected_tokens:
        raise RuntimeError("lane budget does not match the frozen protocol")
    if config.stage != "A1P" or config.model_type != "recurrent_state_space_associative_lane":
        raise RuntimeError("lane config identity mismatch")

    config_hash = sha256_file(config_path)
    script_hash = sha256_file(Path(__file__))
    protocol_hash = sha256_file(REPO_ROOT / PROTOCOL_PATH)
    spec = generator_spec(seed, VALUE_COUNT, ASSOCIATION_COUNT, distance, TARGET_LENGTH)
    generator_hash = sha256_json(spec)
    data_hash = _data_hash(seed, VALUE_COUNT, ASSOCIATION_COUNT, distance, TARGET_LENGTH)
    seed_report = set_reproducible_seed(seed, deterministic=True)
    device = torch.device("cuda")
    model = A1AssociativeLaneLM(config, mode=mode, lane_width=LANE_WIDTH).to(device)
    parameter_count = model.parameter_count()
    if model.base.parameter_count() != EXPECTED_BASE_PARAMETERS:
        raise RuntimeError(f"unexpected base A1 parameter count: {model.base.parameter_count()}")
    if parameter_count != EXPECTED_PARAMETERS:
        raise RuntimeError(f"unexpected lane parameter count: {parameter_count}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.optimizer_beta1, config.optimizer_beta2),
        eps=config.optimizer_epsilon,
        weight_decay=config.weight_decay,
    )

    initial_step = 0
    initial_tokens = 0
    total_cuda_seconds = 0.0
    training_wall_seconds = 0.0
    validation_history: list[dict[str, Any]] = []
    metric_history: list[dict[str, Any]] = []
    last_train_loss: float | None = None
    if resume_path is not None:
        resume_record = verify_checkpoint(resume_path)
        for field, expected in {
            "config_sha256": config_hash,
            "data_manifest_sha256": data_hash,
            "seed": seed,
            "parameter_count": parameter_count,
            "protocol_commit": protocol_commit,
            "lane_mode": mode,
        }.items():
            if resume_record.get(field) != expected:
                raise RuntimeError(f"resume checkpoint metadata mismatch for {field}")
        payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        if payload.get("synthetic_generator") != spec or payload.get("lane_mode") != mode:
            raise RuntimeError("resume checkpoint lane provenance mismatch")
        initial_step = int(payload["step"])
        initial_tokens = int(payload["tokens_processed"])
        if not 0 < initial_step < MAX_STEPS:
            raise RuntimeError("resume checkpoint must be an interior checkpoint")
        if initial_tokens != initial_step * current_sequence_length * EFFECTIVE_BATCH_SIZE:
            raise RuntimeError("resume checkpoint token progress mismatch")
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
        seed, "train", VALUE_COUNT, ASSOCIATION_COUNT, distance, TARGET_LENGTH
    )
    heldout_examples = make_evaluation_examples(
        seed, "heldout", VALUE_COUNT, ASSOCIATION_COUNT, distance, TARGET_LENGTH
    )
    run_metadata = {
        "experiment_id": run_dir.name,
        "status": "running",
        "architecture": "A1P",
        "lane_mode": mode,
        "lane_width": LANE_WIDTH,
        "config_path": str(config_path.relative_to(REPO_ROOT)),
        "config_sha256": config_hash,
        "protocol_commit": protocol_commit,
        "source_commit": source_commit,
        "protocol_sha256": protocol_hash,
        "script_sha256": script_hash,
        "synthetic_generator_version": version,
        "synthetic_generator_sha256": generator_hash,
        "synthetic_data_sha256": data_hash,
        "seed": seed,
        "seed_report": seed_report,
        "parameter_count": parameter_count,
        "base_parameter_count": model.base.parameter_count(),
        "training": {
            "max_steps": MAX_STEPS,
            "sequence_length": current_sequence_length,
            "tokens_per_step": current_sequence_length * EFFECTIVE_BATCH_SIZE,
            "tokens": expected_tokens,
            "fixed_training_examples": FIXED_EXAMPLES,
            "value_count": VALUE_COUNT,
            "association_count": ASSOCIATION_COUNT,
            "distractor_distance": distance,
            "target_length": TARGET_LENGTH,
            "warmup_steps": max(1, int(MAX_STEPS * config.warmup_ratio)),
            "validation_interval_steps": EVAL_INTERVAL,
            "checkpoint_interval_steps": config.checkpoint_interval_steps,
        },
        "initial_step": initial_step,
        "initial_tokens": initial_tokens,
        "resume_checkpoint": None if resume_path is None else str(resume_path),
        "cuda_timing_scope": "CUDA event from optimizer.zero_grad through optimizer.step; evaluation excluded",
    }
    _write_json(run_dir / "run_metadata.json", run_metadata)
    monitor = IntegrityMonitor(
        run_dir=run_dir,
        metadata=run_metadata,
        tokens_per_step=current_sequence_length * EFFECTIVE_BATCH_SIZE,
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
                "tokens_processed": step * current_sequence_length * EFFECTIVE_BATCH_SIZE,
                "seed": seed,
                "rng_state": _rng_state(),
                "synthetic_generator": spec,
                "lane_mode": mode,
                "lane_width": LANE_WIDTH,
                "total_cuda_seconds": total_cuda_seconds,
                "training_wall_seconds": training_wall_seconds,
                "validation_history": validation_history,
                "metric_history": metric_history,
                "last_train_loss": last_train_loss,
            },
            checkpoint_path,
            {
                "step": step,
                "tokens_processed": step * current_sequence_length * EFFECTIVE_BATCH_SIZE,
                "seed": seed,
                "config_sha256": config_hash,
                "data_manifest_sha256": data_hash,
                "checkpoint_kind": kind,
                "parameter_count": parameter_count,
                "protocol_commit": protocol_commit,
                "lane_mode": mode,
                "lane_width": LANE_WIDTH,
                "synthetic_generator_version": version,
                "value_count": VALUE_COUNT,
                "association_count": ASSOCIATION_COUNT,
                "distractor_distance": distance,
                "target_length": TARGET_LENGTH,
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
                seed,
                step,
                VALUE_COUNT,
                ASSOCIATION_COUNT,
                distance,
                TARGET_LENGTH,
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
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=config.gradient_clip_norm, error_if_nonfinite=True
                ).item()
            )
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
            evaluation = None
            if step % EVAL_INTERVAL == 0 or step == MAX_STEPS:
                train_metrics = _metrics(model, train_examples, device)
                heldout_metrics = _metrics(model, heldout_examples, device)
                evaluation = {"train": train_metrics, "heldout": heldout_metrics}
                validation_history.append({"step": step, "metrics": evaluation})
            monitor.record_step(
                step=step,
                tokens_processed=step * current_sequence_length * EFFECTIVE_BATCH_SIZE,
                train_loss=train_loss,
                validation_loss=None if evaluation is None else evaluation["heldout"]["target_loss"],
                gradient_norm=gradient_norm,
                learning_rate=learning_rate,
                tokens_per_second=(current_sequence_length * EFFECTIVE_BATCH_SIZE) / step_wall,
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
        final_lane_train = _lane_metrics(model, train_examples, device, distance)
        final_lane_heldout = _lane_metrics(model, heldout_examples, device, distance)
        if checkpoint_path is None or checkpoint_record is None:
            raise RuntimeError("completed lane run did not produce a final checkpoint")
    except KeyboardInterrupt as exc:
        status = "interrupted"
        failure = {"type": type(exc).__name__, "message": str(exc)}
        if last_step > initial_step:
            save_progress_checkpoint(last_step, "interrupt_recovery")
        raise
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
                "architecture": "A1P",
                "lane_mode": mode,
                "lane_width": LANE_WIDTH,
                "config_path": str(config_path.relative_to(REPO_ROOT)),
                "config_sha256": config_hash,
                "protocol_commit": protocol_commit,
                "source_commit": source_commit,
                "protocol_sha256": protocol_hash,
                "script_sha256": script_hash,
                "synthetic_generator_version": version,
                "synthetic_generator_sha256": generator_hash,
                "synthetic_data_sha256": data_hash,
                "seed": seed,
                "parameter_count": parameter_count,
                "base_parameter_count": model.base.parameter_count(),
                "optimizer_steps": MAX_STEPS,
                "sequence_length": current_sequence_length,
                "association_count": ASSOCIATION_COUNT,
                "distractor_distance": distance,
                "target_length": TARGET_LENGTH,
                "tokens_processed": expected_tokens,
                "warmup_steps": max(1, int(MAX_STEPS * config.warmup_ratio)),
                "learning_rate": config.learning_rate,
                "lr_schedule": config.lr_schedule,
                "value_count": VALUE_COUNT,
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
                "final_lane_train": final_lane_train,
                "final_lane_heldout": final_lane_heldout,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_record["checkpoint_sha256"],
                "checkpoint_integrity_verified": True,
                "resume_checkpoint": None if resume_path is None else str(resume_path),
                "numerical_instability": False,
                "failure": None,
            }
        else:
            summary = {
                "status": status,
                "architecture": "A1P",
                "lane_mode": mode,
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
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--protocol-commit", required=True)
    parser.add_argument("--mode", choices=LANE_MODES, required=True)
    parser.add_argument("--distance", type=int, choices=(0, 4, 16, 64), required=True)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute-approved", action="store_true")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    run_dir = args.run_dir if args.run_dir.is_absolute() else REPO_ROOT / args.run_dir
    resume_path = None if args.resume is None else (
        args.resume if args.resume.is_absolute() else REPO_ROOT / args.resume
    )
    config = load_config(config_path)
    version = generator_version(VALUE_COUNT, ASSOCIATION_COUNT, args.distance, TARGET_LENGTH)
    current_sequence_length = sequence_length(ASSOCIATION_COUNT, args.distance, TARGET_LENGTH)
    expected_tokens = MAX_STEPS * current_sequence_length * EFFECTIVE_BATCH_SIZE
    if config.data_and_tokenizer != version:
        raise SystemExit(f"config data identity mismatch: {config.data_and_tokenizer} != {version}")
    if config.max_steps != MAX_STEPS or config.training_token_budget != expected_tokens:
        raise SystemExit("lane budget does not match frozen protocol")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "PLAN_VALID",
                    "config": str(config_path.relative_to(REPO_ROOT)),
                    "mode": args.mode,
                    "distance": args.distance,
                    "sequence_length": current_sequence_length,
                    "generator_version": version,
                    "max_steps": MAX_STEPS,
                    "training_token_budget": expected_tokens,
                    "parameter_count": EXPECTED_PARAMETERS,
                    "run_dir": str(run_dir),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if not args.execute_approved:
        raise SystemExit("training is gated; pass --execute-approved after protocol authorization")
    print(
        json.dumps(
            run(
                config_path,
                run_dir,
                args.seed,
                args.protocol_commit,
                args.mode,
                args.distance,
                resume_path,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
