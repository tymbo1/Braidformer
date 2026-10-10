#!/usr/bin/env python3
"""Calibrate A0/A1 CUDA execution time without starting a training run.

This bounded benchmark loads the preserved pilot checkpoints, performs a small
number of in-memory optimizer updates, and writes no checkpoint or run log.
CUDA-event time covers the default-stream update with batches already resident
on the device. Host wall time is recorded separately. The in-memory updates
are timing probes only; they are discarded when each arm completes.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from braidformer.config import load_config
from braidformer.training.integrity import (
    set_reproducible_seed,
    sha256_file,
    verify_checkpoint,
)
from braidformer.training.runner import PackedTokenStream, build_model


DEFAULT_A0_CONFIG = REPO_ROOT / "configs" / "braidnano_a0_baseline_proposal.json"
DEFAULT_A1_CONFIG = REPO_ROOT / "configs" / "braidnano_a1_recurrent_state_space_proposal.json"
DEFAULT_A0_CHECKPOINT = REPO_ROOT / "results" / "runs" / "a0_seed17_pilot" / "checkpoint_step_001000.pt"
DEFAULT_A1_CHECKPOINT = REPO_ROOT / "results" / "runs" / "a1_seed17_pilot" / "checkpoint_step_001000.pt"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "a0_a1_cuda_timing_calibration.json"

WARMUP_STEPS = 2
MEASURED_STEPS = 20


def _resolve(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _release_cuda_objects() -> None:
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def _approved_train_batches(
    config: Any,
    *,
    total_steps: int,
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    data_config_path = REPO_ROOT / config.data_and_tokenizer
    data_config = json.loads(data_config_path.read_text(encoding="utf-8"))
    prepared = data_config["prepared_artifacts"]
    manifest_path = REPO_ROOT / prepared["manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_record = manifest["packing"]["streams"]["train"]
    stream_path = REPO_ROOT / train_record["path"]
    actual_hash = sha256_file(stream_path)
    if actual_hash != train_record["sha256"]:
        raise RuntimeError("approved train stream hash does not match its manifest")
    total_batches = total_steps * config.gradient_accumulation_steps
    tokens_per_batch = config.micro_batch_size * config.context_length
    batches: list[torch.Tensor] = []
    with PackedTokenStream(stream_path, expected_tokens=train_record["tokens"]) as stream:
        for batch_index in range(total_batches):
            start = batch_index * tokens_per_batch
            batches.append(
                stream.read_tokens(start, tokens_per_batch).view(
                    config.micro_batch_size,
                    config.context_length,
                )
            )
    return batches, {
        "data_config_path": str(data_config_path.relative_to(REPO_ROOT)),
        "data_config_sha256": sha256_file(data_config_path),
        "manifest_path": str(manifest_path.relative_to(REPO_ROOT)),
        "manifest_sha256": sha256_file(manifest_path),
        "train_stream_path": str(stream_path.relative_to(REPO_ROOT)),
        "train_stream_sha256": actual_hash,
        "batch_source": "approved sequential packed train stream from token 0",
    }


def _estimated_flops(config: Any) -> dict[str, Any]:
    """Return a labeled analytical lower-bound estimate, never a budget."""

    hidden = config.hidden_size
    intermediate = config.intermediate_size
    layers = config.num_layers
    context = config.context_length
    vocabulary = config.vocab_size
    dense_and_lm = layers * (8 * hidden * hidden + 4 * hidden * intermediate) + 2 * hidden * vocabulary
    attention_interaction = layers * 4 * context * hidden
    if config.stage == "A0":
        forward = dense_and_lm + attention_interaction
        omitted = (
            "embedding, layer norm, activations, softmax, masking, optimizer, "
            "memory traffic, and kernel overhead"
        )
    else:
        forward = dense_and_lm
        omitted = (
            "embedding, layer norm, activations, optimizer, memory traffic, "
            "kernel overhead, and all elementwise recurrence/scan operations"
        )
    tokens_per_step = config.micro_batch_size * config.gradient_accumulation_steps * context
    return {
        "convention": "2 FLOPs per counted multiply-add; backward approximated as 2x forward",
        "forward_lower_bound_flops_per_token": forward,
        "training_lower_bound_flops_per_token": 3 * forward,
        "training_lower_bound_flops_per_optimizer_step": 3 * forward * tokens_per_step,
        "dense_and_lm_forward_flops_per_token": dense_and_lm,
        "a0_attention_interaction_forward_flops_per_token": attention_interaction,
        "omitted_terms": omitted,
        "used_for_cuda_budget": False,
    }


def _run_arm(
    *,
    config_path: Path,
    checkpoint_path: Path,
    batches: list[torch.Tensor],
    device: torch.device,
) -> dict[str, Any]:
    config = load_config(config_path)
    checkpoint_record = verify_checkpoint(checkpoint_path)
    config_sha = sha256_file(config_path)
    if checkpoint_record.get("config_sha256") != config_sha:
        raise RuntimeError(f"checkpoint/config hash mismatch for {config.stage}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    set_reproducible_seed(config.seed, deterministic=True)
    model = build_model(config).to(device).train()
    model.load_state_dict(payload["model_state_dict"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.optimizer_beta1, config.optimizer_beta2),
        eps=config.optimizer_epsilon,
        weight_decay=config.weight_decay,
    )
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    _move_optimizer_state_to_device(optimizer, device)
    device_batches = [batch.to(device) for batch in batches]
    tokens_per_step = config.micro_batch_size * config.gradient_accumulation_steps * config.context_length
    amp_dtype = torch.float16 if config.precision == "fp16" else torch.bfloat16

    def run_update(batch_offset: int, measure: bool) -> dict[str, float]:
        torch.cuda.synchronize(device)
        host_started = time.perf_counter()
        start_event = torch.cuda.Event(enable_timing=True) if measure else None
        end_event = torch.cuda.Event(enable_timing=True) if measure else None
        if start_event is not None:
            start_event.record()
        optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        for accumulation_index in range(config.gradient_accumulation_steps):
            input_ids = device_batches[batch_offset + accumulation_index]
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                output = model(input_ids, labels=input_ids)
                loss = output.loss
            if loss is None or not bool(torch.isfinite(loss).item()):
                raise RuntimeError(f"non-finite calibration loss for {config.stage}")
            losses.append(float(loss.item()))
            (loss / config.gradient_accumulation_steps).backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=config.gradient_clip_norm,
            error_if_nonfinite=True,
        )
        if not bool(torch.isfinite(gradient_norm).item()):
            raise RuntimeError(f"non-finite calibration gradient for {config.stage}")
        optimizer.step()
        if end_event is not None:
            end_event.record()
            end_event.synchronize()
            cuda_seconds = start_event.elapsed_time(end_event) / 1_000.0
        else:
            torch.cuda.synchronize(device)
            cuda_seconds = 0.0
        host_seconds = time.perf_counter() - host_started
        return {
            "cuda_event_seconds": cuda_seconds,
            "host_wall_seconds": host_seconds,
            "mean_loss": sum(losses) / len(losses),
            "gradient_norm": float(gradient_norm.item()),
            "tokens_per_second_by_cuda_time": tokens_per_step / cuda_seconds if cuda_seconds else 0.0,
            "tokens_per_second_by_host_time": tokens_per_step / host_seconds,
        }

    for warmup_index in range(WARMUP_STEPS):
        run_update(warmup_index * config.gradient_accumulation_steps, measure=False)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    measured: list[dict[str, float]] = []
    for measured_index in range(MEASURED_STEPS):
        offset = (WARMUP_STEPS + measured_index) * config.gradient_accumulation_steps
        measured.append(run_update(offset, measure=True))

    cuda_values = [item["cuda_event_seconds"] for item in measured]
    host_values = [item["host_wall_seconds"] for item in measured]
    result = {
        "stage": config.stage,
        "model_type": config.model_type,
        "recurrent_path": getattr(config, "recurrent_path", "sequential_reference"),
        "config_path": str(config_path.relative_to(REPO_ROOT)),
        "config_sha256": config_sha,
        "checkpoint_path": str(checkpoint_path.relative_to(REPO_ROOT)),
        "checkpoint_sha256": checkpoint_record["checkpoint_sha256"],
        "checkpoint_step": checkpoint_record["step"],
        "parameter_count": model.parameter_count(),
        "seed": config.seed,
        "warmup_steps_not_measured": WARMUP_STEPS,
        "measured_optimizer_steps": MEASURED_STEPS,
        "tokens_per_optimizer_step": tokens_per_step,
        "tokens_measured": MEASURED_STEPS * tokens_per_step,
        "cuda_event_seconds_total": sum(cuda_values),
        "cuda_event_seconds_mean": statistics.mean(cuda_values),
        "cuda_event_seconds_median": statistics.median(cuda_values),
        "cuda_event_seconds_stdev": statistics.stdev(cuda_values) if len(cuda_values) > 1 else 0.0,
        "host_wall_seconds_total": sum(host_values),
        "host_wall_seconds_mean": statistics.mean(host_values),
        "host_wall_seconds_median": statistics.median(host_values),
        "cuda_to_host_time_ratio": sum(cuda_values) / sum(host_values),
        "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device),
        "finite_updates": True,
        "updates_persisted": False,
        "update_records": measured,
        "estimated_flops": _estimated_flops(config),
    }
    del device_batches, optimizer, model, payload
    _release_cuda_objects()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a0-config", type=Path, default=DEFAULT_A0_CONFIG)
    parser.add_argument("--a1-config", type=Path, default=DEFAULT_A1_CONFIG)
    parser.add_argument("--a0-checkpoint", type=Path, default=DEFAULT_A0_CHECKPOINT)
    parser.add_argument("--a1-checkpoint", type=Path, default=DEFAULT_A1_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    a0_config_path = _resolve(REPO_ROOT, args.a0_config)
    a1_config_path = _resolve(REPO_ROOT, args.a1_config)
    a0_checkpoint_path = _resolve(REPO_ROOT, args.a0_checkpoint)
    a1_checkpoint_path = _resolve(REPO_ROOT, args.a1_checkpoint)
    output_path = _resolve(REPO_ROOT, args.output)
    a0_config = load_config(a0_config_path)
    a1_config = load_config(a1_config_path)
    if a0_config.stage != "A0" or a0_config.model_type != "transformer":
        raise ValueError("A0 timing calibration requires the approved Transformer configuration")
    if (
        a1_config.stage != "A1"
        or a1_config.model_type != "recurrent_state_space"
        or a1_config.recurrent_path != "parallel_scan"
    ):
        raise ValueError("A1 timing calibration requires the approved parallel_scan configuration")
    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA is unavailable")
    for config in (a0_config, a1_config):
        if (
            config.micro_batch_size != 2
            or config.context_length != 512
            or config.gradient_accumulation_steps != 8
            or config.precision != "fp16"
        ):
            raise ValueError("timing calibration requires the approved micro-batch 2/context 512/accumulation 8 FP16 conditions")
    if a0_config.seed != a1_config.seed:
        raise ValueError("A0/A1 calibration seeds must match")

    total_steps = WARMUP_STEPS + MEASURED_STEPS
    batches, data_record = _approved_train_batches(a0_config, total_steps=total_steps)
    if a1_config.data_and_tokenizer != a0_config.data_and_tokenizer:
        raise ValueError("A0/A1 timing calibration requires the same data configuration")

    device = torch.device("cuda")
    a0 = _run_arm(
        config_path=a0_config_path,
        checkpoint_path=a0_checkpoint_path,
        batches=batches,
        device=device,
    )
    a1 = _run_arm(
        config_path=a1_config_path,
        checkpoint_path=a1_checkpoint_path,
        batches=batches,
        device=device,
    )
    reference_budget = a0["cuda_event_seconds_mean"] * 1_000
    proposed_a1_steps = max(1, math.floor(reference_budget / a1["cuda_event_seconds_mean"]))
    tokens_per_step = a0["tokens_per_optimizer_step"]
    proposed = {
        "budget_basis": "A0 measured mean CUDA-event update time multiplied by its existing 1,000-step pilot budget",
        "cuda_compute_budget_seconds": reference_budget,
        "a0_optimizer_steps": 1_000,
        "a0_tokens": 1_000 * tokens_per_step,
        "a1_optimizer_steps": proposed_a1_steps,
        "a1_tokens": proposed_a1_steps * tokens_per_step,
        "a1_projected_cuda_seconds": proposed_a1_steps * a1["cuda_event_seconds_mean"],
        "a1_step_derivation": "floor(A0 CUDA budget / A1 mean CUDA-event seconds per optimizer step)",
        "schedule": {
            "learning_rate": a0_config.learning_rate,
            "schedule": a0_config.lr_schedule,
            "warmup_ratio": a0_config.warmup_ratio,
            "a0_warmup_steps": max(1, math.ceil(a0_config.warmup_ratio * 1_000)),
            "a1_warmup_steps": max(1, math.ceil(a1_config.warmup_ratio * proposed_a1_steps)),
            "rule": "normalize cosine progress to each arm's planned optimizer-step budget and reach zero at its final step",
        },
        "validation_checkpoints": {
            "interval_steps": 100,
            "a0_steps": list(range(100, 1_001, 100)),
            "a1_steps": list(range(100, proposed_a1_steps + 1, 100))
            + ([] if proposed_a1_steps % 100 == 0 else [proposed_a1_steps]),
            "tokens": a0_config.evaluation_token_budget,
        },
        "acceptance": {
            "advantage": "A1 validation loss minus A0 validation loss <= -0.05 at the same measured CUDA budget, with finite metrics and intact checkpoints",
            "parity": "absolute validation-loss difference <= 0.05",
            "failure": "A1 validation loss is >0.05 worse, or a critical numerical/CUDA/OOM/checkpoint failure occurs",
        },
        "status": "proposal_only_requires_review_and_authorization",
    }
    result = {
        "status": "completed",
        "calibration": "a0_a1_cuda_event_timing_only",
        "training_started": False,
        "checkpoint_written": False,
        "updates_persisted": False,
        "warmup_steps": WARMUP_STEPS,
        "measured_steps_per_arm": MEASURED_STEPS,
        "device": torch.cuda.get_device_name(device),
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "conditions": {
            "micro_batch_size": a0_config.micro_batch_size,
            "context_length": a0_config.context_length,
            "gradient_accumulation_steps": a0_config.gradient_accumulation_steps,
            "effective_batch_size": a0_config.micro_batch_size * a0_config.gradient_accumulation_steps,
            "precision": a0_config.precision,
            "optimizer": a0_config.optimizer,
            "learning_rate": a0_config.learning_rate,
            "weight_decay": a0_config.weight_decay,
            "batch_source": data_record["batch_source"],
            "batches_device_resident_during_timing": True,
        },
        "timing_scope": {
            "cuda_event": "default CUDA stream from optimizer.zero_grad through optimizer.step; excludes host-to-device copies, validation, checkpoint I/O, and data loading",
            "host_wall": "host perf_counter for the same device-resident update scope, including synchronization",
            "full_pilot_wall_not_measured": True,
        },
        "data": data_record,
        "arms": {"A0": a0, "A1": a1},
        "proposed_equal_cuda_budget": proposed,
        "energy": {
            "status": "not_measured",
            "reason": "No synchronized power telemetry was sampled; CUDA elapsed time and FLOPs are not energy measurements.",
            "not_used_for_budget": True,
        },
        "interpretation": "bounded timing calibration only; in-memory updates were discarded and no training pilot was started",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
