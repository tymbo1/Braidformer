#!/usr/bin/env python3
"""Run the explicitly approved one-update A1 host-GPU smoke test.

This is an infrastructure and numerical smoke test, not the 1,000-step A1
pilot.  It uses synthetic token batches, performs one AdamW update after the
configured eight-way gradient accumulation, writes no checkpoint, and records
the measured CUDA resources.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from braidformer.config import load_config
from braidformer.training.integrity import set_reproducible_seed, sha256_file
from braidformer.training.runner import build_model


EXPECTED_PARAMETER_COUNT = 41_870_336


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


def _validate_smoke_config(config: Any) -> None:
    expected = {
        "stage": "A1",
        "model_type": "recurrent_state_space",
        "micro_batch_size": 2,
        "context_length": 512,
        "gradient_accumulation_steps": 8,
        "precision": "fp16",
    }
    for field, value in expected.items():
        if getattr(config, field) != value:
            raise ValueError(f"A1 smoke test requires {field}={value!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "braidnano_a1_recurrent_state_space_proposal.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "a1_cuda_smoke.json",
    )
    args = parser.parse_args(argv)

    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    output_path = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    config = load_config(config_path)
    _validate_smoke_config(config)
    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA is unavailable")

    seed_report = set_reproducible_seed(config.seed, deterministic=True)
    device = torch.device("cuda")
    model = build_model(config).to(device)
    parameter_count = model.parameter_count()
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(f"unexpected A1 parameter count: {parameter_count}")

    # Verify that each forward starts with a fresh sequence state before the
    # timed optimizer update.  A1 has no persistent state buffer by design.
    model.eval()
    probe = torch.randint(
        0,
        config.vocab_size,
        (config.micro_batch_size, config.context_length),
        dtype=torch.long,
        device=device,
    )
    with torch.inference_mode():
        first_probe = model(probe).logits
        second_probe = model(probe).logits
    state_reset_max_abs_diff = float((first_probe - second_probe).abs().max().item())
    if not math.isfinite(state_reset_max_abs_diff) or state_reset_max_abs_diff != 0.0:
        raise RuntimeError(
            "A1 sequence-state reset check failed: "
            f"max_abs_diff={state_reset_max_abs_diff}"
        )

    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.optimizer_beta1, config.optimizer_beta2),
        eps=config.optimizer_epsilon,
        weight_decay=config.weight_decay,
    )
    tokens_per_microbatch = config.micro_batch_size * config.context_length
    tokens_processed = tokens_per_microbatch * config.gradient_accumulation_steps
    amp_dtype = torch.float16

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    for _ in range(config.gradient_accumulation_steps):
        input_ids = torch.randint(
            0,
            config.vocab_size,
            (config.micro_batch_size, config.context_length),
            dtype=torch.long,
            device=device,
        )
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            output = model(input_ids, labels=input_ids)
            loss = output.loss
        if loss is None or not bool(torch.isfinite(loss).item()):
            raise RuntimeError("A1 smoke test produced a non-finite loss")
        losses.append(float(loss.item()))
        (loss / config.gradient_accumulation_steps).backward()

    finite_gradients, gradient_tensor_count, gradient_element_count = _finite_gradients(model)
    if not finite_gradients or gradient_tensor_count == 0:
        raise RuntimeError("A1 smoke test produced missing or non-finite gradients")
    if model.token_embedding.weight.grad is None:
        raise RuntimeError("A1 token embedding gradient is missing")
    gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=config.gradient_clip_norm,
        error_if_nonfinite=True,
    )
    gradient_norm = float(gradient_norm_tensor.item())
    if not math.isfinite(gradient_norm):
        raise RuntimeError("A1 smoke test produced a non-finite gradient norm")

    optimizer.step()
    torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - started
    result = {
        "status": "completed",
        "smoke_test": "a1_cuda_one_optimizer_update",
        "training_pilot_started": False,
        "optimizer_steps_executed": 1,
        "checkpoint_written": False,
        "config_path": str(config_path.relative_to(REPO_ROOT)),
        "config_sha256": sha256_file(config_path),
        "seed": config.seed,
        "seed_report": seed_report,
        "stage": config.stage,
        "model_type": config.model_type,
        "recurrent_path": config.recurrent_path,
        "parallel_scan_enabled": model.use_parallel_scan,
        "parameter_count": parameter_count,
        "micro_batch_size": config.micro_batch_size,
        "context_length": config.context_length,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "effective_batch_size": config.micro_batch_size * config.gradient_accumulation_steps,
        "tokens_processed": tokens_processed,
        "finite_loss": True,
        "mean_microbatch_loss": sum(losses) / len(losses),
        "finite_gradients": True,
        "gradient_tensor_count": gradient_tensor_count,
        "gradient_element_count": gradient_element_count,
        "gradient_norm": gradient_norm,
        "state_reset_verified": True,
        "state_reset_max_abs_diff": state_reset_max_abs_diff,
        "optimizer": config.optimizer,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "precision": config.precision,
        "gradient_checkpointing": {
            "configured": config.gradient_checkpointing,
            "implemented": False,
            "enabled": False,
        },
        "optimizer_step_wall_seconds": elapsed_seconds,
        "tokens_per_second": tokens_processed / elapsed_seconds,
        "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device),
        "device": torch.cuda.get_device_name(device),
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "note": "synthetic one-update readiness smoke test; not an A1 training result",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
