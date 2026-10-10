#!/usr/bin/env python3
"""Benchmark A1 reference recurrence versus its optional associative scan.

The benchmark uses identical initial weights, synthetic token batches, FP16
autocast, micro-batch 2, eight-way accumulation, and one AdamW update for each
path.  It never writes a checkpoint and is not the A1 training pilot.
"""

from __future__ import annotations

import argparse
import gc
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
from braidformer.models import A1RecurrentStateSpaceLM
from braidformer.training.integrity import set_reproducible_seed, sha256_file


EXPECTED_PARAMETER_COUNT = 41_870_336


def _assert_finite_gradients(model: torch.nn.Module) -> None:
    for parameter in model.parameters():
        if parameter.grad is None:
            raise RuntimeError("benchmark encountered a missing gradient")
        if not bool(torch.isfinite(parameter.grad).all().item()):
            raise RuntimeError("benchmark encountered a non-finite gradient")


def _gradient_snapshot(model: torch.nn.Module) -> list[torch.Tensor]:
    return [parameter.grad.detach().cpu().clone() for parameter in model.parameters()]


def _compare_snapshots(
    reference: list[torch.Tensor],
    candidate: list[torch.Tensor],
) -> dict[str, float]:
    if len(reference) != len(candidate):
        raise RuntimeError("gradient snapshots have different parameter counts")
    max_absolute = 0.0
    max_relative = 0.0
    absolute_sum = 0.0
    squared_difference_sum = 0.0
    squared_reference_sum = 0.0
    element_count = 0
    for reference_tensor, candidate_tensor in zip(reference, candidate):
        reference_float = reference_tensor.float()
        candidate_float = candidate_tensor.float()
        difference = (candidate_float - reference_float).abs()
        max_absolute = max(max_absolute, float(difference.max().item()))
        denominator = reference_float.abs().clamp_min(1e-8)
        max_relative = max(max_relative, float((difference / denominator).max().item()))
        absolute_sum += float(difference.sum().item())
        squared_difference_sum += float(difference.square().sum().item())
        squared_reference_sum += float(reference_float.square().sum().item())
        element_count += difference.numel()
    return {
        "max_absolute": max_absolute,
        "max_relative": max_relative,
        "mean_absolute": absolute_sum / element_count,
        "relative_l2": math.sqrt(
            squared_difference_sum / max(squared_reference_sum, 1e-24)
        ),
    }


def _release_cuda_objects() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _forward_probe(
    config: Any,
    base_state: dict[str, torch.Tensor],
    input_ids: torch.Tensor,
    *,
    use_parallel_scan: bool,
    device: torch.device,
) -> torch.Tensor:
    model = A1RecurrentStateSpaceLM(config, use_parallel_scan=use_parallel_scan)
    model.load_state_dict(base_state)
    model.to(device).eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = model(input_ids.to(device)).logits.detach().cpu()
    del model
    _release_cuda_objects()
    return logits


def _run_path(
    config: Any,
    base_state: dict[str, torch.Tensor],
    batches: list[torch.Tensor],
    *,
    use_parallel_scan: bool,
    device: torch.device,
) -> dict[str, Any]:
    model = A1RecurrentStateSpaceLM(config, use_parallel_scan=use_parallel_scan)
    model.load_state_dict(base_state)
    model.to(device).train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.optimizer_beta1, config.optimizer_beta2),
        eps=config.optimizer_epsilon,
        weight_decay=config.weight_decay,
    )
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    for batch in batches:
        input_ids = batch.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(input_ids, labels=input_ids)
            loss = output.loss
        if loss is None or not bool(torch.isfinite(loss).item()):
            raise RuntimeError("benchmark encountered a non-finite loss")
        losses.append(float(loss.item()))
        (loss / config.gradient_accumulation_steps).backward()
    _assert_finite_gradients(model)
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=config.gradient_clip_norm,
        error_if_nonfinite=True,
    )
    if not math.isfinite(float(gradient_norm.item())):
        raise RuntimeError("benchmark encountered a non-finite gradient norm")
    optimizer.step()
    torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - started
    gradients = _gradient_snapshot(model)
    result = {
        "path": "parallel_scan" if use_parallel_scan else "sequential_reference",
        "use_parallel_scan": use_parallel_scan,
        "optimizer_steps": 1,
        "tokens_processed": len(batches) * config.micro_batch_size * config.context_length,
        "mean_microbatch_loss": sum(losses) / len(losses),
        "gradient_norm": float(gradient_norm.item()),
        "optimizer_step_wall_seconds": elapsed_seconds,
        "tokens_per_second": (
            len(batches) * config.micro_batch_size * config.context_length / elapsed_seconds
        ),
        "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device),
        "gradients": gradients,
    }
    del model, optimizer
    _release_cuda_objects()
    return result


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
        default=REPO_ROOT / "results" / "a1_recurrence_benchmark.json",
    )
    args = parser.parse_args(argv)

    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    output_path = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    config = load_config(config_path)
    if config.stage != "A1" or config.model_type != "recurrent_state_space":
        raise ValueError("A1 recurrence benchmark requires the explicit A1 configuration")
    if (
        config.micro_batch_size != 2
        or config.context_length != 512
        or config.gradient_accumulation_steps != 8
        or config.precision != "fp16"
    ):
        raise ValueError("benchmark conditions do not match the approved A1 smoke configuration")
    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA is unavailable")

    seed_report = set_reproducible_seed(config.seed, deterministic=True)
    device = torch.device("cuda")
    base_model = A1RecurrentStateSpaceLM(config, use_parallel_scan=False)
    parameter_count = base_model.parameter_count()
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("unexpected A1 parameter count")
    base_state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in base_model.state_dict().items()
    }
    generator = torch.Generator(device="cpu").manual_seed(config.seed + 1_000)
    batches = [
        torch.randint(
            0,
            config.vocab_size,
            (config.micro_batch_size, config.context_length),
            dtype=torch.long,
            generator=generator,
        )
        for _ in range(config.gradient_accumulation_steps)
    ]
    probe_logits_reference = _forward_probe(
        config,
        base_state,
        batches[0],
        use_parallel_scan=False,
        device=device,
    )
    probe_logits_parallel = _forward_probe(
        config,
        base_state,
        batches[0],
        use_parallel_scan=True,
        device=device,
    )
    forward_difference = _compare_snapshots(
        [probe_logits_reference],
        [probe_logits_parallel],
    )
    del base_model
    reference = _run_path(
        config,
        base_state,
        batches,
        use_parallel_scan=False,
        device=device,
    )
    parallel = _run_path(
        config,
        base_state,
        batches,
        use_parallel_scan=True,
        device=device,
    )
    gradient_difference = _compare_snapshots(reference["gradients"], parallel["gradients"])
    del reference["gradients"], parallel["gradients"]

    result = {
        "status": "completed",
        "benchmark": "a1_recurrence_reference_vs_associative_scan",
        "training_pilot_started": False,
        "checkpoint_written": False,
        "config_path": str(config_path.relative_to(REPO_ROOT)),
        "config_sha256": sha256_file(config_path),
        "seed": config.seed,
        "seed_report": seed_report,
        "device": torch.cuda.get_device_name(device),
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "parameter_count": parameter_count,
        "conditions": {
            "micro_batch_size": config.micro_batch_size,
            "context_length": config.context_length,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "effective_batch_size": config.micro_batch_size * config.gradient_accumulation_steps,
            "tokens_per_path": len(batches) * config.micro_batch_size * config.context_length,
            "precision": config.precision,
            "optimizer": config.optimizer,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
        },
        "forward_equivalence": forward_difference,
        "gradient_equivalence": gradient_difference,
        "paths": {
            "sequential_reference": reference,
            "parallel_scan": parallel,
        },
        "speedup_parallel_over_reference": (
            reference["optimizer_step_wall_seconds"] / parallel["optimizer_step_wall_seconds"]
        ),
        "throughput_ratio_parallel_over_reference": (
            parallel["tokens_per_second"] / reference["tokens_per_second"]
        ),
        "interpretation": "one-update implementation benchmark only; not an A1 training result",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
