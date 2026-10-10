#!/usr/bin/env python3
"""Compare ten matched A1 optimizer updates for the two recurrence paths.

The sequential reference and associative scan each start from the same model
weights and an empty AdamW state.  They consume the same pre-generated token
batches under the approved A1 smoke-test conditions.  This is a bounded
numerical-stability check only: it never writes a checkpoint and never starts
the 1,000-step A1 pilot.
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
from torch.nn.utils import parameters_to_vector

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from braidformer.config import load_config
from braidformer.models import A1RecurrentStateSpaceLM
from braidformer.training.integrity import set_reproducible_seed, sha256_file
from braidformer.training.runner import cosine_warmup_learning_rate


EXPECTED_PARAMETER_COUNT = 41_870_336
STABILITY_STEPS = 10


def _release_cuda_objects() -> None:
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def _assert_finite_gradients(model: torch.nn.Module) -> tuple[bool, int]:
    gradient_count = 0
    for parameter in model.parameters():
        if parameter.grad is None:
            raise RuntimeError("stability check encountered a missing gradient")
        gradient_count += 1
        if not bool(torch.isfinite(parameter.grad).all().item()):
            raise RuntimeError("stability check encountered a non-finite gradient")
    return True, gradient_count


def _vector_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    reference_float = reference.float()
    candidate_float = candidate.float()
    difference = candidate_float - reference_float
    difference_abs = difference.abs()
    reference_norm = torch.linalg.vector_norm(reference_float)
    difference_norm = torch.linalg.vector_norm(difference)
    return {
        "max_absolute": float(difference_abs.max().item()),
        "mean_absolute": float(difference_abs.mean().item()),
        "relative_l2": float(
            (difference_norm / reference_norm.clamp_min(1e-24)).item()
        ),
        "reference_l2": float(reference_norm.item()),
        "candidate_l2": float(torch.linalg.vector_norm(candidate_float).item()),
    }


def _run_path(
    config: Any,
    base_state: dict[str, torch.Tensor],
    initial_parameters: torch.Tensor,
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
    warmup_steps = max(1, math.ceil(config.warmup_ratio * config.max_steps))
    tokens_per_update = (
        config.micro_batch_size
        * config.gradient_accumulation_steps
        * config.context_length
    )
    losses: list[float] = []
    gradient_norms: list[float] = []
    learning_rates: list[float] = []
    update_wall_seconds: list[float] = []
    nonfinite_loss_steps: list[int] = []
    nonfinite_gradient_steps: list[int] = []

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for update_index in range(STABILITY_STEPS):
        step = update_index + 1
        learning_rate = cosine_warmup_learning_rate(
            config.learning_rate,
            step,
            warmup_steps,
            config.max_steps,
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate
        learning_rates.append(learning_rate)

        update_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        microbatch_losses: list[float] = []
        for accumulation_index in range(config.gradient_accumulation_steps):
            batch = batches[update_index * config.gradient_accumulation_steps + accumulation_index]
            input_ids = batch.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model(input_ids, labels=input_ids)
                loss = output.loss
            if loss is None or not bool(torch.isfinite(loss).item()):
                nonfinite_loss_steps.append(step)
                raise RuntimeError(f"non-finite loss at {use_parallel_scan=} {step=}")
            microbatch_losses.append(float(loss.item()))
            (loss / config.gradient_accumulation_steps).backward()

        gradients_finite, _ = _assert_finite_gradients(model)
        if not gradients_finite:
            nonfinite_gradient_steps.append(step)
            raise RuntimeError(f"non-finite gradient at {use_parallel_scan=} {step=}")
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=config.gradient_clip_norm,
            error_if_nonfinite=True,
        )
        if not bool(torch.isfinite(gradient_norm).item()):
            nonfinite_gradient_steps.append(step)
            raise RuntimeError(f"non-finite gradient norm at {use_parallel_scan=} {step=}")
        optimizer.step()
        torch.cuda.synchronize(device)
        update_wall_seconds.append(time.perf_counter() - update_started)
        losses.append(sum(microbatch_losses) / len(microbatch_losses))
        gradient_norms.append(float(gradient_norm.item()))

    total_wall_seconds = time.perf_counter() - started
    final_parameters = parameters_to_vector(model.parameters()).detach().cpu().clone()
    final_gradients = parameters_to_vector(
        [parameter.grad for parameter in model.parameters()]
    ).detach().cpu().clone()
    result = {
        "path": "parallel_scan" if use_parallel_scan else "sequential_reference",
        "use_parallel_scan": use_parallel_scan,
        "optimizer_steps": STABILITY_STEPS,
        "tokens_processed": STABILITY_STEPS * tokens_per_update,
        "tokens_per_update": tokens_per_update,
        "loss_trajectory": losses,
        "gradient_norm_trajectory": gradient_norms,
        "learning_rate_trajectory": learning_rates,
        "update_wall_seconds": update_wall_seconds,
        "total_wall_seconds": total_wall_seconds,
        "tokens_per_second": STABILITY_STEPS * tokens_per_update / total_wall_seconds,
        "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device),
        "nonfinite_loss_steps": nonfinite_loss_steps,
        "nonfinite_gradient_steps": nonfinite_gradient_steps,
        "final_parameters": final_parameters,
        "final_gradients": final_gradients,
        "parameter_change_from_initial": _vector_metrics(
            initial_parameters,
            final_parameters,
        ),
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
        default=REPO_ROOT / "results" / "a1_scan_stability.json",
    )
    args = parser.parse_args(argv)

    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    output_path = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    config = load_config(config_path)
    if config.stage != "A1" or config.model_type != "recurrent_state_space":
        raise ValueError("A1 scan stability requires the explicit A1 configuration")
    if (
        config.micro_batch_size != 2
        or config.context_length != 512
        or config.gradient_accumulation_steps != 8
        or config.precision != "fp16"
    ):
        raise ValueError("stability conditions do not match the approved A1 smoke configuration")
    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA is unavailable")

    seed_report = set_reproducible_seed(config.seed, deterministic=True)
    device = torch.device("cuda")
    base_model = A1RecurrentStateSpaceLM(config, use_parallel_scan=False)
    parameter_count = base_model.parameter_count()
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(f"unexpected A1 parameter count: {parameter_count}")
    initial_parameters = parameters_to_vector(base_model.parameters()).detach().cpu().clone()
    base_state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in base_model.state_dict().items()
    }
    del base_model

    generator = torch.Generator(device="cpu").manual_seed(config.seed + 2_000)
    batches = [
        torch.randint(
            0,
            config.vocab_size,
            (config.micro_batch_size, config.context_length),
            dtype=torch.long,
            generator=generator,
        )
        for _ in range(STABILITY_STEPS * config.gradient_accumulation_steps)
    ]

    reference = _run_path(
        config,
        base_state,
        initial_parameters,
        batches,
        use_parallel_scan=False,
        device=device,
    )
    parallel = _run_path(
        config,
        base_state,
        initial_parameters,
        batches,
        use_parallel_scan=True,
        device=device,
    )

    loss_comparison = []
    gradient_norm_comparison = []
    for step, (reference_loss, parallel_loss, reference_norm, parallel_norm) in enumerate(
        zip(
            reference["loss_trajectory"],
            parallel["loss_trajectory"],
            reference["gradient_norm_trajectory"],
            parallel["gradient_norm_trajectory"],
        ),
        start=1,
    ):
        loss_comparison.append(
            {
                "step": step,
                "sequential_reference": reference_loss,
                "parallel_scan": parallel_loss,
                "absolute_difference": abs(parallel_loss - reference_loss),
                "relative_difference": abs(parallel_loss - reference_loss)
                / max(abs(reference_loss), 1e-12),
            }
        )
        gradient_norm_comparison.append(
            {
                "step": step,
                "sequential_reference": reference_norm,
                "parallel_scan": parallel_norm,
                "absolute_difference": abs(parallel_norm - reference_norm),
                "relative_difference": abs(parallel_norm - reference_norm)
                / max(abs(reference_norm), 1e-12),
            }
        )

    final_parameter_difference = _vector_metrics(
        reference["final_parameters"],
        parallel["final_parameters"],
    )
    final_gradient_difference = _vector_metrics(
        reference["final_gradients"],
        parallel["final_gradients"],
    )
    result = {
        "status": "completed",
        "comparison": "a1_sequential_reference_vs_associative_scan_10_updates",
        "training_pilot_started": False,
        "checkpoint_written": False,
        "config_path": str(config_path.relative_to(REPO_ROOT)),
        "config_sha256": sha256_file(config_path),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "seed": config.seed,
        "seed_report": seed_report,
        "batch_seed": config.seed + 2_000,
        "device": torch.cuda.get_device_name(device),
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "parameter_count": parameter_count,
        "conditions": {
            "optimizer_steps": STABILITY_STEPS,
            "micro_batch_size": config.micro_batch_size,
            "context_length": config.context_length,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "effective_batch_size": config.micro_batch_size * config.gradient_accumulation_steps,
            "tokens_per_update": (
                config.micro_batch_size
                * config.gradient_accumulation_steps
                * config.context_length
            ),
            "tokens_per_path": reference["tokens_processed"],
            "precision": config.precision,
            "optimizer": config.optimizer,
            "learning_rate": config.learning_rate,
            "lr_schedule": config.lr_schedule,
            "warmup_steps": max(1, math.ceil(config.warmup_ratio * config.max_steps)),
            "weight_decay": config.weight_decay,
            "gradient_clip_norm": config.gradient_clip_norm,
            "optimizer_state_initial": "empty AdamW state for both paths",
            "input_batches": "one pre-generated CPU batch sequence shared by both paths",
        },
        "paths": {
            "sequential_reference": {
                key: value
                for key, value in reference.items()
                if key not in {"final_parameters", "final_gradients"}
            },
            "parallel_scan": {
                key: value
                for key, value in parallel.items()
                if key not in {"final_parameters", "final_gradients"}
            },
        },
        "loss_comparison": loss_comparison,
        "gradient_norm_comparison": gradient_norm_comparison,
        "final_parameter_difference": final_parameter_difference,
        "final_gradient_difference": final_gradient_difference,
        "interpretation": (
            "bounded numerical-stability comparison only; no A1 training pilot was started"
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
