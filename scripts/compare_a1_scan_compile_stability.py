#!/usr/bin/env python3
"""Compare eager and default-mode compiled A1 for ten in-memory updates."""

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


PROTOCOL_PATH = "docs/A1_SCAN_COMPILE_STABILITY_PROTOCOL.md"
CONFIG_PATH = "configs/braidnano_a1_recurrent_state_space_proposal.json"
EXPECTED_PARAMETER_COUNT = 41_870_336
STABILITY_UPDATES = 10
WARMUP_UPDATE = 1


def _finite(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"non-finite {label}: {value!r}")
    return result


def _vector_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    difference = (candidate.float() - reference.float()).abs()
    reference_float = reference.float()
    candidate_float = candidate.float()
    return {
        "max_absolute": _finite(difference.max().item(), "max parameter difference"),
        "mean_absolute": _finite(difference.mean().item(), "mean parameter difference"),
        "reference_l2": _finite(reference_float.norm().item(), "reference parameter norm"),
        "candidate_l2": _finite(candidate_float.norm().item(), "candidate parameter norm"),
        "relative_l2": _finite(
            difference.norm().div(reference_float.norm().clamp_min(1e-12)).item(),
            "relative parameter difference",
        ),
    }


def _release_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _make_batches(config: Any) -> list[torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(config.seed + 2_000)
    return [
        torch.randint(
            0,
            config.vocab_size,
            (config.micro_batch_size, config.context_length),
            dtype=torch.long,
            generator=generator,
        )
        for _ in range(STABILITY_UPDATES * config.gradient_accumulation_steps)
    ]


def _update(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batches: list[torch.Tensor],
    config: Any,
    device: torch.device,
) -> tuple[float, float]:
    model.zero_grad(set_to_none=True)
    losses: list[float] = []
    for batch in batches:
        input_ids = batch.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(input_ids, labels=input_ids)
            loss = output.loss
        if loss is None or not bool(torch.isfinite(loss).item()):
            raise RuntimeError("non-finite loss in compile stability comparison")
        losses.append(float(loss.item()))
        (loss / config.gradient_accumulation_steps).backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=config.gradient_clip_norm,
        error_if_nonfinite=True,
    )
    if not bool(torch.isfinite(gradient_norm).item()):
        raise RuntimeError("non-finite gradient norm in compile stability comparison")
    optimizer.step()
    return (
        _finite(sum(losses) / len(losses), "mean update loss"),
        _finite(gradient_norm.item(), "update gradient norm"),
    )


def _run_path(
    config: Any,
    base_state: dict[str, torch.Tensor],
    batches: list[torch.Tensor],
    *,
    compiled: bool,
    device: torch.device,
) -> dict[str, Any]:
    model: torch.nn.Module = A1RecurrentStateSpaceLM(config, use_parallel_scan=True)
    model.load_state_dict(base_state)
    model.to(device).train()
    if compiled:
        model = torch.compile(model, mode="default")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.optimizer_beta1, config.optimizer_beta2),
        eps=config.optimizer_epsilon,
        weight_decay=config.weight_decay,
    )

    torch.cuda.synchronize(device)
    warmup_start = torch.cuda.Event(enable_timing=True)
    warmup_end = torch.cuda.Event(enable_timing=True)
    warmup_start.record()
    warmup_loss, warmup_gradient = _update(
        model,
        optimizer,
        batches[: config.gradient_accumulation_steps],
        config,
        device,
    )
    warmup_end.record()
    warmup_end.synchronize()
    warmup_cuda_seconds = float(warmup_start.elapsed_time(warmup_end) / 1_000.0)

    steady_start = torch.cuda.Event(enable_timing=True)
    steady_end = torch.cuda.Event(enable_timing=True)
    steady_start.record()
    losses: list[float] = []
    gradients: list[float] = []
    for update_index in range(1, STABILITY_UPDATES):
        start = update_index * config.gradient_accumulation_steps
        end = (update_index + 1) * config.gradient_accumulation_steps
        loss, gradient = _update(model, optimizer, batches[start:end], config, device)
        losses.append(loss)
        gradients.append(gradient)
    steady_end.record()
    steady_end.synchronize()
    steady_cuda_seconds = float(steady_start.elapsed_time(steady_end) / 1_000.0)
    tokens_per_update = (
        config.micro_batch_size
        * config.gradient_accumulation_steps
        * config.context_length
    )
    return_value = {
        "path": "compiled_default" if compiled else "eager_parallel_scan",
        "compiled": compiled,
        "compile_mode": "default" if compiled else None,
        "updates": STABILITY_UPDATES,
        "warmup_update": {
            "loss": warmup_loss,
            "gradient_norm": warmup_gradient,
            "cuda_seconds": _finite(warmup_cuda_seconds, "warm-up CUDA time"),
        },
        "steady_update_count": STABILITY_UPDATES - WARMUP_UPDATE,
        "steady_cuda_seconds": _finite(steady_cuda_seconds, "steady CUDA time"),
        "mean_steady_cuda_seconds": _finite(
            steady_cuda_seconds / (STABILITY_UPDATES - WARMUP_UPDATE),
            "mean steady CUDA time",
        ),
        "tokens_processed_steady": (STABILITY_UPDATES - WARMUP_UPDATE) * tokens_per_update,
        "tokens_per_second_steady": _finite(
            (STABILITY_UPDATES - WARMUP_UPDATE) * tokens_per_update / steady_cuda_seconds,
            "steady throughput",
        ),
        "loss_trajectory": [warmup_loss, *losses],
        "gradient_norm_trajectory": [warmup_gradient, *gradients],
        "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved(device)),
        "final_parameters": parameters_to_vector(model.parameters()).detach().cpu().clone(),
    }
    del model, optimizer
    _release_cuda()
    return return_value


def _report(result: dict[str, Any]) -> str:
    eager = result["paths"]["eager_parallel_scan"]
    compiled = result["paths"]["compiled_default"]
    lines = [
        "# A1 scan compilation stability result",
        "",
        "Status: complete; engineering-only ten-update in-memory comparison.",
        "",
        f"Device: {result['device']}; PyTorch {result['torch_version']}; CUDA build {result['torch_cuda_build']}.",
        "",
        "| path | warm-up CUDA s | steady CUDA s/update | steady tok/s | peak reserved VRAM |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| eager parallel scan | {eager['warmup_update']['cuda_seconds']:.6f} | {eager['mean_steady_cuda_seconds']:.6f} | {eager['tokens_per_second_steady']:.1f} | {eager['peak_reserved_vram_bytes']} |",
        f"| torch.compile default | {compiled['warmup_update']['cuda_seconds']:.6f} | {compiled['mean_steady_cuda_seconds']:.6f} | {compiled['tokens_per_second_steady']:.1f} | {compiled['peak_reserved_vram_bytes']} |",
        "",
        f"Steady-state speedup: {result['speedup_compiled_over_eager']:.3f}x.",
        f"Final-parameter relative L2 difference: {result['final_parameter_difference']['relative_l2']:.6f}.",
        "",
        "Per-update loss and gradient comparisons are in the machine-readable aggregate. This result does not authorize normal-runner integration.",
        "",
        "Integrity: both paths completed ten finite updates; no checkpoint or model artifact was written.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results/a1_scan_compile_stability.json")
    parser.add_argument("--report", type=Path, default=REPO_ROOT / "docs/A1_SCAN_COMPILE_STABILITY_RESULT.md")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    output_path = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    report_path = args.report if args.report.is_absolute() else REPO_ROOT / args.report
    config = load_config(config_path)
    if config.stage != "A1" or config.model_type != "recurrent_state_space":
        raise ValueError("compile stability requires the explicit A1 configuration")
    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA is unavailable")

    seed_report = set_reproducible_seed(config.seed, deterministic=True)
    device = torch.device("cuda")
    base_model = A1RecurrentStateSpaceLM(config, use_parallel_scan=True)
    parameter_count = base_model.parameter_count()
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(f"unexpected A1 parameter count: {parameter_count}")
    base_state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in base_model.state_dict().items()
    }
    del base_model
    batches = _make_batches(config)
    eager = _run_path(config, base_state, batches, compiled=False, device=device)
    compiled = _run_path(config, base_state, batches, compiled=True, device=device)
    loss_comparison = []
    gradient_comparison = []
    for step, (eager_loss, compiled_loss, eager_gradient, compiled_gradient) in enumerate(
        zip(
            eager["loss_trajectory"],
            compiled["loss_trajectory"],
            eager["gradient_norm_trajectory"],
            compiled["gradient_norm_trajectory"],
        ),
        start=1,
    ):
        loss_comparison.append(
            {
                "step": step,
                "compiled_minus_eager": _finite(compiled_loss - eager_loss, "loss delta"),
                "relative_difference": _finite(
                    abs(compiled_loss - eager_loss) / max(abs(eager_loss), 1e-12),
                    "relative loss difference",
                ),
            }
        )
        gradient_comparison.append(
            {
                "step": step,
                "compiled_minus_eager": _finite(
                    compiled_gradient - eager_gradient,
                    "gradient norm delta",
                ),
                "relative_difference": _finite(
                    abs(compiled_gradient - eager_gradient)
                    / max(abs(eager_gradient), 1e-12),
                    "relative gradient difference",
                ),
            }
        )
    result = {
        "version": "a1_scan_compile_stability_v1",
        "status": "complete",
        "protocol_path": PROTOCOL_PATH,
        "protocol_sha256": sha256_file(REPO_ROOT / PROTOCOL_PATH),
        "script_path": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "config_path": str(config_path.relative_to(REPO_ROOT)),
        "config_sha256": sha256_file(config_path),
        "seed": config.seed,
        "batch_seed": config.seed + 2_000,
        "seed_report": seed_report,
        "device": torch.cuda.get_device_name(device),
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "parameter_count": parameter_count,
        "paths": {
            "eager_parallel_scan": {
                key: value for key, value in eager.items() if key != "final_parameters"
            },
            "compiled_default": {
                key: value for key, value in compiled.items() if key != "final_parameters"
            },
        },
        "speedup_compiled_over_eager": _finite(
            eager["mean_steady_cuda_seconds"] / compiled["mean_steady_cuda_seconds"],
            "compile stability speedup",
        ),
        "loss_comparison": loss_comparison,
        "gradient_norm_comparison": gradient_comparison,
        "final_parameter_difference": _vector_metrics(
            eager["final_parameters"], compiled["final_parameters"]
        ),
        "training_pilot_started": False,
        "checkpoint_written": False,
        "weights_written": False,
        "all_values_finite": True,
        "interpretation": "engineering stability comparison only; compiled runner adoption remains separately gated",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_report(result), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
