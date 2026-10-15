#!/usr/bin/env python3
"""Measure isolated mixer and end-to-end A1 compile speed retention."""

from __future__ import annotations

import argparse
import gc
import hashlib
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
from braidformer.models import A1RecurrentStateSpaceLM, GatedDiagonalStateSpaceMixer
from braidformer.training.integrity import set_reproducible_seed, sha256_file
from braidformer.training.runner import build_forward_model


PROTOCOL_PATH = "docs/A1_END_TO_END_COMPILE_BENCHMARK_PROTOCOL.md"
CONFIG_PATH = "configs/braidnano_a1_recurrent_state_space_proposal.json"
EXPECTED_PARAMETER_COUNT = 41_870_336
WARMUP_UPDATES = 1
MEASURED_UPDATES = 5
TOKEN_BATCH_SEED_OFFSET = 3_000
MIXER_BATCH_SEED_OFFSET = 4_000
LOSS_RELATIVE_TOLERANCE = 1e-4
GRADIENT_RELATIVE_TOLERANCE = 1e-2
PARAMETER_RELATIVE_L2_TOLERANCE = 0.02


def _finite(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"non-finite {label}: {value!r}")
    return result


def _vector_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    reference_float = reference.float()
    candidate_float = candidate.float()
    difference = (candidate_float - reference_float).abs()
    return {
        "max_absolute": _finite(difference.max().item(), "maximum vector difference"),
        "mean_absolute": _finite(difference.mean().item(), "mean vector difference"),
        "reference_l2": _finite(reference_float.norm().item(), "reference vector norm"),
        "candidate_l2": _finite(candidate_float.norm().item(), "candidate vector norm"),
        "relative_l2": _finite(
            difference.norm().div(reference_float.norm().clamp_min(1e-12)).item(),
            "relative vector difference",
        ),
    }


def _trajectory_metrics(
    reference: list[float],
    candidate: list[float],
    label: str,
) -> dict[str, float]:
    if len(reference) != len(candidate) or not reference:
        raise RuntimeError(f"{label} trajectories are not aligned")
    relative = [
        abs(float(c_value) - float(r_value)) / max(abs(float(r_value)), 1e-12)
        for r_value, c_value in zip(reference, candidate)
    ]
    absolute = [
        abs(float(c_value) - float(r_value))
        for r_value, c_value in zip(reference, candidate)
    ]
    return {
        "max_absolute": _finite(max(absolute), f"{label} maximum absolute difference"),
        "max_relative": _finite(max(relative), f"{label} maximum relative difference"),
        "mean_absolute": _finite(
            sum(absolute) / len(absolute),
            f"{label} mean absolute difference",
        ),
    }


def _state_key_digest(keys: list[str]) -> str:
    return hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()


def _optimizer_structure(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    state = optimizer.state_dict()
    value_keys = sorted(
        {
            str(key)
            for parameter_state in state["state"].values()
            for key in parameter_state
        }
    )
    return {
        "top_level_keys": sorted(str(key) for key in state),
        "state_entry_count": len(state["state"]),
        "state_value_keys": value_keys,
        "param_group_count": len(state["param_groups"]),
        "parameters_per_group": [len(group["params"]) for group in state["param_groups"]],
    }


def _release_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _reset_compiler() -> None:
    compiler_namespace = getattr(torch, "compiler", None)
    reset = getattr(compiler_namespace, "reset", None)
    if callable(reset):
        reset()


def _copy_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def _make_token_batches(config: Any) -> list[torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(
        config.seed + TOKEN_BATCH_SEED_OFFSET
    )
    return [
        torch.randint(
            0,
            config.vocab_size,
            (config.micro_batch_size, config.context_length),
            dtype=torch.long,
            generator=generator,
        )
        for _ in range(
            (WARMUP_UPDATES + MEASURED_UPDATES)
            * config.gradient_accumulation_steps
        )
    ]


def _make_mixer_batches(config: Any) -> list[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator(device="cpu").manual_seed(
        config.seed + MIXER_BATCH_SEED_OFFSET
    )
    return [
        (
            torch.randn(
                (config.micro_batch_size, config.context_length, config.hidden_size),
                generator=generator,
            ),
            torch.randn(
                (config.micro_batch_size, config.context_length, config.hidden_size),
                generator=generator,
            ),
        )
        for _ in range(
            (WARMUP_UPDATES + MEASURED_UPDATES)
            * config.gradient_accumulation_steps
        )
    ]


def _device_token_batches(
    batches: list[torch.Tensor], device: torch.device
) -> list[torch.Tensor]:
    return [batch.to(device) for batch in batches]


def _device_mixer_batches(
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [(hidden.to(device), target.to(device)) for hidden, target in batches]


def _assert_finite_gradients(model: torch.nn.Module, label: str) -> None:
    for parameter in model.parameters():
        if parameter.grad is None:
            raise RuntimeError(f"{label} encountered a missing gradient")
        if not bool(torch.isfinite(parameter.grad).all().item()):
            raise RuntimeError(f"{label} encountered a non-finite gradient")


def _compile_mixer_forwards(model: A1RecurrentStateSpaceLM) -> None:
    for block in model.blocks:
        mixer = block.state_space
        mixer.forward = torch.compile(mixer.forward, mode="default")


def _run_complete_update(
    *,
    base_model: torch.nn.Module,
    forward_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batches: list[torch.Tensor],
    config: Any,
    label: str,
) -> tuple[float, float]:
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    for batch in batches:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = forward_model(batch, labels=batch)
            loss = output.loss
        if loss is None or not bool(torch.isfinite(loss).item()):
            raise RuntimeError(f"{label} encountered a non-finite loss")
        losses.append(float(loss.item()))
        (loss / config.gradient_accumulation_steps).backward()
    _assert_finite_gradients(base_model, label)
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        base_model.parameters(),
        max_norm=config.gradient_clip_norm,
        error_if_nonfinite=True,
    )
    if not bool(torch.isfinite(gradient_norm).item()):
        raise RuntimeError(f"{label} encountered a non-finite gradient norm")
    optimizer.step()
    return (
        _finite(sum(losses) / len(losses), f"{label} loss"),
        _finite(gradient_norm.item(), f"{label} gradient norm"),
    )


def _run_mixer_update(
    *,
    base_model: torch.nn.Module,
    forward_model: Any,
    optimizer: torch.optim.Optimizer,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    config: Any,
    label: str,
) -> tuple[float, float]:
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    for hidden, target in batches:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = forward_model(hidden)
            loss = (output.float() - target).square().mean()
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError(f"{label} encountered a non-finite loss")
        losses.append(float(loss.item()))
        (loss / config.gradient_accumulation_steps).backward()
    _assert_finite_gradients(base_model, label)
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        base_model.parameters(),
        max_norm=config.gradient_clip_norm,
        error_if_nonfinite=True,
    )
    if not bool(torch.isfinite(gradient_norm).item()):
        raise RuntimeError(f"{label} encountered a non-finite gradient norm")
    optimizer.step()
    return (
        _finite(sum(losses) / len(losses), f"{label} loss"),
        _finite(gradient_norm.item(), f"{label} gradient norm"),
    )


def _run_isolated_forward(
    config: Any,
    base_state: dict[str, torch.Tensor],
    hidden: torch.Tensor,
    *,
    compiled: bool,
    device: torch.device,
) -> dict[str, Any]:
    mixer = GatedDiagonalStateSpaceMixer(
        hidden_size=config.hidden_size,
        dropout=config.dropout,
        use_bias=config.use_bias,
        state_transition_initialization=config.state_transition_initialization,
    )
    mixer.load_state_dict(base_state)
    mixer.to(device).eval()
    forward_model: Any = mixer
    mode = None
    if compiled:
        mode = "reduce-overhead"
        forward_model = torch.compile(mixer.forward, mode=mode)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    warmup_start = time.perf_counter()
    warmup_cuda_start = torch.cuda.Event(enable_timing=True)
    warmup_cuda_end = torch.cuda.Event(enable_timing=True)
    warmup_cuda_start.record()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        warmup_output = forward_model(hidden)
    warmup_cuda_end.record()
    warmup_cuda_end.synchronize()
    warmup_wall_seconds = time.perf_counter() - warmup_start
    warmup_cuda_seconds = warmup_cuda_start.elapsed_time(warmup_cuda_end) / 1_000.0

    torch.cuda.reset_peak_memory_stats(device)
    measured_start = torch.cuda.Event(enable_timing=True)
    measured_end = torch.cuda.Event(enable_timing=True)
    measured_start.record()
    output = warmup_output
    for _ in range(5):
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            output = forward_model(hidden)
    measured_end.record()
    measured_end.synchronize()
    measured_cuda_seconds = measured_start.elapsed_time(measured_end) / 1_000.0
    output_cpu = output.detach().cpu().clone()
    result = {
        "path": "isolated_mixer_compiled_forward" if compiled else "isolated_mixer_eager_forward",
        "compiled": compiled,
        "compile_mode": mode,
        "warmup_wall_seconds": _finite(
            warmup_wall_seconds,
            "isolated forward warm-up wall time",
        ),
        "warmup_cuda_seconds": _finite(
            warmup_cuda_seconds,
            "isolated forward warm-up CUDA time",
        ),
        "measured_calls": 5,
        "measured_cuda_seconds": _finite(
            measured_cuda_seconds,
            "isolated forward CUDA time",
        ),
        "mean_measured_cuda_seconds": _finite(
            measured_cuda_seconds / 5.0,
            "isolated forward mean CUDA time",
        ),
        "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved(device)),
        "output": output_cpu,
        "state_key_digest": _state_key_digest(list(mixer.state_dict().keys())),
        "state_key_count": len(mixer.state_dict()),
        "all_values_finite": bool(torch.isfinite(output_cpu).all().item()),
    }
    del forward_model, mixer
    _reset_compiler()
    _release_cuda()
    return result


def _run_isolated_training(
    config: Any,
    base_state: dict[str, torch.Tensor],
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    compiled: bool,
    device: torch.device,
) -> dict[str, Any]:
    mixer = GatedDiagonalStateSpaceMixer(
        hidden_size=config.hidden_size,
        dropout=config.dropout,
        use_bias=config.use_bias,
        state_transition_initialization=config.state_transition_initialization,
    )
    mixer.load_state_dict(base_state)
    mixer.to(device).train()
    forward_model: Any = mixer
    mode = None
    if compiled:
        mode = "default"
        forward_model = torch.compile(mixer.forward, mode=mode)
    optimizer = torch.optim.AdamW(
        mixer.parameters(),
        lr=config.learning_rate,
        betas=(config.optimizer_beta1, config.optimizer_beta2),
        eps=config.optimizer_epsilon,
        weight_decay=config.weight_decay,
    )

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    warmup_start = time.perf_counter()
    warmup_cuda_start = torch.cuda.Event(enable_timing=True)
    warmup_cuda_end = torch.cuda.Event(enable_timing=True)
    warmup_cuda_start.record()
    warmup_loss, warmup_gradient = _run_mixer_update(
        base_model=mixer,
        forward_model=forward_model,
        optimizer=optimizer,
        batches=batches[: config.gradient_accumulation_steps],
        config=config,
        label="isolated mixer training",
    )
    warmup_cuda_end.record()
    warmup_cuda_end.synchronize()
    warmup_wall_seconds = time.perf_counter() - warmup_start
    warmup_cuda_seconds = warmup_cuda_start.elapsed_time(warmup_cuda_end) / 1_000.0

    warmup_peak_allocated = int(torch.cuda.max_memory_allocated(device))
    warmup_peak_reserved = int(torch.cuda.max_memory_reserved(device))
    torch.cuda.reset_peak_memory_stats(device)
    measured_start = torch.cuda.Event(enable_timing=True)
    measured_end = torch.cuda.Event(enable_timing=True)
    measured_start.record()
    losses: list[float] = []
    gradients: list[float] = []
    for update_index in range(MEASURED_UPDATES):
        start = (update_index + WARMUP_UPDATES) * config.gradient_accumulation_steps
        end = start + config.gradient_accumulation_steps
        loss, gradient = _run_mixer_update(
            base_model=mixer,
            forward_model=forward_model,
            optimizer=optimizer,
            batches=batches[start:end],
            config=config,
            label="isolated mixer training",
        )
        losses.append(loss)
        gradients.append(gradient)
    measured_end.record()
    measured_end.synchronize()
    measured_cuda_seconds = measured_start.elapsed_time(measured_end) / 1_000.0
    result = {
        "path": "isolated_mixer_compiled_training" if compiled else "isolated_mixer_eager_training",
        "compiled": compiled,
        "compile_mode": mode,
        "warmup_update": {
            "loss": warmup_loss,
            "gradient_norm": warmup_gradient,
            "wall_seconds": _finite(
                warmup_wall_seconds,
                "isolated training warm-up wall time",
            ),
            "cuda_seconds": _finite(
                warmup_cuda_seconds,
                "isolated training warm-up CUDA time",
            ),
        },
        "measured_updates": MEASURED_UPDATES,
        "mean_measured_cuda_seconds": _finite(
            measured_cuda_seconds / MEASURED_UPDATES,
            "isolated training mean CUDA time",
        ),
        "tokens_per_second_not_applicable": True,
        "loss_trajectory": losses,
        "gradient_norm_trajectory": gradients,
        "peak_allocated_vram_bytes_all": warmup_peak_allocated,
        "peak_reserved_vram_bytes_all": warmup_peak_reserved,
        "peak_allocated_vram_bytes_measured": int(
            torch.cuda.max_memory_allocated(device)
        ),
        "peak_reserved_vram_bytes_measured": int(
            torch.cuda.max_memory_reserved(device)
        ),
        "final_parameters": parameters_to_vector(mixer.parameters()).detach().cpu().clone(),
        "state_key_digest": _state_key_digest(list(mixer.state_dict().keys())),
        "state_key_count": len(mixer.state_dict()),
        "optimizer_structure": _optimizer_structure(optimizer),
        "all_values_finite": True,
    }
    del forward_model, mixer, optimizer
    _reset_compiler()
    _release_cuda()
    return result


def _run_complete_path(
    config: Any,
    base_state: dict[str, torch.Tensor],
    batches: list[torch.Tensor],
    *,
    variant: str,
    device: torch.device,
) -> dict[str, Any]:
    model = A1RecurrentStateSpaceLM(config, use_parallel_scan=True)
    model.load_state_dict(base_state)
    model.to(device).train()
    forward_model: torch.nn.Module = model
    mode = None
    if variant == "mixers_compiled":
        mode = "default"
        _compile_mixer_forwards(model)
    elif variant == "full_compiled":
        mode = "default"
        forward_model = build_forward_model(model, compile_model=True)
    elif variant != "eager_full":
        raise ValueError(f"unknown complete-model benchmark variant: {variant}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.optimizer_beta1, config.optimizer_beta2),
        eps=config.optimizer_epsilon,
        weight_decay=config.weight_decay,
    )
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    warmup_start = time.perf_counter()
    warmup_cuda_start = torch.cuda.Event(enable_timing=True)
    warmup_cuda_end = torch.cuda.Event(enable_timing=True)
    warmup_cuda_start.record()
    warmup_loss, warmup_gradient = _run_complete_update(
        base_model=model,
        forward_model=forward_model,
        optimizer=optimizer,
        batches=batches[: config.gradient_accumulation_steps],
        config=config,
        label=f"A1 {variant}",
    )
    warmup_cuda_end.record()
    warmup_cuda_end.synchronize()
    warmup_wall_seconds = time.perf_counter() - warmup_start
    warmup_cuda_seconds = warmup_cuda_start.elapsed_time(warmup_cuda_end) / 1_000.0

    warmup_peak_allocated = int(torch.cuda.max_memory_allocated(device))
    warmup_peak_reserved = int(torch.cuda.max_memory_reserved(device))
    torch.cuda.reset_peak_memory_stats(device)
    measured_start = torch.cuda.Event(enable_timing=True)
    measured_end = torch.cuda.Event(enable_timing=True)
    measured_start.record()
    losses: list[float] = []
    gradients: list[float] = []
    for update_index in range(MEASURED_UPDATES):
        start = (update_index + WARMUP_UPDATES) * config.gradient_accumulation_steps
        end = start + config.gradient_accumulation_steps
        loss, gradient = _run_complete_update(
            base_model=model,
            forward_model=forward_model,
            optimizer=optimizer,
            batches=batches[start:end],
            config=config,
            label=f"A1 {variant}",
        )
        losses.append(loss)
        gradients.append(gradient)
    measured_end.record()
    measured_end.synchronize()
    measured_cuda_seconds = measured_start.elapsed_time(measured_end) / 1_000.0
    tokens_per_update = (
        config.micro_batch_size
        * config.gradient_accumulation_steps
        * config.context_length
    )
    state_keys = list(model.state_dict().keys())
    result = {
        "path": variant,
        "compiled": variant != "eager_full",
        "compile_mode": mode,
        "warmup_update": {
            "loss": warmup_loss,
            "gradient_norm": warmup_gradient,
            "wall_seconds": _finite(
                warmup_wall_seconds,
                f"{variant} warm-up wall time",
            ),
            "cuda_seconds": _finite(
                warmup_cuda_seconds,
                f"{variant} warm-up CUDA time",
            ),
        },
        "measured_updates": MEASURED_UPDATES,
        "mean_measured_cuda_seconds": _finite(
            measured_cuda_seconds / MEASURED_UPDATES,
            f"{variant} mean CUDA time",
        ),
        "tokens_per_second_measured": _finite(
            MEASURED_UPDATES * tokens_per_update / measured_cuda_seconds,
            f"{variant} measured throughput",
        ),
        "loss_trajectory": losses,
        "gradient_norm_trajectory": gradients,
        "peak_allocated_vram_bytes_all": warmup_peak_allocated,
        "peak_reserved_vram_bytes_all": warmup_peak_reserved,
        "peak_allocated_vram_bytes_measured": int(
            torch.cuda.max_memory_allocated(device)
        ),
        "peak_reserved_vram_bytes_measured": int(
            torch.cuda.max_memory_reserved(device)
        ),
        "final_parameters": parameters_to_vector(model.parameters()).detach().cpu().clone(),
        "state_key_digest": _state_key_digest(state_keys),
        "state_key_count": len(state_keys),
        "wrapped_state_key_present": any("_orig_mod." in key for key in state_keys),
        "optimizer_structure": _optimizer_structure(optimizer),
        "all_values_finite": True,
    }
    del forward_model, model, optimizer
    _reset_compiler()
    _release_cuda()
    return result


def _path_without_tensors(path: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in path.items()
        if key not in {"final_parameters", "output"}
    }


def _report(result: dict[str, Any]) -> str:
    forward_eager = result["paths"]["isolated_mixer_eager_forward"]
    forward_compiled = result["paths"]["isolated_mixer_compiled_forward"]
    complete_eager = result["paths"]["eager_full"]
    complete_mixers = result["paths"]["mixers_compiled"]
    complete_full = result["paths"]["full_compiled"]
    lines = [
        "# A1 end-to-end compile benchmark result",
        "",
        "Status: complete; engineering-only benchmark.",
        "",
        f"Device: {result['device']}; PyTorch {result['torch_version']}; CUDA build {result['torch_cuda_build']}.",
        "",
        "## Isolated component",
        "",
        "| path | mode | warm-up wall s | measured CUDA s/call |",
        "| --- | --- | ---: | ---: |",
        f"| eager forward | eager | {forward_eager['warmup_wall_seconds']:.3f} | {forward_eager['mean_measured_cuda_seconds']:.6f} |",
        f"| compiled forward | {forward_compiled['compile_mode']} | {forward_compiled['warmup_wall_seconds']:.3f} | {forward_compiled['mean_measured_cuda_seconds']:.6f} |",
        "",
        f"Isolated forward speedup: {result['speedups']['isolated_forward']:.3f}x.",
        f"Isolated accumulated-training speedup: {result['speedups']['isolated_training']:.3f}x.",
        "",
        "## Complete A1 update",
        "",
        "| path | mode | warm-up wall s | steady CUDA s/update | steady tok/s |",
        "| --- | --- | ---: | ---: | ---: |",
        f"| eager full model | eager | {complete_eager['warmup_update']['wall_seconds']:.3f} | {complete_eager['mean_measured_cuda_seconds']:.6f} | {complete_eager['tokens_per_second_measured']:.1f} |",
        f"| mixers compiled only | {complete_mixers['compile_mode']} | {complete_mixers['warmup_update']['wall_seconds']:.3f} | {complete_mixers['mean_measured_cuda_seconds']:.6f} | {complete_mixers['tokens_per_second_measured']:.1f} |",
        f"| full model compiled | {complete_full['compile_mode']} | {complete_full['warmup_update']['wall_seconds']:.3f} | {complete_full['mean_measured_cuda_seconds']:.6f} | {complete_full['tokens_per_second_measured']:.1f} |",
        "",
        f"Mixer-only end-to-end speedup: {result['speedups']['mixers_compiled_end_to_end']:.3f}x.",
        f"Full-model end-to-end speedup: {result['speedups']['full_compiled_end_to_end']:.3f}x.",
        f"Mixer-only speedup retention versus isolated forward: {result['speedups']['mixers_compiled_forward_retention']:.3f}.",
        f"Full-model speedup retention versus isolated forward: {result['speedups']['full_compiled_forward_retention']:.3f}.",
        "",
        "## Numerical and state checks",
        "",
        f"Mixer-only complete loss max relative difference: {result['comparisons']['mixers_compiled']['loss']['max_relative']:.6g}.",
        f"Mixer-only complete gradient max relative difference: {result['comparisons']['mixers_compiled']['gradient_norm']['max_relative']:.6g}.",
        f"Full-compiled loss max relative difference: {result['comparisons']['full_compiled']['loss']['max_relative']:.6g}.",
        f"Full-compiled gradient max relative difference: {result['comparisons']['full_compiled']['gradient_norm']['max_relative']:.6g}.",
        f"Full-compiled final parameter relative L2: {result['comparisons']['full_compiled']['final_parameters']['relative_l2']:.6f}.",
        f"Compiled complete-model paths stayed within declared short-horizon tolerances: {result['numerical_tolerance']['pass']}.",
        f"Complete-model state-key invariants preserved: {result['state_invariants']['model_state_keys_match']}.",
        f"Complete-model optimizer-state structures match: {result['state_invariants']['optimizer_structures_match']}.",
        "",
        "No checkpoint, model artifact, or quality-training result was written.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / CONFIG_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results/a1_end_to_end_compile_benchmark.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "docs/A1_END_TO_END_COMPILE_BENCHMARK_RESULT.md",
    )
    args = parser.parse_args(argv)
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    output_path = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    report_path = args.report if args.report.is_absolute() else REPO_ROOT / args.report
    config = load_config(config_path)
    if config.stage != "A1" or config.model_type != "recurrent_state_space":
        raise ValueError(
            "end-to-end compile benchmark requires the explicit A1 configuration"
        )
    if config.compile_model:
        raise ValueError("benchmark baseline config must keep compile_model=false")
    if (
        config.micro_batch_size != 2
        or config.context_length != 512
        or config.gradient_accumulation_steps != 8
        or config.precision != "fp16"
    ):
        raise ValueError("benchmark conditions do not match the approved A1 geometry")
    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA is unavailable")

    seed_report = set_reproducible_seed(config.seed, deterministic=True)
    device = torch.device("cuda")
    base_model = A1RecurrentStateSpaceLM(config, use_parallel_scan=True)
    parameter_count = base_model.parameter_count()
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(f"unexpected A1 parameter count: {parameter_count}")
    base_state = _copy_state(base_model)
    del base_model

    token_batches = _device_token_batches(_make_token_batches(config), device)
    mixer_batches = _device_mixer_batches(_make_mixer_batches(config), device)
    mixer_base = GatedDiagonalStateSpaceMixer(
        hidden_size=config.hidden_size,
        dropout=config.dropout,
        use_bias=config.use_bias,
        state_transition_initialization=config.state_transition_initialization,
    )
    mixer_parameter_count = sum(
        parameter.numel() for parameter in mixer_base.parameters()
    )
    mixer_state = _copy_state(mixer_base)
    del mixer_base

    isolated_forward_eager = _run_isolated_forward(
        config,
        mixer_state,
        mixer_batches[0][0],
        compiled=False,
        device=device,
    )
    isolated_forward_compiled = _run_isolated_forward(
        config,
        mixer_state,
        mixer_batches[0][0],
        compiled=True,
        device=device,
    )
    isolated_training_eager = _run_isolated_training(
        config,
        mixer_state,
        mixer_batches,
        compiled=False,
        device=device,
    )
    isolated_training_compiled = _run_isolated_training(
        config,
        mixer_state,
        mixer_batches,
        compiled=True,
        device=device,
    )

    complete_eager = _run_complete_path(
        config,
        base_state,
        token_batches,
        variant="eager_full",
        device=device,
    )
    complete_mixers = _run_complete_path(
        config,
        base_state,
        token_batches,
        variant="mixers_compiled",
        device=device,
    )
    complete_full = _run_complete_path(
        config,
        base_state,
        token_batches,
        variant="full_compiled",
        device=device,
    )

    forward_difference = _vector_metrics(
        isolated_forward_eager["output"],
        isolated_forward_compiled["output"],
    )
    isolated_training_comparisons = {
        "loss": _trajectory_metrics(
            isolated_training_eager["loss_trajectory"],
            isolated_training_compiled["loss_trajectory"],
            "isolated training loss",
        ),
        "gradient_norm": _trajectory_metrics(
            isolated_training_eager["gradient_norm_trajectory"],
            isolated_training_compiled["gradient_norm_trajectory"],
            "isolated training gradient norm",
        ),
        "final_parameters": _vector_metrics(
            isolated_training_eager["final_parameters"],
            isolated_training_compiled["final_parameters"],
        ),
    }
    complete_comparisons: dict[str, Any] = {}
    for label, candidate in (
        ("mixers_compiled", complete_mixers),
        ("full_compiled", complete_full),
    ):
        complete_comparisons[label] = {
            "loss": _trajectory_metrics(
                complete_eager["loss_trajectory"],
                candidate["loss_trajectory"],
                f"{label} loss",
            ),
            "gradient_norm": _trajectory_metrics(
                complete_eager["gradient_norm_trajectory"],
                candidate["gradient_norm_trajectory"],
                f"{label} gradient norm",
            ),
            "final_parameters": _vector_metrics(
                complete_eager["final_parameters"],
                candidate["final_parameters"],
            ),
        }

    state_digests = {
        label: path["state_key_digest"]
        for label, path in (
            ("eager_full", complete_eager),
            ("mixers_compiled", complete_mixers),
            ("full_compiled", complete_full),
        )
    }
    optimizer_structures = {
        label: path["optimizer_structure"]
        for label, path in (
            ("eager_full", complete_eager),
            ("mixers_compiled", complete_mixers),
            ("full_compiled", complete_full),
        )
    }
    model_state_keys_match = len(set(state_digests.values())) == 1
    optimizer_structures_match = len(
        {json.dumps(value, sort_keys=True) for value in optimizer_structures.values()}
    ) == 1
    max_compiled_loss_relative = max(
        complete_comparisons["mixers_compiled"]["loss"]["max_relative"],
        complete_comparisons["full_compiled"]["loss"]["max_relative"],
    )
    max_compiled_gradient_relative = max(
        complete_comparisons["mixers_compiled"]["gradient_norm"]["max_relative"],
        complete_comparisons["full_compiled"]["gradient_norm"]["max_relative"],
    )
    max_compiled_parameter_relative = max(
        complete_comparisons["mixers_compiled"]["final_parameters"]["relative_l2"],
        complete_comparisons["full_compiled"]["final_parameters"]["relative_l2"],
    )
    result = {
        "version": "a1_end_to_end_compile_benchmark_v1",
        "status": "complete",
        "protocol_path": PROTOCOL_PATH,
        "protocol_sha256": sha256_file(REPO_ROOT / PROTOCOL_PATH),
        "script_path": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "config_path": str(config_path.relative_to(REPO_ROOT)),
        "config_sha256": sha256_file(config_path),
        "seed": config.seed,
        "seed_report": seed_report,
        "device": torch.cuda.get_device_name(device),
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "parameter_count": parameter_count,
        "isolated_mixer_parameter_count": mixer_parameter_count,
        "conditions": {
            "micro_batch_size": config.micro_batch_size,
            "context_length": config.context_length,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "tokens_per_update": config.micro_batch_size
            * config.gradient_accumulation_steps
            * config.context_length,
            "precision": config.precision,
            "optimizer": config.optimizer,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "warmup_updates": WARMUP_UPDATES,
            "measured_updates": MEASURED_UPDATES,
            "token_batch_seed": config.seed + TOKEN_BATCH_SEED_OFFSET,
            "mixer_batch_seed": config.seed + MIXER_BATCH_SEED_OFFSET,
        },
        "paths": {
            "isolated_mixer_eager_forward": _path_without_tensors(
                isolated_forward_eager
            ),
            "isolated_mixer_compiled_forward": _path_without_tensors(
                isolated_forward_compiled
            ),
            "isolated_mixer_eager_training": _path_without_tensors(
                isolated_training_eager
            ),
            "isolated_mixer_compiled_training": _path_without_tensors(
                isolated_training_compiled
            ),
            "eager_full": _path_without_tensors(complete_eager),
            "mixers_compiled": _path_without_tensors(complete_mixers),
            "full_compiled": _path_without_tensors(complete_full),
        },
        "speedups": {
            "isolated_forward": _finite(
                isolated_forward_eager["mean_measured_cuda_seconds"]
                / isolated_forward_compiled["mean_measured_cuda_seconds"],
                "isolated forward speedup",
            ),
            "isolated_training": _finite(
                isolated_training_eager["mean_measured_cuda_seconds"]
                / isolated_training_compiled["mean_measured_cuda_seconds"],
                "isolated training speedup",
            ),
            "mixers_compiled_end_to_end": _finite(
                complete_eager["mean_measured_cuda_seconds"]
                / complete_mixers["mean_measured_cuda_seconds"],
                "mixer-only end-to-end speedup",
            ),
            "full_compiled_end_to_end": _finite(
                complete_eager["mean_measured_cuda_seconds"]
                / complete_full["mean_measured_cuda_seconds"],
                "full compile end-to-end speedup",
            ),
            "mixers_compiled_forward_retention": _finite(
                (
                    complete_eager["mean_measured_cuda_seconds"]
                    / complete_mixers["mean_measured_cuda_seconds"]
                )
                / (
                    isolated_forward_eager["mean_measured_cuda_seconds"]
                    / isolated_forward_compiled["mean_measured_cuda_seconds"]
                ),
                "mixer forward speedup retention",
            ),
            "full_compiled_forward_retention": _finite(
                (
                    complete_eager["mean_measured_cuda_seconds"]
                    / complete_full["mean_measured_cuda_seconds"]
                )
                / (
                    isolated_forward_eager["mean_measured_cuda_seconds"]
                    / isolated_forward_compiled["mean_measured_cuda_seconds"]
                ),
                "full forward speedup retention",
            ),
        },
        "comparisons": {
            "isolated_forward": forward_difference,
            "isolated_training": isolated_training_comparisons,
            **complete_comparisons,
        },
        "state_invariants": {
            "model_state_keys_match": model_state_keys_match,
            "model_state_key_digests": state_digests,
            "optimizer_structures_match": optimizer_structures_match,
            "optimizer_structures": optimizer_structures,
            "compiled_paths_wrapped_state_key_present": {
                "mixers_compiled": complete_mixers["wrapped_state_key_present"],
                "full_compiled": complete_full["wrapped_state_key_present"],
            },
        },
        "numerical_tolerance": {
            "loss_relative_tolerance": LOSS_RELATIVE_TOLERANCE,
            "gradient_relative_tolerance": GRADIENT_RELATIVE_TOLERANCE,
            "parameter_relative_l2_tolerance": PARAMETER_RELATIVE_L2_TOLERANCE,
            "max_compiled_loss_relative": max_compiled_loss_relative,
            "max_compiled_gradient_relative": max_compiled_gradient_relative,
            "max_compiled_parameter_relative_l2": max_compiled_parameter_relative,
            "pass": (
                max_compiled_loss_relative <= LOSS_RELATIVE_TOLERANCE
                and max_compiled_gradient_relative <= GRADIENT_RELATIVE_TOLERANCE
                and max_compiled_parameter_relative <= PARAMETER_RELATIVE_L2_TOLERANCE
            ),
        },
        "training_pilot_started": False,
        "checkpoint_written": False,
        "weights_written": False,
        "all_values_finite": True,
        "interpretation": "engineering-only isolated and end-to-end compile comparison",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_report(result), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
