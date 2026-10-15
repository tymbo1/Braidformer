#!/usr/bin/env python3
"""Audit a matched fifty-update eager versus compiled A1 trajectory."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
import traceback
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
from braidformer.training.runner import build_forward_model


PROTOCOL_PATH = "docs/A1_COMPILED_TRAJECTORY_AUDIT_PROTOCOL.md"
CONFIG_PATH = "configs/braidnano_a1_recurrent_state_space_proposal.json"
EXPECTED_PARAMETER_COUNT = 41_870_336
EXPECTED_SEED = 17
BATCH_SEED = 5_017
TOTAL_UPDATES = 50
WARMUP_UPDATES = 1
PROBE_COUNT = 1_024
LOSS_RELATIVE_TOLERANCE = 1e-4
GRADIENT_RELATIVE_TOLERANCE = 1e-2
PARAMETER_RELATIVE_L2_TOLERANCE = 0.02


def _finite(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"non-finite {label}: {value!r}")
    return result


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
    step_values: list[int] = []
    for parameter_state in state["state"].values():
        step = parameter_state.get("step")
        if step is not None:
            step_values.append(int(step.item() if torch.is_tensor(step) else step))
    return {
        "top_level_keys": sorted(str(key) for key in state),
        "param_group_count": len(state["param_groups"]),
        "parameters_per_group": [
            len(group["params"]) for group in state["param_groups"]
        ],
        "state_entry_count": len(state["state"]),
        "state_value_keys": value_keys,
        "step_values": sorted(set(step_values)),
    }


def _copy_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _release_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _reset_compiler() -> None:
    compiler = getattr(torch, "compiler", None)
    reset = getattr(compiler, "reset", None) if compiler is not None else None
    if reset is not None:
        reset()


def _make_batches(config: Any, device: torch.device) -> tuple[list[torch.Tensor], str]:
    generator = torch.Generator(device="cpu").manual_seed(BATCH_SEED)
    cpu_batches = [
        torch.randint(
            0,
            config.vocab_size,
            (config.micro_batch_size, config.context_length),
            dtype=torch.long,
            generator=generator,
        )
        for _ in range(TOTAL_UPDATES * config.gradient_accumulation_steps)
    ]
    digest = hashlib.sha256(
        b"".join(batch.numpy().tobytes() for batch in cpu_batches)
    ).hexdigest()
    return [batch.to(device) for batch in cpu_batches], digest


def _probe_indices(parameter_count: int) -> torch.Tensor:
    indices = torch.linspace(
        0,
        parameter_count - 1,
        steps=PROBE_COUNT,
        dtype=torch.float64,
    ).round().to(dtype=torch.long)
    if int(indices.unique().numel()) != PROBE_COUNT:
        raise RuntimeError("parameter probe indices are not unique")
    return indices


def _parameter_snapshot(
    model: torch.nn.Module,
    probe_indices: torch.Tensor,
) -> tuple[float, list[float], torch.Tensor]:
    vector = parameters_to_vector(model.parameters()).detach()
    vector_float = vector.float()
    probe = vector_float.index_select(0, probe_indices).cpu()
    return (
        _finite(vector_float.norm().item(), "parameter L2 norm"),
        [_finite(value, "parameter probe value") for value in probe.tolist()],
        vector,
    )


def _assert_finite_model(model: torch.nn.Module) -> None:
    for name, parameter in model.named_parameters():
        if not bool(torch.isfinite(parameter).all().item()):
            raise RuntimeError(f"non-finite parameter after optimizer step: {name}")


def _update(
    model: torch.nn.Module,
    forward_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batches: list[torch.Tensor],
    config: Any,
    update_index: int,
) -> tuple[float, float]:
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    start = (update_index - 1) * config.gradient_accumulation_steps
    end = update_index * config.gradient_accumulation_steps
    for batch in batches[start:end]:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = forward_model(batch, labels=batch)
            loss = output.loss
        if loss is None or not bool(torch.isfinite(loss).item()):
            raise RuntimeError(f"non-finite loss at update {update_index}")
        losses.append(_finite(loss.item(), "microbatch loss"))
        (loss / config.gradient_accumulation_steps).backward()

    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=config.gradient_clip_norm,
        error_if_nonfinite=True,
    )
    gradient_value = _finite(gradient_norm.item(), "gradient norm")
    optimizer.step()
    _assert_finite_model(model)
    return (
        _finite(sum(losses) / len(losses), "update loss"),
        gradient_value,
    )


def _run_path(
    config: Any,
    base_state: dict[str, torch.Tensor],
    batches: list[torch.Tensor],
    probe_indices: torch.Tensor,
    *,
    compiled: bool,
    device: torch.device,
) -> dict[str, Any]:
    model: torch.nn.Module | None = None
    forward_model: torch.nn.Module | None = None
    optimizer: torch.optim.Optimizer | None = None
    try:
        model = A1RecurrentStateSpaceLM(config, use_parallel_scan=True)
        model.load_state_dict(base_state)
        state_keys = list(model.state_dict().keys())
        model.to(device).train()
        forward_model = build_forward_model(model, compile_model=compiled)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(config.optimizer_beta1, config.optimizer_beta2),
            eps=config.optimizer_epsilon,
            weight_decay=config.weight_decay,
        )
        if optimizer.state:
            raise RuntimeError("fresh optimizer unexpectedly has state")

        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        losses: list[float] = []
        gradients: list[float] = []
        parameter_norms: list[float] = []
        parameter_probes: list[list[float]] = []
        cuda_seconds: list[float] = []
        wall_seconds: list[float] = []
        optimizer_steps: list[dict[str, Any]] = []
        final_parameters: torch.Tensor | None = None

        for update_index in range(1, TOTAL_UPDATES + 1):
            cuda_start = torch.cuda.Event(enable_timing=True)
            cuda_end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(device)
            wall_start = time.perf_counter()
            cuda_start.record()
            loss, gradient = _update(
                model,
                forward_model,
                optimizer,
                batches,
                config,
                update_index,
            )
            cuda_end.record()
            cuda_end.synchronize()
            wall_elapsed = time.perf_counter() - wall_start
            cuda_elapsed = cuda_start.elapsed_time(cuda_end) / 1_000.0
            parameter_norm, parameter_probe, final_parameters = _parameter_snapshot(
                model,
                probe_indices,
            )
            losses.append(loss)
            gradients.append(gradient)
            parameter_norms.append(parameter_norm)
            parameter_probes.append(parameter_probe)
            cuda_seconds.append(_finite(cuda_elapsed, "update CUDA time"))
            wall_seconds.append(_finite(wall_elapsed, "update wall time"))
            optimizer_steps.append(_optimizer_structure(optimizer))

        if final_parameters is None:
            raise RuntimeError("trajectory produced no final parameters")
        tokens_per_update = (
            config.micro_batch_size
            * config.gradient_accumulation_steps
            * config.context_length
        )
        steady_cuda = sum(cuda_seconds[WARMUP_UPDATES:])
        return {
            "path": "compiled_default" if compiled else "eager_reference",
            "compiled": compiled,
            "compile_mode": "default" if compiled else None,
            "updates": TOTAL_UPDATES,
            "warmup_update": {
                "loss": losses[0],
                "gradient_norm": gradients[0],
                "cuda_seconds": cuda_seconds[0],
                "wall_seconds": wall_seconds[0],
            },
            "loss_trajectory": losses,
            "gradient_norm_trajectory": gradients,
            "parameter_norm_trajectory": parameter_norms,
            "parameter_probe_trajectory": parameter_probes,
            "cuda_seconds_trajectory": cuda_seconds,
            "wall_seconds_trajectory": wall_seconds,
            "steady_update_count": TOTAL_UPDATES - WARMUP_UPDATES,
            "mean_steady_cuda_seconds": _finite(
                steady_cuda / (TOTAL_UPDATES - WARMUP_UPDATES),
                "mean steady CUDA time",
            ),
            "tokens_per_second_steady": _finite(
                (TOTAL_UPDATES - WARMUP_UPDATES) * tokens_per_update / steady_cuda,
                "steady throughput",
            ),
            "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved(device)),
            "state_key_count": len(state_keys),
            "state_key_digest": _state_key_digest(state_keys),
            "wrapped_state_key_present": any(
                "_orig_mod." in key for key in state_keys
            ),
            "optimizer_structure": _optimizer_structure(optimizer),
            "optimizer_step_trajectory": optimizer_steps,
            "all_values_finite": True,
            "final_parameters": final_parameters.detach().cpu().clone(),
        }
    finally:
        del forward_model, model, optimizer
        if compiled:
            _reset_compiler()
        _release_cuda()


def _relative_difference(reference: float, candidate: float) -> float:
    return _finite(
        abs(candidate - reference) / max(abs(reference), 1e-12),
        "relative trajectory difference",
    )


def _probe_difference(
    reference: list[float],
    candidate: list[float],
) -> dict[str, float]:
    reference_tensor = torch.tensor(reference, dtype=torch.float32)
    candidate_tensor = torch.tensor(candidate, dtype=torch.float32)
    difference = (candidate_tensor - reference_tensor).abs()
    return {
        "max_absolute": _finite(difference.max().item(), "probe max difference"),
        "mean_absolute": _finite(difference.mean().item(), "probe mean difference"),
        "relative_l2": _finite(
            (candidate_tensor - reference_tensor).norm().item()
            / reference_tensor.norm().clamp_min(1e-12).item(),
            "probe relative L2 difference",
        ),
    }


def _vector_difference(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> dict[str, float]:
    reference_float = reference.float()
    difference = (candidate.float() - reference_float).abs()
    return {
        "max_absolute": _finite(difference.max().item(), "parameter max difference"),
        "mean_absolute": _finite(difference.mean().item(), "parameter mean difference"),
        "reference_l2": _finite(reference_float.norm().item(), "reference parameter norm"),
        "candidate_l2": _finite(candidate.float().norm().item(), "candidate parameter norm"),
        "relative_l2": _finite(
            difference.norm().div(reference_float.norm().clamp_min(1e-12)).item(),
            "parameter relative L2 difference",
        ),
    }


def _compare_paths(
    eager: dict[str, Any],
    compiled: dict[str, Any],
) -> dict[str, Any]:
    per_update: list[dict[str, Any]] = []
    for step in range(TOTAL_UPDATES):
        eager_loss = eager["loss_trajectory"][step]
        compiled_loss = compiled["loss_trajectory"][step]
        eager_gradient = eager["gradient_norm_trajectory"][step]
        compiled_gradient = compiled["gradient_norm_trajectory"][step]
        probe = _probe_difference(
            eager["parameter_probe_trajectory"][step],
            compiled["parameter_probe_trajectory"][step],
        )
        per_update.append(
            {
                "update": step + 1,
                "loss": {
                    "eager": eager_loss,
                    "compiled": compiled_loss,
                    "compiled_minus_eager": _finite(
                        compiled_loss - eager_loss,
                        "loss delta",
                    ),
                    "relative_difference": _relative_difference(
                        eager_loss,
                        compiled_loss,
                    ),
                },
                "gradient_norm": {
                    "eager": eager_gradient,
                    "compiled": compiled_gradient,
                    "compiled_minus_eager": _finite(
                        compiled_gradient - eager_gradient,
                        "gradient delta",
                    ),
                    "relative_difference": _relative_difference(
                        eager_gradient,
                        compiled_gradient,
                    ),
                },
                "parameter_norm": {
                    "eager": eager["parameter_norm_trajectory"][step],
                    "compiled": compiled["parameter_norm_trajectory"][step],
                    "relative_difference": _relative_difference(
                        eager["parameter_norm_trajectory"][step],
                        compiled["parameter_norm_trajectory"][step],
                    ),
                },
                "parameter_probe": probe,
            }
        )
    return {
        "per_update": per_update,
        "max_loss_relative": max(
            entry["loss"]["relative_difference"] for entry in per_update
        ),
        "max_gradient_relative": max(
            entry["gradient_norm"]["relative_difference"] for entry in per_update
        ),
        "max_parameter_probe_relative_l2": max(
            entry["parameter_probe"]["relative_l2"] for entry in per_update
        ),
        "final_parameters": _vector_difference(
            eager["final_parameters"],
            compiled["final_parameters"],
        ),
    }


def _report(result: dict[str, Any]) -> str:
    eager = result["paths"]["eager_reference"]
    compiled = result["paths"]["compiled_default"]
    comparison = result["trajectory_comparison"]
    gates = result["audit_gates"]
    lines = [
        "# A1 compiled trajectory audit result",
        "",
        "Status: complete; engineering-only matched in-memory trajectory audit.",
        "",
        f"Device: {result['device']}; PyTorch {result['torch_version']}; CUDA build {result['torch_cuda_build']}.",
        "",
        "| path | update-1 CUDA s | steady CUDA s/update | steady tok/s | peak reserved VRAM |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| eager reference | {eager['warmup_update']['cuda_seconds']:.6f} | {eager['mean_steady_cuda_seconds']:.6f} | {eager['tokens_per_second_steady']:.1f} | {eager['peak_reserved_vram_bytes']} |",
        f"| compiled default | {compiled['warmup_update']['cuda_seconds']:.6f} | {compiled['mean_steady_cuda_seconds']:.6f} | {compiled['tokens_per_second_steady']:.1f} | {compiled['peak_reserved_vram_bytes']} |",
        "",
        f"Steady-state compiled/eager speedup: {result['speedup_compiled_over_eager']:.3f}x.",
        f"Maximum per-update relative loss difference: {comparison['max_loss_relative']:.6g}.",
        f"Maximum per-update relative gradient difference: {comparison['max_gradient_relative']:.6g}.",
        f"Maximum parameter-probe relative L2 difference: {comparison['max_parameter_probe_relative_l2']:.6g}.",
        f"Final parameter relative L2 difference: {comparison['final_parameters']['relative_l2']:.6g}.",
        f"Predeclared audit gates passed: {gates['pass']}.",
        f"State-key and optimizer invariants passed: {result['state_invariants']['pass']}.",
        "",
        "The raw per-update trajectories are in the machine-readable aggregate.",
        "No checkpoint, model artifact, validation result, or quality-training result was written.",
        "",
    ]
    return "\n".join(lines)


def _failure_report(result: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# A1 compiled trajectory audit result",
            "",
            "Status: failed; engineering-only audit; no quality-training result.",
            "",
            f"Failure type: `{result['failure']['type']}`.",
            f"Failure message: `{result['failure']['message']}`.",
            "",
            "The failure was recorded in the machine-readable aggregate. No checkpoint or model artifact was written.",
            "",
        ]
    )


def _paths_for_output(args: argparse.Namespace) -> tuple[Path, Path]:
    output_path = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    report_path = args.report if args.report.is_absolute() else REPO_ROOT / args.report
    return output_path, report_path


def _execute(config_path: Path, output_path: Path, report_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    if config.stage != "A1" or config.model_type != "recurrent_state_space":
        raise ValueError("trajectory audit requires the explicit A1 configuration")
    if config.recurrent_path != "parallel_scan":
        raise ValueError("trajectory audit requires A1 parallel_scan")
    if config.seed != EXPECTED_SEED:
        raise ValueError(f"trajectory audit requires seed {EXPECTED_SEED}")
    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA is unavailable")

    seed_report = set_reproducible_seed(config.seed, deterministic=True)
    device = torch.device("cuda")
    base_model = A1RecurrentStateSpaceLM(config, use_parallel_scan=True)
    parameter_count = base_model.parameter_count()
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(f"unexpected A1 parameter count: {parameter_count}")
    base_state = _copy_state(base_model)
    state_keys = list(base_model.state_dict().keys())
    state_key_digest = _state_key_digest(state_keys)
    del base_model
    batches, batch_digest = _make_batches(config, device)
    probe_indices = _probe_indices(parameter_count)

    eager = _run_path(
        config,
        base_state,
        batches,
        probe_indices.to(device),
        compiled=False,
        device=device,
    )
    compiled = _run_path(
        config,
        base_state,
        batches,
        probe_indices.to(device),
        compiled=True,
        device=device,
    )
    comparison = _compare_paths(eager, compiled)
    state_invariants = {
        "model_state_keys_match": (
            eager["state_key_digest"] == compiled["state_key_digest"] == state_key_digest
            and eager["state_key_count"] == compiled["state_key_count"] == len(state_keys)
        ),
        "eager_wrapped_state_key_present": eager["wrapped_state_key_present"],
        "compiled_wrapped_state_key_present": compiled["wrapped_state_key_present"],
        "optimizer_structures_match": (
            {
                key: value
                for key, value in eager["optimizer_structure"].items()
                if key != "step_values"
            }
            == {
                key: value
                for key, value in compiled["optimizer_structure"].items()
                if key != "step_values"
            }
        ),
        "eager_step_sequence": [
            item["step_values"] for item in eager["optimizer_step_trajectory"]
        ],
        "compiled_step_sequence": [
            item["step_values"] for item in compiled["optimizer_step_trajectory"]
        ],
    }
    state_invariants["optimizer_step_sequences_match"] = (
        state_invariants["eager_step_sequence"]
        == state_invariants["compiled_step_sequence"]
        == [[update] for update in range(1, TOTAL_UPDATES + 1)]
    )
    state_invariants["pass"] = (
        state_invariants["model_state_keys_match"]
        and not state_invariants["eager_wrapped_state_key_present"]
        and not state_invariants["compiled_wrapped_state_key_present"]
        and state_invariants["optimizer_structures_match"]
        and state_invariants["optimizer_step_sequences_match"]
    )
    audit_gates = {
        "all_values_finite": eager["all_values_finite"] and compiled["all_values_finite"],
        "exact_update_count": (
            eager["updates"] == compiled["updates"] == TOTAL_UPDATES
        ),
        "batch_digest_match": True,
        "loss_relative_tolerance": LOSS_RELATIVE_TOLERANCE,
        "gradient_relative_tolerance": GRADIENT_RELATIVE_TOLERANCE,
        "parameter_relative_l2_tolerance": PARAMETER_RELATIVE_L2_TOLERANCE,
        "max_loss_relative": comparison["max_loss_relative"],
        "max_gradient_relative": comparison["max_gradient_relative"],
        "final_parameter_relative_l2": comparison["final_parameters"]["relative_l2"],
    }
    audit_gates["pass"] = (
        audit_gates["all_values_finite"]
        and audit_gates["exact_update_count"]
        and audit_gates["batch_digest_match"]
        and audit_gates["max_loss_relative"] <= LOSS_RELATIVE_TOLERANCE
        and audit_gates["max_gradient_relative"] <= GRADIENT_RELATIVE_TOLERANCE
        and audit_gates["final_parameter_relative_l2"]
        <= PARAMETER_RELATIVE_L2_TOLERANCE
        and state_invariants["pass"]
    )
    tokens_per_update = (
        config.micro_batch_size
        * config.gradient_accumulation_steps
        * config.context_length
    )
    result = {
        "version": "a1_compiled_trajectory_audit_v1",
        "status": "complete",
        "interpretation": "engineering-only matched trajectory audit",
        "protocol_path": PROTOCOL_PATH,
        "protocol_sha256": sha256_file(REPO_ROOT / PROTOCOL_PATH),
        "script_path": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "config_path": str(config_path.relative_to(REPO_ROOT)),
        "config_sha256": sha256_file(config_path),
        "seed": config.seed,
        "batch_seed": BATCH_SEED,
        "batch_sequence_sha256": batch_digest,
        "seed_report": seed_report,
        "device": torch.cuda.get_device_name(device),
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "parameter_count": parameter_count,
        "total_updates": TOTAL_UPDATES,
        "warmup_updates": WARMUP_UPDATES,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "tokens_per_update": tokens_per_update,
        "paths": {
            "eager_reference": {
                key: value for key, value in eager.items() if key != "final_parameters"
            },
            "compiled_default": {
                key: value for key, value in compiled.items() if key != "final_parameters"
            },
        },
        "speedup_compiled_over_eager": _finite(
            eager["mean_steady_cuda_seconds"] / compiled["mean_steady_cuda_seconds"],
            "compiled/eager steady-state speedup",
        ),
        "trajectory_comparison": comparison,
        "state_invariants": state_invariants,
        "audit_gates": audit_gates,
        "training_pilot_started": False,
        "checkpoint_written": False,
        "weights_written": False,
        "quality_training_result_written": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_report(result), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-approved", action="store_true")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / CONFIG_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results/a1_compiled_trajectory_audit.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "docs/A1_COMPILED_TRAJECTORY_AUDIT_RESULT.md",
    )
    args = parser.parse_args()
    output_path, report_path = _paths_for_output(args)
    if not args.execute_approved:
        raise SystemExit(
            "trajectory audit is gated; pass --execute-approved after protocol authorization"
        )
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    try:
        result = _execute(config_path, output_path, report_path)
    except Exception as exc:
        failure = {
            "version": "a1_compiled_trajectory_audit_v1",
            "status": "failed",
            "interpretation": "engineering-only matched trajectory audit failure",
            "protocol_path": PROTOCOL_PATH,
            "protocol_sha256": sha256_file(REPO_ROOT / PROTOCOL_PATH),
            "script_path": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "failure": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            "training_pilot_started": False,
            "checkpoint_written": False,
            "weights_written": False,
            "quality_training_result_written": False,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(_failure_report(failure), encoding="utf-8")
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
