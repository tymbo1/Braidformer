#!/usr/bin/env python3
"""Read-only A1 distractor-interval state-update intervention.

The analysis compares native A1 recurrence with two functional inference-time
state-update controls during the neutral distractor interval:

* write-protected: retain learned decay but suppress new distractor writes;
* fully frozen: suppress both decay and new writes.

No checkpoint or parameter is modified.  Source runs, protocol hashes,
parameter digests, finite tensors, and a distance-zero no-op control are
verified before a result is accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from braidformer.config import load_config
from braidformer.models import A1RecurrentStateSpaceLM
from braidformer.training.integrity import sha256_file, verify_checkpoint
from scripts.analyze_a1_state_decay_forensics import (
    PROBE_BATCH_SIZE,
    PROBE_EVAL_EXAMPLES,
    PROBE_TRAIN_EXAMPLES,
    _fit_ridge_probe,
    _probe_batch,
    _probe_metrics,
)
from scripts.analyze_synthetic_solvability_distances_values8000 import (
    ARCHITECTURES,
    DISTANCES,
    EXPECTED,
    PROTOCOL_COMMIT,
    _baseline,
    _load_json,
    _load_run,
)
from scripts.run_synthetic_overfit_calibration import (
    FIXED_EXAMPLES,
    VALUE_BASE,
    make_evaluation_examples,
)


ANALYSIS_PLAN_PATH = "docs/A0_A1_INTERFERENCE_ISOLATION_DESIGN.md"
DISTANCE_RESULT_PATH = "results/a0_a1_solvability_distances_values8000.json"
DISTANCE_RESULT_SHA256 = "3a40736a783fd1ac977fe2b6651a6e015891d85699f6fee2e3698cc0917a8261"
RUNNER_PATH = "scripts/run_synthetic_overfit_calibration.py"
RUNNER_SHA256 = "2a1a1c84f1ebfabc11cf66f0defa2e2ad4db2d1741c89c3d12660d3ac066d3e5"
DISTANCE_VERIFIER_PATH = "scripts/analyze_synthetic_solvability_distances_values8000.py"
DISTANCE_VERIFIER_SHA256 = "79b3abd5f40fab4646c3308b1b8daf6ec54b44653b08ce81ef471e435d386ff3"
EXPECTED_PARAMS = 41_870_336
SEED = 17
VALUE_COUNT = 8_000
BATCH_SIZE = 16
VARIANTS = ("native", "write_protected", "fully_frozen")
SITES = ("post_value", "post_distractor", "query", "target_logit")
OUTPUT_VERSION = "a1_interference_isolation_v1"


def _finite(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite {label}: {value!r}")
    return number


def _finite_tensor(value: torch.Tensor, label: str) -> torch.Tensor:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"non-finite tensor in {label}")
    return value


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _parameter_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT.resolve()))


def _mean_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left_norm = left.norm(dim=-1).clamp_min(1e-8)
    right_norm = right.norm(dim=-1).clamp_min(1e-8)
    cosine = (left * right).sum(dim=-1) / (left_norm * right_norm)
    return _finite(cosine.mean(), "mean cosine")


def _logit_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float | int]:
    logits = _finite_tensor(logits.float(), "logit metrics").cpu()
    targets = targets.long().cpu()
    if logits.ndim != 2 or targets.ndim != 1 or logits.shape[0] != targets.shape[0]:
        raise ValueError("invalid metric tensor shapes")
    correct = logits.gather(-1, targets[:, None]).squeeze(-1)
    log_probability = correct - torch.logsumexp(logits, dim=-1)
    probability = torch.exp(log_probability)
    incorrect = logits.clone()
    incorrect.scatter_(-1, targets[:, None], float("-inf"))
    margin = correct - incorrect.amax(dim=-1)
    ranks = 1 + (logits > correct[:, None]).sum(dim=-1)
    return {
        "examples": int(targets.shape[0]),
        "cross_entropy": _finite((-log_probability).mean(), "cross entropy"),
        "correct_token_probability": _finite(probability.mean(), "correct probability"),
        "top1_accuracy": _finite(logits.argmax(dim=-1).eq(targets).float().mean(), "top1"),
        "correct_token_logit_margin": _finite(margin.mean(), "logit margin"),
        "mean_target_rank": _finite(ranks.float().mean(), "mean target rank"),
        "median_target_rank": _finite(torch.median(ranks.float()), "median target rank"),
    }


def _target_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, Any]:
    value_logits = logits[:, VALUE_BASE : VALUE_BASE + VALUE_COUNT]
    value_targets = targets - VALUE_BASE
    return {
        "full_vocabulary": _logit_metrics(logits, targets),
        "value_candidate_range": _logit_metrics(value_logits, value_targets),
        "greedy_answer_token_accuracy": _finite(
            logits.argmax(dim=-1).eq(targets).float().mean(),
            "greedy answer-token accuracy",
        ),
        "greedy_exact_match": _finite(
            logits.argmax(dim=-1).eq(targets).float().mean(),
            "greedy exact match",
        ),
    }


def _metric_delta(patched: dict[str, Any], native: dict[str, Any]) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    for scope in ("full_vocabulary", "value_candidate_range"):
        delta[scope] = {}
        for field in (
            "cross_entropy",
            "correct_token_probability",
            "top1_accuracy",
            "correct_token_logit_margin",
            "mean_target_rank",
            "median_target_rank",
        ):
            delta[scope][field] = _finite(
                patched[scope][field] - native[scope][field],
                f"delta {scope}.{field}",
            )
    for field in ("greedy_answer_token_accuracy", "greedy_exact_match"):
        delta[field] = _finite(patched[field] - native[field], f"delta {field}")
    return delta


def _affine_scan(
    coefficients: torch.Tensor,
    updates: torch.Tensor,
) -> torch.Tensor:
    """Reference scan for a per-timestep affine recurrence."""

    if coefficients.ndim != 3 or updates.ndim != 3:
        raise ValueError("affine scan inputs must be rank 3")
    if coefficients.shape != updates.shape:
        raise ValueError("affine scan input shapes must match")
    state = updates.new_zeros((updates.shape[0], updates.shape[2]))
    states: list[torch.Tensor] = []
    for timestep in range(updates.shape[1]):
        state = coefficients[:, timestep, :] * state + updates[:, timestep, :]
        states.append(state)
    return torch.stack(states, dim=1)


def _functional_forward(
    model: A1RecurrentStateSpaceLM,
    input_ids: torch.Tensor,
    distance: int,
    variant: str,
    *,
    collect: bool,
) -> tuple[torch.Tensor, dict[str, list[dict[str, torch.Tensor]]]]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown intervention variant: {variant}")
    if distance not in (0, 4, 16, 64):
        raise ValueError(f"unsupported distance: {distance}")

    sequence_length = input_ids.shape[1]
    expected_length = 8 + distance
    if sequence_length != expected_length:
        raise ValueError(
            f"unexpected sequence length {sequence_length}; expected {expected_length}"
        )
    distractor_positions = set(range(4, 4 + distance))
    site_positions = {
        "post_value": 3,
        "post_distractor": 3 + distance,
        "query": 4 + distance,
        "target_logit": 6 + distance,
    }
    hidden_states = model.token_embedding(input_ids)
    positions = torch.arange(sequence_length, device=input_ids.device)
    hidden_states = hidden_states + model.position_embedding(positions).unsqueeze(0)
    hidden_states = model.embedding_dropout(hidden_states)
    layer_traces: list[dict[str, torch.Tensor]] = []

    for block in model.blocks:
        normalized = block.state_space_norm(hidden_states)
        module = block.state_space
        projected = module.input_projection(normalized)
        candidate, input_gate, output_gate = projected.chunk(3, dim=-1)
        decay = torch.sigmoid(module.state_transition_logit).to(dtype=normalized.dtype)
        updates = (1.0 - decay).view(1, 1, module.hidden_size)
        updates = updates * torch.sigmoid(input_gate) * torch.tanh(candidate)

        if variant == "native" or not distractor_positions:
            states = module._parallel_scan(decay, updates)
            effective_coefficients = decay.view(1, 1, module.hidden_size).expand_as(updates)
            effective_updates = updates
        else:
            effective_coefficients = decay.view(1, 1, module.hidden_size).expand(
                normalized.shape[0], sequence_length, module.hidden_size
            ).clone()
            effective_updates = updates.clone()
            for timestep in distractor_positions:
                effective_updates[:, timestep, :] = 0.0
                if variant == "fully_frozen":
                    effective_coefficients[:, timestep, :] = 1.0
            states = _affine_scan(effective_coefficients, effective_updates)

        mixed = torch.sigmoid(output_gate) * states
        hidden_states = hidden_states + module.residual_dropout(module.output_projection(mixed))
        hidden_states = hidden_states + block.feed_forward(
            block.feed_forward_norm(hidden_states)
        )

        if collect:
            layer_traces.append(
                {
                    "states": {
                        site: states[:, position, :].detach().float().cpu()
                        for site, position in site_positions.items()
                    },
                    "effective_updates": {
                        site: effective_updates[:, position, :].detach().float().cpu()
                        for site, position in site_positions.items()
                    },
                }
            )

    normalized_final = model.final_norm(hidden_states)
    logits = model.lm_head(normalized_final)
    target_logits = logits[:, 6 + distance, :]
    _finite_tensor(target_logits, f"{variant} target logits")
    return target_logits, {"layers": layer_traces}


def _append_trace(
    destination: dict[str, list[torch.Tensor]],
    trace: dict[str, list[dict[str, torch.Tensor]]],
) -> None:
    for layer_index, layer_trace in enumerate(trace["layers"]):
        for kind in ("states", "effective_updates"):
            for site in SITES:
                key = f"layer_{layer_index}.{kind}.{site}"
                destination.setdefault(key, []).append(layer_trace[kind][site])


def _summarize_traces(
    trace_parts: dict[str, dict[str, list[torch.Tensor]]],
) -> dict[str, Any]:
    tensors: dict[str, dict[str, torch.Tensor]] = {}
    for variant, parts in trace_parts.items():
        tensors[variant] = {
            key: torch.cat(values, dim=0)
            for key, values in parts.items()
        }
        for key, value in tensors[variant].items():
            _finite_tensor(value, f"trace {variant} {key}")

    result: dict[str, Any] = {}
    for variant in VARIANTS:
        variant_result: dict[str, Any] = {}
        for layer_index in range(8):
            layer_result: dict[str, Any] = {}
            post_value = tensors[variant][f"layer_{layer_index}.states.post_value"]
            native_layer = tensors["native"]
            for site in SITES:
                state = tensors[variant][f"layer_{layer_index}.states.{site}"]
                update = tensors[variant][f"layer_{layer_index}.effective_updates.{site}"]
                state_norm = state.norm(dim=-1).clamp_min(1e-8)
                update_norm = update.norm(dim=-1)
                row = {
                    "examples": int(state.shape[0]),
                    "mean_state_norm": _finite(state_norm.mean(), "state norm"),
                    "mean_update_norm": _finite(update_norm.mean(), "update norm"),
                    "mean_cosine_to_post_value": _mean_cosine(state, post_value),
                }
                native_state = native_layer[f"layer_{layer_index}.states.{site}"]
                row["mean_l2_to_native"] = _finite(
                    (state - native_state).norm(dim=-1).mean(),
                    "state l2 to native",
                )
                row["mean_cosine_to_native"] = _mean_cosine(state, native_state)
                layer_result[site] = row
            variant_result[f"layer_{layer_index}"] = layer_result
        result[variant] = variant_result
    return result


def _collect_task_variant(
    model: A1RecurrentStateSpaceLM,
    examples: list[Any],
    distance: int,
    variant: str,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, list[torch.Tensor]]]:
    logit_parts: list[torch.Tensor] = []
    trace_parts: dict[str, list[torch.Tensor]] = {}
    targets = torch.tensor([example.target for example in examples], dtype=torch.long)
    for start in range(0, len(examples), BATCH_SIZE):
        batch = examples[start : start + BATCH_SIZE]
        inputs = torch.tensor([example.input_ids for example in batch], dtype=torch.long, device=device)
        with torch.inference_mode(), _autocast(device):
            logits, trace = _functional_forward(
                model, inputs, distance, variant, collect=True
            )
        logit_parts.append(_finite_tensor(logits.float().cpu(), "task logits"))
        _append_trace(trace_parts, trace)
    all_logits = torch.cat(logit_parts, dim=0)
    if all_logits.shape[0] != len(examples):
        raise ValueError("task logit count mismatch")
    return _target_metrics(all_logits, targets), trace_parts


def _collect_probe_variant(
    model: A1RecurrentStateSpaceLM,
    split: str,
    count: int,
    distance: int,
    variant: str,
    device: torch.device,
) -> tuple[dict[str, dict[str, torch.Tensor]], torch.Tensor]:
    trace_parts: dict[str, list[torch.Tensor]] = {}
    target_parts: list[torch.Tensor] = []
    for start in range(0, count, PROBE_BATCH_SIZE):
        end = min(start + PROBE_BATCH_SIZE, count)
        inputs, targets = _probe_batch(split, start, end, distance)
        with torch.inference_mode(), _autocast(device):
            _logits, trace = _functional_forward(
                model, inputs.to(device), distance, variant, collect=True
            )
        _append_trace(trace_parts, trace)
        target_parts.append(targets)
    states: dict[str, dict[str, torch.Tensor]] = {}
    for layer_index in range(8):
        states[f"layer_{layer_index}"] = {
            site: torch.cat(
                trace_parts[f"layer_{layer_index}.states.{site}"], dim=0
            )
            for site in SITES
        }
    return states, torch.cat(target_parts, dim=0).long()


def _fixed_probe_metrics(
    model: A1RecurrentStateSpaceLM,
    distance: int,
    heldout_states: dict[str, dict[str, torch.Tensor]],
    heldout_targets: torch.Tensor,
    train_states: dict[str, dict[str, torch.Tensor]],
    train_targets: torch.Tensor,
) -> dict[str, Any]:
    value_embeddings = model.token_embedding.weight.detach().float()[
        VALUE_BASE : VALUE_BASE + VALUE_COUNT
    ].cpu()
    result: dict[str, Any] = {
        "train_examples": int(train_targets.shape[0]),
        "heldout_examples": int(heldout_targets.shape[0]),
        "distance": distance,
        "layers": {},
    }
    for layer_index in range(8):
        layer_key = f"layer_{layer_index}"
        probe = _fit_ridge_probe(
            train_states[layer_key]["post_value"],
            value_embeddings[train_targets],
        )
        layer_result: dict[str, Any] = {}
        for site in SITES:
            layer_result[site] = _probe_metrics(
                heldout_states[layer_key][site],
                heldout_targets,
                probe,
                value_embeddings,
            )
        result["layers"][layer_key] = layer_result
    return result


def _check_distance_zero(
    metrics: dict[str, dict[str, Any]],
    trace_summary: dict[str, Any],
    *,
    logit_tolerance: float = 1e-3,
    metric_tolerance: float = 1e-7,
) -> dict[str, Any]:
    native = metrics["native"]
    checks: dict[str, Any] = {}
    for variant in ("write_protected", "fully_frozen"):
        metric_deltas = _metric_delta(metrics[variant], native)
        max_metric_delta = max(
            abs(float(value))
            for scope in ("full_vocabulary", "value_candidate_range")
            for value in metric_deltas[scope].values()
        )
        checks[variant] = {
            "max_metric_delta": max_metric_delta,
            "passed_metric_tolerance": max_metric_delta <= metric_tolerance,
        }
        if max_metric_delta > metric_tolerance:
            raise RuntimeError(
                f"distance-zero {variant} metric delta exceeds tolerance: {max_metric_delta}"
            )

    for layer_index in range(8):
        for site in SITES:
            native_state = trace_summary["native"][f"layer_{layer_index}"][site]
            for variant in ("write_protected", "fully_frozen"):
                row = trace_summary[variant][f"layer_{layer_index}"][site]
                if row["mean_l2_to_native"] > logit_tolerance:
                    raise RuntimeError(
                        f"distance-zero {variant} state delta exceeds tolerance at "
                        f"layer {layer_index}, site {site}: {row['mean_l2_to_native']}"
                    )
                _ = native_state
    checks["passed"] = True
    return checks


def _verify_distance_zero_native_reference(
    model: A1RecurrentStateSpaceLM,
    inputs: torch.Tensor,
    distance: int,
    functional_logits: torch.Tensor,
    device: torch.device,
) -> float:
    with torch.inference_mode(), _autocast(device):
        reference = model(inputs).logits[:, 6 + distance, :].float().cpu()
    difference = (reference - functional_logits.cpu()).abs().max()
    value = _finite(difference, "native implementation max difference")
    if value > 1e-3:
        raise RuntimeError(f"functional native forward diverges from model forward: {value}")
    return value


def _source_records() -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    plan_sha = sha256_file(REPO_ROOT / ANALYSIS_PLAN_PATH)
    distance_sha = sha256_file(REPO_ROOT / DISTANCE_RESULT_PATH)
    if distance_sha != DISTANCE_RESULT_SHA256:
        raise ValueError("distance aggregate hash drifted")
    if sha256_file(REPO_ROOT / RUNNER_PATH) != RUNNER_SHA256:
        raise ValueError("synthetic runner hash drifted")
    if sha256_file(REPO_ROOT / DISTANCE_VERIFIER_PATH) != DISTANCE_VERIFIER_SHA256:
        raise ValueError("distance verifier hash drifted")

    aggregate = _load_json(REPO_ROOT / DISTANCE_RESULT_PATH)
    if aggregate.get("status") != "complete":
        raise ValueError("distance aggregate is not complete")
    baseline = _baseline()
    records: dict[int, dict[str, Any]] = {}
    baseline_a1 = next(
        run for run in baseline["runs"] if run.get("architecture") == "A1"
    )
    records[0] = baseline_a1
    for distance in DISTANCES:
        records[distance] = _load_run(distance, "A1")

    for distance, record in records.items():
        if int(record.get("seed", SEED)) != SEED:
            raise ValueError(f"source seed mismatch at distance {distance}")
        if int(record.get("parameter_count", -1)) != EXPECTED_PARAMS:
            raise ValueError(f"source parameter count mismatch at distance {distance}")
        checkpoint_path = REPO_ROOT / record["checkpoint"]["path"]
        checkpoint_record = verify_checkpoint(checkpoint_path)
        if checkpoint_record.get("checkpoint_sha256") != record["checkpoint"]["sha256"]:
            raise ValueError(f"source checkpoint hash mismatch at distance {distance}")
        config_path = REPO_ROOT / record["config_path"]
        if sha256_file(config_path) != record["config_sha256"]:
            raise ValueError(f"source config hash mismatch at distance {distance}")
        config = load_config(config_path)
        if config.stage != "A1" or config.model_type != "recurrent_state_space":
            raise ValueError(f"source config is not original A1 at distance {distance}")
        if int(record.get("distractor_distance", distance)) != distance:
            raise ValueError(f"source distance mismatch at distance {distance}")
    return records, {
        "analysis_plan_sha256": plan_sha,
        "distance_result_sha256": distance_sha,
        "distance_result_path": DISTANCE_RESULT_PATH,
        "runner_sha256": RUNNER_SHA256,
        "distance_verifier_sha256": DISTANCE_VERIFIER_SHA256,
    }


def _load_model(record: dict[str, Any], device: torch.device) -> A1RecurrentStateSpaceLM:
    config = load_config(REPO_ROOT / record["config_path"])
    model = A1RecurrentStateSpaceLM(config, use_parallel_scan=True)
    if model.parameter_count() != EXPECTED_PARAMS:
        raise ValueError(f"unexpected A1 parameter count: {model.parameter_count()}")
    checkpoint_path = REPO_ROOT / record["checkpoint"]["path"]
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.eval().to(device)
    return model


def _run_distance(
    record: dict[str, Any],
    distance: int,
    device: torch.device,
) -> dict[str, Any]:
    model = _load_model(record, device)
    before_digest = _parameter_digest(model)
    examples = make_evaluation_examples(
        SEED,
        "heldout",
        VALUE_COUNT,
        1,
        distance,
        1,
    )
    metrics: dict[str, dict[str, Any]] = {}
    trace_parts_by_variant: dict[str, dict[str, list[torch.Tensor]]] = {}
    native_reference_max_difference: float | None = None
    for variant in VARIANTS:
        metric, trace_parts = _collect_task_variant(
            model, examples, distance, variant, device
        )
        metrics[variant] = metric
        trace_parts_by_variant[variant] = trace_parts
        if distance == 0 and variant == "native":
            first_inputs = torch.tensor(
                [examples[0].input_ids], dtype=torch.long, device=device
            )
            with torch.inference_mode(), _autocast(device):
                functional_logits, _ = _functional_forward(
                    model, first_inputs, distance, variant, collect=False
                )
            native_reference_max_difference = _verify_distance_zero_native_reference(
                model, first_inputs, distance, functional_logits, device
            )

    trace_summary = _summarize_traces(trace_parts_by_variant)
    zero_control = None
    if distance == 0:
        zero_control = _check_distance_zero(metrics, trace_summary)

    train_states, train_targets = _collect_probe_variant(
        model,
        "train",
        PROBE_TRAIN_EXAMPLES,
        distance,
        "native",
        device,
    )
    probe_results: dict[str, Any] = {}
    for variant in VARIANTS:
        heldout_states, heldout_targets = _collect_probe_variant(
            model,
            "heldout",
            PROBE_EVAL_EXAMPLES,
            distance,
            variant,
            device,
        )
        probe_results[variant] = _fixed_probe_metrics(
            model,
            distance,
            heldout_states,
            heldout_targets,
            train_states,
            train_targets,
        )

    after_digest = _parameter_digest(model)
    if before_digest != after_digest:
        raise RuntimeError(f"functional intervention changed model parameters at distance {distance}")
    result = {
        "distance": distance,
        "checkpoint": record["checkpoint"],
        "source_run": record["run_dir"],
        "parameter_digest_before": before_digest,
        "parameter_digest_after": after_digest,
        "metrics": metrics,
        "deltas_from_native": {
            variant: _metric_delta(metrics[variant], metrics["native"])
            for variant in VARIANTS
            if variant != "native"
        },
        "state_metrics": trace_summary,
        "fixed_probe_metrics": probe_results,
        "distance_zero_control": zero_control,
        "native_reference_max_abs_difference": native_reference_max_difference,
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _reference_a0(aggregate: dict[str, Any], distance: int) -> dict[str, Any]:
    if distance == 0:
        paired = aggregate["baseline_distance0"]["paired"]["heldout"]["A0"]
    else:
        paired = aggregate["distances"][str(distance)]["paired"]["heldout"]["A0"]
    return {
        "target_loss": _finite(paired["target_loss"], "A0 target loss"),
        "greedy_answer_token_accuracy": _finite(
            paired["greedy_answer_token_accuracy"], "A0 answer accuracy"
        ),
        "greedy_exact_match": _finite(paired["greedy_exact_match"], "A0 exact match"),
    }


def _report(result: dict[str, Any]) -> str:
    lines = [
        "# A0/A1 inference-only interference-isolation result",
        "",
        "Status: **complete; reversible inference-only intervention**.",
        "",
        f"Protocol: `{result['analysis_plan_path']}` (sha256 `{result['analysis_plan_sha256']}`).",
        f"Source distance aggregate sha256: `{result['source_distance_result_sha256']}`.",
        "",
        "The write-protected variant suppressed new recurrent writes only between the stored value and query. The fully frozen variant additionally suppressed passive decay during that interval. Query/key/answer processing remained native.",
        "",
        "## Native and intervention metrics",
        "",
        "Negative CE deltas are improvements relative to native A1. Primary rows use the 8,000-value candidate range.",
        "",
        "| distance | variant | value CE | Δ CE | value top-1 | Δ top-1 | mean rank | Δ rank | exact | Δ exact |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in result["runs"]:
        distance = run["distance"]
        native = run["metrics"]["native"]["value_candidate_range"]
        for variant in VARIANTS:
            metrics = run["metrics"][variant]["value_candidate_range"]
            if variant == "native":
                delta = {"cross_entropy": 0.0, "top1_accuracy": 0.0, "mean_target_rank": 0.0}
                exact_delta = 0.0
            else:
                delta = run["deltas_from_native"][variant]["value_candidate_range"]
                exact_delta = run["deltas_from_native"][variant]["greedy_exact_match"]
            lines.append(
                f"| {distance} | {variant} | {metrics['cross_entropy']:.6f} | "
                f"{delta['cross_entropy']:+.6f} | {metrics['top1_accuracy']:.4f} | "
                f"{delta['top1_accuracy']:+.4f} | {metrics['mean_target_rank']:.1f} | "
                f"{delta['mean_target_rank']:+.1f} | {metrics['top1_accuracy']:.4f} | "
                f"{exact_delta:+.4f} |"
            )
    lines.extend(
        [
            "",
            "## Predeclared interpretation",
            "",
            "The machine-readable deltas are authoritative. Write-protected improvement at distances 16/64 supports active distractor-write interference; improvement only under full freezing supports a decay or decay/update interaction; a null result leaves query-conditioned corruption, pre-query binding/generalisation, and other recurrent transformations as the leading explanations.",
            "",
            "## Integrity",
            "",
            "All source checkpoints and hashes were verified. The distance-zero protected variants reproduced native metrics within tolerance, functional native logits matched the model forward, all tensors were finite, and parameter digests were unchanged. No checkpoint was written.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results/a0_a1_interference_isolation_values8000.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "docs/A0_A1_INTERFERENCE_ISOLATION_RESULT.md",
    )
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    records, provenance = _source_records()
    aggregate = _load_json(REPO_ROOT / DISTANCE_RESULT_PATH)

    runs = []
    for distance in (0, *DISTANCES):
        run = _run_distance(records[distance], distance, device)
        run["a0_native_reference"] = _reference_a0(aggregate, distance)
        runs.append(run)

    result = {
        "status": "complete",
        "version": OUTPUT_VERSION,
        "read_only": True,
        "analysis_device": str(device),
        "analysis_plan_path": ANALYSIS_PLAN_PATH,
        "analysis_plan_sha256": provenance["analysis_plan_sha256"],
        "source_distance_result_path": DISTANCE_RESULT_PATH,
        "source_distance_result_sha256": provenance["distance_result_sha256"],
        "source_runner_path": RUNNER_PATH,
        "source_runner_sha256": RUNNER_SHA256,
        "source_distance_verifier_path": DISTANCE_VERIFIER_PATH,
        "source_distance_verifier_sha256": DISTANCE_VERIFIER_SHA256,
        "source_protocol_commit": PROTOCOL_COMMIT,
        "seed": SEED,
        "value_count": VALUE_COUNT,
        "association_count": 1,
        "target_length": 1,
        "distances": [0, *DISTANCES],
        "variants": list(VARIANTS),
        "heldout_examples": FIXED_EXAMPLES,
        "probe_train_examples": PROBE_TRAIN_EXAMPLES,
        "probe_heldout_examples": PROBE_EVAL_EXAMPLES,
        "probe_batch_size": PROBE_BATCH_SIZE,
        "intervention": {
            "write_protected": "s_t = decay * s_(t-1) at distractor positions",
            "fully_frozen": "s_t = s_(t-1) at distractor positions",
            "patched_layers": 8,
            "distractor_positions": "4 through 3+d inclusive",
            "native_query_and_after": True,
        },
        "source_runs": {
            str(distance): records[distance]
            for distance in (0, *DISTANCES)
        },
        "runs": runs,
    }
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    report = args.report if args.report.is_absolute() else REPO_ROOT / args.report
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_report(result), encoding="utf-8")
    print(json.dumps({"output": str(output), "report": str(report), "runs": len(runs)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
