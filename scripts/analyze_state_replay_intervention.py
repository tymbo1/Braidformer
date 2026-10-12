#!/usr/bin/env python3
"""Read-only layer-7 A1 state-replay intervention on frozen checkpoints."""

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
from braidformer.training.integrity import sha256_file
from scripts.analyze_synthetic_solvability_target_lengths_values8000 import (
    ARCHITECTURES,
    CONFIGS,
    PROTOCOL_COMMIT,
    PROTOCOL_PATH,
    PROTOCOL_SHA256,
    RUNNER_SHA256,
    SEED,
    TARGET_LENGTHS,
    _verify_run,
)
from scripts.run_synthetic_overfit_calibration import (
    FIXED_EXAMPLES,
    VALUE_BASE,
    make_evaluation_examples,
)


VALUE_COUNT = 8_000
ASSOCIATION_COUNT = 1
DISTRACTOR_DISTANCE = 0
BATCH_SIZE = 16
PATCH_LAYER = 7
EXPECTED_PARAMS = 41_870_336
PLAN_PATH = "docs/A0_A1_STATE_REPLAY_INTERVENTION_PLAN.md"
TARGET_RUNNER_PATH = "scripts/run_synthetic_overfit_calibration.py"
OUTPUT_VERSION = "state_replay_intervention_v1"


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


def _load_model(
    target_length: int,
    run: dict[str, Any],
    device: torch.device,
) -> A1RecurrentStateSpaceLM:
    config = load_config(REPO_ROOT / CONFIGS[(target_length, "A1")])
    model = A1RecurrentStateSpaceLM(config, use_parallel_scan=True)
    if model.parameter_count() != EXPECTED_PARAMS:
        raise ValueError(f"unexpected A1 parameter count: {model.parameter_count()}")
    checkpoint_path = REPO_ROOT / run["checkpoint"]["path"]
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.eval().to(device)
    return model


def _target_ids(examples: list[Any], target_length: int) -> torch.Tensor:
    targets = []
    for example in examples:
        if not isinstance(example.target, tuple) or len(example.target) != target_length:
            raise ValueError("target metadata does not match target length")
        targets.append(example.target)
    return torch.tensor(targets, dtype=torch.long, device="cpu")


def _logit_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float | int]:
    logits = _finite_tensor(logits.float(), "logit metric input")
    targets = targets.long().to(logits.device)
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
        "mean_target_rank": _finite(ranks.float().mean(), "target rank"),
        "median_target_rank": _finite(torch.median(ranks.float()), "median target rank"),
    }


def _metrics_by_position(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_length: int,
) -> list[dict[str, Any]]:
    value_logits = logits[:, :, VALUE_BASE : VALUE_BASE + VALUE_COUNT]
    value_targets = targets - VALUE_BASE
    rows = []
    for position in range(target_length):
        rows.append(
            {
                "position": position + 1,
                "full_vocabulary": _logit_metrics(logits[:, position, :], targets[:, position]),
                "value_candidate_range": _logit_metrics(
                    value_logits[:, position, :],
                    value_targets[:, position],
                ),
            }
        )
    return rows


def _state_replay_forward(
    model: A1RecurrentStateSpaceLM,
    input_ids: torch.Tensor,
    target_length: int,
    patch_layer: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run A1 with one functional state-replay rule at the chosen layer."""

    sequence_length = input_ids.shape[1]
    positions = torch.arange(sequence_length, device=input_ids.device)
    hidden_states = model.token_embedding(input_ids)
    hidden_states = hidden_states + model.position_embedding(positions).unsqueeze(0)
    hidden_states = model.embedding_dropout(hidden_states)

    source_positions = {2 + position: 4 + target_length + position for position in range(1, target_length + 1)}
    target_positions = {target: source for source, target in source_positions.items()}
    if any(target >= sequence_length or source >= sequence_length for source, target in source_positions.items()):
        raise ValueError("state replay positions exceed sequence length")
    patched_state_records: dict[str, torch.Tensor] = {}

    for layer_index, block in enumerate(model.blocks):
        if layer_index != patch_layer:
            hidden_states = block(hidden_states)
            continue

        normalized = block.state_space_norm(hidden_states)
        module = block.state_space
        projected = module.input_projection(normalized)
        candidate, input_gate, output_gate = projected.chunk(3, dim=-1)
        decay = torch.sigmoid(module.state_transition_logit).to(dtype=normalized.dtype)
        updates = (1.0 - decay).view(1, 1, module.hidden_size)
        updates = updates * torch.sigmoid(input_gate) * torch.tanh(candidate)

        state = normalized.new_zeros((normalized.shape[0], module.hidden_size))
        states: list[torch.Tensor] = []
        unpatched_target_states: dict[int, torch.Tensor] = {}
        replayed_target_states: dict[int, torch.Tensor] = {}
        source_states: dict[int, torch.Tensor] = {}
        for timestep in range(sequence_length):
            state = decay * state + updates[:, timestep, :]
            if timestep in source_positions:
                source_states[timestep] = state.clone()
            if timestep in target_positions:
                unpatched_target_states[timestep] = state.clone()
                source_position = target_positions[timestep]
                if source_position not in source_states:
                    raise RuntimeError(f"missing source state at position {source_position}")
                state = source_states[source_position].clone()
                replayed_target_states[timestep] = state.clone()
            states.append(state)

        state_sequence = torch.stack(states, dim=1)
        mixed = torch.sigmoid(output_gate) * state_sequence
        hidden_states = hidden_states + module.residual_dropout(module.output_projection(mixed))
        hidden_states = hidden_states + block.feed_forward(block.feed_forward_norm(hidden_states))

        for target_position, replayed in replayed_target_states.items():
            original = unpatched_target_states[target_position]
            patched_state_records[f"target_{target_position - (4 + target_length)}"] = torch.stack(
                (original, replayed), dim=1
            ).detach().float().cpu()

    logits = model.lm_head(model.final_norm(hidden_states))
    target_indices = sorted(source_positions.values())
    target_logits = logits[:, target_indices, :]
    _finite_tensor(target_logits, "patched target logits")
    return target_logits, patched_state_records


def _collect_run(
    model: A1RecurrentStateSpaceLM,
    examples: list[Any],
    target_length: int,
    device: torch.device,
) -> dict[str, Any]:
    targets = _target_ids(examples, target_length)
    baseline_parts: list[torch.Tensor] = []
    patched_parts: list[torch.Tensor] = []
    state_parts: dict[str, list[torch.Tensor]] = {}
    for start in range(0, len(examples), BATCH_SIZE):
        batch = examples[start : start + BATCH_SIZE]
        inputs = torch.tensor(
            [example.input_ids for example in batch],
            dtype=torch.long,
            device=device,
        )
        with torch.inference_mode(), _autocast(device):
            baseline = model(inputs).logits[:, -(target_length + 1) : -1, :].float().cpu()
            patched, states = _state_replay_forward(model, inputs, target_length, PATCH_LAYER)
        baseline_parts.append(_finite_tensor(baseline, "baseline target logits"))
        patched_parts.append(_finite_tensor(patched.float().cpu(), "patched target logits"))
        for name, state in states.items():
            state_parts.setdefault(name, []).append(state)

    baseline_logits = torch.cat(baseline_parts, dim=0)
    patched_logits = torch.cat(patched_parts, dim=0)
    baseline_metrics = _metrics_by_position(baseline_logits, targets, target_length)
    patched_metrics = _metrics_by_position(patched_logits, targets, target_length)
    position_rows = []
    for baseline_row, patched_row in zip(baseline_metrics, patched_metrics):
        baseline_value = baseline_row["value_candidate_range"]
        patched_value = patched_row["value_candidate_range"]
        position_rows.append(
            {
                "position": baseline_row["position"],
                "baseline": baseline_row,
                "patched": patched_row,
                "delta_value_candidate_range": {
                    "cross_entropy": _finite(
                        patched_value["cross_entropy"] - baseline_value["cross_entropy"],
                        "delta CE",
                    ),
                    "correct_token_probability": _finite(
                        patched_value["correct_token_probability"] - baseline_value["correct_token_probability"],
                        "delta probability",
                    ),
                    "top1_accuracy": _finite(
                        patched_value["top1_accuracy"] - baseline_value["top1_accuracy"],
                        "delta top1",
                    ),
                    "correct_token_logit_margin": _finite(
                        patched_value["correct_token_logit_margin"] - baseline_value["correct_token_logit_margin"],
                        "delta margin",
                    ),
                    "mean_target_rank": _finite(
                        patched_value["mean_target_rank"] - baseline_value["mean_target_rank"],
                        "delta rank",
                    ),
                },
            }
        )

    state_metrics = {}
    for name, parts in state_parts.items():
        state = torch.cat(parts, dim=0)
        original = state[:, 0, :]
        replayed = state[:, 1, :]
        original_norm = original.norm(dim=-1).clamp_min(1e-8)
        replayed_norm = replayed.norm(dim=-1).clamp_min(1e-8)
        cosine = (original * replayed).sum(dim=-1) / (original_norm * replayed_norm)
        state_metrics[name] = {
            "examples": int(state.shape[0]),
            "original_mean_norm": _finite(original_norm.mean(), "original state norm"),
            "replayed_mean_norm": _finite(replayed_norm.mean(), "replayed state norm"),
            "replayed_vs_original_mean_cosine": _finite(cosine.mean(), "state replay cosine"),
        }

    return {
        "heldout_examples": len(examples),
        "patch_layer": PATCH_LAYER,
        "source_positions": [2 + position for position in range(1, target_length + 1)],
        "target_positions": [4 + target_length + position for position in range(1, target_length + 1)],
        "positions": position_rows,
        "state_metrics": state_metrics,
    }


def _report(result: dict[str, Any]) -> str:
    lines = [
        "# A0/A1 state-replay intervention result",
        "",
        "Status: **complete; reversible inference-only intervention**.",
        "",
        f"Intervention plan: `{result['analysis_plan_path']}` (sha256 `{result['analysis_plan_sha256']}`).",
        f"Source protocol commit: `{PROTOCOL_COMMIT}`; source runner sha256 `{RUNNER_SHA256}`.",
        "",
        "The intervention copied each target value's layer-7 recurrent state from its stored-value position to the corresponding target-output position. No model parameter or checkpoint was changed.",
        "",
        "## Held-out A1 baseline versus replay",
        "",
        "| L | target position | baseline value top1 | replay value top1 | Δ top1 | baseline value CE | replay value CE | Δ mean rank |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in result["runs"]:
        for row in run["positions"]:
            baseline = row["baseline"]["value_candidate_range"]
            patched = row["patched"]["value_candidate_range"]
            delta = row["delta_value_candidate_range"]
            lines.append(
                f"| {run['target_length']} | {row['position']} | {baseline['top1_accuracy']:.4f} | "
                f"{patched['top1_accuracy']:.4f} | {delta['top1_accuracy']:+.4f} | "
                f"{baseline['cross_entropy']:.4f} | {patched['cross_entropy']:.4f} | "
                f"{delta['mean_target_rank']:+.1f} |"
            )
    lines.extend(
        [
            "",
            "## Predeclared prediction",
            "",
            "The prediction was that replay would improve value rank/top-1 at one or more positions, especially at lengths 4 and 8, if temporal retention/state transformation were causal. The machine-readable deltas above are authoritative. A null or worsening result means this particular layer-7 state-preservation operation did not restore retrieval; no second intervention is authorized in this campaign.",
            "",
            "## Integrity",
            "",
            "All six A0/A1 source runs were sidecar/hash verified. A1 parameter digests before and after each functional intervention were unchanged; source/target positions and all baseline/patched tensors were finite. Raw checkpoints/logs remain ignored under `results/runs/`.",
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
        default=REPO_ROOT / "results/a0_a1_state_replay_intervention_values8000.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "docs/A0_A1_STATE_REPLAY_INTERVENTION_RESULT.md",
    )
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    if sha256_file(REPO_ROOT / PROTOCOL_PATH) != PROTOCOL_SHA256:
        raise ValueError("target-length protocol hash changed from its frozen value")
    if sha256_file(REPO_ROOT / TARGET_RUNNER_PATH) != RUNNER_SHA256:
        raise ValueError("synthetic runner hash changed from its frozen value")

    verified_runs: dict[tuple[int, str], dict[str, Any]] = {}
    for target_length in TARGET_LENGTHS:
        for architecture in ARCHITECTURES:
            verified_runs[(target_length, architecture)] = _verify_run(
                target_length,
                architecture,
            )

    runs = []
    source_digests = []
    for target_length in TARGET_LENGTHS:
        run = verified_runs[(target_length, "A1")]
        model = _load_model(target_length, run, device)
        before_digest = _parameter_digest(model)
        examples = make_evaluation_examples(
            SEED,
            "heldout",
            VALUE_COUNT,
            ASSOCIATION_COUNT,
            DISTRACTOR_DISTANCE,
            target_length,
        )
        run_result = _collect_run(model, examples, target_length, device)
        after_digest = _parameter_digest(model)
        if before_digest != after_digest:
            raise RuntimeError("functional intervention changed model parameters")
        run_result.update(
            {
                "architecture": "A1",
                "target_length": target_length,
                "checkpoint": run["checkpoint"],
                "source_run": run["run_dir"],
                "parameter_digest_before": before_digest,
                "parameter_digest_after": after_digest,
            }
        )
        runs.append(run_result)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for (target_length, architecture), run in verified_runs.items():
        source_digests.append(
            {
                "architecture": architecture,
                "target_length": target_length,
                "checkpoint": run["checkpoint"],
            }
        )

    plan_path = REPO_ROOT / PLAN_PATH
    result = {
        "status": "complete",
        "version": OUTPUT_VERSION,
        "read_only": True,
        "analysis_device": str(device),
        "analysis_plan_path": PLAN_PATH,
        "analysis_plan_sha256": sha256_file(plan_path),
        "source_protocol_commit": PROTOCOL_COMMIT,
        "source_protocol_path": PROTOCOL_PATH,
        "source_protocol_sha256": PROTOCOL_SHA256,
        "source_runner_path": TARGET_RUNNER_PATH,
        "source_runner_sha256": sha256_file(REPO_ROOT / TARGET_RUNNER_PATH),
        "seed": SEED,
        "value_count": VALUE_COUNT,
        "association_count": ASSOCIATION_COUNT,
        "distractor_distance": DISTRACTOR_DISTANCE,
        "target_lengths": list(TARGET_LENGTHS),
        "architectures_verified": list(ARCHITECTURES),
        "heldout_examples": FIXED_EXAMPLES,
        "patch_layer": PATCH_LAYER,
        "intervention": "copy_layer7_recurrent_state_from_each_stored_value_to_matching_target_output",
        "verified_source_runs": source_digests,
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
