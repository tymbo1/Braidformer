#!/usr/bin/env python3
"""Read-only target-position decomposition for the frozen target-length runs.

The script verifies the six existing target-length runs, loads their final
checkpoints without gradients, and separates teacher-forced target positions
from greedy autoregressive target positions. It never writes checkpoints or
modifies model parameters.
"""

from __future__ import annotations

import argparse
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
from braidformer.models import A0TransformerLM, A1RecurrentStateSpaceLM
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
from scripts.run_synthetic_overfit_calibration import make_evaluation_examples


VALUE_COUNT = 8_000
ASSOCIATION_COUNT = 1
DISTRACTOR_DISTANCE = 0
BATCH_SIZE = 16
EXPECTED_PARAMS = {"A0": 41_866_240, "A1": 41_870_336}
PLAN_PATH = "docs/A0_A1_TARGET_POSITION_DECOMPOSITION_PLAN.md"
TARGET_PROTOCOL_PATH = PROTOCOL_PATH
TARGET_RUNNER_PATH = "scripts/run_synthetic_overfit_calibration.py"
OUTPUT_VERSION = "target_position_decomposition_v1"


def _finite(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite {label}: {value!r}")
    return number


def _finite_tensor(values: torch.Tensor, label: str) -> torch.Tensor:
    if not bool(torch.isfinite(values).all()):
        raise ValueError(f"non-finite tensor in {label}")
    return values


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _load_model(
    architecture: str,
    target_length: int,
    run: dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    config = load_config(REPO_ROOT / CONFIGS[(target_length, architecture)])
    if architecture == "A0":
        model: torch.nn.Module = A0TransformerLM(config)
    elif architecture == "A1":
        model = A1RecurrentStateSpaceLM(config, use_parallel_scan=True)
    else:
        raise ValueError(f"unsupported architecture: {architecture}")
    if model.parameter_count() != EXPECTED_PARAMS[architecture]:  # type: ignore[attr-defined]
        raise ValueError(
            f"unexpected {architecture} parameter count for target length {target_length}"
        )
    checkpoint_path = REPO_ROOT / run["checkpoint"]["path"]
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.eval().to(device)
    return model


def _target_tensor(examples: list[Any], target_length: int, device: torch.device) -> torch.Tensor:
    targets = []
    for example in examples:
        target = example.target
        if target_length == 1:
            if not isinstance(target, int):
                raise ValueError("length-1 target was not an integer")
            targets.append((target,))
        else:
            if not isinstance(target, tuple) or len(target) != target_length:
                raise ValueError("target tuple length mismatch")
            targets.append(target)
    return torch.tensor(targets, dtype=torch.long, device=device)


def _teacher_forced(
    model: torch.nn.Module,
    examples: list[Any],
    target_length: int,
    device: torch.device,
) -> dict[str, Any]:
    per_position: list[dict[str, float | int]] = []
    total_examples = len(examples)
    correct_counts = torch.zeros(target_length, dtype=torch.float64)
    loss_sums = torch.zeros(target_length, dtype=torch.float64)
    probability_sums = torch.zeros(target_length, dtype=torch.float64)
    margin_sums = torch.zeros(target_length, dtype=torch.float64)
    sequence_correct = 0
    token_correct = 0
    for start in range(0, total_examples, BATCH_SIZE):
        batch = examples[start : start + BATCH_SIZE]
        inputs = torch.tensor([example.input_ids for example in batch], dtype=torch.long, device=device)
        targets = _target_tensor(batch, target_length, device)
        with torch.inference_mode(), _autocast(device):
            logits = model(inputs).logits[:, -(target_length + 1) : -1, :]
        logits = _finite_tensor(logits.float(), "teacher-forced logits")
        correct_logits = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        log_probability = correct_logits - torch.logsumexp(logits, dim=-1)
        probabilities = torch.exp(log_probability)
        top1 = logits.argmax(dim=-1).eq(targets)
        incorrect_logits = logits.clone()
        incorrect_logits.scatter_(-1, targets.unsqueeze(-1), float("-inf"))
        margins = correct_logits - incorrect_logits.amax(dim=-1)
        _finite_tensor(log_probability, "teacher-forced log probability")
        _finite_tensor(probabilities, "teacher-forced probability")
        _finite_tensor(margins, "teacher-forced logit margin")
        loss_sums += (-log_probability).sum(dim=0).double().cpu()
        probability_sums += probabilities.sum(dim=0).double().cpu()
        margin_sums += margins.sum(dim=0).double().cpu()
        correct_counts += top1.sum(dim=0).double().cpu()
        sequence_correct += int(top1.all(dim=1).sum().item())
        token_correct += int(top1.sum().item())

    for position in range(target_length):
        count = float(total_examples)
        per_position.append(
            {
                "position": position + 1,
                "examples": total_examples,
                "cross_entropy": _finite(loss_sums[position] / count, "teacher-forced CE"),
                "top1_accuracy": _finite(correct_counts[position] / count, "teacher-forced top1"),
                "correct_token_probability": _finite(
                    probability_sums[position] / count,
                    "teacher-forced correct-token probability",
                ),
                "correct_token_logit_margin": _finite(
                    margin_sums[position] / count,
                    "teacher-forced correct-token margin",
                ),
            }
        )
    return {
        "examples": total_examples,
        "greedy_exact_match": _finite(sequence_correct / total_examples, "teacher-forced exact match"),
        "token_accuracy": _finite(
            token_correct / (total_examples * target_length),
            "teacher-forced token accuracy",
        ),
        "positions": per_position,
    }


def _autoregressive(
    model: torch.nn.Module,
    examples: list[Any],
    target_length: int,
    device: torch.device,
) -> dict[str, Any]:
    total_examples = len(examples)
    targets_all = _target_tensor(examples, target_length, device)
    correct_counts = torch.zeros(target_length, dtype=torch.float64)
    probability_sums = torch.zeros(target_length, dtype=torch.float64)
    margin_sums = torch.zeros(target_length, dtype=torch.float64)
    predictions_all = torch.empty((total_examples, target_length), dtype=torch.long, device=device)

    for start in range(0, total_examples, BATCH_SIZE):
        batch = examples[start : start + BATCH_SIZE]
        prefixes = torch.tensor(
            [example.input_ids[:-target_length] for example in batch],
            dtype=torch.long,
            device=device,
        )
        targets = targets_all[start : start + len(batch)]
        generated = prefixes
        for position in range(target_length):
            with torch.inference_mode(), _autocast(device):
                logits = model(generated).logits[:, -1, :]
            logits = _finite_tensor(logits.float(), "autoregressive logits")
            correct_logits = logits.gather(-1, targets[:, position : position + 1]).squeeze(-1)
            log_probability = correct_logits - torch.logsumexp(logits, dim=-1)
            probabilities = torch.exp(log_probability)
            predictions = logits.argmax(dim=-1)
            incorrect_logits = logits.clone()
            incorrect_logits.scatter_(-1, targets[:, position : position + 1], float("-inf"))
            margins = correct_logits - incorrect_logits.amax(dim=-1)
            _finite_tensor(probabilities, "autoregressive probability")
            _finite_tensor(margins, "autoregressive logit margin")
            correct_counts[position] += predictions.eq(targets[:, position]).sum().double().cpu()
            probability_sums[position] += probabilities.sum().double().cpu()
            margin_sums[position] += margins.sum().double().cpu()
            predictions_all[start : start + len(batch), position] = predictions
            generated = torch.cat((generated, predictions.unsqueeze(-1)), dim=1)

    sequence_correct = predictions_all.eq(targets_all).all(dim=1)
    positions = []
    for position in range(target_length):
        count = float(total_examples)
        positions.append(
            {
                "position": position + 1,
                "examples": total_examples,
                "top1_accuracy": _finite(correct_counts[position] / count, "autoregressive top1"),
                "correct_token_probability": _finite(
                    probability_sums[position] / count,
                    "autoregressive correct-token probability",
                ),
                "correct_token_logit_margin": _finite(
                    margin_sums[position] / count,
                    "autoregressive correct-token margin",
                ),
            }
        )
    return {
        "examples": total_examples,
        "positions": positions,
        "greedy_exact_match": _finite(sequence_correct.double().mean(), "autoregressive exact match"),
        "greedy_token_accuracy": _finite(
            predictions_all.eq(targets_all).double().mean(),
            "autoregressive token accuracy",
        ),
    }


def _run_analysis(
    architecture: str,
    target_length: int,
    run: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    model = _load_model(architecture, target_length, run, device)
    split_results: dict[str, Any] = {}
    for split in ("train", "heldout"):
        examples = make_evaluation_examples(
            SEED,
            split,
            VALUE_COUNT,
            ASSOCIATION_COUNT,
            DISTRACTOR_DISTANCE,
            target_length,
        )
        teacher = _teacher_forced(model, examples, target_length, device)
        autoregressive = _autoregressive(model, examples, target_length, device)
        split_results[split] = {
            "teacher_forced": teacher,
            "autoregressive": autoregressive,
        }
    return {
        "architecture": architecture,
        "target_length": target_length,
        "checkpoint": run["checkpoint"],
        "source_run": run["run_dir"],
        "splits": split_results,
    }


def _fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def _report(result: dict[str, Any]) -> str:
    lines = [
        "# A0/A1 target-position decomposition result",
        "",
        "Status: **complete; read-only frozen-checkpoint analysis**.",
        "",
        f"Analysis plan: `{result['analysis_plan_path']}` (sha256 `{result['analysis_plan_sha256']}`).",
        f"Source protocol: `{PROTOCOL_COMMIT}`; source runner sha256 `{RUNNER_SHA256}`.",
        "",
        "No model parameters or source checkpoints were modified. The tables below use the held-out split; the machine-readable result also preserves the training split.",
        "",
        "## Held-out teacher-forced and greedy autoregressive positions",
        "",
        "| L | pos | A0 TF CE | A1 TF CE | A0 TF top1 | A1 TF top1 | A0 TF p(correct) | A1 TF p(correct) | A0 AR top1 | A1 AR top1 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    by_key = {
        (entry["architecture"], entry["target_length"]): entry
        for entry in result["runs"]
    }
    for target_length in TARGET_LENGTHS:
        a0 = by_key[("A0", target_length)]["splits"]["heldout"]
        a1 = by_key[("A1", target_length)]["splits"]["heldout"]
        for position in range(target_length):
            a0_tf = a0["teacher_forced"]["positions"][position]
            a1_tf = a1["teacher_forced"]["positions"][position]
            a0_ar = a0["autoregressive"]["positions"][position]
            a1_ar = a1["autoregressive"]["positions"][position]
            lines.append(
                f"| {target_length} | {position + 1} | "
                f"{_fmt(a0_tf['cross_entropy'])} | {_fmt(a1_tf['cross_entropy'])} | "
                f"{_fmt(a0_tf['top1_accuracy'], 4)} | {_fmt(a1_tf['top1_accuracy'], 4)} | "
                f"{_fmt(a0_tf['correct_token_probability'], 6)} | "
                f"{_fmt(a1_tf['correct_token_probability'], 6)} | "
                f"{_fmt(a0_ar['top1_accuracy'], 4)} | {_fmt(a1_ar['top1_accuracy'], 4)} |"
            )
    lines.extend(
        [
            "",
            "## Held-out endpoint summaries",
            "",
        "| L | architecture | TF token accuracy | TF exact match | AR token accuracy | AR exact match |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for target_length in TARGET_LENGTHS:
        for architecture in ARCHITECTURES:
            split = by_key[(architecture, target_length)]["splits"]["heldout"]
            teacher = split["teacher_forced"]
            ar = split["autoregressive"]
            lines.append(
                f"| {target_length} | {architecture} | {_fmt(teacher['token_accuracy'], 6)} | "
                f"{_fmt(teacher['greedy_exact_match'], 6)} | "
                f"{_fmt(ar['greedy_token_accuracy'], 6)} | {_fmt(ar['greedy_exact_match'], 6)} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The position-resolved metrics are diagnostic rather than a new training result. A1 performance at position 1 is compared directly with later positions under both teacher forcing and greedy generation. A first-position teacher-forced deficit cannot be attributed solely to autoregressive error accumulation; a teacher-forced/autoregressive divergence supports a target-state feedback contribution. Layerwise and query/readout forensics remain required before assigning a mechanism.",
            "",
            "## Integrity",
            "",
            "All six existing target-length runs were sidecar/hash verified before analysis. The analysis required the expected parameter counts, finite logits and metrics, and used evaluation mode with gradients disabled. Raw logs and checkpoints remain under ignored `results/runs/` paths.",
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
        default=REPO_ROOT / "results/a0_a1_target_position_decomposition_values8000.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "docs/A0_A1_TARGET_POSITION_DECOMPOSITION_RESULT.md",
    )
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    if sha256_file(REPO_ROOT / TARGET_PROTOCOL_PATH) != PROTOCOL_SHA256:
        raise ValueError("target-length protocol hash changed from the frozen value")
    if sha256_file(REPO_ROOT / TARGET_RUNNER_PATH) != RUNNER_SHA256:
        raise ValueError("synthetic runner hash changed from the frozen value")

    verified_runs: dict[tuple[int, str], dict[str, Any]] = {}
    for target_length in TARGET_LENGTHS:
        for architecture in ARCHITECTURES:
            verified_runs[(target_length, architecture)] = _verify_run(
                target_length,
                architecture,
            )

    analysis_runs = []
    for target_length in TARGET_LENGTHS:
        for architecture in ARCHITECTURES:
            analysis_runs.append(
                _run_analysis(
                    architecture,
                    target_length,
                    verified_runs[(target_length, architecture)],
                    device,
                )
            )

    plan_path = REPO_ROOT / PLAN_PATH
    runner_path = REPO_ROOT / TARGET_RUNNER_PATH
    result = {
        "status": "complete",
        "version": OUTPUT_VERSION,
        "read_only": True,
        "analysis_device": str(device),
        "analysis_plan_path": PLAN_PATH,
        "analysis_plan_sha256": sha256_file(plan_path),
        "source_protocol_commit": PROTOCOL_COMMIT,
        "source_protocol_path": TARGET_PROTOCOL_PATH,
        "source_protocol_sha256": PROTOCOL_SHA256,
        "source_runner_path": TARGET_RUNNER_PATH,
        "source_runner_sha256": sha256_file(runner_path),
        "seed": SEED,
        "value_count": VALUE_COUNT,
        "association_count": ASSOCIATION_COUNT,
        "distractor_distance": DISTRACTOR_DISTANCE,
        "target_lengths": list(TARGET_LENGTHS),
        "architectures": list(ARCHITECTURES),
        "batch_size": BATCH_SIZE,
        "runs": analysis_runs,
        "source_run_verification": {
            f"L{target_length}_{architecture}": verified_runs[(target_length, architecture)]
            for target_length in TARGET_LENGTHS
            for architecture in ARCHITECTURES
        },
    }
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    report = args.report if args.report.is_absolute() else REPO_ROOT / args.report
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_report(result), encoding="utf-8")
    print(json.dumps({"output": str(output), "report": str(report), "runs": len(analysis_runs)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

