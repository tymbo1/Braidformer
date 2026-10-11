#!/usr/bin/env python3
"""Verify and aggregate the completed synthetic A0/A1 task battery.

This is a read-only analysis of ignored run artifacts.  It does not load model
weights, alter checkpoints, or start training.  The existing checkpoint
integrity utility is used for every completed arm.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from braidformer.training.integrity import verify_checkpoint


SEEDS = (17, 23, 29)
ARCHITECTURES = ("A0", "A1")
TASKS = ("copy", "associative", "selective")
PROTOCOL_COMMIT = "a0f2a95a65c696053d9b2d98b4f6106d421c0daa"
PROTOCOL_PATH = REPO_ROOT / "docs" / "A0_A1_SYNTHETIC_RECALL_PROTOCOL.md"
EXPECTED_PARAMS = {"A0": 41_866_240, "A1": 41_870_336}
EXPECTED_CONFIGS = {
    "A0": "configs/braidnano_a0_synthetic_recall_battery.json",
    "A1": "configs/braidnano_a1_synthetic_recall_battery.json",
}
EXPECTED_CONFIG_HASHES = {
    "A0": "be0445199083ad6923e556e3bd117b692969c58048a65f3496b0ea5f34f933fb",
    "A1": "1538b1aea0add4b309f9824f93ace8f00fd8847157c7fdc5f0cbd423941891b0",
}
EXPECTED_STEPS = 1_000
EXPECTED_TOKENS = 8_192_000
TOKENS_PER_STEP = 8_192


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _finite(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite {label}: {value!r}")
    return number


def _sample_sd(values: list[float]) -> float | None:
    return stdev(values) if len(values) > 1 else None


def _mean_metric(records: list[dict[str, Any]], task: str, metric: str) -> float:
    return mean(float(record["test_metrics"][task][metric]) for record in records)


def _load_run(architecture: str, seed: int) -> dict[str, Any]:
    run_dir = REPO_ROOT / "results" / "runs" / f"synthetic_{architecture.lower()}_seed{seed}"
    summary_path = run_dir / "summary.json"
    summary = _load_json(summary_path)
    if summary.get("status") != "completed":
        raise ValueError(f"run is not complete: {run_dir}")
    if summary.get("architecture") != architecture or int(summary.get("seed", -1)) != seed:
        raise ValueError(f"run identity mismatch: {run_dir}")
    if summary.get("protocol_commit") != PROTOCOL_COMMIT:
        raise ValueError(f"protocol commit mismatch: {run_dir}")
    if summary.get("config_path") != EXPECTED_CONFIGS[architecture]:
        raise ValueError(f"config path mismatch: {run_dir}")
    if summary.get("config_sha256") != EXPECTED_CONFIG_HASHES[architecture]:
        raise ValueError(f"config hash mismatch: {run_dir}")
    if int(summary.get("parameter_count", -1)) != EXPECTED_PARAMS[architecture]:
        raise ValueError(f"parameter count mismatch: {run_dir}")
    if int(summary.get("optimizer_steps", -1)) != EXPECTED_STEPS:
        raise ValueError(f"step budget mismatch: {run_dir}")
    if int(summary.get("tokens_processed", -1)) != EXPECTED_TOKENS:
        raise ValueError(f"token budget mismatch: {run_dir}")
    if summary.get("failure") is not None or summary.get("numerical_instability") is not False:
        raise ValueError(f"run reports a failure or instability: {run_dir}")

    checkpoint_path = Path(summary["checkpoint"])
    checkpoint_record = verify_checkpoint(checkpoint_path)
    expected_checkpoint = {
        "config_sha256": summary["config_sha256"],
        "data_manifest_sha256": summary["synthetic_data_sha256"],
        "seed": seed,
        "step": EXPECTED_STEPS,
        "tokens_processed": EXPECTED_TOKENS,
        "parameter_count": EXPECTED_PARAMS[architecture],
        "protocol_commit": PROTOCOL_COMMIT,
    }
    for field, expected in expected_checkpoint.items():
        if checkpoint_record.get(field) != expected:
            raise ValueError(
                f"checkpoint metadata mismatch in {checkpoint_path}: "
                f"{field}={checkpoint_record.get(field)!r}, expected={expected!r}"
            )

    step_records: list[dict[str, Any]] = []
    validation_records: list[dict[str, Any]] = []
    with (run_dir / "steps.jsonl").open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            if record.get("event") != "step":
                continue
            step = int(record["step"])
            if step != len(step_records) + 1:
                raise ValueError(f"non-contiguous step log at {run_dir}:{line_number}")
            if int(record["tokens_processed"]) != step * TOKENS_PER_STEP:
                raise ValueError(f"token progress mismatch at {run_dir}:{line_number}")
            _finite(record["train_loss"], "training loss")
            cuda_seconds = _finite(record["cuda_elapsed_seconds"], "CUDA duration")
            if cuda_seconds <= 0:
                raise ValueError(f"non-positive CUDA duration at {run_dir}:{line_number}")
            _finite(record["wall_clock_seconds"], "wall time")
            if record.get("validation_loss") is not None:
                validation_records.append(
                    {
                        "step": step,
                        "tokens_processed": int(record["tokens_processed"]),
                        "validation_loss": _finite(record["validation_loss"], "validation loss"),
                        "learning_rate": _finite(record["learning_rate"], "learning rate"),
                        "cumulative_cuda_seconds": sum(
                            float(item["cuda_elapsed_seconds"]) for item in [*step_records, record]
                        ),
                    }
                )
            step_records.append(record)
    if len(step_records) != EXPECTED_STEPS:
        raise ValueError(f"expected {EXPECTED_STEPS} step records in {run_dir}, got {len(step_records)}")
    expected_validation_steps = list(range(100, EXPECTED_STEPS + 1, 100))
    if [record["step"] for record in validation_records] != expected_validation_steps:
        raise ValueError(f"validation cadence mismatch in {run_dir}")
    logged_cuda_seconds = sum(float(record["cuda_elapsed_seconds"]) for record in step_records)
    reported_cuda_seconds = _finite(summary["cuda_training_seconds"], "reported CUDA time")
    if not math.isclose(logged_cuda_seconds, reported_cuda_seconds, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"CUDA total mismatch in {run_dir}")

    for task in TASKS:
        for metric in (
            "target_loss",
            "teacher_forced_answer_token_accuracy",
            "teacher_forced_exact_match",
            "greedy_answer_token_accuracy",
            "greedy_exact_match",
        ):
            _finite(summary["test_metrics"][task][metric], f"{task} test {metric}")
        for metric in (
            "target_loss",
            "teacher_forced_answer_token_accuracy",
            "teacher_forced_exact_match",
        ):
            _finite(summary["final_validation"][task][metric], f"{task} validation {metric}")

    return {
        "architecture": architecture,
        "seed": seed,
        "experiment_id": summary["experiment_id"],
        "run_dir": str(run_dir.relative_to(REPO_ROOT)),
        "parameter_count": int(summary["parameter_count"]),
        "optimizer_steps": int(summary["optimizer_steps"]),
        "tokens_processed": int(summary["tokens_processed"]),
        "config_path": summary["config_path"],
        "config_sha256": summary["config_sha256"],
        "synthetic_data_sha256": summary["synthetic_data_sha256"],
        "synthetic_generator_sha256": summary["synthetic_generator_sha256"],
        "training_wall_seconds": _finite(summary["training_wall_seconds"], "training wall time"),
        "elapsed_wall_seconds": _finite(summary["elapsed_wall_seconds"], "elapsed wall time"),
        "cuda_training_seconds": reported_cuda_seconds,
        "average_tokens_per_second": _finite(summary["average_tokens_per_second"], "throughput"),
        "average_cuda_tokens_per_second": _finite(
            summary["average_cuda_tokens_per_second"], "CUDA throughput"
        ),
        "peak_allocated_vram_bytes": int(summary["peak_allocated_vram_bytes"]),
        "peak_reserved_vram_bytes": int(summary["peak_reserved_vram_bytes"]),
        "warmup_steps": int(summary["warmup_steps"]),
        "learning_rate": _finite(summary["learning_rate"], "learning rate"),
        "test_metrics": summary["test_metrics"],
        "final_validation": summary["final_validation"],
        "validation_history": summary["validation_history"],
        "validation_log": validation_records,
        "checkpoint": {
            "path": str(checkpoint_path.relative_to(REPO_ROOT)),
            "sha256": checkpoint_record["checkpoint_sha256"],
            "bytes": int(checkpoint_record["checkpoint_bytes"]),
            "complete": bool(checkpoint_record["complete"]),
            "verified": True,
        },
        "gradient_checkpointing": summary["gradient_checkpointing"],
        "numerical_instability": bool(summary["numerical_instability"]),
        "failure": summary["failure"],
    }


def _paired(records: dict[tuple[str, int], dict[str, Any]], seed: int, task: str) -> dict[str, Any]:
    a0 = records[("A0", seed)]["test_metrics"][task]
    a1 = records[("A1", seed)]["test_metrics"][task]
    validation_a0 = records[("A0", seed)]["final_validation"][task]
    validation_a1 = records[("A1", seed)]["final_validation"][task]
    return {
        "seed": seed,
        "task": task,
        "test": {
            "A0": a0,
            "A1": a1,
            "A1_minus_A0": {
                "target_loss": float(a1["target_loss"]) - float(a0["target_loss"]),
                "greedy_answer_token_accuracy": float(a1["greedy_answer_token_accuracy"])
                - float(a0["greedy_answer_token_accuracy"]),
                "greedy_exact_match": float(a1["greedy_exact_match"])
                - float(a0["greedy_exact_match"]),
            },
        },
        "final_validation": {
            "A0": validation_a0,
            "A1": validation_a1,
            "A1_minus_A0_target_loss": float(validation_a1["target_loss"])
            - float(validation_a0["target_loss"]),
        },
    }


def _aggregate(records: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    pairs = [_paired(records, seed, task) for seed in SEEDS for task in TASKS]
    per_seed: list[dict[str, Any]] = []
    for seed in SEEDS:
        task_pairs = [pair for pair in pairs if pair["seed"] == seed]
        a0_losses = [float(pair["test"]["A0"]["target_loss"]) for pair in task_pairs]
        a1_losses = [float(pair["test"]["A1"]["target_loss"]) for pair in task_pairs]
        per_seed.append(
            {
                "seed": seed,
                "test_target_loss_mean": {
                    "A0": mean(a0_losses),
                    "A1": mean(a1_losses),
                    "A1_minus_A0": mean(a1 - a0 for a0, a1 in zip(a0_losses, a1_losses)),
                },
                "task_pairs": task_pairs,
                "cuda_seconds": {
                    "A0": records[("A0", seed)]["cuda_training_seconds"],
                    "A1": records[("A1", seed)]["cuda_training_seconds"],
                },
            }
        )

    per_task: dict[str, Any] = {}
    for task in TASKS:
        task_pairs = [pair for pair in pairs if pair["task"] == task]
        a0_target = [float(pair["test"]["A0"]["target_loss"]) for pair in task_pairs]
        a1_target = [float(pair["test"]["A1"]["target_loss"]) for pair in task_pairs]
        target_deltas = [float(pair["test"]["A1_minus_A0"]["target_loss"]) for pair in task_pairs]
        per_task[task] = {
            "test_target_loss": {
                "A0_mean": mean(a0_target),
                "A1_mean": mean(a1_target),
                "A1_minus_A0_mean": mean(target_deltas),
                "A1_minus_A0_sample_sd": _sample_sd(target_deltas),
                "A1_lower_loss_count": sum(delta < 0 for delta in target_deltas),
            },
            "greedy_answer_token_accuracy": {
                "A0_mean": mean(float(pair["test"]["A0"]["greedy_answer_token_accuracy"]) for pair in task_pairs),
                "A1_mean": mean(float(pair["test"]["A1"]["greedy_answer_token_accuracy"]) for pair in task_pairs),
            },
            "greedy_exact_match": {
                "A0_mean": mean(float(pair["test"]["A0"]["greedy_exact_match"]) for pair in task_pairs),
                "A1_mean": mean(float(pair["test"]["A1"]["greedy_exact_match"]) for pair in task_pairs),
            },
        }

    all_target_deltas = [float(pair["test"]["A1_minus_A0"]["target_loss"]) for pair in pairs]
    a0_all = [float(pair["test"]["A0"]["target_loss"]) for pair in pairs]
    a1_all = [float(pair["test"]["A1"]["target_loss"]) for pair in pairs]
    resource_summary: dict[str, Any] = {}
    for architecture in ARCHITECTURES:
        arms = [records[(architecture, seed)] for seed in SEEDS]
        resource_summary[architecture] = {
            "cuda_training_seconds_mean": mean(arm["cuda_training_seconds"] for arm in arms),
            "cuda_training_seconds_sample_sd": _sample_sd(
                [arm["cuda_training_seconds"] for arm in arms]
            ),
            "elapsed_wall_seconds_mean": mean(arm["elapsed_wall_seconds"] for arm in arms),
            "average_tokens_per_second_mean": mean(
                arm["average_tokens_per_second"] for arm in arms
            ),
            "average_cuda_tokens_per_second_mean": mean(
                arm["average_cuda_tokens_per_second"] for arm in arms
            ),
            "peak_allocated_vram_bytes_max": max(arm["peak_allocated_vram_bytes"] for arm in arms),
            "peak_reserved_vram_bytes_max": max(arm["peak_reserved_vram_bytes"] for arm in arms),
        }
    resource_summary["A1_to_A0_cuda_time_ratio"] = (
        resource_summary["A1"]["cuda_training_seconds_mean"]
        / resource_summary["A0"]["cuda_training_seconds_mean"]
    )

    validation_curve_mean: dict[str, Any] = {}
    for architecture in ARCHITECTURES:
        validation_curve_mean[architecture] = {}
        for step_index in range(10):
            step = (step_index + 1) * 100
            validation_curve_mean[architecture][str(step)] = {
                task: mean(
                    float(records[(architecture, seed)]["validation_history"][step_index]["metrics"][task]["target_loss"])
                    for seed in SEEDS
                )
                for task in TASKS
            }

    return {
        "status": "complete",
        "experiment": "a0_a1_synthetic_recall_battery",
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_path": str(PROTOCOL_PATH.relative_to(REPO_ROOT)),
        "seeds": list(SEEDS),
        "architectures": list(ARCHITECTURES),
        "tasks": list(TASKS),
        "budget": {
            "optimizer_steps": EXPECTED_STEPS,
            "tokens_processed": EXPECTED_TOKENS,
            "context_length": 512,
            "micro_batch_size": 2,
            "gradient_accumulation_steps": 8,
            "effective_batch_size": 16,
            "learning_rate": 3e-4,
            "warmup_steps": 10,
            "schedule": "cosine indexed by optimizer steps",
            "comparison_axis": "equal token positions and equal optimizer steps; CUDA time measured but not matched",
        },
        "runs": [records[(architecture, seed)] for architecture in ARCHITECTURES for seed in SEEDS],
        "paired_seed_task_results": pairs,
        "per_seed": per_seed,
        "per_task": per_task,
        "aggregate_test_target_loss": {
            "A0_mean_over_9_seed_task_pairs": mean(a0_all),
            "A1_mean_over_9_seed_task_pairs": mean(a1_all),
            "A1_minus_A0_mean": mean(all_target_deltas),
            "A1_minus_A0_sample_sd": _sample_sd(all_target_deltas),
            "A1_lower_loss_pairs": sum(delta < 0 for delta in all_target_deltas),
            "pairs": len(all_target_deltas),
        },
        "aggregate_greedy_metrics": {
            "A0_mean_answer_token_accuracy_over_9_pairs": mean(
                float(pair["test"]["A0"]["greedy_answer_token_accuracy"]) for pair in pairs
            ),
            "A1_mean_answer_token_accuracy_over_9_pairs": mean(
                float(pair["test"]["A1"]["greedy_answer_token_accuracy"]) for pair in pairs
            ),
            "A0_mean_exact_match_over_9_pairs": mean(
                float(pair["test"]["A0"]["greedy_exact_match"]) for pair in pairs
            ),
            "A1_mean_exact_match_over_9_pairs": mean(
                float(pair["test"]["A1"]["greedy_exact_match"]) for pair in pairs
            ),
        },
        "resource_summary": resource_summary,
        "validation_curve_mean_target_loss_by_step": validation_curve_mean,
        "integrity": {
            "run_count": len(records),
            "completed_run_count": sum(record["failure"] is None for record in records.values()),
            "checkpoint_verified_count": sum(record["checkpoint"]["verified"] for record in records.values()),
            "checkpoint_integrity_all_verified": all(record["checkpoint"]["verified"] for record in records.values()),
            "numerical_failure_count": sum(record["numerical_instability"] for record in records.values()),
            "failure_count": sum(record["failure"] is not None for record in records.values()),
            "gradient_checkpointing_implemented": False,
            "gradient_checkpointing_enabled": False,
        },
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _report(aggregate: dict[str, Any]) -> str:
    resource = aggregate["resource_summary"]
    target = aggregate["aggregate_test_target_loss"]
    greedy = aggregate["aggregate_greedy_metrics"]
    lines = [
        "# A0/A1 synthetic recall-battery result",
        "",
        "Status: complete. This report covers only the six fresh runs frozen in "
        f"protocol commit `{aggregate['protocol_commit']}`.",
        "",
        "The experiment used unchanged 41.87M-parameter A0/A1 implementations, "
        "fresh seeds 17, 23, and 29, the deterministic synthetic generator v1, "
        "1,000 optimizer steps, and 8,192,000 token positions per arm. It is an "
        "equal-token/equal-update comparison; CUDA time was measured, not matched.",
        "",
        "## Primary autoregressive result",
        "",
        "Greedy generation was evaluated without exposing answer tokens. Both "
        "architectures produced zero exact whole-answer matches on every task and "
        "seed. Mean answer-token accuracy was "
        f"{greedy['A0_mean_answer_token_accuracy_over_9_pairs']:.6f} for A0 and "
        f"{greedy['A1_mean_answer_token_accuracy_over_9_pairs']:.6f} for A1 "
        "(macro-average over nine seed/task pairs). The primary exact-recall test "
        "therefore does not show a winner at this training budget.",
        "",
        "| Task | A0 greedy token accuracy | A1 greedy token accuracy | A0 exact match | A1 exact match |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for task in TASKS:
        metrics = aggregate["per_task"][task]
        lines.append(
            f"| {task} | {metrics['greedy_answer_token_accuracy']['A0_mean']:.6f} | "
            f"{metrics['greedy_answer_token_accuracy']['A1_mean']:.6f} | "
            f"{metrics['greedy_exact_match']['A0_mean']:.6f} | "
            f"{metrics['greedy_exact_match']['A1_mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Secondary target-loss result",
            "",
            "Target-only cross-entropy is secondary because it does not establish "
            "successful exact recall. Lower is better.",
            "",
            "| Task | A0 test loss mean | A1 test loss mean | A1 − A0 | A1 lower-loss seeds |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for task in TASKS:
        metrics = aggregate["per_task"][task]["test_target_loss"]
        lines.append(
            f"| {task} | {metrics['A0_mean']:.6f} | {metrics['A1_mean']:.6f} | "
            f"{metrics['A1_minus_A0_mean']:.6f} | {metrics['A1_lower_loss_count']}/3 |"
        )
    lines.extend(
        [
            "",
            f"Across all nine seed/task pairs, A0 mean target loss was "
            f"`{target['A0_mean_over_9_seed_task_pairs']:.6f}` and A1 was "
            f"`{target['A1_mean_over_9_seed_task_pairs']:.6f}`. The paired mean "
            f"delta was `{target['A1_minus_A0_mean']:.6f}` with sample SD "
            f"`{target['A1_minus_A0_sample_sd']:.6f}`; A1 had lower target loss in "
            f"{target['A1_lower_loss_pairs']}/{target['pairs']} pairs.",
            "",
            "| Seed | A0 mean task loss | A1 mean task loss | A1 − A0 |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for record in aggregate["per_seed"]:
        losses = record["test_target_loss_mean"]
        lines.append(
            f"| {record['seed']} | {losses['A0']:.6f} | {losses['A1']:.6f} | "
            f"{losses['A1_minus_A0']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Resource measurements",
            "",
            "| Architecture | Mean CUDA training s | Mean elapsed wall s | Mean CUDA tok/s | Peak allocated VRAM | Peak reserved VRAM |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for architecture in ARCHITECTURES:
        metrics = resource[architecture]
        lines.append(
            f"| {architecture} | {metrics['cuda_training_seconds_mean']:.3f} | "
            f"{metrics['elapsed_wall_seconds_mean']:.3f} | "
            f"{metrics['average_cuda_tokens_per_second_mean']:.1f} | "
            f"{metrics['peak_allocated_vram_bytes_max'] / (1024 ** 3):.3f} GiB | "
            f"{metrics['peak_reserved_vram_bytes_max'] / (1024 ** 3):.3f} GiB |"
        )
    lines.extend(
        [
            "",
            f"A1 used `{resource['A1_to_A0_cuda_time_ratio']:.3f}×` the mean A0 CUDA "
            "training time for the same token/update budget. This is not a matched-"
            "compute, matched-FLOP, or energy comparison.",
            "",
            "## Validation and integrity",
            "",
            "Validation target loss was recorded every 100 steps and at step 1,000 "
            "for each task. The complete validation curves and per-run records are "
            "in the machine-readable aggregate. All six runs completed with finite "
            "values; all six final checkpoints passed the existing SHA-256 sidecar "
            "verification. Gradient checkpointing remains unimplemented and was not "
            "enabled.",
            "",
            "## Interpretation",
            "",
            "At this 1,000-step budget, the synthetic battery does not support an "
            "A1 advantage claim. The primary autoregressive exact-recall metric is "
            "zero for both models, while the secondary target loss favors A0 in all "
            "nine seed/task pairs. This result is specific to the declared generator, "
            "context, budget, optimizer, and 42M-parameter implementations; it does "
            "not overturn or generalize the prior TinyStories results. No further "
            "training, architecture change, long-context, second-corpus, A2, memory, "
            "or WNF work was started.",
            "",
            "Raw logs, task metrics, checkpoints, and checkpoint sidecars remain in "
            "ignored `results/runs/` directories and are not tracked by Git.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "a0_a1_synthetic_recall_battery.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "docs" / "A0_A1_SYNTHETIC_RECALL_RESULT.md",
    )
    args = parser.parse_args()

    records = {(architecture, seed): _load_run(architecture, seed) for architecture in ARCHITECTURES for seed in SEEDS}
    aggregate = _aggregate(records)
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    report = args.report if args.report.is_absolute() else REPO_ROOT / args.report
    _write_json(output, aggregate)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_report(aggregate), encoding="utf-8")
    print(json.dumps({"output": str(output), "report": str(report), "runs": len(records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
