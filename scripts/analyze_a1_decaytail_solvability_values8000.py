#!/usr/bin/env python3
"""Verify and summarize the A1 long-decay-tail intervention runs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from braidformer.training.integrity import sha256_file, sha256_json, verify_checkpoint
from scripts.run_synthetic_overfit_calibration import (
    EFFECTIVE_BATCH_SIZE,
    _data_hash,
    generator_spec,
)


SEED = 17
VALUE_COUNT = 8_000
ASSOCIATION_COUNT = 1
EXPECTED_STEPS = 2_000
DISTANCES = (4, 16, 64)
PROTOCOL_COMMIT = "515f9d99e359789ee65e0291d25904ab8717768c"
PROTOCOL_PATH = "docs/A1_SOLVABILITY_DECAYTAIL_VALUES8000_PROTOCOL.md"
PROTOCOL_SHA256 = "ed14d7b6ac06193f587c8b476c69b437bac606635bf0c12baf57a04530ecdf1a"
RUNNER_SHA256 = "75744163eac3551f06010295ecd2d9716bf2977887883b4c066e13b96300f515"
INITIALIZATION = "tail_75_original_25_long_v1"
CONFIGS = {
    4: "configs/braidnano_a1_solvability_decaytail_distance4_values8000.json",
    16: "configs/braidnano_a1_solvability_decaytail_distance16_values8000.json",
    64: "configs/braidnano_a1_solvability_decaytail_distance64_values8000.json",
}
CONFIG_SHA256 = {
    4: "22d9de38ad6066eb8975de2a94cf132cae887f4f9b4facfa24a3fb0c79b469fc",
    16: "9737681059a39220fa89a30d2d6b20dd9bb002803a04fe7a423eb5fff2ec96ab",
    64: "b3dd6745fa93b1bd3cc8effc5abc6f57dc3e1c856b66638d7d46137abe15ddc9",
}
CHECKPOINT_SHA256 = {
    4: "79bdbae2ecd56e6963f019e018f087de98948d1e03a9d7d0876160cb5881a7df",
    16: "10b650f6fcf44b88267178845cea8fc0ca8d775562f5cbc61780c06497194cb7",
    64: "42ee4a206f7dec3d14c70afa181d629d6da7af90588f1dd79f46ac2696a1a682",
}
EXPECTED_SEQUENCE_LENGTH = {4: 12, 16: 24, 64: 72}
EXPECTED_TOKENS = {distance: EXPECTED_SEQUENCE_LENGTH[distance] * 16 * EXPECTED_STEPS for distance in DISTANCES}


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


def _metrics(metrics: dict[str, Any], label: str) -> dict[str, Any]:
    if int(metrics.get("examples", -1)) != 128:
        raise ValueError(f"unexpected example count in {label}")
    for field in ("target_loss", "greedy_answer_token_accuracy", "greedy_exact_match"):
        _finite(metrics[field], f"{label}.{field}")
    return {
        "examples": 128,
        "target_loss": float(metrics["target_loss"]),
        "greedy_answer_token_accuracy": float(metrics["greedy_answer_token_accuracy"]),
        "greedy_exact_match": float(metrics["greedy_exact_match"]),
    }


def _run_dir(distance: int) -> Path:
    return REPO_ROOT / "results" / "runs" / (
        f"solvability_decaytail_distance{distance}_values8000_a1_seed{SEED}"
    )


def _verify_run(distance: int) -> dict[str, Any]:
    run_dir = _run_dir(distance)
    summary = _load_json(run_dir / "summary.json")
    if summary.get("status") != "completed":
        raise ValueError(f"run is not complete: {run_dir}")
    expected_generator = generator_spec(SEED, VALUE_COUNT, ASSOCIATION_COUNT, distance)
    expected_data_hash = _data_hash(SEED, VALUE_COUNT, ASSOCIATION_COUNT, distance)
    expected = {
        "architecture": "A1",
        "state_transition_initialization": INITIALIZATION,
        "seed": SEED,
        "parameter_count": 41_870_336,
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "script_sha256": RUNNER_SHA256,
        "config_path": CONFIGS[distance],
        "config_sha256": CONFIG_SHA256[distance],
        "synthetic_generator_version": expected_generator["version"],
        "synthetic_generator_sha256": sha256_json(expected_generator),
        "synthetic_data_sha256": expected_data_hash,
        "distractor_distance": distance,
        "sequence_length": EXPECTED_SEQUENCE_LENGTH[distance],
        "optimizer_steps": EXPECTED_STEPS,
        "tokens_processed": EXPECTED_TOKENS[distance],
        "value_count": VALUE_COUNT,
        "association_count": ASSOCIATION_COUNT,
    }
    for field, expected_value in expected.items():
        if summary.get(field) != expected_value:
            raise ValueError(
                f"summary mismatch for distance {distance}: {field}="
                f"{summary.get(field)!r}, expected={expected_value!r}"
            )
    config_path = REPO_ROOT / CONFIGS[distance]
    if sha256_file(config_path) != CONFIG_SHA256[distance]:
        raise ValueError(f"working-tree config hash mismatch: {config_path}")
    protocol_path = REPO_ROOT / PROTOCOL_PATH
    if sha256_file(protocol_path) != PROTOCOL_SHA256:
        raise ValueError(f"working-tree protocol hash mismatch: {protocol_path}")

    checkpoint_path = Path(summary["checkpoint"])
    checkpoint_record = verify_checkpoint(checkpoint_path)
    if checkpoint_record.get("checkpoint_sha256") != CHECKPOINT_SHA256[distance]:
        raise ValueError(f"checkpoint digest mismatch: {checkpoint_path}")
    for field, expected_value in {
        "config_sha256": CONFIG_SHA256[distance],
        "data_manifest_sha256": expected_data_hash,
        "seed": SEED,
        "step": EXPECTED_STEPS,
        "tokens_processed": EXPECTED_TOKENS[distance],
        "parameter_count": 41_870_336,
        "protocol_commit": PROTOCOL_COMMIT,
        "state_transition_initialization": INITIALIZATION,
        "value_count": VALUE_COUNT,
        "association_count": ASSOCIATION_COUNT,
        "distractor_distance": distance,
    }.items():
        if checkpoint_record.get(field) != expected_value:
            raise ValueError(f"checkpoint metadata mismatch: {checkpoint_path}: {field}")
    if summary.get("checkpoint_sha256") != checkpoint_record.get("checkpoint_sha256"):
        raise ValueError(f"summary/checkpoint digest mismatch: {run_dir}")

    tokens_per_step = EXPECTED_SEQUENCE_LENGTH[distance] * EFFECTIVE_BATCH_SIZE
    step_count = 0
    cumulative_cuda = 0.0
    validation_steps: list[int] = []
    with (run_dir / "steps.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("event") != "step":
                continue
            step_count += 1
            step = int(record["step"])
            if step != step_count or int(record["tokens_processed"]) != step * tokens_per_step:
                raise ValueError(f"step progress mismatch in {run_dir} at step {step}")
            cuda_seconds = _finite(record["cuda_elapsed_seconds"], "CUDA duration")
            if cuda_seconds <= 0:
                raise ValueError(f"non-positive CUDA duration in {run_dir}")
            _finite(record["train_loss"], "training loss")
            _finite(record["gradient_norm"], "gradient norm")
            cumulative_cuda += cuda_seconds
            if record.get("validation_loss") is not None:
                validation_steps.append(step)
    if step_count != EXPECTED_STEPS or validation_steps != list(range(100, 2001, 100)):
        raise ValueError(f"step/validation cadence mismatch in {run_dir}")
    if not math.isclose(cumulative_cuda, float(summary["cuda_training_seconds"]), abs_tol=1e-6):
        raise ValueError(f"CUDA total mismatch in {run_dir}")

    history = []
    for record in summary["validation_history"]:
        history.append(
            {
                "step": int(record["step"]),
                "train": _metrics(record["metrics"]["train"], f"{run_dir}:train"),
                "heldout": _metrics(record["metrics"]["heldout"], f"{run_dir}:heldout"),
            }
        )
    if [record["step"] for record in history] != list(range(100, 2001, 100)):
        raise ValueError(f"summary validation cadence mismatch in {run_dir}")
    return {
        "distance": distance,
        "run_dir": str(run_dir.relative_to(REPO_ROOT)),
        "config_path": CONFIGS[distance],
        "config_sha256": CONFIG_SHA256[distance],
        "checkpoint": {
            "path": str(checkpoint_path.relative_to(REPO_ROOT)),
            "sha256": CHECKPOINT_SHA256[distance],
            "verified": True,
        },
        "sequence_length": EXPECTED_SEQUENCE_LENGTH[distance],
        "tokens_processed": EXPECTED_TOKENS[distance],
        "cuda_training_seconds": _finite(summary["cuda_training_seconds"], "CUDA time"),
        "final_train": _metrics(summary["final_train"], f"{run_dir}:final_train"),
        "final_heldout": _metrics(summary["final_heldout"], f"{run_dir}:final_heldout"),
        "validation_history": history,
    }


def _load_control(distance: int) -> dict[str, Any]:
    payload = _load_json(REPO_ROOT / "results/a0_a1_solvability_distances_values8000.json")
    pair = payload["distances"][str(distance)]
    return next(run for run in pair["runs"] if run["architecture"] == "A1")


def _report(result: dict[str, Any]) -> str:
    lines = [
        "# A1 decay-tail intervention on the 8000-value distance axis",
        "",
        "Status: **complete; all three runs and final checkpoints verified**.",
        "",
        f"Frozen intervention commit: `{PROTOCOL_COMMIT}`.",
        "",
        "The intervention changes only fresh A1 state-transition initialization: 75% of each layer remains at the original decay initialization and 25% is split across initial τ=16, 32, 64, and 128 token channels. The original A1 controls remain unchanged.",
        "",
        "| Distance | Control EM | Tail EM | Δ EM | Control loss | Tail loss | Δ loss | Train EM |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for point in result["curve"]:
        lines.append(
            f"| {point['distance']} | {point['control_heldout_em']:.4f} | "
            f"{point['intervention_heldout_em']:.4f} | {point['heldout_em_delta']:+.4f} | "
            f"{point['control_heldout_loss']:.6f} | {point['intervention_heldout_loss']:.6f} | "
            f"{point['heldout_loss_delta']:+.6f} | {point['intervention_train_em']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Immediate task-level result",
            "",
            "All three intervention models reached 100% training exact match, so the initialization change did not prevent memorization. It did not improve held-out exact match on this seed: distance 4 fell from 1.5625% to 0.78125%, while distances 16 and 64 remained at 0%. Target loss was worse at distances 4 and 16, but improved at distance 64 by 0.116425 despite no exact-match gain.",
            "",
            "This is not yet the final mechanistic conclusion. The fixed state probe below is the predeclared discriminator: improved probe retention with unchanged task performance would mean storage improved but query/readout remained limiting; unchanged probe retention would weaken the decay-causality hypothesis.",
            "",
            "## Integrity",
            "",
            "Each run has 2,000 contiguous step records, the expected 100-step validation cadence, matching generator/data hashes, a frozen protocol hash, and a sidecar-verified final checkpoint.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results/a1_decaytail_solvability_values8000.json")
    parser.add_argument("--report", type=Path, default=REPO_ROOT / "docs/A1_DECAYTAIL_SOLVABILITY_VALUES8000_RESULT.md")
    args = parser.parse_args()

    runs = [_verify_run(distance) for distance in DISTANCES]
    curve = []
    for run in runs:
        control = _load_control(run["distance"])
        intervention_heldout = run["final_heldout"]
        control_heldout = control["final_heldout"]
        curve.append(
            {
                "distance": run["distance"],
                "control_heldout_em": float(control_heldout["greedy_exact_match"]),
                "intervention_heldout_em": intervention_heldout["greedy_exact_match"],
                "heldout_em_delta": intervention_heldout["greedy_exact_match"]
                - float(control_heldout["greedy_exact_match"]),
                "control_heldout_loss": float(control_heldout["target_loss"]),
                "intervention_heldout_loss": intervention_heldout["target_loss"],
                "heldout_loss_delta": intervention_heldout["target_loss"]
                - float(control_heldout["target_loss"]),
                "intervention_train_em": run["final_train"]["greedy_exact_match"],
                "cuda_training_seconds": run["cuda_training_seconds"],
            }
        )
    result = {
        "status": "complete",
        "experiment": "a1_decaytail_solvability_values8000",
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_path": PROTOCOL_PATH,
        "protocol_sha256": PROTOCOL_SHA256,
        "initialization": INITIALIZATION,
        "seed": SEED,
        "value_count": VALUE_COUNT,
        "association_count": ASSOCIATION_COUNT,
        "runs": runs,
        "curve": curve,
        "interpretation": {
            "training_memorization": "All intervention runs reached 100% training exact match.",
            "task_result": "The decay-tail initialization did not improve held-out retrieval on the tested seed and distances.",
            "next": "Run the fixed-probe state/decay forensics before deciding whether the tail repaired storage without repairing retrieval, or failed to change the relevant state signature.",
        },
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
