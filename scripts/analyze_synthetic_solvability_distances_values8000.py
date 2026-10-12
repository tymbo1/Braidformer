#!/usr/bin/env python3
"""Verify and aggregate the completed distractor-distance solvability axis."""

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
    _data_hash,
    generator_spec,
)


ARCHITECTURES = ("A0", "A1")
DISTANCES = (4, 16, 64)
SEED = 17
PROTOCOL_COMMIT = "3e8e8623d596d0878d0e7f49dea280d37335fcd7"
SCRIPT_SHA256 = "35768a1d83fc3a75a3362285a2afe1f62c932d5f15af0a00d12ac36e72960bf1"
EXPECTED_PARAMS = {"A0": 41_866_240, "A1": 41_870_336}
EXPECTED_VALUE_COUNT = 8_000
EXPECTED_ASSOCIATION_COUNT = 1
EXPECTED_STEPS = 2_000
EFFECTIVE_BATCH_SIZE = 16
EVAL_STEPS = tuple(range(100, EXPECTED_STEPS + 1, 100))
EXPECTED = {
    4: {
        "sequence_length": 12,
        "tokens_processed": 384_000,
        "generator_version": "synthetic_solvability_distance4_values8000_v1",
        "generator_sha256": "d64e431072089ac7d04ea967dca6e2062eeab64b5013199ceef2f387aee4801d",
        "data_sha256": "156f4f713ac86e59b78fed23b528601e1de254636c92055656b61d7e1de8e220",
        "protocol_path": "docs/A0_A1_SOLVABILITY_DISTANCE4_VALUES8000_PROTOCOL.md",
        "protocol_sha256": "62b163beea87780b26bff186b533db8dfbf3a59109e90d05fdcb5494307538a6",
        "configs": {
            "A0": "configs/braidnano_a0_solvability_distance4_values8000.json",
            "A1": "configs/braidnano_a1_solvability_distance4_values8000.json",
        },
        "config_sha256": {
            "A0": "02f89c9d0561c2fd6933bf8e2baeca1a3f7d2a8f7f337e70b3e792cd9207e41b",
            "A1": "fc7abd875f1c37649721b85059e4fac31e34d5044bac8f1f851927961281b091",
        },
    },
    16: {
        "sequence_length": 24,
        "tokens_processed": 768_000,
        "generator_version": "synthetic_solvability_distance16_values8000_v1",
        "generator_sha256": "57a50a8cc01ea8d36ca93c4a067df754c0a2f36ac77c3d1f6dcb0c110c99ad5e",
        "data_sha256": "47105d980e02bd0163e05a6be326a4a138a07f400358b0c3ccae7366405903ab",
        "protocol_path": "docs/A0_A1_SOLVABILITY_DISTANCE16_VALUES8000_PROTOCOL.md",
        "protocol_sha256": "7d78ec444cbea500f7f998df011a3a11c3a66ab47f23e40973c41987313fc62e",
        "configs": {
            "A0": "configs/braidnano_a0_solvability_distance16_values8000.json",
            "A1": "configs/braidnano_a1_solvability_distance16_values8000.json",
        },
        "config_sha256": {
            "A0": "8ea76d405996dc7d596c073bb1b607e0da43151a10c2097c9d5d39e3a823a8e1",
            "A1": "16c5962b6440ef324b7728c81e07361a13ec1066b349fb69610bdbb07c42e04b",
        },
    },
    64: {
        "sequence_length": 72,
        "tokens_processed": 2_304_000,
        "generator_version": "synthetic_solvability_distance64_values8000_v1",
        "generator_sha256": "3741d5cc25b60dac0605f381f2ca6a87ab5adead7b84b219c871b524dca267a0",
        "data_sha256": "ab184dfe1ce3c3c8f9b219f236c8b2fe09860fb451dc6a26dec997dd32d1c1e0",
        "protocol_path": "docs/A0_A1_SOLVABILITY_DISTANCE64_VALUES8000_PROTOCOL.md",
        "protocol_sha256": "d65503f460e1ecd5efc7329f30d868927cfb86d49ce3213bfa6ca60f9614dc8f",
        "configs": {
            "A0": "configs/braidnano_a0_solvability_distance64_values8000.json",
            "A1": "configs/braidnano_a1_solvability_distance64_values8000.json",
        },
        "config_sha256": {
            "A0": "71cb68054033a70448577344d3d1443f1aca4113471e48d69ace2cdb372d52a8",
            "A1": "d7b23162e6390e9eaefb4d131d72b36b5157a2df8e70ce287fdf779731466d82",
        },
    },
}
BASELINE_RESULT = "results/a0_a1_solvability_values8000.json"
BASELINE_DATA_SHA256 = "bf361a9dd1519f5441bff61a5e26f854adfd72ee4650f3c9525170ab0af376b1"
BASELINE_GENERATOR_VERSION = "synthetic_solvability_values8000_v1"


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


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT.resolve()))


def _metrics(metrics: dict[str, Any], label: str) -> dict[str, Any]:
    if int(metrics.get("examples", -1)) != 128:
        raise ValueError(f"unexpected example count in {label}")
    for field in ("target_loss", "greedy_answer_token_accuracy", "greedy_exact_match"):
        _finite(metrics[field], f"{label}.{field}")
    for field in ("greedy_answer_token_accuracy", "greedy_exact_match"):
        if not 0.0 <= float(metrics[field]) <= 1.0:
            raise ValueError(f"invalid accuracy in {label}: {metrics[field]!r}")
    return {
        "examples": int(metrics["examples"]),
        "target_loss": float(metrics["target_loss"]),
        "greedy_answer_token_accuracy": float(metrics["greedy_answer_token_accuracy"]),
        "greedy_exact_match": float(metrics["greedy_exact_match"]),
    }


def _expected_generator(distance: int) -> tuple[str, str, str]:
    spec = generator_spec(SEED, EXPECTED_VALUE_COUNT, EXPECTED_ASSOCIATION_COUNT, distance)
    return spec["version"], sha256_json(spec), _data_hash(
        SEED, EXPECTED_VALUE_COUNT, EXPECTED_ASSOCIATION_COUNT, distance
    )


def _load_run(distance: int, architecture: str) -> dict[str, Any]:
    expected = EXPECTED[distance]
    run_dir = REPO_ROOT / "results" / "runs" / (
        f"solvability_distance{distance}_values8000_{architecture.lower()}_seed{SEED}"
    )
    summary = _load_json(run_dir / "summary.json")
    if summary.get("status") != "completed":
        raise ValueError(f"run is not complete: {run_dir}")
    if summary.get("architecture") != architecture or int(summary.get("seed", -1)) != SEED:
        raise ValueError(f"run identity mismatch: {run_dir}")
    for field, expected_value in (
        ("association_count", EXPECTED_ASSOCIATION_COUNT),
        ("distractor_distance", distance),
        ("sequence_length", expected["sequence_length"]),
        ("optimizer_steps", EXPECTED_STEPS),
        ("tokens_processed", expected["tokens_processed"]),
        ("value_count", EXPECTED_VALUE_COUNT),
        ("parameter_count", EXPECTED_PARAMS[architecture]),
    ):
        if int(summary.get(field, -1)) != expected_value:
            raise ValueError(f"{field} mismatch: {run_dir}")
    if summary.get("protocol_commit") != PROTOCOL_COMMIT:
        raise ValueError(f"protocol commit mismatch: {run_dir}")
    if summary.get("script_sha256") != SCRIPT_SHA256:
        raise ValueError(f"runner script hash mismatch: {run_dir}")
    if summary.get("protocol_sha256") != expected["protocol_sha256"]:
        raise ValueError(f"protocol hash mismatch: {run_dir}")
    protocol_path = REPO_ROOT / expected["protocol_path"]
    if sha256_file(protocol_path) != expected["protocol_sha256"]:
        raise ValueError(f"working-tree protocol hash mismatch: {protocol_path}")
    if summary.get("config_path") != expected["configs"][architecture]:
        raise ValueError(f"config path mismatch: {run_dir}")
    if summary.get("config_sha256") != expected["config_sha256"][architecture]:
        raise ValueError(f"config hash mismatch: {run_dir}")
    config_path = REPO_ROOT / expected["configs"][architecture]
    if sha256_file(config_path) != expected["config_sha256"][architecture]:
        raise ValueError(f"working-tree config hash mismatch: {config_path}")

    version, generator_hash, data_hash = _expected_generator(distance)
    if version != expected["generator_version"] or generator_hash != expected["generator_sha256"]:
        raise ValueError(f"working-tree generator identity mismatch for distance {distance}")
    if data_hash != expected["data_sha256"]:
        raise ValueError(f"working-tree data identity mismatch for distance {distance}")
    for field, expected_value in (
        ("synthetic_generator_version", expected["generator_version"]),
        ("synthetic_generator_sha256", expected["generator_sha256"]),
        ("synthetic_data_sha256", expected["data_sha256"]),
    ):
        if summary.get(field) != expected_value:
            raise ValueError(f"{field} mismatch: {run_dir}")
    if summary.get("failure") is not None or summary.get("numerical_instability") is not False:
        raise ValueError(f"run reports failure or numerical instability: {run_dir}")

    checkpoint_path = Path(summary["checkpoint"])
    checkpoint_record = verify_checkpoint(checkpoint_path)
    expected_checkpoint = {
        "config_sha256": expected["config_sha256"][architecture],
        "data_manifest_sha256": expected["data_sha256"],
        "seed": SEED,
        "step": EXPECTED_STEPS,
        "tokens_processed": expected["tokens_processed"],
        "parameter_count": EXPECTED_PARAMS[architecture],
        "protocol_commit": PROTOCOL_COMMIT,
        "value_count": EXPECTED_VALUE_COUNT,
        "association_count": EXPECTED_ASSOCIATION_COUNT,
        "distractor_distance": distance,
    }
    for field, expected_value in expected_checkpoint.items():
        if checkpoint_record.get(field) != expected_value:
            raise ValueError(
                f"checkpoint metadata mismatch in {checkpoint_path}: "
                f"{field}={checkpoint_record.get(field)!r}, expected={expected_value!r}"
            )
    if summary.get("checkpoint_sha256") != checkpoint_record.get("checkpoint_sha256"):
        raise ValueError(f"summary/checkpoint hash mismatch: {run_dir}")

    tokens_per_step = expected["sequence_length"] * EFFECTIVE_BATCH_SIZE
    step_count = 0
    cumulative_cuda = 0.0
    validation_steps: set[int] = set()
    with (run_dir / "steps.jsonl").open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            if record.get("event") != "step":
                continue
            step_count += 1
            step = int(record["step"])
            if step != step_count:
                raise ValueError(f"non-contiguous step log at {run_dir}:{line_number}")
            if int(record["tokens_processed"]) != step * tokens_per_step:
                raise ValueError(f"token progress mismatch at {run_dir}:{line_number}")
            _finite(record["train_loss"], "training loss")
            cuda_seconds = _finite(record["cuda_elapsed_seconds"], "CUDA duration")
            if cuda_seconds <= 0:
                raise ValueError(f"non-positive CUDA duration at {run_dir}:{line_number}")
            _finite(record["wall_clock_seconds"], "wall time")
            _finite(record["gradient_norm"], "gradient norm")
            _finite(record["learning_rate"], "learning rate")
            cumulative_cuda += cuda_seconds
            if record.get("validation_loss") is not None:
                validation_steps.add(step)
                _finite(record["validation_loss"], "validation loss")
    if step_count != EXPECTED_STEPS:
        raise ValueError(f"expected {EXPECTED_STEPS} step records in {run_dir}, got {step_count}")
    if sorted(validation_steps) != list(EVAL_STEPS):
        raise ValueError(f"validation cadence mismatch in {run_dir}")
    if not math.isclose(
        cumulative_cuda, float(summary["cuda_training_seconds"]), rel_tol=0.0, abs_tol=1e-6
    ):
        raise ValueError(f"CUDA total mismatch in {run_dir}")

    metric_count = 0
    evaluation_steps: list[int] = []
    with (run_dir / "calibration_metrics.jsonl").open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            metric_count += 1
            if int(record["step"]) != metric_count:
                raise ValueError(f"non-contiguous calibration metric at {run_dir}:{line_number}")
            _finite(record["train_loss"], "calibration metric training loss")
            _finite(record["learning_rate"], "calibration metric learning rate")
            _finite(record["cuda_elapsed_seconds"], "calibration metric CUDA duration")
            _finite(record["wall_clock_seconds"], "calibration metric wall time")
            if record.get("evaluation") is not None:
                evaluation_steps.append(int(record["step"]))
    if metric_count != EXPECTED_STEPS or evaluation_steps != list(EVAL_STEPS):
        raise ValueError(f"calibration metric cadence mismatch in {run_dir}")

    history = []
    for record in summary["validation_history"]:
        step = int(record["step"])
        if step not in EVAL_STEPS:
            raise ValueError(f"unexpected validation step in {run_dir}: {step}")
        history.append(
            {
                "step": step,
                "tokens_processed": step * tokens_per_step,
                "train": _metrics(record["metrics"]["train"], f"{run_dir}:{step}:train"),
                "heldout": _metrics(record["metrics"]["heldout"], f"{run_dir}:{step}:heldout"),
            }
        )
    if [record["step"] for record in history] != list(EVAL_STEPS):
        raise ValueError(f"summary validation history mismatch in {run_dir}")

    return {
        "architecture": architecture,
        "distance": distance,
        "seed": SEED,
        "experiment_id": summary["experiment_id"],
        "run_dir": _relative(run_dir),
        "config_path": summary["config_path"],
        "config_sha256": summary["config_sha256"],
        "parameter_count": EXPECTED_PARAMS[architecture],
        "value_count": EXPECTED_VALUE_COUNT,
        "association_count": EXPECTED_ASSOCIATION_COUNT,
        "distractor_distance": distance,
        "sequence_length": expected["sequence_length"],
        "optimizer_steps": EXPECTED_STEPS,
        "tokens_processed": expected["tokens_processed"],
        "final_train": _metrics(summary["final_train"], f"{run_dir}:final_train"),
        "final_heldout": _metrics(summary["final_heldout"], f"{run_dir}:final_heldout"),
        "cuda_training_seconds": _finite(summary["cuda_training_seconds"], "CUDA training time"),
        "elapsed_wall_seconds": _finite(summary["elapsed_wall_seconds"], "elapsed wall time"),
        "average_tokens_per_second": _finite(summary["average_tokens_per_second"], "throughput"),
        "average_cuda_tokens_per_second": _finite(
            summary["average_cuda_tokens_per_second"], "CUDA throughput"
        ),
        "peak_allocated_vram_bytes": int(summary["peak_allocated_vram_bytes"]),
        "peak_reserved_vram_bytes": int(summary["peak_reserved_vram_bytes"]),
        "validation_history": history,
        "checkpoint": {
            "path": _relative(checkpoint_path),
            "sha256": checkpoint_record["checkpoint_sha256"],
            "bytes": int(checkpoint_record["checkpoint_bytes"]),
            "complete": bool(checkpoint_record["complete"]),
            "verified": True,
        },
        "gradient_checkpointing": summary["gradient_checkpointing"],
        "numerical_instability": False,
        "failure": None,
    }


def _pair(distance: int, runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    a0, a1 = runs["A0"], runs["A1"]
    curve = []
    for left, right in zip(a0["validation_history"], a1["validation_history"]):
        if left["step"] != right["step"]:
            raise ValueError(f"A0/A1 validation cadence mismatch at distance {distance}")
        curve.append(
            {
                "step": left["step"],
                "tokens_processed": left["tokens_processed"],
                "A0_heldout_target_loss": left["heldout"]["target_loss"],
                "A1_heldout_target_loss": right["heldout"]["target_loss"],
                "A1_minus_A0_heldout_target_loss": right["heldout"]["target_loss"]
                - left["heldout"]["target_loss"],
                "A0_heldout_greedy_exact_match": left["heldout"]["greedy_exact_match"],
                "A1_heldout_greedy_exact_match": right["heldout"]["greedy_exact_match"],
                "A0_train_greedy_exact_match": left["train"]["greedy_exact_match"],
                "A1_train_greedy_exact_match": right["train"]["greedy_exact_match"],
            }
        )
    return {
        "distance": distance,
        "sequence_length": a0["sequence_length"],
        "tokens_processed": a0["tokens_processed"],
        "tokens_per_step": a0["sequence_length"] * EFFECTIVE_BATCH_SIZE,
        "runs": [a0, a1],
        "paired": {
            "heldout": {
                "A0": a0["final_heldout"],
                "A1": a1["final_heldout"],
                "A1_minus_A0_target_loss": a1["final_heldout"]["target_loss"]
                - a0["final_heldout"]["target_loss"],
                "A1_minus_A0_greedy_exact_match": a1["final_heldout"]["greedy_exact_match"]
                - a0["final_heldout"]["greedy_exact_match"],
            },
            "train": {
                "A0": a0["final_train"],
                "A1": a1["final_train"],
                "A1_minus_A0_target_loss": a1["final_train"]["target_loss"]
                - a0["final_train"]["target_loss"],
                "A1_minus_A0_greedy_exact_match": a1["final_train"]["greedy_exact_match"]
                - a0["final_train"]["greedy_exact_match"],
            },
            "resources": {
                "A0": {
                    key: a0[key]
                    for key in (
                        "cuda_training_seconds",
                        "elapsed_wall_seconds",
                        "average_tokens_per_second",
                        "average_cuda_tokens_per_second",
                    )
                },
                "A1": {
                    key: a1[key]
                    for key in (
                        "cuda_training_seconds",
                        "elapsed_wall_seconds",
                        "average_tokens_per_second",
                        "average_cuda_tokens_per_second",
                    )
                },
                "A1_to_A0_cuda_time_ratio": a1["cuda_training_seconds"]
                / a0["cuda_training_seconds"],
                "A1_to_A0_average_throughput_ratio": a1["average_tokens_per_second"]
                / a0["average_tokens_per_second"],
            },
        },
        "validation_curve": curve,
    }


def _baseline() -> dict[str, Any]:
    payload = _load_json(REPO_ROOT / BASELINE_RESULT)
    if payload.get("status") != "complete":
        raise ValueError("completed distance-0 baseline result is not marked complete")
    if payload.get("protocol_path") != "docs/A0_A1_SOLVABILITY_VALUES8000_PROTOCOL.md":
        raise ValueError("unexpected distance-0 baseline protocol")
    expected_data_hash = _data_hash(SEED, EXPECTED_VALUE_COUNT, EXPECTED_ASSOCIATION_COUNT, 0)
    if expected_data_hash != BASELINE_DATA_SHA256:
        raise ValueError("working-tree distance-0 generator no longer matches the baseline")
    runs = {run["architecture"]: run for run in payload["runs"]}
    if set(runs) != set(ARCHITECTURES):
        raise ValueError("distance-0 baseline does not contain both architectures")
    for architecture in ARCHITECTURES:
        run = runs[architecture]
        if run.get("synthetic_data_sha256", BASELINE_DATA_SHA256) != BASELINE_DATA_SHA256:
            raise ValueError(f"distance-0 baseline data hash mismatch for {architecture}")
        if run.get("synthetic_generator_version", BASELINE_GENERATOR_VERSION) != BASELINE_GENERATOR_VERSION:
            raise ValueError(f"distance-0 baseline generator mismatch for {architecture}")
        if int(run.get("value_count", EXPECTED_VALUE_COUNT)) != EXPECTED_VALUE_COUNT:
            raise ValueError(f"distance-0 baseline value count mismatch for {architecture}")
    paired = payload["paired"]
    resources = paired["resources"]
    return {
        "distance": 0,
        "sequence_length": 8,
        "tokens_processed": 256_000,
        "tokens_per_step": 128,
        "runs": [runs["A0"], runs["A1"]],
        "paired": paired,
        "validation_curve": payload.get("validation_curve", []),
        "source_result": BASELINE_RESULT,
        "source_protocol_commit": payload["protocol_commit"],
        "resources": resources,
    }


def _aggregate(pairs: dict[int, dict[str, Any]], baseline: dict[str, Any]) -> dict[str, Any]:
    all_pairs = {0: baseline, **pairs}
    return {
        "status": "complete",
        "experiment": "a0_a1_synthetic_solvability_distractor_distances_values8000",
        "protocol_commit": PROTOCOL_COMMIT,
        "protocols": {
            str(distance): {
                "path": EXPECTED[distance]["protocol_path"],
                "sha256": EXPECTED[distance]["protocol_sha256"],
            }
            for distance in DISTANCES
        },
        "seed": SEED,
        "value_count": EXPECTED_VALUE_COUNT,
        "association_count": EXPECTED_ASSOCIATION_COUNT,
        "architectures": list(ARCHITECTURES),
        "baseline_distance0": baseline,
        "distances": {str(distance): pairs[distance] for distance in DISTANCES},
        "final_curve": [
            {
                "distance": distance,
                "sequence_length": all_pairs[distance]["sequence_length"],
                "tokens_processed": all_pairs[distance]["tokens_processed"],
                "A0_heldout_greedy_exact_match": all_pairs[distance]["paired"]["heldout"]["A0"][
                    "greedy_exact_match"
                ],
                "A1_heldout_greedy_exact_match": all_pairs[distance]["paired"]["heldout"]["A1"][
                    "greedy_exact_match"
                ],
                "A0_heldout_target_loss": all_pairs[distance]["paired"]["heldout"]["A0"][
                    "target_loss"
                ],
                "A1_heldout_target_loss": all_pairs[distance]["paired"]["heldout"]["A1"][
                    "target_loss"
                ],
                "A1_minus_A0_heldout_target_loss": all_pairs[distance]["paired"]["heldout"][
                    "A1_minus_A0_target_loss"
                ],
                "A1_to_A0_cuda_time_ratio": all_pairs[distance]["paired"]["resources"][
                    "A1_to_A0_cuda_time_ratio"
                ],
            }
            for distance in (0, *DISTANCES)
        ],
        "interpretation": {
            "training_memorization": "Both architectures reached 100% autoregressive greedy exact match on the repeated training set at every distance rung.",
            "distance_axis": "A0 held-out exact match and target loss remain broadly flat across distances 4, 16, and 64, while A1 held-out exact match falls to zero and target loss worsens as distance increases.",
            "binding_vs_retention": "The association axis established a binding/interference weakness for both models; this distance axis adds evidence of a stronger long-range retention or state-utilization weakness in the present A1 implementation.",
            "token_budget_limitation": "Distance rungs use the same optimizer-step budget but longer sequences and therefore more token positions; they are matched-step retention probes, not equal-token comparisons.",
            "scope": "This is one seed and one model size. It does not identify the internal mechanism or authorize an A1 architectural change.",
            "next_axis": "Keep the A0/A1 architecture frozen and proceed to the predeclared target-length axis or non-invasive A1 state/decay forensics; defer relational curriculum and A2 until this baseline is recorded.",
        },
        "raw_artifacts": "Ignored results/runs directories; not tracked by Git.",
    }


def _report(aggregate: dict[str, Any]) -> str:
    lines = [
        "# BraidFormer A0/A1 distractor-distance solvability result",
        "",
        "Status: **complete**. This records the one-association, 8000-value distractor-distance axis after the completed value-space and association-count tests.",
        "",
        f"Frozen execution commit: `{PROTOCOL_COMMIT}`.",
        "",
        "## Final distance curve",
        "",
        "The distance-0 row is the previously completed one-association values8000 baseline. Distances 4, 16, and 64 are the fresh runs in this axis.",
        "",
        "| Distractor distance | Sequence length | Token positions | A0 held-out EM | A1 held-out EM | A0 held-out loss | A1 held-out loss | A1 − A0 loss | A1/A0 CUDA |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for point in aggregate["final_curve"]:
        lines.append(
            f"| {point['distance']} | {point['sequence_length']} | {point['tokens_processed']:,} | "
            f"{point['A0_heldout_greedy_exact_match']:.4f} | "
            f"{point['A1_heldout_greedy_exact_match']:.4f} | "
            f"{point['A0_heldout_target_loss']:.6f} | "
            f"{point['A1_heldout_target_loss']:.6f} | "
            f"{point['A1_minus_A0_heldout_target_loss']:+.6f} | "
            f"{point['A1_to_A0_cuda_time_ratio']:.3f}× |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Both unchanged 42M models reached 100% training exact match at every distance rung. The distance manipulation therefore did not create a training-memorization failure.",
            "",
            "A0 remained essentially flat: its held-out exact match was 4.69% at distances 4, 16, and 64, with target loss 9.7791, 9.6737, and 9.7342. A1 fell from 1.56% at distance 4 to 0% at distances 16 and 64, while its target loss rose from 10.4665 to 10.8242 and 10.9740.",
            "",
            "This separates the present behaviors more clearly than the association axis alone. Both models still show the previously observed binding/generalization weakness, but A1 also shows a distance-sensitive degradation under this one-seed, fixed-step protocol. The result is consistent with a recurrent-state retention or state-utilization weakness; it does not by itself distinguish decay, interference, optimization, or readout failure.",
            "",
            "The distance-0 baseline is not an equal-token control for the longer arms. Every fresh rung keeps 2,000 optimizer steps and effective batch 16, so longer distances process more token positions. The comparison is therefore a matched-step retention probe, with actual CUDA time and token exposure reported explicitly.",
            "",
            "## Training memorization and resource summary",
            "",
            "| Distance | A0 train EM | A1 train EM | A0 CUDA seconds | A1 CUDA seconds |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    all_pairs = {0: aggregate["baseline_distance0"], **{
        int(distance): pair for distance, pair in aggregate["distances"].items()
    }}
    for distance in (0, *DISTANCES):
        pair = all_pairs[distance]
        runs = {run["architecture"]: run for run in pair["runs"]}
        lines.append(
            f"| {distance} | {runs['A0']['final_train']['greedy_exact_match']:.4f} | "
            f"{runs['A1']['final_train']['greedy_exact_match']:.4f} | "
            f"{runs['A0']['cuda_training_seconds']:.3f} | "
            f"{runs['A1']['cuda_training_seconds']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Selected validation checkpoints",
            "",
            "| Distance | Step | A0 held-out loss | A1 held-out loss | A1 − A0 | A0 exact | A1 exact |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for distance in (0, *DISTANCES):
        pair = all_pairs[distance]
        for point in pair["validation_curve"]:
            if point["step"] in {100, 500, 1_000, 1_500, 2_000}:
                lines.append(
                    f"| {distance} | {point['step']:,} | "
                    f"{point.get('A0_heldout_target_loss', point.get('A0_heldout_target_loss')):.6f} | "
                    f"{point.get('A1_heldout_target_loss', point.get('A1_heldout_target_loss')):.6f} | "
                    f"{point.get('A1_minus_A0_heldout_target_loss', 0.0):+.6f} | "
                    f"{point.get('A0_heldout_greedy_exact_match', 0.0):.4f} | "
                    f"{point.get('A1_heldout_greedy_exact_match', 0.0):.4f} |"
                )
    lines.extend(
        [
            "",
            "## Integrity and scope",
            "",
            "All six fresh runs completed with finite losses and gradients, no CUDA/OOM failure, and SHA-256-verified final checkpoints containing optimizer and RNG state. The verifier checked every optimizer step, token counter, evaluation/checkpoint cadence, config hash, protocol hash, generator hash, data hash, checkpoint metadata, and final checkpoint digest.",
            "",
            "- Fresh raw artifacts remain in ignored `results/runs/` directories.",
            "- The distance-0 baseline remains the tracked result in `results/a0_a1_solvability_values8000.json`.",
            "- No A1 architecture change, relational curriculum, A2 store, predictive-learning branch, scaling run, or memory work was introduced.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "a0_a1_solvability_distances_values8000.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "docs" / "A0_A1_SOLVABILITY_DISTANCES_VALUES8000_RESULT.md",
    )
    args = parser.parse_args()
    pairs = {
        distance: _pair(
            distance,
            {architecture: _load_run(distance, architecture) for architecture in ARCHITECTURES},
        )
        for distance in DISTANCES
    }
    aggregate = _aggregate(pairs, _baseline())
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    report = args.report if args.report.is_absolute() else REPO_ROOT / args.report
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_report(aggregate), encoding="utf-8")
    print(json.dumps({"output": str(output), "report": str(report), "runs": 6}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
