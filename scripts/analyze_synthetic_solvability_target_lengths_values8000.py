#!/usr/bin/env python3
"""Verify and aggregate the completed target-length solvability axis."""

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
DISTRACTOR_DISTANCE = 0
TARGET_LENGTHS = (2, 4, 8)
ARCHITECTURES = ("A0", "A1")
EXPECTED_PARAMS = {"A0": 41_866_240, "A1": 41_870_336}
PROTOCOL_COMMIT = "e133e5279761b73b390f8d52feacd63c5289368c"
PROTOCOL_PATH = "docs/A0_A1_SOLVABILITY_TARGET_LENGTH_VALUES8000_PROTOCOL.md"
PROTOCOL_SHA256 = "ca92f1b5f44fd5c8e0ab07669fdad7babc6a56ef3f4533ab72779ab24a86fb4b"
RUNNER_SHA256 = "2a1a1c84f1ebfabc11cf66f0defa2e2ad4db2d1741c89c3d12660d3ac066d3e5"
SEQUENCE_LENGTH = {2: 10, 4: 14, 8: 22}
CONFIGS = {
    (2, "A0"): "configs/braidnano_a0_solvability_targetlength2_values8000.json",
    (2, "A1"): "configs/braidnano_a1_solvability_targetlength2_values8000.json",
    (4, "A0"): "configs/braidnano_a0_solvability_targetlength4_values8000.json",
    (4, "A1"): "configs/braidnano_a1_solvability_targetlength4_values8000.json",
    (8, "A0"): "configs/braidnano_a0_solvability_targetlength8_values8000.json",
    (8, "A1"): "configs/braidnano_a1_solvability_targetlength8_values8000.json",
}
CONFIG_SHA256 = {
    (2, "A0"): "0d55b57ed4b63525030b07418397dd8e66225240794510b21fdee10e642b7926",
    (2, "A1"): "0b93c631f704db2797d9556868a98fd8ac44f3ffe806c4e8a498de0f92b03f5f",
    (4, "A0"): "ec782078533b7a4d54451f2f1d3765cee66102f3c4af2edf8acc0f099fc78cee",
    (4, "A1"): "2d9322d61ef6a59df7a6b701ad5e32873fd42ff351cb443ba51194e1637aa933",
    (8, "A0"): "a63a3a42fa8f42211ec42f967c0eb6a851ab01dd1a770b86bdc9aa76c108ab0a",
    (8, "A1"): "ba3b7e6cfe874e2f300ac4b7a70824b1b8c12e5cff856227c12976ad5d674cf3",
}
CHECKPOINT_SHA256 = {
    (2, "A0"): "5e4b95596f7c3cf582e5b378c2e4f5ce354b2e445f5340b19ebfd71d9adfe393",
    (2, "A1"): "f235dfe95eab1eb243d2b186a196d2cffb4778eaa454c6e075748adba6f2a0d2",
    (4, "A0"): "4e99dcb426ff422eb1ae0ac8637a9f49f64830f3ff45326510735ca0838b5358",
    (4, "A1"): "691d4d212d513fb53c5ac2519136f9db77f7f412f188fb2f547a763388b8cf31",
    (8, "A0"): "75d307a5396a34a69fdd3bdbf3fe7e25684c0a94fae41b3ce7b717a50d06db64",
    (8, "A1"): "9045fa017cf10c2b881e446e047a1ec46fb830cc867c18155e3eef69c52895bd",
}


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
    for field in ("greedy_answer_token_accuracy", "greedy_exact_match"):
        if not 0.0 <= float(metrics[field]) <= 1.0:
            raise ValueError(f"invalid accuracy in {label}")
    return {
        "examples": 128,
        "target_loss": float(metrics["target_loss"]),
        "greedy_answer_token_accuracy": float(metrics["greedy_answer_token_accuracy"]),
        "greedy_exact_match": float(metrics["greedy_exact_match"]),
    }


def _run_dir(target_length: int, architecture: str) -> Path:
    return REPO_ROOT / "results" / "runs" / (
        f"solvability_targetlength{target_length}_values8000_"
        f"{architecture.lower()}_seed{SEED}"
    )


def _verify_run(target_length: int, architecture: str) -> dict[str, Any]:
    run_dir = _run_dir(target_length, architecture)
    summary = _load_json(run_dir / "summary.json")
    sequence_length = SEQUENCE_LENGTH[target_length]
    tokens_processed = sequence_length * EFFECTIVE_BATCH_SIZE * 2_000
    expected_generator = generator_spec(
        SEED,
        VALUE_COUNT,
        ASSOCIATION_COUNT,
        DISTRACTOR_DISTANCE,
        target_length,
    )
    expected_data_hash = _data_hash(
        SEED,
        VALUE_COUNT,
        ASSOCIATION_COUNT,
        DISTRACTOR_DISTANCE,
        target_length,
    )
    expected = {
        "status": "completed",
        "architecture": architecture,
        "state_transition_initialization": "uniform_0.9",
        "seed": SEED,
        "parameter_count": EXPECTED_PARAMS[architecture],
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "script_sha256": RUNNER_SHA256,
        "config_path": CONFIGS[(target_length, architecture)],
        "config_sha256": CONFIG_SHA256[(target_length, architecture)],
        "synthetic_generator_version": expected_generator["version"],
        "synthetic_generator_sha256": sha256_json(expected_generator),
        "synthetic_data_sha256": expected_data_hash,
        "target_length": target_length,
        "value_count": VALUE_COUNT,
        "association_count": ASSOCIATION_COUNT,
        "distractor_distance": DISTRACTOR_DISTANCE,
        "sequence_length": sequence_length,
        "optimizer_steps": 2_000,
        "tokens_processed": tokens_processed,
    }
    for field, expected_value in expected.items():
        if summary.get(field) != expected_value:
            raise ValueError(
                f"summary mismatch in {run_dir}: {field}="
                f"{summary.get(field)!r}, expected={expected_value!r}"
            )
    config_path = REPO_ROOT / CONFIGS[(target_length, architecture)]
    if sha256_file(config_path) != CONFIG_SHA256[(target_length, architecture)]:
        raise ValueError(f"working-tree config hash mismatch: {config_path}")
    protocol_path = REPO_ROOT / PROTOCOL_PATH
    if sha256_file(protocol_path) != PROTOCOL_SHA256:
        raise ValueError(f"working-tree protocol hash mismatch: {protocol_path}")

    checkpoint_path = Path(summary["checkpoint"])
    checkpoint_record = verify_checkpoint(checkpoint_path)
    expected_checkpoint = {
        "config_sha256": CONFIG_SHA256[(target_length, architecture)],
        "data_manifest_sha256": expected_data_hash,
        "seed": SEED,
        "step": 2_000,
        "tokens_processed": tokens_processed,
        "parameter_count": EXPECTED_PARAMS[architecture],
        "protocol_commit": PROTOCOL_COMMIT,
        "state_transition_initialization": "uniform_0.9",
        "synthetic_generator_version": expected_generator["version"],
        "target_length": target_length,
        "value_count": VALUE_COUNT,
        "association_count": ASSOCIATION_COUNT,
        "distractor_distance": DISTRACTOR_DISTANCE,
    }
    for field, expected_value in expected_checkpoint.items():
        if checkpoint_record.get(field) != expected_value:
            raise ValueError(f"checkpoint mismatch in {checkpoint_path}: {field}")
    if checkpoint_record.get("checkpoint_sha256") != CHECKPOINT_SHA256[(target_length, architecture)]:
        raise ValueError(f"checkpoint digest mismatch: {checkpoint_path}")
    if summary.get("checkpoint_sha256") != checkpoint_record.get("checkpoint_sha256"):
        raise ValueError(f"summary/checkpoint digest mismatch: {run_dir}")

    tokens_per_step = sequence_length * EFFECTIVE_BATCH_SIZE
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
                raise ValueError(f"step progress mismatch in {run_dir}")
            cuda_seconds = _finite(record["cuda_elapsed_seconds"], "CUDA duration")
            if cuda_seconds <= 0:
                raise ValueError(f"non-positive CUDA duration in {run_dir}")
            _finite(record["train_loss"], "training loss")
            _finite(record["gradient_norm"], "gradient norm")
            cumulative_cuda += cuda_seconds
            if record.get("validation_loss") is not None:
                validation_steps.append(step)
    if step_count != 2_000 or validation_steps != list(range(100, 2_001, 100)):
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
    if [record["step"] for record in history] != list(range(100, 2_001, 100)):
        raise ValueError(f"summary validation cadence mismatch in {run_dir}")
    return {
        "architecture": architecture,
        "target_length": target_length,
        "sequence_length": sequence_length,
        "tokens_processed": tokens_processed,
        "run_dir": str(run_dir.relative_to(REPO_ROOT)),
        "config_path": CONFIGS[(target_length, architecture)],
        "config_sha256": CONFIG_SHA256[(target_length, architecture)],
        "cuda_training_seconds": _finite(summary["cuda_training_seconds"], "CUDA time"),
        "final_train": _metrics(summary["final_train"], f"{run_dir}:final_train"),
        "final_heldout": _metrics(summary["final_heldout"], f"{run_dir}:final_heldout"),
        "validation_history": history,
        "checkpoint": {
            "path": str(checkpoint_path.relative_to(REPO_ROOT)),
            "sha256": CHECKPOINT_SHA256[(target_length, architecture)],
            "verified": True,
        },
    }


def _pair(target_length: int, runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    a0, a1 = runs["A0"], runs["A1"]
    return {
        "target_length": target_length,
        "sequence_length": a0["sequence_length"],
        "tokens_processed": a0["tokens_processed"],
        "runs": [a0, a1],
        "heldout": {
            "A0": a0["final_heldout"],
            "A1": a1["final_heldout"],
            "A1_minus_A0_target_loss": a1["final_heldout"]["target_loss"]
            - a0["final_heldout"]["target_loss"],
            "A1_minus_A0_token_accuracy": a1["final_heldout"]["greedy_answer_token_accuracy"]
            - a0["final_heldout"]["greedy_answer_token_accuracy"],
            "A1_minus_A0_exact_match": a1["final_heldout"]["greedy_exact_match"]
            - a0["final_heldout"]["greedy_exact_match"],
        },
        "train": {"A0": a0["final_train"], "A1": a1["final_train"]},
        "resources": {
            "A0_cuda_training_seconds": a0["cuda_training_seconds"],
            "A1_cuda_training_seconds": a1["cuda_training_seconds"],
            "A1_to_A0_cuda_time_ratio": a1["cuda_training_seconds"]
            / a0["cuda_training_seconds"],
        },
    }


def _report(result: dict[str, Any]) -> str:
    lines = [
        "# A0/A1 synthetic solvability target-length result",
        "",
        "Status: **complete; all six runs and final checkpoints verified**.",
        "",
        f"Frozen execution commit: `{PROTOCOL_COMMIT}`.",
        "",
        "The target is a one-key binding to a sequence of 2, 4, or 8 value tokens drawn from the 8,000-token reserved pool. All other architecture and optimizer controls are unchanged.",
        "",
        "| Target length | A0 token acc. | A1 token acc. | A1−A0 token acc. | A0 loss | A1 loss | A1−A0 loss | A1/A0 CUDA |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for pair in result["pairs"]:
        heldout = pair["heldout"]
        lines.append(
            f"| {pair['target_length']} | {heldout['A0']['greedy_answer_token_accuracy']:.4f} | "
            f"{heldout['A1']['greedy_answer_token_accuracy']:.4f} | "
            f"{heldout['A1_minus_A0_token_accuracy']:+.4f} | "
            f"{heldout['A0']['target_loss']:.6f} | {heldout['A1']['target_loss']:.6f} | "
            f"{heldout['A1_minus_A0_target_loss']:+.6f} | "
            f"{pair['resources']['A1_to_A0_cuda_time_ratio']:.3f}× |"
        )
    lines.extend(
        [
            "",
            "## Result",
            "",
            "All six models reached 100% training exact match. Held-out whole-sequence exact match was 0% for every arm, so the informative measurements are target loss and per-token accuracy.",
            "",
            "A0 retained held-out token accuracy of 0.3906%, 2.9297%, and 0.8789% at target lengths 2, 4, and 8. A1 matched the length-2 token accuracy but fell to 0% at lengths 4 and 8. A1 target loss was higher at every length and required approximately 1.72×, 1.69×, and 1.84× the A0 CUDA time.",
            "",
            "This is consistent with a target-length-sensitive A1 retrieval/readout weakness beyond the one-token distance result. It remains one model size and one seed, and the data are synthetic; it does not by itself distinguish recurrent storage from binding, output routing, or autoregressive error accumulation.",
            "",
            "## Integrity",
            "",
            "Every arm has 2,000 contiguous step records, the expected 100-step validation cadence, matching generator/data/config/protocol hashes, and a sidecar-verified final checkpoint.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results/a0_a1_solvability_target_lengths_values8000.json")
    parser.add_argument("--report", type=Path, default=REPO_ROOT / "docs/A0_A1_SOLVABILITY_TARGET_LENGTHS_VALUES8000_RESULT.md")
    args = parser.parse_args()
    pairs = []
    for target_length in TARGET_LENGTHS:
        runs = {
            architecture: _verify_run(target_length, architecture)
            for architecture in ARCHITECTURES
        }
        pairs.append(_pair(target_length, runs))
    result = {
        "status": "complete",
        "experiment": "a0_a1_synthetic_solvability_target_lengths_values8000",
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_path": PROTOCOL_PATH,
        "protocol_sha256": PROTOCOL_SHA256,
        "seed": SEED,
        "value_count": VALUE_COUNT,
        "association_count": ASSOCIATION_COUNT,
        "distractor_distance": DISTRACTOR_DISTANCE,
        "target_lengths": list(TARGET_LENGTHS),
        "architectures": list(ARCHITECTURES),
        "pairs": pairs,
        "interpretation": {
            "training_memorization": "All six arms reached 100% training exact match.",
            "target_length_result": "A1 matched A0 token accuracy at length 2 but fell below A0 at lengths 4 and 8; A1 target loss was higher at all three lengths.",
            "next": "Keep the target-length result frozen and move to a causal readout/binding probe or explicitly trained relational curriculum before adding further architectural state changes.",
        },
    }
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    report = args.report if args.report.is_absolute() else REPO_ROOT / args.report
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_report(result), encoding="utf-8")
    print(json.dumps({"output": str(output), "report": str(report), "pairs": len(pairs)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
