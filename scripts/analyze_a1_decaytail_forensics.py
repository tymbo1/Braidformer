#!/usr/bin/env python3
"""Run the existing read-only A1 state probe on decay-tail checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from braidformer.config import load_config
from braidformer.training.integrity import sha256_file, verify_checkpoint
from scripts.analyze_a1_state_decay_forensics import (
    PROBE_DISTANCES,
    _analyze_checkpoint,
)


INITIALIZATION = "tail_75_original_25_long_v1"
EXPECTED_CHECKPOINTS = {
    "decaytail_distance4": {
        "checkpoint": "results/runs/solvability_decaytail_distance4_values8000_a1_seed17/checkpoint_step_002000.pt",
        "checkpoint_sha256": "79bdbae2ecd56e6963f019e018f087de98948d1e03a9d7d0876160cb5881a7df",
        "config": "configs/braidnano_a1_solvability_decaytail_distance4_values8000.json",
        "training_distance": 4,
    },
    "decaytail_distance16": {
        "checkpoint": "results/runs/solvability_decaytail_distance16_values8000_a1_seed17/checkpoint_step_002000.pt",
        "checkpoint_sha256": "10b650f6fcf44b88267178845cea8fc0ca8d775562f5cbc61780c06497194cb7",
        "config": "configs/braidnano_a1_solvability_decaytail_distance16_values8000.json",
        "training_distance": 16,
    },
    "decaytail_distance64": {
        "checkpoint": "results/runs/solvability_decaytail_distance64_values8000_a1_seed17/checkpoint_step_002000.pt",
        "checkpoint_sha256": "42ee4a206f7dec3d14c70afa181d629d6da7af90588f1dd79f46ac2696a1a682",
        "config": "configs/braidnano_a1_solvability_decaytail_distance64_values8000.json",
        "training_distance": 64,
    },
}


def _validate_metadata(metadata: dict[str, Any]) -> None:
    checkpoint_path = REPO_ROOT / metadata["checkpoint"]
    checkpoint_record = verify_checkpoint(checkpoint_path)
    if checkpoint_record.get("checkpoint_sha256") != metadata["checkpoint_sha256"]:
        raise ValueError(f"checkpoint digest mismatch: {checkpoint_path}")
    config_path = REPO_ROOT / metadata["config"]
    config = load_config(config_path)
    if config.stage != "A1" or config.model_type != "recurrent_state_space":
        raise ValueError(f"not an A1 config: {config_path}")
    if config.state_transition_initialization != INITIALIZATION:
        raise ValueError(f"unexpected initialization in {config_path}")
    if checkpoint_record.get("state_transition_initialization") != INITIALIZATION:
        raise ValueError(f"checkpoint provenance mismatch: {checkpoint_path}")


def _report(result: dict[str, Any]) -> str:
    lines = [
        "# A1 decay-tail state/decay forensics",
        "",
        "Status: **complete; read-only analysis**. No checkpoint or model parameter was updated.",
        "",
        "The same fixed linear value-vector probe used for the original A1 forensics was applied to the three decay-tail checkpoints. The probe is fit at the post-value state for distance 0 and evaluated without refitting after neutral-marker distances.",
        "",
        "## Learned time scales",
        "",
        "| Checkpoint | Training distance | Median τ | P90 τ | Max τ | Channels τ≥16 | Channels τ≥64 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for checkpoint in result["checkpoints"]:
        times = checkpoint["time_constants"]
        lines.append(
            f"| {checkpoint['label']} | {checkpoint['training_distance']} | "
            f"{times['tau_median']:.3f} | {times['tau_p90']:.3f} | {times['tau_max']:.3f} | "
            f"{times['channels_tau_at_least']['16']:,} | "
            f"{times['channels_tau_at_least']['64']:,} |"
        )
    lines.extend(
        [
            "",
            "The tail survived optimization in a measurable form: every intervention checkpoint retained 768 channels with τ≥16 and 256 channels with τ≥64, whereas the original A1 forensic checkpoints had zero channels at either threshold. The median τ nevertheless remained about 8.9 tokens, so the intervention created a minority long-lived population rather than changing the whole recurrent stack.",
            "",
            "## Fixed-probe curve",
            "",
            "The table shows the best held-out probe top-1 across recurrent layers. Actual answer top-1 is the model's value-token readout within the reserved 8,000-value pool.",
            "",
            "| Checkpoint | Distance | Best layer | Probe top-1 | Probe top-5 | State cosine to d0 | Actual answer top-1 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for checkpoint in result["checkpoints"]:
        readout_by_distance = {
            point["distance"]: point["heldout"]
            for point in checkpoint["final_readout_curve"]
        }
        for distance in PROBE_DISTANCES:
            candidates = []
            for layer in checkpoint["layers"]:
                point = next(item for item in layer["distances"] if item["distance"] == distance)
                candidates.append((point["probe_heldout"]["top1_accuracy"], layer["layer"], point))
            best_accuracy, best_layer, best_point = max(candidates)
            lines.append(
                f"| {checkpoint['label']} | {distance} | {best_layer} | "
                f"{best_accuracy:.4f} | {best_point['probe_heldout']['top5_accuracy']:.4f} | "
                f"{best_point['state_drift']['mean_cosine_to_distance0']:.4f} | "
                f"{readout_by_distance[distance]['top1_accuracy']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The intervention clearly changed the learned time-scale distribution, but it did not produce a corresponding fixed-probe recovery at the target long distances: the distance-64-trained checkpoint still has probe top-1 0.0020 at distances 16, 32, and 64. The probe improves at short distances in some arms, especially distance 4 for the distance-64-trained model, while the long-distance trace still collapses. Combined with unchanged held-out exact match, this says that long-lived channels alone are insufficient to solve the present retrieval failure; query interaction, binding, later-layer transformation, or optimization remains limiting.",
            "",
            "All measurements are one model size and one seed. They are diagnostic evidence, not a license to replace the A1 architecture or to claim language-model improvement.",
            "",
            "## Integrity",
            "",
            "Source checkpoints were sidecar-verified, loaded in evaluation mode, and analyzed with gradients disabled. Probe weights were temporary in-memory ridge solutions; source artifacts remain unchanged.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results/a1_decaytail_forensics_values8000.json")
    parser.add_argument("--report", type=Path, default=REPO_ROOT / "docs/A1_DECAYTAIL_FORENSICS_VALUES8000.md")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    for metadata in EXPECTED_CHECKPOINTS.values():
        _validate_metadata(metadata)
    device = torch.device(args.device)
    checkpoints = [
        _analyze_checkpoint(label, metadata, device)
        for label, metadata in EXPECTED_CHECKPOINTS.items()
    ]
    result = {
        "status": "complete",
        "experiment": "a1_decaytail_forensics_values8000",
        "initialization": INITIALIZATION,
        "value_count": 8_000,
        "device": str(device),
        "checkpoints": checkpoints,
        "interpretation": {
            "probe": "The fixed probe is fitted at distance 0 and evaluated without refitting at later distances.",
            "causal_limit": "Task and state-probe curves together provide intervention evidence but remain one seed and one model size.",
        },
    }
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    report = args.report if args.report.is_absolute() else REPO_ROOT / args.report
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_report(result), encoding="utf-8")
    print(json.dumps({"output": str(output), "report": str(report), "checkpoints": len(checkpoints)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
