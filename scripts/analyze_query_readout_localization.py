#!/usr/bin/env python3
"""Read-only query/readout localization for frozen A0/A1 checkpoints.

The analysis compares native teacher-forced logits with cross-validated
diagnostic value readouts from pre-query, post-query, and target-position
representations. It never updates a model parameter or writes a checkpoint.
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
from scripts.analyze_a1_state_decay_forensics import F_normalize
from scripts.analyze_layerwise_representation_forensics import (
    _capture_batch,
    _capture_sites,
)
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
    KEY_HELDOUT_BASE,
    VALUE_BASE,
    make_evaluation_examples,
)


VALUE_COUNT = 8_000
ASSOCIATION_COUNT = 1
DISTRACTOR_DISTANCE = 0
BATCH_SIZE = 16
FOLD_COUNT = 4
EXPECTED_PARAMS = {"A0": 41_866_240, "A1": 41_870_336}
PLAN_PATH = "docs/A0_A1_QUERY_READOUT_LOCALIZATION_PLAN.md"
TARGET_RUNNER_PATH = "scripts/run_synthetic_overfit_calibration.py"
OUTPUT_VERSION = "query_readout_localization_v1"
RIDGE_STRENGTH = 1e-3


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
    parameter_count = model.parameter_count()  # type: ignore[attr-defined]
    if parameter_count != EXPECTED_PARAMS[architecture]:
        raise ValueError(f"unexpected {architecture} parameter count: {parameter_count}")
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
    return torch.tensor(targets, dtype=torch.long) - VALUE_BASE


def _collect_features_and_logits(
    model: torch.nn.Module,
    architecture: str,
    examples: list[Any],
    target_length: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, list[str], list[int]]:
    site_names, positions = _capture_sites(target_length)
    target_positions = positions[2:]
    feature_parts: dict[str, list[torch.Tensor]] = {}
    logits_parts: list[torch.Tensor] = []
    for start in range(0, len(examples), BATCH_SIZE):
        batch = examples[start : start + BATCH_SIZE]
        inputs = torch.tensor(
            [example.input_ids for example in batch],
            dtype=torch.long,
            device=device,
        )
        batch_features = _capture_batch(model, inputs, architecture, positions)
        for name, value in batch_features.items():
            feature_parts.setdefault(name, []).append(value)
        with torch.inference_mode(), _autocast(device):
            logits = model(inputs).logits[:, target_positions, :].float().cpu()
        logits_parts.append(_finite_tensor(logits, "target-output logits"))

    features = {name: torch.cat(parts, dim=0) for name, parts in feature_parts.items()}
    logits = torch.cat(logits_parts, dim=0)
    if len(features) == 0 or any(value.shape[0] != len(examples) for value in features.values()):
        raise RuntimeError("feature collection count mismatch")
    if logits.shape[:2] != (len(examples), target_length):
        raise RuntimeError(f"logit shape mismatch: {tuple(logits.shape)}")
    if sorted(features) != sorted(
        [f"block_output_{index}" for index in range(len(model.blocks))]
        + ([f"recurrent_state_{index}" for index in range(len(model.blocks))] if architecture == "A1" else [])
        + ["final_norm"]
    ):
        raise RuntimeError("capture path set mismatch")
    for value in features.values():
        _finite_tensor(value, "captured representation")
    return features, logits, site_names, positions


def _logit_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float | int]:
    logits = _finite_tensor(logits.float(), "logit metric input")
    targets = targets.long()
    correct = logits.gather(-1, targets[:, None]).squeeze(-1)
    log_probability = correct - torch.logsumexp(logits, dim=-1)
    probability = torch.exp(log_probability)
    incorrect = logits.clone()
    incorrect.scatter_(-1, targets[:, None], float("-inf"))
    margin = correct - incorrect.amax(dim=-1)
    rank = 1 + (logits > correct[:, None]).sum(dim=-1)
    return {
        "examples": int(targets.shape[0]),
        "cross_entropy": _finite((-log_probability).mean(), "native cross entropy"),
        "correct_token_probability": _finite(probability.mean(), "native probability"),
        "top1_accuracy": _finite(logits.argmax(dim=-1).eq(targets).float().mean(), "native top1"),
        "correct_token_logit_margin": _finite(margin.mean(), "native margin"),
        "mean_target_rank": _finite(rank.float().mean(), "native rank"),
        "median_target_rank": _finite(torch.median(rank.float()), "native median rank"),
    }


def _native_metrics(logits: torch.Tensor, targets: torch.Tensor, target_length: int) -> list[dict[str, Any]]:
    result = []
    value_logits = logits[:, :, VALUE_BASE : VALUE_BASE + VALUE_COUNT]
    value_targets = targets
    for position in range(target_length):
        result.append(
            {
                "position": position + 1,
                "full_vocabulary": _logit_metrics(logits[:, position, :], targets[:, position] + VALUE_BASE),
                "value_candidate_range": _logit_metrics(
                    value_logits[:, position, :],
                    value_targets[:, position],
                ),
            }
        )
    return result


def _fit_ridge_multi(features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Fit one ridge matrix for all target positions at once."""

    features = features.float()
    targets = targets.float()
    augmented = torch.cat((features, torch.ones((features.shape[0], 1))), dim=1)
    gram = augmented.T @ augmented / features.shape[0]
    scale = torch.diag(gram).mean().clamp_min(1e-6)
    gram = gram + RIDGE_STRENGTH * scale * torch.eye(gram.shape[0])
    cross = torch.einsum("nd,nlh->dlh", augmented, targets) / features.shape[0]
    solution = torch.linalg.solve(gram, cross.reshape(cross.shape[0], -1))
    return solution.reshape(solution.shape[0], cross.shape[1], cross.shape[2])


def _probe_metrics_multi(
    features: torch.Tensor,
    targets: torch.Tensor,
    probe: torch.Tensor,
    value_embeddings: torch.Tensor,
) -> list[dict[str, float | int]]:
    augmented = torch.cat((features.float(), torch.ones((features.shape[0], 1))), dim=1)
    predictions = torch.einsum("nd,dlh->nlh", augmented, probe)
    predictions = F_normalize(predictions)
    embeddings = F_normalize(value_embeddings.float())
    records = []
    for position in range(targets.shape[1]):
        scores = predictions[:, position, :] @ embeddings.T
        position_targets = targets[:, position]
        target_scores = scores.gather(1, position_targets[:, None]).squeeze(1)
        ranks = 1 + (scores > target_scores[:, None]).sum(dim=1)
        topk = scores.topk(k=5, dim=-1).indices
        records.append(
            {
                "examples": int(position_targets.shape[0]),
                "top1_accuracy": _finite(
                    topk[:, :1].eq(position_targets[:, None]).any(dim=1).float().mean(),
                    "probe top1",
                ),
                "top5_accuracy": _finite(
                    topk.eq(position_targets[:, None]).any(dim=1).float().mean(),
                    "probe top5",
                ),
                "mean_target_rank": _finite(ranks.float().mean(), "probe rank"),
                "median_target_rank": _finite(torch.median(ranks.float()), "probe median rank"),
                "mean_target_cosine": _finite(
                    scores.gather(1, position_targets[:, None]).mean(),
                    "probe target cosine",
                ),
            }
        )
    return records


def _average_records(records: list[dict[str, float | int]]) -> dict[str, float | int]:
    if not records:
        raise ValueError("cannot average empty metric records")
    keys = [key for key in records[0] if key != "examples"]
    result: dict[str, float | int] = {"examples": sum(int(record["examples"]) for record in records)}
    for key in keys:
        result[key] = _finite(sum(float(record[key]) for record in records) / len(records), f"average {key}")
    result["folds"] = len(records)
    return result


def _cross_validated_probes(
    features: dict[str, torch.Tensor],
    targets: torch.Tensor,
    value_embeddings: torch.Tensor,
    site_names: list[str],
) -> dict[str, Any]:
    if targets.shape[0] != FIXED_EXAMPLES:
        raise ValueError("unexpected heldout example count")
    fold_ids = torch.arange(targets.shape[0]) % FOLD_COUNT
    if sorted(torch.bincount(fold_ids).tolist()) != [targets.shape[0] // FOLD_COUNT] * FOLD_COUNT:
        raise ValueError("folds are not balanced")

    result: dict[str, Any] = {}
    for path in sorted(features):
        path_features = features[path]
        if path_features.shape[1] != len(site_names):
            raise ValueError(f"site count mismatch for {path}")
        target_records: list[dict[str, Any]] = []
        for target_position in range(targets.shape[1]):
            fixed_records = {site: [] for site in site_names}
            site_records = {site: [] for site in site_names}
            for fold in range(FOLD_COUNT):
                eval_mask = fold_ids.eq(fold)
                train_mask = ~eval_mask
                train_targets = value_embeddings[targets[train_mask]]
                pre_probe = _fit_ridge_multi(path_features[train_mask, 0, :], train_targets)
                for site_index, site_name in enumerate(site_names):
                    site_probe = _fit_ridge_multi(
                        path_features[train_mask, site_index, :],
                        train_targets,
                    )
                    fixed_fold = _probe_metrics_multi(
                        path_features[eval_mask, site_index, :],
                        targets[eval_mask],
                        pre_probe,
                        value_embeddings,
                    )[target_position]
                    site_fold = _probe_metrics_multi(
                        path_features[eval_mask, site_index, :],
                        targets[eval_mask],
                        site_probe,
                        value_embeddings,
                    )[target_position]
                    fixed_records[site_name].append(fixed_fold)
                    site_records[site_name].append(site_fold)
            target_records.append(
                {
                    "target_position": target_position + 1,
                    "fixed_pre_query_cv": {
                        site: _average_records(metrics)
                        for site, metrics in fixed_records.items()
                    },
                    "site_specific_cv": {
                        site: _average_records(metrics)
                        for site, metrics in site_records.items()
                    },
                }
            )
        result[path] = {"target_positions": target_records}
    return {
        "fold_count": FOLD_COUNT,
        "fold_sizes": [int((fold_ids == fold).sum()) for fold in range(FOLD_COUNT)],
        "paths": result,
    }


def _run_analysis(
    architecture: str,
    target_length: int,
    run: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    model = _load_model(architecture, target_length, run, device)
    examples = make_evaluation_examples(
        SEED,
        "heldout",
        VALUE_COUNT,
        ASSOCIATION_COUNT,
        DISTRACTOR_DISTANCE,
        target_length,
    )
    features, logits, site_names, positions = _collect_features_and_logits(
        model,
        architecture,
        examples,
        target_length,
        device,
    )
    targets = _target_ids(examples, target_length)
    embeddings = model.token_embedding.weight.detach().float().cpu()
    value_embeddings = embeddings[VALUE_BASE : VALUE_BASE + VALUE_COUNT]
    if value_embeddings.shape != (VALUE_COUNT, model.config.hidden_size):
        raise ValueError("unexpected value embedding shape")
    probe_result = _cross_validated_probes(features, targets, value_embeddings, site_names)
    return {
        "architecture": architecture,
        "target_length": target_length,
        "checkpoint": run["checkpoint"],
        "source_run": run["run_dir"],
        "heldout_examples": len(examples),
        "capture_sites": [
            {"site": name, "position": position}
            for name, position in zip(site_names, positions)
        ],
        "native_teacher_forced": _native_metrics(logits, targets, target_length),
        "cross_validated_diagnostic_readouts": probe_result,
    }


def _summary_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    selected = {"A0": {"final_norm"}, "A1": {"final_norm", "recurrent_state_0", "recurrent_state_7"}}
    rows = []
    for run in result["runs"]:
        arch = run["architecture"]
        native = {row["position"]: row for row in run["native_teacher_forced"]}
        paths = run["cross_validated_diagnostic_readouts"]["paths"]
        for path in sorted(selected[arch]):
            for position in paths[path]["target_positions"]:
                target_position = position["target_position"]
                fixed = position["fixed_pre_query_cv"]
                site = position["site_specific_cv"]
                rows.append(
                    {
                        "architecture": arch,
                        "target_length": run["target_length"],
                        "path": path,
                        "target_position": target_position,
                        "native_value_top1": native[target_position]["value_candidate_range"]["top1_accuracy"],
                        "fixed_pre_query_pre_top1": fixed["pre_query"]["top1_accuracy"],
                        "fixed_pre_query_post_top1": fixed["post_query"]["top1_accuracy"],
                        "fixed_pre_query_target_top1": fixed[f"target_{target_position}"]["top1_accuracy"],
                        "site_specific_target_top1": site[f"target_{target_position}"]["top1_accuracy"],
                    }
                )
    return rows


def _report(result: dict[str, Any]) -> str:
    lines = [
        "# A0/A1 query/readout localization result",
        "",
        "Status: **complete; read-only frozen-checkpoint analysis**.",
        "",
        f"Analysis plan: `{result['analysis_plan_path']}` (sha256 `{result['analysis_plan_sha256']}`).",
        f"Source protocol commit: `{PROTOCOL_COMMIT}`; source runner sha256 `{RUNNER_SHA256}`.",
        "",
        "The native metrics use teacher-forced target positions. Diagnostic readouts are four-fold cross-validated within the heldout set; no same-sample fitted ceiling is reported.",
        "",
        "## Native held-out output metrics",
        "",
        "| arch | L | target position | full-vocab top1 | value-range top1 | value-range CE | value-range mean rank |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in result["runs"]:
        for row in run["native_teacher_forced"]:
            value_metrics = row["value_candidate_range"]
            full_metrics = row["full_vocabulary"]
            lines.append(
                f"| {run['architecture']} | {run['target_length']} | {row['position']} | "
                f"{full_metrics['top1_accuracy']:.4f} | {value_metrics['top1_accuracy']:.4f} | "
                f"{value_metrics['cross_entropy']:.4f} | {value_metrics['mean_target_rank']:.1f} |"
            )
    lines.extend(
        [
            "",
            "## Selected fixed-pre-query and diagnostic readouts",
            "",
            "The fixed columns use the same pre-query probe at each later site. The site-specific column fits the probe at that site. `target` means the site corresponding to the requested target position.",
            "",
            "| arch | L | path | target position | native value top1 | fixed pre top1 | fixed post top1 | fixed target top1 | site-specific target top1 |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in result["summary_rows"]:
        lines.append(
            f"| {row['architecture']} | {row['target_length']} | {row['path']} | {row['target_position']} | "
            f"{row['native_value_top1']:.4f} | {row['fixed_pre_query_pre_top1']:.4f} | "
            f"{row['fixed_pre_query_post_top1']:.4f} | {row['fixed_pre_query_target_top1']:.4f} | "
            f"{row['site_specific_target_top1']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "A fixed-probe drop from pre-query to post-query/target supports a changed or damaged value code; a strong site-specific probe with weak native output would instead support a readout/routing gap. These are correlational frozen-checkpoint measurements. Queue item 4 is required for causal evidence.",
            "",
            "## Integrity",
            "",
            "All six source runs were sidecar/hash verified before loading. Expected parameter counts, capture sites, finite logits, finite representations, balanced folds, and finite diagnostic metrics were required. Raw checkpoints/logs remain ignored under `results/runs/`.",
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
        default=REPO_ROOT / "results/a0_a1_query_readout_localization_values8000.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "docs/A0_A1_QUERY_READOUT_LOCALIZATION_RESULT.md",
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
    for target_length in TARGET_LENGTHS:
        for architecture in ARCHITECTURES:
            runs.append(
                _run_analysis(
                    architecture,
                    target_length,
                    verified_runs[(target_length, architecture)],
                    device,
                )
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

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
        "architectures": list(ARCHITECTURES),
        "heldout_examples": FIXED_EXAMPLES,
        "batch_size": BATCH_SIZE,
        "fold_count": FOLD_COUNT,
        "fold_assignment": "heldout_index_modulo_fold_count",
        "runs": runs,
    }
    result["summary_rows"] = _summary_rows(result)
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
