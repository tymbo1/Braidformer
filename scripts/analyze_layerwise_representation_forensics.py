#!/usr/bin/env python3
"""Read-only layerwise representation forensics for frozen A0/A1 checkpoints.

This analysis captures block outputs, A1 internal recurrent states, and the
final normalized state at the pre-query, post-query, and target-output
positions. Temporary ridge probes are fit in memory only; no checkpoint or
model parameter is modified.
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
from scripts.analyze_a1_state_decay_forensics import (
    F_normalize,
    _fit_ridge_probe,
    _probe_metrics,
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
    KEY_TRAIN_BASE,
    VALUE_BASE,
    make_evaluation_examples,
)


VALUE_COUNT = 8_000
ASSOCIATION_COUNT = 1
DISTRACTOR_DISTANCE = 0
BATCH_SIZE = 16
EXPECTED_PARAMS = {"A0": 41_866_240, "A1": 41_870_336}
PLAN_PATH = "docs/A0_A1_LAYERWISE_REPRESENTATION_FORENSICS_PLAN.md"
TARGET_RUNNER_PATH = "scripts/run_synthetic_overfit_calibration.py"
OUTPUT_VERSION = "layerwise_representation_forensics_v1"
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


def _capture_sites(target_length: int) -> tuple[list[str], list[int]]:
    names = ["pre_query", "post_query"]
    positions = [2 + target_length, 4 + target_length]
    for target_position in range(1, target_length + 1):
        names.append(f"target_{target_position}")
        positions.append(4 + target_length + target_position)
    return names, positions


def _select_positions(value: torch.Tensor, positions: list[int]) -> torch.Tensor:
    index = torch.tensor(positions, device=value.device, dtype=torch.long)
    return value.index_select(1, index).detach().float().cpu()


def _capture_batch(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    architecture: str,
    positions: list[int],
) -> dict[str, torch.Tensor]:
    captured: dict[str, torch.Tensor] = {}
    hooks = []

    def make_block_hook(layer_index: int):
        def hook(_module: Any, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            captured[f"block_output_{layer_index}"] = _select_positions(output, positions)

        return hook

    def make_state_hook(layer_index: int):
        def hook(
            module: Any,
            inputs: tuple[torch.Tensor, ...],
            _output: torch.Tensor,
        ) -> None:
            hidden_states = inputs[0]
            projected = module.input_projection(hidden_states)
            candidate, input_gate, _output_gate = projected.chunk(3, dim=-1)
            decay = torch.sigmoid(module.state_transition_logit).to(dtype=hidden_states.dtype)
            updates = (1.0 - decay).view(1, 1, module.hidden_size)
            updates = updates * torch.sigmoid(input_gate) * torch.tanh(candidate)
            if module.use_parallel_scan:
                states = module._parallel_scan(decay, updates)
            else:
                states = module._sequential_scan(candidate, input_gate, decay.unsqueeze(0))
            captured[f"recurrent_state_{layer_index}"] = _select_positions(states, positions)

        return hook

    for layer_index, block in enumerate(model.blocks):
        hooks.append(block.register_forward_hook(make_block_hook(layer_index)))
        if architecture == "A1":
            hooks.append(
                block.state_space.register_forward_hook(make_state_hook(layer_index))
            )

    def final_norm_hook(
        _module: Any,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        captured["final_norm"] = _select_positions(output, positions)

    hooks.append(model.final_norm.register_forward_hook(final_norm_hook))
    try:
        with torch.inference_mode(), _autocast(input_ids.device):
            _finite_tensor(model(input_ids).logits.float(), "model logits")
    finally:
        for hook in hooks:
            hook.remove()

    expected = [f"block_output_{index}" for index in range(len(model.blocks))]
    if architecture == "A1":
        expected += [f"recurrent_state_{index}" for index in range(len(model.blocks))]
    expected.append("final_norm")
    missing = [name for name in expected if name not in captured]
    if missing:
        raise RuntimeError(f"missing capture sites: {missing}")
    return captured


def _collect_features(
    model: torch.nn.Module,
    architecture: str,
    examples: list[Any],
    target_length: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    _names, positions = _capture_sites(target_length)
    parts: dict[str, list[torch.Tensor]] = {}
    for start in range(0, len(examples), BATCH_SIZE):
        batch = examples[start : start + BATCH_SIZE]
        inputs = torch.tensor(
            [example.input_ids for example in batch],
            dtype=torch.long,
            device=device,
        )
        batch_features = _capture_batch(model, inputs, architecture, positions)
        for name, value in batch_features.items():
            parts.setdefault(name, []).append(value)

    features = {name: torch.cat(values, dim=0) for name, values in parts.items()}
    keys = torch.tensor(
        [example.input_ids[2] for example in examples],
        dtype=torch.long,
    )
    targets = []
    for example in examples:
        if not isinstance(example.target, tuple) or len(example.target) != target_length:
            raise ValueError("target metadata does not match target length")
        targets.append(example.target)
    values = torch.tensor(targets, dtype=torch.long)
    if len(features) == 0 or any(value.shape[0] != len(examples) for value in features.values()):
        raise RuntimeError("feature collection count mismatch")
    return features, keys, values


def _fit_probe(features: torch.Tensor, targets: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
    if features.shape[0] != targets.shape[0]:
        raise ValueError("probe feature/target count mismatch")
    return _fit_ridge_probe(features.float(), embeddings[targets].float())


def _probe_correct_mask(
    features: torch.Tensor,
    targets: torch.Tensor,
    probe: torch.Tensor,
    embeddings: torch.Tensor,
) -> torch.Tensor:
    augmented = torch.cat((features.float(), torch.ones((features.shape[0], 1))), dim=1)
    prediction = F_normalize(augmented @ probe)
    scores = prediction @ F_normalize(embeddings).T
    return scores.argmax(dim=-1).eq(targets)


def _probe_record(
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    eval_features: torch.Tensor,
    eval_targets: torch.Tensor,
    embeddings: torch.Tensor,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    train_probe = _fit_probe(train_features, train_targets, embeddings)
    train_metrics = _probe_metrics(
        train_features,
        train_targets,
        train_probe,
        embeddings,
    )
    eval_metrics = _probe_metrics(
        eval_features,
        eval_targets,
        train_probe,
        embeddings,
    )
    eval_probe = _fit_probe(eval_features, eval_targets, embeddings)
    eval_fit_metrics = _probe_metrics(
        eval_features,
        eval_targets,
        eval_probe,
        embeddings,
    )
    return (
        {
            "train_fit_train": train_metrics,
            "train_fit_heldout": eval_metrics,
            "heldout_fit_heldout": eval_fit_metrics,
        },
        train_probe,
        eval_probe,
    )


def _key_probe_record(
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    eval_features: torch.Tensor,
    eval_targets: torch.Tensor,
    train_embeddings: torch.Tensor,
    eval_embeddings: torch.Tensor,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    """Probe key identity while preserving separate train/held-out key bases."""

    train_probe = _fit_probe(train_features, train_targets, train_embeddings)
    train_metrics = _probe_metrics(
        train_features,
        train_targets,
        train_probe,
        train_embeddings,
    )
    transfer_metrics = _probe_metrics(
        eval_features,
        eval_targets,
        train_probe,
        eval_embeddings,
    )
    eval_probe = _fit_probe(eval_features, eval_targets, eval_embeddings)
    eval_metrics = _probe_metrics(
        eval_features,
        eval_targets,
        eval_probe,
        eval_embeddings,
    )
    return (
        {
            "train_fit_train": train_metrics,
            "train_fit_heldout": transfer_metrics,
            "heldout_fit_heldout": eval_metrics,
        },
        train_probe,
        eval_probe,
    )


def _position_probe(
    train_features: torch.Tensor,
    eval_features: torch.Tensor,
    target_length: int,
) -> dict[str, float | int]:
    train_flat = train_features[:, 2:, :].reshape(-1, train_features.shape[-1])
    eval_flat = eval_features[:, 2:, :].reshape(-1, eval_features.shape[-1])
    train_labels = torch.arange(target_length).repeat(train_features.shape[0])
    eval_labels = torch.arange(target_length).repeat(eval_features.shape[0])
    train_targets = torch.nn.functional.one_hot(train_labels, num_classes=target_length).float()
    probe = _fit_ridge_probe(train_flat, train_targets)
    train_predictions = (torch.cat((train_flat, torch.ones((train_flat.shape[0], 1))), dim=1) @ probe).argmax(dim=-1)
    eval_predictions = (torch.cat((eval_flat, torch.ones((eval_flat.shape[0], 1))), dim=1) @ probe).argmax(dim=-1)
    return {
        "train_examples": int(train_flat.shape[0]),
        "heldout_examples": int(eval_flat.shape[0]),
        "train_accuracy": _finite(train_predictions.eq(train_labels).float().mean(), "position train accuracy"),
        "heldout_accuracy": _finite(eval_predictions.eq(eval_labels).float().mean(), "position heldout accuracy"),
    }


def _geometry(features: torch.Tensor) -> list[dict[str, float | int | str]]:
    baseline = features[:, 0, :]
    baseline_norm = baseline.norm(dim=-1).clamp_min(1e-8)
    records = []
    names, _positions = _capture_sites_from_count(features.shape[1])
    for index, name in enumerate(names):
        current = features[:, index, :]
        current_norm = current.norm(dim=-1).clamp_min(1e-8)
        cosine = (baseline * current).sum(dim=-1) / (baseline_norm * current_norm)
        records.append(
            {
                "site": name,
                "site_index": index,
                "mean_norm": _finite(current_norm.mean(), "representation norm"),
                "mean_cosine_to_pre_query": _finite(cosine.mean(), "representation cosine"),
                "p10_cosine_to_pre_query": _finite(torch.quantile(cosine, 0.10), "representation cosine p10"),
                "p90_cosine_to_pre_query": _finite(torch.quantile(cosine, 0.90), "representation cosine p90"),
            }
        )
    return records


def _capture_sites_from_count(site_count: int) -> tuple[list[str], list[int]]:
    if site_count < 3:
        raise ValueError("expected pre-query, post-query, and target capture sites")
    names = ["pre_query", "post_query"] + [f"target_{index}" for index in range(1, site_count - 1)]
    return names, list(range(site_count))


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


def _analyze_model(
    architecture: str,
    target_length: int,
    run: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    model = _load_model(architecture, target_length, run, device)
    train_examples = make_evaluation_examples(
        SEED,
        "train",
        VALUE_COUNT,
        ASSOCIATION_COUNT,
        DISTRACTOR_DISTANCE,
        target_length,
    )
    heldout_examples = make_evaluation_examples(
        SEED,
        "heldout",
        VALUE_COUNT,
        ASSOCIATION_COUNT,
        DISTRACTOR_DISTANCE,
        target_length,
    )
    train_features, train_keys, train_values = _collect_features(
        model, architecture, train_examples, target_length, device
    )
    heldout_features, heldout_keys, heldout_values = _collect_features(
        model, architecture, heldout_examples, target_length, device
    )
    site_names, site_positions = _capture_sites(target_length)
    if any(value.shape[1] != len(site_names) for value in train_features.values()):
        raise RuntimeError("capture site count mismatch")

    embeddings = model.token_embedding.weight.detach().float().cpu()
    value_embeddings = embeddings[VALUE_BASE : VALUE_BASE + VALUE_COUNT]
    train_key_embeddings = embeddings[KEY_TRAIN_BASE : KEY_TRAIN_BASE + FIXED_EXAMPLES]
    heldout_key_embeddings = embeddings[KEY_HELDOUT_BASE : KEY_HELDOUT_BASE + FIXED_EXAMPLES]
    train_key_targets = train_keys - KEY_TRAIN_BASE
    heldout_key_targets = heldout_keys - KEY_HELDOUT_BASE
    train_value_targets = train_values - VALUE_BASE
    heldout_value_targets = heldout_values - VALUE_BASE

    paths = sorted(train_features)
    probe_results: dict[str, Any] = {}
    for path in paths:
        train_path = train_features[path]
        heldout_path = heldout_features[path]
        site_records = []
        for site_index, site_name in enumerate(site_names):
            train_site = train_path[:, site_index, :]
            heldout_site = heldout_path[:, site_index, :]
            key_record, train_key_probe, heldout_key_probe = _key_probe_record(
                train_site,
                train_key_targets,
                heldout_site,
                heldout_key_targets,
                train_key_embeddings,
                heldout_key_embeddings,
            )

            value_records = []
            for target_position in range(target_length):
                train_targets = train_value_targets[:, target_position]
                heldout_targets = heldout_value_targets[:, target_position]
                value_record, train_value_probe, heldout_value_probe = _probe_record(
                    train_site,
                    train_targets,
                    heldout_site,
                    heldout_targets,
                    value_embeddings,
                )
                pre_query_train = train_path[:, 0, :]
                pre_query_probe = _fit_probe(
                    pre_query_train,
                    train_targets,
                    value_embeddings,
                )
                transfer_metrics = _probe_metrics(
                    heldout_site,
                    heldout_targets,
                    pre_query_probe,
                    value_embeddings,
                )
                train_key_mask = _probe_correct_mask(
                    heldout_site,
                    heldout_key_targets,
                    train_key_probe,
                    train_key_embeddings,
                )
                train_value_mask = _probe_correct_mask(
                    heldout_site,
                    heldout_targets,
                    train_value_probe,
                    value_embeddings,
                )
                heldout_key_mask = _probe_correct_mask(
                    heldout_site,
                    heldout_key_targets,
                    heldout_key_probe,
                    heldout_key_embeddings,
                )
                heldout_value_mask = _probe_correct_mask(
                    heldout_site,
                    heldout_targets,
                    heldout_value_probe,
                    value_embeddings,
                )
                value_records.append(
                    {
                        "target_position": target_position + 1,
                        "train_fit": value_record,
                        "fixed_pre_query_train_fit_heldout": transfer_metrics,
                        "binding_pair_top1": {
                            "train_fit_heldout": _finite(
                                (train_key_mask & train_value_mask).float().mean(),
                                "train-fit binding pair accuracy",
                            ),
                            "heldout_fit_heldout": _finite(
                                (heldout_key_mask & heldout_value_mask).float().mean(),
                                "heldout-fit binding pair accuracy",
                            ),
                        },
                    }
                )
            site_records.append(
                {
                    "site": site_name,
                    "site_index": site_index,
                    "position": site_positions[site_index],
                "key_identity": key_record,
                    "associated_value": value_records,
                }
            )

        probe_results[path] = {
            "sites": site_records,
            "target_position_probe": _position_probe(
                train_path,
                heldout_path,
                target_length,
            ),
            "geometry_heldout": _geometry(heldout_path),
        }

    return {
        "architecture": architecture,
        "target_length": target_length,
        "checkpoint": run["checkpoint"],
        "source_run": run["run_dir"],
        "capture_sites": [
            {"site": name, "position": position}
            for name, position in zip(site_names, site_positions)
        ],
        "paths": probe_results,
    }


def _summary_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for run in result["runs"]:
        for path, path_result in run["paths"].items():
            value_site_fit = []
            value_transfer = []
            key_site_fit = []
            for site in path_result["sites"]:
                key_site_fit.append(site["key_identity"]["heldout_fit_heldout"]["top1_accuracy"])
                for value in site["associated_value"]:
                    value_site_fit.append(value["train_fit"]["heldout_fit_heldout"]["top1_accuracy"])
                    value_transfer.append(value["fixed_pre_query_train_fit_heldout"]["top1_accuracy"])
            rows.append(
                {
                    "architecture": run["architecture"],
                    "target_length": run["target_length"],
                    "path": path,
                    "mean_key_heldout_fit_top1": _finite(sum(key_site_fit) / len(key_site_fit), "summary key probe"),
                    "mean_value_heldout_fit_top1": _finite(sum(value_site_fit) / len(value_site_fit), "summary value probe"),
                    "mean_value_fixed_pre_query_transfer_top1": _finite(sum(value_transfer) / len(value_transfer), "summary transfer probe"),
                    "target_position_train_fit_heldout_accuracy": path_result["target_position_probe"]["heldout_accuracy"],
                }
            )
    return rows


def _report(result: dict[str, Any]) -> str:
    lines = [
        "# A0/A1 layerwise representation forensics result",
        "",
        "Status: **complete; read-only frozen-checkpoint analysis**.",
        "",
        f"Analysis plan: `{result['analysis_plan_path']}` (sha256 `{result['analysis_plan_sha256']}`).",
        f"Source protocol commit: `{PROTOCOL_COMMIT}`; source runner sha256 `{RUNNER_SHA256}`.",
        "",
        "The machine-readable aggregate contains per-site key/value/binding probes, fixed pre-query transfer probes, target-position probes, geometry, and all source checkpoint hashes. Probe weights were temporary in-memory diagnostics; no checkpoint or model parameter was modified.",
        "",
        "## Summary by representation path",
        "",
        "| arch | L | path | key heldout-fit top1 | value heldout-fit top1 | value fixed-pre-query transfer top1 | target-position heldout acc. |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in result["summary_rows"]:
        lines.append(
            f"| {row['architecture']} | {row['target_length']} | {row['path']} | "
            f"{row['mean_key_heldout_fit_top1']:.4f} | "
            f"{row['mean_value_heldout_fit_top1']:.4f} | "
            f"{row['mean_value_fixed_pre_query_transfer_top1']:.4f} | "
            f"{row['target_position_train_fit_heldout_accuracy']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Site-fitted held-out probes measure whether a linear decoder can recover information at a representation site; fixed pre-query transfer probes measure whether the same value coordinate survives later transformations. A drop in both supports information loss or destructive transformation. Strong site-fitted value probes with weak model output would shift attention toward query-conditioned use, routing, or readout. These are correlational findings; queue item 4 is required for causal evidence.",
            "",
            "## Integrity",
            "",
            "All six source runs were sidecar/hash verified before loading. Expected A0/A1 parameter counts, capture sites, recurrent layers, finite tensors, and finite probe metrics were required. Raw checkpoints/logs remain ignored under `results/runs/`.",
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
        default=REPO_ROOT / "results/a0_a1_layerwise_representation_forensics_values8000.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "docs/A0_A1_LAYERWISE_REPRESENTATION_FORENSICS_RESULT.md",
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
                _analyze_model(
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
        "fixed_examples": FIXED_EXAMPLES,
        "batch_size": BATCH_SIZE,
        "capture_definition": {
            str(target_length): [
                {"site": name, "position": position}
                for name, position in zip(*_capture_sites(target_length))
            ]
            for target_length in TARGET_LENGTHS
        },
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

