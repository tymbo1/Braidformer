#!/usr/bin/env python3
"""Localize protected-lane storage, binding, query selection, and readout.

This is a read-only analysis of the completed A1 protected/ordinary lane
checkpoints.  It reconstructs the frozen forward pass, captures four causal
positions, fits temporary ridge probes, and writes only aggregate evidence and
an analysis report.  No model parameter, checkpoint, or optimizer state is
modified.
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
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from braidformer.config import load_config
from braidformer.models import A1AssociativeLaneLM, EXPECTED_LANE_PARAMETERS, LANE_WIDTH
from braidformer.training.integrity import sha256_file, verify_checkpoint
from scripts.run_synthetic_overfit_calibration import (
    FIXED_EXAMPLES,
    KEY_HELDOUT_BASE,
    KEY_TRAIN_BASE,
    VALUE_BASE,
    make_evaluation_examples,
)


ANALYSIS_PLAN_PATH = "docs/A1_PROTECTED_LANE_QUERY_BINDING_LOCALIZATION_PROTOCOL.md"
ANALYSIS_PLAN_COMMIT = "5d60713e8e8a176933562f64964ab3c35904003b"
SOURCE_AGGREGATE_PATH = "results/a1_protected_lane_distance_axis_values8000.json"
SOURCE_AGGREGATE_SHA256 = "afdeeb42b2379f38ac7a592b805dcad1b2acadbeace0eaa57c43472f4472a158"
RUNNER_PATH = "scripts/run_synthetic_overfit_calibration.py"
RUNNER_SHA256 = "2a1a1c84f1ebfabc11cf66f0defa2e2ad4db2d1741c89c3d12660d3ac066d3e5"
MODEL_PATH = "src/braidformer/models/protected_lane.py"
MODEL_SHA256 = "d776dea620662187c2698e604b8aa171a52e5b911adaa738daea7ffca13f421c"
OUTPUT_VERSION = "a1_protected_lane_query_binding_localization_v1"
SEED = 17
VALUE_COUNT = 8_000
EXPECTED_BASE_PARAMETERS = 41_870_336
EXPECTED_PARAMETERS = EXPECTED_BASE_PARAMETERS + EXPECTED_LANE_PARAMETERS
DISTANCES = (0, 4, 16, 64)
MODES = ("protected", "ordinary")
BATCH_SIZE = 16
FOLD_COUNT = 4
RIDGE_STRENGTH = 1e-3
PRIMARY_STAGES = ("post_value", "post_distractor", "post_query_key", "answer_marker")
ALL_STAGES = (
    "post_value",
    "post_distractor",
    "post_query_marker",
    "post_query_key",
    "answer_marker",
)


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


def _finite_tensor(value: torch.Tensor, label: str) -> torch.Tensor:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"non-finite tensor: {label}")
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


def _fit_ridge_probe(features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    features = features.float()
    targets = targets.float()
    augmented = torch.cat((features, torch.ones((features.shape[0], 1))), dim=1)
    gram = augmented.T @ augmented / features.shape[0]
    scale = torch.diag(gram).mean().clamp_min(1e-6)
    gram = gram + RIDGE_STRENGTH * scale * torch.eye(gram.shape[0])
    cross = augmented.T @ targets / features.shape[0]
    probe = torch.linalg.solve(gram, cross)
    return _finite_tensor(probe, "ridge probe")


def _probe_metrics(
    features: torch.Tensor,
    target_indices: torch.Tensor,
    probe: torch.Tensor,
    candidate_embeddings: torch.Tensor,
) -> dict[str, float | int]:
    if features.shape[0] != target_indices.shape[0]:
        raise ValueError("probe feature/target count mismatch")
    augmented = torch.cat(
        (features.float(), torch.ones((features.shape[0], 1))), dim=1
    )
    prediction = F.normalize(augmented @ probe, dim=-1)
    embeddings = F.normalize(candidate_embeddings.float(), dim=-1)
    scores = prediction @ embeddings.T
    target_scores = scores.gather(1, target_indices[:, None]).squeeze(1)
    ranks = 1 + (scores > target_scores[:, None]).sum(dim=1)
    topk = scores.topk(k=min(5, scores.shape[1]), dim=-1).indices
    return {
        "examples": int(features.shape[0]),
        "top1_accuracy": _finite(
            (topk[:, :1] == target_indices[:, None]).any(dim=1).float().mean(),
            "probe top1",
        ),
        "top5_accuracy": _finite(
            (topk == target_indices[:, None]).any(dim=1).float().mean(),
            "probe top5",
        ),
        "mean_target_rank": _finite(ranks.float().mean(), "probe rank"),
        "median_target_rank": _finite(torch.median(ranks.float()), "probe median rank"),
        "mean_target_cosine": _finite(target_scores.mean(), "probe target cosine"),
    }


def _mean_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float()
    right = right.float()
    cosine = F.cosine_similarity(left, right, dim=-1)
    return _finite(cosine.mean(), "mean cosine")


def _stage_positions(distance: int) -> dict[str, int]:
    positions = {
        "post_value": 3,
        "post_distractor": 3 + distance,
        "post_query_marker": 4 + distance,
        "post_query_key": 5 + distance,
        "answer_marker": 6 + distance,
    }
    if sorted(set(positions.values())) != sorted(positions.values()):
        raise ValueError(f"stage positions overlap unexpectedly at distance {distance}")
    return positions


def _capture_split(
    model: A1AssociativeLaneLM,
    examples: list[Any],
    distance: int,
    device: torch.device,
) -> dict[str, Any]:
    positions = _stage_positions(distance)
    feature_parts: dict[str, dict[str, list[torch.Tensor]]] = {
        stage: {} for stage in ALL_STAGES
    }
    native_logits_parts: list[torch.Tensor] = []
    base_logits_parts: list[torch.Tensor] = []
    value_targets: list[int] = []
    key_targets: list[int] = []
    diagnostics: dict[str, dict[str, list[float]]] = {
        stage: {
            "memory_key_norm": [],
            "memory_value_norm": [],
            "memory_value_cosine_to_post_value": [],
            "similarity": [],
            "read_gate": [],
            "lane_delta_norm": [],
        }
        for stage in ALL_STAGES
    }

    for start in range(0, len(examples), BATCH_SIZE):
        batch = examples[start : start + BATCH_SIZE]
        inputs = torch.tensor(
            [example.input_ids for example in batch], dtype=torch.long, device=device
        )
        with torch.inference_mode(), _autocast(device):
            base_hidden = model._base_hidden(inputs)
            lane_delta, trace = model._lane_forward(
                base_hidden, inputs, return_trace=True
            )
            combined_hidden = base_hidden + lane_delta
            final_hidden = model.base.final_norm(combined_hidden)
            base_norm = model.base.final_norm(base_hidden)
            native_logits = model.base.lm_head(final_hidden)
            base_logits = model.base.lm_head(base_norm)
            key_candidates = model.key_projection(base_hidden)
            value_candidates = model.value_projection(base_hidden)

        answer_position = positions["answer_marker"]
        native_logits_parts.append(native_logits[:, answer_position, :].float().cpu())
        base_logits_parts.append(base_logits[:, answer_position, :].float().cpu())
        query_key_position = positions["post_query_key"]
        for example in batch:
            value_targets.append(int(example.target) - VALUE_BASE)
            key_targets.append(int(example.input_ids[2]) - KEY_TRAIN_BASE)

        for stage in ALL_STAGES:
            position = positions[stage]
            memory_key = trace["memory_key"][:, position, :].float()
            memory_value = trace["memory_value"][:, position, :].float()
            stage_features = {
                "base_hidden": base_hidden[:, position, :].float(),
                "base_normalized": base_norm[:, position, :].float(),
                "final_normalized": final_hidden[:, position, :].float(),
                "key_candidate": key_candidates[:, position, :].float(),
                "value_candidate": value_candidates[:, position, :].float(),
                "memory_key": memory_key,
                "memory_value": memory_value,
                "lane_delta": lane_delta[:, position, :].float(),
                "pair_state": torch.cat((memory_key, memory_value), dim=-1),
            }
            if stage in {"post_query_key", "answer_marker"}:
                query_key = key_candidates[:, query_key_position, :].float()
                stage_features["query_key_candidate"] = query_key
                stage_features["query_conditioned"] = torch.cat(
                    (query_key, memory_key, memory_value), dim=-1
                )
            for name, tensor in stage_features.items():
                feature_parts[stage].setdefault(name, []).append(
                    _finite_tensor(tensor.detach().float().cpu(), f"{stage}/{name}")
                )
            post_value_value = trace["memory_value"][:, positions["post_value"], :].float()
            diagnostics[stage]["memory_key_norm"].extend(
                memory_key.norm(dim=-1).detach().cpu().tolist()
            )
            diagnostics[stage]["memory_value_norm"].extend(
                memory_value.norm(dim=-1).detach().cpu().tolist()
            )
            diagnostics[stage]["memory_value_cosine_to_post_value"].extend(
                F.cosine_similarity(memory_value, post_value_value, dim=-1)
                .detach()
                .cpu()
                .tolist()
            )
            diagnostics[stage]["similarity"].extend(
                trace["similarity"][:, position].float().detach().cpu().tolist()
            )
            diagnostics[stage]["read_gate"].extend(
                trace["read_gate"][:, position].float().detach().cpu().tolist()
            )
            diagnostics[stage]["lane_delta_norm"].extend(
                lane_delta[:, position, :].float().norm(dim=-1).detach().cpu().tolist()
            )

    features = {
        stage: {
            name: torch.cat(parts, dim=0)
            for name, parts in stage_parts.items()
        }
        for stage, stage_parts in feature_parts.items()
    }
    for stage, stage_features in features.items():
        for name, tensor in stage_features.items():
            _finite_tensor(tensor, f"collected {stage}/{name}")
    diagnostic_means = {
        stage: {
            name: _finite(sum(values) / len(values), f"diagnostic {stage}/{name}")
            for name, values in values_by_name.items()
        }
        for stage, values_by_name in diagnostics.items()
    }
    native_logits = torch.cat(native_logits_parts, dim=0)
    base_logits = torch.cat(base_logits_parts, dim=0)
    return {
        "features": features,
        "native_logits": _finite_tensor(native_logits, "native answer logits"),
        "base_logits": _finite_tensor(base_logits, "base answer logits"),
        "value_targets": torch.tensor(value_targets, dtype=torch.long),
        "key_targets": torch.tensor(key_targets, dtype=torch.long),
        "diagnostic_means": diagnostic_means,
        "stage_positions": positions,
    }


def _output_metrics(
    logits: torch.Tensor,
    value_targets: torch.Tensor,
    target_tokens: torch.Tensor,
) -> dict[str, Any]:
    logits = _finite_tensor(logits.float(), "output metric logits")
    value_logits = logits[:, VALUE_BASE : VALUE_BASE + VALUE_COUNT]
    target_logits = logits.gather(1, target_tokens[:, None]).squeeze(1)
    incorrect = logits.clone()
    incorrect.scatter_(1, target_tokens[:, None], float("-inf"))
    full_ranks = 1 + (logits > target_logits[:, None]).sum(dim=1)
    value_target_logits = value_logits.gather(1, value_targets[:, None]).squeeze(1)
    value_incorrect = value_logits.clone()
    value_incorrect.scatter_(1, value_targets[:, None], float("-inf"))
    value_ranks = 1 + (value_logits > value_target_logits[:, None]).sum(dim=1)
    full_log_prob = target_logits - torch.logsumexp(logits, dim=-1)
    value_log_prob = value_target_logits - torch.logsumexp(value_logits, dim=-1)
    return {
        "examples": int(logits.shape[0]),
        "full_vocabulary": {
            "cross_entropy": _finite((-full_log_prob).mean(), "full CE"),
            "top1_accuracy": _finite(
                logits.argmax(dim=-1).eq(target_tokens).float().mean(), "full top1"
            ),
            "mean_target_probability": _finite(full_log_prob.exp().mean(), "full probability"),
            "mean_target_logit": _finite(target_logits.mean(), "full target logit"),
            "mean_target_margin": _finite(
                (target_logits - incorrect.amax(dim=-1)).mean(), "full target margin"
            ),
            "mean_target_rank": _finite(full_ranks.float().mean(), "full rank"),
        },
        "value_candidate_range": {
            "cross_entropy": _finite((-value_log_prob).mean(), "value CE"),
            "top1_accuracy": _finite(
                value_logits.argmax(dim=-1).eq(value_targets).float().mean(), "value top1"
            ),
            "mean_target_probability": _finite(value_log_prob.exp().mean(), "value probability"),
            "mean_target_logit": _finite(value_target_logits.mean(), "value target logit"),
            "mean_target_margin": _finite(
                (value_target_logits - value_incorrect.amax(dim=-1)).mean(),
                "value target margin",
            ),
            "mean_target_rank": _finite(value_ranks.float().mean(), "value rank"),
        },
    }


def _fixed_probe_results(
    train: dict[str, Any],
    heldout: dict[str, Any],
    value_embeddings: torch.Tensor,
    key_embeddings: torch.Tensor,
) -> dict[str, Any]:
    value_targets_train = train["value_targets"]
    value_targets_heldout = heldout["value_targets"]
    key_targets_train = train["key_targets"]
    key_targets_heldout = heldout["key_targets"]
    value_target_vectors = value_embeddings[value_targets_train]
    key_target_vectors = key_embeddings[key_targets_train]

    value_paths = (
        "base_hidden",
        "base_normalized",
        "final_normalized",
        "value_candidate",
        "memory_value",
        "pair_state",
        "query_conditioned",
    )
    key_paths = ("base_hidden", "key_candidate", "memory_key", "query_key_candidate")
    result: dict[str, Any] = {
        "train_examples": int(value_targets_train.shape[0]),
        "heldout_examples": int(value_targets_heldout.shape[0]),
        "value_identity": {},
        "key_identity": {},
        "query_conditioned_selection": {},
        "output_representation": {},
    }
    for stage in ALL_STAGES:
        stage_result: dict[str, Any] = {}
        for path in value_paths:
            if path not in train["features"].get(stage, {}):
                continue
            probe = _fit_ridge_probe(
                train["features"][stage][path], value_target_vectors
            )
            stage_result[path] = _probe_metrics(
                heldout["features"][stage][path],
                value_targets_heldout,
                probe,
                value_embeddings,
            )
        result["value_identity"][stage] = stage_result

        key_result: dict[str, Any] = {}
        for path in key_paths:
            if path not in train["features"].get(stage, {}):
                continue
            probe = _fit_ridge_probe(
                train["features"][stage][path], key_target_vectors
            )
            key_result[path] = _probe_metrics(
                heldout["features"][stage][path],
                key_targets_heldout,
                probe,
                key_embeddings,
            )
        result["key_identity"][stage] = key_result

        selection: dict[str, Any] = {}
        for path in ("memory_value", "pair_state", "query_conditioned"):
            if path not in train["features"].get(stage, {}):
                continue
            probe = _fit_ridge_probe(
                train["features"][stage][path], value_target_vectors
            )
            selection[path] = _probe_metrics(
                heldout["features"][stage][path],
                value_targets_heldout,
                probe,
                value_embeddings,
            )
        result["query_conditioned_selection"][stage] = selection

        output_path = "final_normalized"
        if output_path in train["features"].get(stage, {}):
            probe = _fit_ridge_probe(
                train["features"][stage][output_path], value_target_vectors
            )
            result["output_representation"][stage] = _probe_metrics(
                heldout["features"][stage][output_path],
                value_targets_heldout,
                probe,
                value_embeddings,
            )
    return result


def _cv_probe_results(
    heldout: dict[str, Any],
    value_embeddings: torch.Tensor,
    key_embeddings: torch.Tensor,
) -> dict[str, Any]:
    value_targets = heldout["value_targets"]
    key_targets = heldout["key_targets"]
    fold_ids = torch.arange(value_targets.shape[0]) % FOLD_COUNT
    result: dict[str, Any] = {
        "fold_count": FOLD_COUNT,
        "fold_sizes": [int(fold_ids.eq(fold).sum()) for fold in range(FOLD_COUNT)],
        "value_identity": {},
        "key_identity": {},
        "query_conditioned_selection": {},
        "output_representation": {},
    }

    def average(records: list[dict[str, float | int]]) -> dict[str, float | int]:
        if not records:
            raise ValueError("empty CV metric records")
        keys = [key for key in records[0] if key != "examples"]
        return {
            "examples": sum(int(record["examples"]) for record in records),
            **{
                key: _finite(
                    sum(float(record[key]) for record in records) / len(records),
                    f"CV {key}",
                )
                for key in keys
            },
            "folds": len(records),
        }

    value_paths = (
        "base_hidden",
        "base_normalized",
        "final_normalized",
        "value_candidate",
        "memory_value",
        "pair_state",
        "query_conditioned",
    )
    key_paths = ("base_hidden", "key_candidate", "memory_key", "query_key_candidate")
    for stage in ALL_STAGES:
        value_stage: dict[str, Any] = {}
        key_stage: dict[str, Any] = {}
        selection_stage: dict[str, Any] = {}
        output_stage: dict[str, Any] = {}
        for path in value_paths:
            if path not in heldout["features"].get(stage, {}):
                continue
            records = []
            for fold in range(FOLD_COUNT):
                eval_mask = fold_ids.eq(fold)
                train_mask = ~eval_mask
                probe = _fit_ridge_probe(
                    heldout["features"][stage][path][train_mask],
                    value_embeddings[value_targets[train_mask]],
                )
                records.append(
                    _probe_metrics(
                        heldout["features"][stage][path][eval_mask],
                        value_targets[eval_mask],
                        probe,
                        value_embeddings,
                    )
                )
            value_stage[path] = average(records)
        for path in key_paths:
            if path not in heldout["features"].get(stage, {}):
                continue
            records = []
            for fold in range(FOLD_COUNT):
                eval_mask = fold_ids.eq(fold)
                train_mask = ~eval_mask
                probe = _fit_ridge_probe(
                    heldout["features"][stage][path][train_mask],
                    key_embeddings[key_targets[train_mask]],
                )
                records.append(
                    _probe_metrics(
                        heldout["features"][stage][path][eval_mask],
                        key_targets[eval_mask],
                        probe,
                        key_embeddings,
                    )
                )
            key_stage[path] = average(records)
        for path in ("memory_value", "pair_state", "query_conditioned"):
            if path not in heldout["features"].get(stage, {}):
                continue
            records = []
            for fold in range(FOLD_COUNT):
                eval_mask = fold_ids.eq(fold)
                train_mask = ~eval_mask
                probe = _fit_ridge_probe(
                    heldout["features"][stage][path][train_mask],
                    value_embeddings[value_targets[train_mask]],
                )
                records.append(
                    _probe_metrics(
                        heldout["features"][stage][path][eval_mask],
                        value_targets[eval_mask],
                        probe,
                        value_embeddings,
                    )
                )
            selection_stage[path] = average(records)
        if "final_normalized" in heldout["features"].get(stage, {}):
            records = []
            for fold in range(FOLD_COUNT):
                eval_mask = fold_ids.eq(fold)
                train_mask = ~eval_mask
                probe = _fit_ridge_probe(
                    heldout["features"][stage]["final_normalized"][train_mask],
                    value_embeddings[value_targets[train_mask]],
                )
                records.append(
                    _probe_metrics(
                        heldout["features"][stage]["final_normalized"][eval_mask],
                        value_targets[eval_mask],
                        probe,
                        value_embeddings,
                    )
                )
            output_stage["final_normalized"] = average(records)
        result["value_identity"][stage] = value_stage
        result["key_identity"][stage] = key_stage
        result["query_conditioned_selection"][stage] = selection_stage
        result["output_representation"][stage] = output_stage
    return result


def _verify_sources() -> dict[str, Any]:
    aggregate_path = REPO_ROOT / SOURCE_AGGREGATE_PATH
    if sha256_file(aggregate_path) != SOURCE_AGGREGATE_SHA256:
        raise ValueError("protected-lane source aggregate hash drifted")
    if sha256_file(REPO_ROOT / ANALYSIS_PLAN_PATH) != (
        "63456020d89b2f5003559b233d196894d2cf8b35752e279e4c0b839123b49da9"
    ):
        raise ValueError("analysis protocol hash drifted")
    if sha256_file(REPO_ROOT / RUNNER_PATH) != RUNNER_SHA256:
        raise ValueError("synthetic runner hash drifted")
    if sha256_file(REPO_ROOT / MODEL_PATH) != MODEL_SHA256:
        raise ValueError("protected-lane model source hash drifted")
    aggregate = _load_json(aggregate_path)
    if aggregate.get("status") != "completed" or aggregate.get("completed_arm_count") != 8:
        raise ValueError("protected-lane aggregate is not the completed eight-arm axis")
    if aggregate.get("expected_parameter_count") != EXPECTED_PARAMETERS:
        raise ValueError("aggregate parameter count mismatch")
    return aggregate


def _load_arm(
    arm: dict[str, Any],
    device: torch.device,
) -> tuple[A1AssociativeLaneLM, dict[str, Any]]:
    checkpoint_info = arm["final_checkpoint"]
    checkpoint_path = REPO_ROOT / checkpoint_info["path"]
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    verified = verify_checkpoint(checkpoint_path)
    actual_hash = sha256_file(checkpoint_path)
    if actual_hash != checkpoint_info["sha256"]:
        raise ValueError(f"checkpoint hash mismatch: {checkpoint_path}")
    if verified.get("checkpoint_sha256") != checkpoint_info["sha256"]:
        raise ValueError(f"checkpoint sidecar hash mismatch: {checkpoint_path}")
    config_path = REPO_ROOT / arm["config_path"]
    config = load_config(config_path)
    model = A1AssociativeLaneLM(
        config,
        mode=arm["mode"],
        lane_width=LANE_WIDTH,
        use_parallel_scan=True,
    )
    if model.base.parameter_count() != EXPECTED_BASE_PARAMETERS:
        raise ValueError("base parameter count mismatch")
    if model.parameter_count() != EXPECTED_PARAMETERS:
        raise ValueError("lane parameter count mismatch")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.eval().to(device)
    metadata = {
        "distance": int(arm["distance"]),
        "mode": arm["mode"],
        "run_name": arm["run_name"],
        "checkpoint": checkpoint_info,
        "checkpoint_verified": True,
        "config_path": arm["config_path"],
        "config_sha256": arm["config_sha256"],
        "parameter_count": model.parameter_count(),
    }
    return model, metadata


def _run_arm(arm: dict[str, Any], device: torch.device) -> dict[str, Any]:
    model, metadata = _load_arm(arm, device)
    before_digest = _parameter_digest(model)
    distance = int(arm["distance"])
    train_examples = make_evaluation_examples(
        SEED, "train", VALUE_COUNT, 1, distance, 1
    )
    heldout_examples = make_evaluation_examples(
        SEED, "heldout", VALUE_COUNT, 1, distance, 1
    )
    train = _capture_split(model, train_examples, distance, device)
    heldout = _capture_split(model, heldout_examples, distance, device)
    embedding_weight = model.base.token_embedding.weight.detach().float().cpu()
    value_embeddings = embedding_weight[VALUE_BASE : VALUE_BASE + VALUE_COUNT]
    key_start = KEY_TRAIN_BASE
    key_end = KEY_HELDOUT_BASE + FIXED_EXAMPLES
    key_embeddings = embedding_weight[key_start:key_end]
    if value_embeddings.shape[0] != VALUE_COUNT:
        raise ValueError("value embedding pool shape mismatch")
    if key_embeddings.shape[0] != key_end - key_start:
        raise ValueError("key embedding pool shape mismatch")
    value_probe = _fixed_probe_results(train, heldout, value_embeddings, key_embeddings)
    cv_probe = _cv_probe_results(heldout, value_embeddings, key_embeddings)
    target_tokens = heldout["value_targets"] + VALUE_BASE
    native_metrics = _output_metrics(
        heldout["native_logits"], heldout["value_targets"], target_tokens
    )
    base_metrics = _output_metrics(
        heldout["base_logits"], heldout["value_targets"], target_tokens
    )
    target_native = heldout["native_logits"].gather(1, target_tokens[:, None]).squeeze(1)
    target_base = heldout["base_logits"].gather(1, target_tokens[:, None]).squeeze(1)
    lane_target_delta = target_native - target_base
    after_digest = _parameter_digest(model)
    if before_digest != after_digest:
        raise RuntimeError(f"analysis changed model parameters: {metadata['run_name']}")
    result = {
        **metadata,
        "source_train_examples": len(train_examples),
        "source_heldout_examples": len(heldout_examples),
        "stage_positions": heldout["stage_positions"],
        "native_output": native_metrics,
        "base_only_output": base_metrics,
        "lane_delta_output_effect": {
            "mean_target_logit_delta": _finite(lane_target_delta.mean(), "lane target logit delta"),
            "mean_target_probability_delta": _finite(
                (
                    (heldout["native_logits"] - heldout["native_logits"].logsumexp(dim=-1, keepdim=True)).exp()
                    .gather(1, target_tokens[:, None])
                    .squeeze(1)
                    - (heldout["base_logits"] - heldout["base_logits"].logsumexp(dim=-1, keepdim=True)).exp()
                    .gather(1, target_tokens[:, None])
                    .squeeze(1)
                ).mean(),
                "lane target probability delta",
            ),
            "mean_target_rank_delta_full": _finite(
                (
                    1
                    + (heldout["native_logits"] > target_native[:, None]).sum(dim=1)
                    - 1
                    - (heldout["base_logits"] > target_base[:, None]).sum(dim=1)
                ).float().mean(),
                "lane full rank delta",
            ),
        },
        "state_diagnostics": {
            "train": train["diagnostic_means"],
            "heldout": heldout["diagnostic_means"],
        },
        "fixed_transfer_probes": value_probe,
        "within_heldout_cv_probes": cv_probe,
        "parameter_digest_before": before_digest,
        "parameter_digest_after": after_digest,
        "weights_unchanged": True,
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _summary_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        fixed = run["fixed_transfer_probes"]
        rows.append(
            {
                "distance": run["distance"],
                "mode": run["mode"],
                "native_value_top1": run["native_output"]["value_candidate_range"]["top1_accuracy"],
                "native_value_ce": run["native_output"]["value_candidate_range"]["cross_entropy"],
                "memory_value_post_value_top1": fixed["value_identity"]["post_value"].get("memory_value", {}).get("top1_accuracy"),
                "memory_value_post_distractor_top1": fixed["value_identity"]["post_distractor"].get("memory_value", {}).get("top1_accuracy"),
                "memory_value_post_query_top1": fixed["value_identity"]["post_query_key"].get("memory_value", {}).get("top1_accuracy"),
                "memory_value_answer_top1": fixed["value_identity"]["answer_marker"].get("memory_value", {}).get("top1_accuracy"),
                "pair_answer_top1": fixed["query_conditioned_selection"]["answer_marker"].get("pair_state", {}).get("top1_accuracy"),
                "query_conditioned_answer_top1": fixed["query_conditioned_selection"]["answer_marker"].get("query_conditioned", {}).get("top1_accuracy"),
                "output_representation_answer_top1": fixed["output_representation"]["answer_marker"].get("top1_accuracy"),
                "lane_target_logit_delta": run["lane_delta_output_effect"]["mean_target_logit_delta"],
            }
        )
    return rows


def _report(result: dict[str, Any]) -> str:
    lines = [
        "# A1 protected-lane query/binding localization result",
        "",
        "Status: **complete; read-only frozen-checkpoint analysis**.",
        "",
        f"Protocol: `{result['analysis_plan_path']}` (sha256 `{result['analysis_plan_sha256']}`).",
        f"Source aggregate: `{SOURCE_AGGREGATE_PATH}` (sha256 `{result['source_aggregate_sha256']}`).",
        "",
        "The primary stages are post-value, post-distractor, post-query-key, and answer-marker. Fixed probes are fit on the 128-example training split and evaluated on the disjoint held-out split; within-held-out cross-validation is retained as a local decodability diagnostic.",
        "",
        "## Summary",
        "",
        "| distance | mode | native value top-1 | native value CE | memory value post-write | post-distractor | post-query | answer | pair answer | query-conditioned answer | output-representation answer | lane target-logit delta |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["summary_rows"]:
        fmt = lambda value: "n/a" if value is None else f"{float(value):.4f}"
        lines.append(
            f"| {row['distance']} | {row['mode']} | {fmt(row['native_value_top1'])} | {fmt(row['native_value_ce'])} | "
            f"{fmt(row['memory_value_post_value_top1'])} | {fmt(row['memory_value_post_distractor_top1'])} | "
            f"{fmt(row['memory_value_post_query_top1'])} | {fmt(row['memory_value_answer_top1'])} | "
            f"{fmt(row['pair_answer_top1'])} | {fmt(row['query_conditioned_answer_top1'])} | "
            f"{fmt(row['output_representation_answer_top1'])} | {fmt(row['lane_target_logit_delta'])} |"
        )
    lines.extend(
        [
            "",
            "## Reading boundary",
            "",
            "A fixed-probe drop before the query localizes a cross-example storage/coordinate problem. Survival through post-distractor followed by a drop at post-query-key supports query-conditioned transformation. A strong pair/query-conditioned diagnostic with weak native output supports downstream integration or readout failure. These are frozen-state localization results, not causal claims about a new intervention.",
            "",
            "## Integrity",
            "",
            f"Analyzed arms: {len(result['runs'])}; every final checkpoint and sidecar was hash verified. Parameter digests were unchanged for every arm. All tensors, probes, logits, and aggregate metrics were finite. No checkpoint was written.",
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
        default=REPO_ROOT / "results/a1_protected_lane_query_binding_localization.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "docs/A1_PROTECTED_LANE_QUERY_BINDING_LOCALIZATION_RESULT.md",
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    aggregate = _verify_sources()
    arms = sorted(
        aggregate["arms"], key=lambda arm: (int(arm["distance"]), str(arm["mode"]))
    )
    if [(int(arm["distance"]), arm["mode"]) for arm in arms] != [
        (distance, mode) for distance in DISTANCES for mode in MODES
    ]:
        raise ValueError("source aggregate arm order/content mismatch")
    runs = [_run_arm(arm, device) for arm in arms]
    result = {
        "version": OUTPUT_VERSION,
        "status": "complete",
        "analysis_plan_path": ANALYSIS_PLAN_PATH,
        "analysis_plan_commit": ANALYSIS_PLAN_COMMIT,
        "analysis_plan_sha256": sha256_file(REPO_ROOT / ANALYSIS_PLAN_PATH),
        "analysis_script": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
        "analysis_script_sha256": sha256_file(Path(__file__)),
        "source_aggregate_path": SOURCE_AGGREGATE_PATH,
        "source_aggregate_sha256": SOURCE_AGGREGATE_SHA256,
        "runner_sha256": RUNNER_SHA256,
        "protected_lane_model_sha256": MODEL_SHA256,
        "seed": SEED,
        "value_count": VALUE_COUNT,
        "distances": list(DISTANCES),
        "modes": list(MODES),
        "primary_stages": list(PRIMARY_STAGES),
        "all_stages": list(ALL_STAGES),
        "runs": runs,
        "summary_rows": _summary_rows(runs),
        "integrity": {
            "source_aggregate_verified": True,
            "all_final_checkpoints_verified": True,
            "all_metrics_finite": True,
            "all_parameter_digests_unchanged": True,
            "weights_modified_by_analysis": False,
            "checkpoints_written_by_analysis": False,
        },
    }
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    report = args.report if args.report.is_absolute() else REPO_ROOT / args.report
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_report(result), encoding="utf-8")
    print(json.dumps(result["summary_rows"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
