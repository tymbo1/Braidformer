#!/usr/bin/env python3
"""Read-only A1 decay and frozen-state forensic analysis.

The script never updates a model checkpoint.  It reconstructs each diagonal
state-space state from the mixer input, extracts learned transition time
scales, and fits a small ridge probe only on states immediately after the
stored value.  The same fixed probe is then evaluated after neutral markers
and at the answer readout position.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
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
from braidformer.training.integrity import sha256_file, verify_checkpoint


SEED = 17
VALUE_BASE = 8_000
VALUE_COUNT = 8_000
DISTRACTOR_MARKER = 30_004
MARKERS = {"task": 30_000, "record": 30_001, "query": 30_002, "answer": 30_003}
PROBE_TRAIN_EXAMPLES = 1_024
PROBE_EVAL_EXAMPLES = 512
PROBE_BATCH_SIZE = 128
PROBE_DISTANCES = (0, 4, 8, 16, 32, 64)
RIDGE_STRENGTH = 1e-3
EXPECTED_PARAMS = 41_870_336
EXPECTED_CHECKPOINTS = {
    "distance0": {
        "checkpoint": "results/runs/solvability_values8000_a1_seed17/checkpoint_step_002000.pt",
        "checkpoint_sha256": "550c669c663d63db56765df4ee4512edbecbc5bcd6552bcb922bd236b41a12e7",
        "config": "configs/braidnano_a1_solvability_values8000.json",
        "training_distance": 0,
    },
    "distance4": {
        "checkpoint": "results/runs/solvability_distance4_values8000_a1_seed17/checkpoint_step_002000.pt",
        "checkpoint_sha256": "8d370cb799e5f200528561c0473d2a794e847f81748a218b89096dc7bc94c981",
        "config": "configs/braidnano_a1_solvability_distance4_values8000.json",
        "training_distance": 4,
    },
    "distance16": {
        "checkpoint": "results/runs/solvability_distance16_values8000_a1_seed17/checkpoint_step_002000.pt",
        "checkpoint_sha256": "d6fd707bd81be88cd5111af7e36e103f4508bdc62acf26fd00cfc3874d1bf6c2",
        "config": "configs/braidnano_a1_solvability_distance16_values8000.json",
        "training_distance": 16,
    },
    "distance64": {
        "checkpoint": "results/runs/solvability_distance64_values8000_a1_seed17/checkpoint_step_002000.pt",
        "checkpoint_sha256": "c0b3ba35b668a02c6f9dfcff5e6951d7b971763b022b8f0bc1cb3f78975ebcae",
        "config": "configs/braidnano_a1_solvability_distance64_values8000.json",
        "training_distance": 64,
    },
}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT.resolve()))


def _finite(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite {label}: {value!r}")
    return number


def _probe_seed(split: str, index: int) -> int:
    offset = 0x13579BDF if split == "train" else 0x2468ACE0
    payload = f"a1_forensic_probe:{SEED}:{split}:{index}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") ^ offset


def _probe_example(split: str, index: int, distance: int) -> tuple[tuple[int, ...], int]:
    if split not in {"train", "heldout"}:
        raise ValueError(f"unknown probe split: {split}")
    if distance not in PROBE_DISTANCES:
        raise ValueError(f"unsupported probe distance: {distance}")
    rng = random.Random(_probe_seed(split, index))
    key_base = 1_000 if split == "train" else 20_000
    key = key_base + index
    value = VALUE_BASE + rng.randrange(VALUE_COUNT)
    input_ids = (
        MARKERS["task"],
        MARKERS["record"],
        key,
        value,
        *(DISTRACTOR_MARKER for _ in range(distance)),
        MARKERS["query"],
        key,
        MARKERS["answer"],
        value,
    )
    return tuple(input_ids), value


def _probe_batch(split: str, start: int, end: int, distance: int) -> tuple[torch.Tensor, torch.Tensor]:
    examples = [_probe_example(split, index, distance) for index in range(start, end)]
    inputs = torch.tensor([example[0] for example in examples], dtype=torch.long)
    targets = torch.tensor([example[1] - VALUE_BASE for example in examples], dtype=torch.long)
    return inputs, targets


def _time_constants(model: A1RecurrentStateSpaceLM) -> dict[str, Any]:
    layers = []
    all_taus = []
    all_decays = []
    for layer_index, block in enumerate(model.blocks):
        decay = torch.sigmoid(block.state_space.state_transition_logit.detach().float()).cpu()
        tau = (-1.0 / torch.log(decay)).clamp_min(0.0)
        half_life = (math.log(0.5) / torch.log(decay)).clamp_min(0.0)
        all_taus.append(tau)
        all_decays.append(decay)
        layers.append(
            {
                "layer": layer_index,
                "channels": int(decay.numel()),
                "decay_min": float(decay.min()),
                "decay_p10": float(torch.quantile(decay, 0.10)),
                "decay_median": float(torch.median(decay)),
                "decay_p90": float(torch.quantile(decay, 0.90)),
                "decay_max": float(decay.max()),
                "tau_min": float(tau.min()),
                "tau_median": float(torch.median(tau)),
                "tau_p90": float(torch.quantile(tau, 0.90)),
                "tau_max": float(tau.max()),
                "half_life_median": float(torch.median(half_life)),
                "channels_tau_at_least": {
                    str(threshold): int((tau >= threshold).sum())
                    for threshold in (4, 8, 16, 32, 64)
                },
            }
        )
    flat_tau = torch.cat(all_taus)
    flat_decay = torch.cat(all_decays)
    return {
        "channels": int(flat_tau.numel()),
        "decay_min": float(flat_decay.min()),
        "decay_median": float(torch.median(flat_decay)),
        "decay_max": float(flat_decay.max()),
        "tau_min": float(flat_tau.min()),
        "tau_median": float(torch.median(flat_tau)),
        "tau_p90": float(torch.quantile(flat_tau, 0.90)),
        "tau_max": float(flat_tau.max()),
        "channels_tau_at_least": {
            str(threshold): int((flat_tau >= threshold).sum())
            for threshold in (4, 8, 16, 32, 64)
        },
        "layers": layers,
    }


def _trace_batch(
    model: A1RecurrentStateSpaceLM,
    input_ids: torch.Tensor,
    state_position: int,
    answer_position: int,
    device: torch.device,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Return each layer's internal state and final readout state at positions."""

    captured: dict[int, torch.Tensor] = {}
    hooks = []

    def make_hook(layer_index: int):
        def hook(module: Any, inputs: tuple[torch.Tensor, ...], _output: torch.Tensor) -> None:
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
            captured[layer_index] = states[:, state_position, :].detach().float().cpu()

        return hook

    for layer_index, block in enumerate(model.blocks):
        hooks.append(block.state_space.register_forward_hook(make_hook(layer_index)))

    try:
        inputs = input_ids.to(device)
        positions = torch.arange(inputs.shape[1], device=device)
        hidden_states = model.token_embedding(inputs) + model.position_embedding(positions).unsqueeze(0)
        hidden_states = model.embedding_dropout(hidden_states)
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if device.type == "cuda"
            else nullcontext()
        )
        with torch.no_grad(), autocast:
            for block in model.blocks:
                hidden_states = block(hidden_states)
            final_hidden = model.final_norm(hidden_states)[:, answer_position, :]
            final_hidden = final_hidden.detach().float().cpu()
    finally:
        for hook in hooks:
            hook.remove()

    if sorted(captured) != list(range(len(model.blocks))):
        raise RuntimeError("did not capture every A1 recurrent layer")
    return [captured[index] for index in range(len(model.blocks))], final_hidden


def _collect_traces(
    model: A1RecurrentStateSpaceLM,
    split: str,
    count: int,
    distance: int,
    device: torch.device,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    state_position = 3 + distance
    answer_position = 6 + distance
    state_parts: list[list[torch.Tensor]] = [[] for _ in model.blocks]
    readout_parts: list[torch.Tensor] = []
    for start in range(0, count, PROBE_BATCH_SIZE):
        end = min(start + PROBE_BATCH_SIZE, count)
        inputs, _targets = _probe_batch(split, start, end, distance)
        layer_states, readout = _trace_batch(
            model, inputs, state_position, answer_position, device
        )
        for layer_index, state in enumerate(layer_states):
            state_parts[layer_index].append(state)
        readout_parts.append(readout)
    return [torch.cat(parts, dim=0) for parts in state_parts], torch.cat(readout_parts, dim=0)


def _fit_ridge_probe(features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    features = features.float()
    targets = targets.float()
    augmented = torch.cat((features, torch.ones((features.shape[0], 1))), dim=1)
    gram = augmented.T @ augmented / features.shape[0]
    scale = torch.diag(gram).mean().clamp_min(1e-6)
    gram = gram + RIDGE_STRENGTH * scale * torch.eye(gram.shape[0])
    cross = augmented.T @ targets / features.shape[0]
    return torch.linalg.solve(gram, cross)


def _probe_metrics(
    features: torch.Tensor,
    targets: torch.Tensor,
    probe: torch.Tensor,
    value_embeddings: torch.Tensor,
) -> dict[str, float]:
    augmented = torch.cat((features.float(), torch.ones((features.shape[0], 1))), dim=1)
    prediction = augmented @ probe
    prediction = F_normalize(prediction)
    embeddings = F_normalize(value_embeddings)
    scores = prediction @ embeddings.T
    topk = scores.topk(k=5, dim=-1).indices
    top1 = (topk[:, :1] == targets[:, None]).any(dim=1).float().mean()
    top5 = (topk == targets[:, None]).any(dim=1).float().mean()
    return {
        "top1_accuracy": float(top1),
        "top5_accuracy": float(top5),
        "mean_cosine_to_predicted_value": float(scores.gather(1, targets[:, None]).mean()),
    }


def F_normalize(value: torch.Tensor) -> torch.Tensor:
    return value / value.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _readout_metrics(
    readout: torch.Tensor,
    targets: torch.Tensor,
    value_embeddings: torch.Tensor,
) -> dict[str, float]:
    scores = readout @ value_embeddings.T
    target_scores = scores.gather(1, targets[:, None]).squeeze(1)
    ranks = 1 + (scores > target_scores[:, None]).sum(dim=1)
    topk = scores.topk(k=5, dim=-1).indices
    return {
        "top1_accuracy": float((topk[:, :1] == targets[:, None]).any(dim=1).float().mean()),
        "top5_accuracy": float((topk == targets[:, None]).any(dim=1).float().mean()),
        "mean_target_rank": float(ranks.float().mean()),
        "median_target_rank": float(torch.median(ranks.float())),
        "mean_target_logit": float(target_scores.mean()),
    }


def _state_drift(
    baseline: torch.Tensor,
    current: torch.Tensor,
) -> dict[str, float]:
    baseline_norm = baseline.norm(dim=-1).clamp_min(1e-8)
    current_norm = current.norm(dim=-1).clamp_min(1e-8)
    cosine = (baseline * current).sum(dim=-1) / (baseline_norm * current_norm)
    return {
        "mean_norm": float(current_norm.mean()),
        "mean_cosine_to_distance0": float(cosine.mean()),
        "p10_cosine_to_distance0": float(torch.quantile(cosine, 0.10)),
        "p90_cosine_to_distance0": float(torch.quantile(cosine, 0.90)),
    }


def _analyze_checkpoint(label: str, metadata: dict[str, Any], device: torch.device) -> dict[str, Any]:
    checkpoint_path = REPO_ROOT / metadata["checkpoint"]
    config_path = REPO_ROOT / metadata["config"]
    checkpoint_record = verify_checkpoint(checkpoint_path)
    if checkpoint_record.get("checkpoint_sha256") != metadata["checkpoint_sha256"]:
        raise ValueError(f"checkpoint digest mismatch: {checkpoint_path}")
    config = load_config(config_path)
    if config.stage != "A1" or config.model_type != "recurrent_state_space":
        raise ValueError(f"not an A1 config: {config_path}")
    model = A1RecurrentStateSpaceLM(config, use_parallel_scan=True)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    if model.parameter_count() != EXPECTED_PARAMS:
        raise ValueError(f"unexpected A1 parameter count: {model.parameter_count()}")
    model.eval().to(device)

    value_embeddings = model.token_embedding.weight.detach().float()[
        VALUE_BASE : VALUE_BASE + VALUE_COUNT
    ].cpu()
    time_constants = _time_constants(model)
    traces: dict[int, dict[str, Any]] = {}
    for distance in PROBE_DISTANCES:
        train_states, train_readout = _collect_traces(
            model, "train", PROBE_TRAIN_EXAMPLES, distance, device
        )
        eval_states, eval_readout = _collect_traces(
            model, "heldout", PROBE_EVAL_EXAMPLES, distance, device
        )
        _inputs, train_targets = _probe_batch("train", 0, PROBE_TRAIN_EXAMPLES, distance)
        _inputs, eval_targets = _probe_batch("heldout", 0, PROBE_EVAL_EXAMPLES, distance)
        traces[distance] = {
            "train_states": train_states,
            "eval_states": eval_states,
            "train_readout": train_readout,
            "eval_readout": eval_readout,
            "train_targets": train_targets,
            "eval_targets": eval_targets,
        }

    layer_results = []
    baseline = traces[0]
    for layer_index in range(len(model.blocks)):
        probe = _fit_ridge_probe(
            baseline["train_states"][layer_index],
            value_embeddings[baseline["train_targets"]],
        )
        distances = []
        for distance in PROBE_DISTANCES:
            current = traces[distance]
            distances.append(
                {
                    "distance": distance,
                    "state_drift": _state_drift(
                        baseline["eval_states"][layer_index], current["eval_states"][layer_index]
                    ),
                    "probe_train": _probe_metrics(
                        current["train_states"][layer_index],
                        current["train_targets"],
                        probe,
                        value_embeddings,
                    ),
                    "probe_heldout": _probe_metrics(
                        current["eval_states"][layer_index],
                        current["eval_targets"],
                        probe,
                        value_embeddings,
                    ),
                }
            )
        layer_results.append({"layer": layer_index, "distances": distances})

    readout_curve = []
    for distance in PROBE_DISTANCES:
        current = traces[distance]
        readout_curve.append(
            {
                "distance": distance,
                "train": _readout_metrics(
                    current["train_readout"], current["train_targets"], value_embeddings
                ),
                "heldout": _readout_metrics(
                    current["eval_readout"], current["eval_targets"], value_embeddings
                ),
            }
        )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "label": label,
        "checkpoint": metadata["checkpoint"],
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "config": metadata["config"],
        "config_sha256": sha256_file(config_path),
        "training_distance": metadata["training_distance"],
        "probe": {
            "train_examples": PROBE_TRAIN_EXAMPLES,
            "heldout_examples": PROBE_EVAL_EXAMPLES,
            "batch_size": PROBE_BATCH_SIZE,
            "distances": list(PROBE_DISTANCES),
            "ridge_strength": RIDGE_STRENGTH,
            "target": "frozen token-embedding vector for the correct value token",
            "fit_position": "state immediately after the VALUE token at distance 0",
        },
        "time_constants": time_constants,
        "layers": layer_results,
        "final_readout_curve": readout_curve,
    }


def _report(result: dict[str, Any]) -> str:
    lines = [
        "# A1 state/decay forensics on the 8000-value retrieval task",
        "",
        "Status: **complete; read-only analysis**. No checkpoint or model parameter was updated.",
        "",
        "The analysis uses the completed distance-0, distance-4, distance-16, and distance-64 A1 checkpoints. It extracts the learned diagonal decay scales, reconstructs internal recurrent states from forward hooks, fits a fixed ridge probe only at the post-value distance-0 state, and evaluates that same probe after neutral-marker distances.",
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
            f"{times['channels_tau_at_least']['16']:,} | {times['channels_tau_at_least']['64']:,} |"
        )
    lines.extend(
        [
            "",
            "The τ values are characteristic decay times for `state[t] = a state[t−1] + ...`, computed as `−1/log(a)`. They describe the learned transition coefficients, not a guarantee that information survives for exactly τ tokens because input gates, residual paths, feed-forward blocks, and later layers also contribute.",
            "",
            "## Frozen-state probe summary",
            "",
        "The table reports the best held-out value-vector probe accuracy across the eight recurrent layers at each distance. The probe is fit once at distance 0 and is not refit at later distances; this prevents later positions from hiding decay through a newly adapted decoder. Actual answer readout metrics are top-1/rank within the reserved 8,000-value candidate pool, not the full vocabulary.",
            "",
            "| Checkpoint | Distance | Best layer | Probe top-1 | Probe top-5 | Mean state cosine to d0 | Actual answer top-1 | Mean answer rank |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
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
            drift = best_point["state_drift"]
            readout = readout_by_distance[distance]
            lines.append(
                f"| {checkpoint['label']} | {distance} | {best_layer} | {best_accuracy:.4f} | "
                f"{best_point['probe_heldout']['top5_accuracy']:.4f} | "
                f"{drift['mean_cosine_to_distance0']:.4f} | "
                f"{readout['top1_accuracy']:.4f} | {readout['mean_target_rank']:.1f} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "A falling fixed-probe score together with falling state cosine would support loss or transformation of the binding trace inside the recurrent stack. A stable probe score with poor actual answer readout would instead point toward query interaction or readout failure. These measurements are diagnostic signatures, not causal proof: the probe is linear, its decoder is fit at distance 0, and this remains one model size and one seed.",
            "",
            "No longer-decay initialization, architectural change, target-length run, relational curriculum, or A2 intervention was performed. The next action should be selected from the measured signature rather than assumed in advance.",
            "",
            "## Integrity",
            "",
            "Each source checkpoint was sidecar-verified before loading. The model was placed in evaluation mode, gradients were disabled, and only temporary probe weights were fitted in memory. Source checkpoint files remain unchanged.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "a1_state_decay_forensics_values8000.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "docs" / "A1_STATE_DECAY_FORENSICS_VALUES8000.md",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    checkpoints = [
        _analyze_checkpoint(label, metadata, device)
        for label, metadata in EXPECTED_CHECKPOINTS.items()
    ]
    result = {
        "status": "complete",
        "experiment": "a1_state_decay_forensics_values8000",
        "seed": SEED,
        "value_count": VALUE_COUNT,
        "device": str(device),
        "checkpoints": checkpoints,
        "interpretation": {
            "state_probe": "The frozen linear probe is fit at the post-value state for distance 0 and evaluated without refitting after each neutral-marker distance.",
            "causal_limit": "Decay extraction and probe curves are correlational diagnostics; no intervention was run.",
            "next": "Use the measured time-scale and probe signature to choose a narrowly targeted A1 intervention, while keeping the original A1 and distance benchmark frozen.",
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
