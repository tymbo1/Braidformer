#!/usr/bin/env python3
"""Inference-only exact-address intervention for the protected lane.

The intervention keeps the frozen lane value state and trained readout, but
replaces the learned key-similarity gate at the answer marker with exact
record-key/query-key token equality.  It writes no checkpoint and never
updates a parameter.
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


PROTOCOL_PATH = "docs/A1_PROTECTED_LANE_EXACT_ADDRESS_PROTOCOL.md"
PROTOCOL_COMMIT = "f9badbf69dc016f1bd2a29ccae5e9ca30dbb86ee"
PROTOCOL_SHA256 = "9a7a7b0105eeefe8032e2b77d24b2e382876445c444ee269fece2d2bdd95cc43"
SOURCE_AGGREGATE_PATH = "results/a1_protected_lane_distance_axis_values8000.json"
SOURCE_AGGREGATE_SHA256 = "afdeeb42b2379f38ac7a592b805dcad1b2acadbeace0eaa57c43472f4472a158"
LOCALIZATION_PATH = "results/a1_protected_lane_query_binding_localization.json"
LOCALIZATION_SHA256 = "97d2b0ecf6659dffec052a1a31cc83b3af57fc50a6cbbe32fb391c2d6124f5c0"
RUNNER_PATH = "scripts/run_synthetic_overfit_calibration.py"
RUNNER_SHA256 = "2a1a1c84f1ebfabc11cf66f0defa2e2ad4db2d1741c89c3d12660d3ac066d3e5"
MODEL_PATH = "src/braidformer/models/protected_lane.py"
MODEL_SHA256 = "d776dea620662187c2698e604b8aa171a52e5b911adaa738daea7ffca13f421c"
OUTPUT_VERSION = "a1_protected_lane_exact_address_intervention_v1"
SEED = 17
VALUE_COUNT = 8_000
EXPECTED_BASE_PARAMETERS = 41_870_336
EXPECTED_PARAMETERS = EXPECTED_BASE_PARAMETERS + EXPECTED_LANE_PARAMETERS
DISTANCES = (0, 4, 16, 64)
MODES = ("protected", "ordinary")
BATCH_SIZE = 16


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


def _positions(distance: int) -> tuple[int, int]:
    return 6 + distance, 5 + distance


def _output_metrics(
    logits: torch.Tensor,
    value_targets: torch.Tensor,
    target_tokens: torch.Tensor,
) -> dict[str, Any]:
    logits = _finite_tensor(logits.float(), "answer logits")
    values = logits[:, VALUE_BASE : VALUE_BASE + VALUE_COUNT]
    full_target = logits.gather(1, target_tokens[:, None]).squeeze(1)
    value_target = values.gather(1, value_targets[:, None]).squeeze(1)
    full_incorrect = logits.clone()
    full_incorrect.scatter_(1, target_tokens[:, None], float("-inf"))
    value_incorrect = values.clone()
    value_incorrect.scatter_(1, value_targets[:, None], float("-inf"))
    full_rank = 1 + (logits > full_target[:, None]).sum(dim=1)
    value_rank = 1 + (values > value_target[:, None]).sum(dim=1)
    full_log_prob = full_target - torch.logsumexp(logits, dim=-1)
    value_log_prob = value_target - torch.logsumexp(values, dim=-1)
    return {
        "examples": int(logits.shape[0]),
        "full_vocabulary": {
            "cross_entropy": _finite((-full_log_prob).mean(), "full CE"),
            "top1_accuracy": _finite(logits.argmax(dim=-1).eq(target_tokens).float().mean(), "full top1"),
            "mean_target_probability": _finite(full_log_prob.exp().mean(), "full probability"),
            "mean_target_margin": _finite(
                (full_target - full_incorrect.amax(dim=-1)).mean(), "full margin"
            ),
            "mean_target_rank": _finite(full_rank.float().mean(), "full rank"),
        },
        "value_candidate_range": {
            "cross_entropy": _finite((-value_log_prob).mean(), "value CE"),
            "top1_accuracy": _finite(values.argmax(dim=-1).eq(value_targets).float().mean(), "value top1"),
            "mean_target_probability": _finite(value_log_prob.exp().mean(), "value probability"),
            "mean_target_margin": _finite(
                (value_target - value_incorrect.amax(dim=-1)).mean(), "value margin"
            ),
            "mean_target_rank": _finite(value_rank.float().mean(), "value rank"),
        },
    }


def _load_arm(arm: dict[str, Any], device: torch.device):
    checkpoint_info = arm["final_checkpoint"]
    checkpoint_path = REPO_ROOT / checkpoint_info["path"]
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    verified = verify_checkpoint(checkpoint_path)
    if sha256_file(checkpoint_path) != checkpoint_info["sha256"]:
        raise ValueError(f"checkpoint hash mismatch: {checkpoint_path}")
    if verified.get("checkpoint_sha256") != checkpoint_info["sha256"]:
        raise ValueError(f"checkpoint sidecar hash mismatch: {checkpoint_path}")
    config = load_config(REPO_ROOT / arm["config_path"])
    model = A1AssociativeLaneLM(
        config, mode=arm["mode"], lane_width=LANE_WIDTH, use_parallel_scan=True
    )
    if model.base.parameter_count() != EXPECTED_BASE_PARAMETERS:
        raise ValueError("base parameter count mismatch")
    if model.parameter_count() != EXPECTED_PARAMETERS:
        raise ValueError("lane parameter count mismatch")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.eval().to(device)
    return model, {
        "distance": int(arm["distance"]),
        "mode": arm["mode"],
        "run_name": arm["run_name"],
        "checkpoint": checkpoint_info,
        "config_path": arm["config_path"],
        "config_sha256": arm["config_sha256"],
        "parameter_count": model.parameter_count(),
    }


def _run_arm(arm: dict[str, Any], device: torch.device) -> dict[str, Any]:
    model, metadata = _load_arm(arm, device)
    before_digest = _parameter_digest(model)
    distance = int(arm["distance"])
    answer_position, query_key_position = _positions(distance)
    examples = make_evaluation_examples(SEED, "heldout", VALUE_COUNT, 1, distance, 1)
    native_logit_parts: list[torch.Tensor] = []
    patched_logit_parts: list[torch.Tensor] = []
    value_targets: list[int] = []
    target_tokens: list[int] = []
    native_gates: list[float] = []
    exact_gates: list[float] = []
    value_norms: list[float] = []
    value_cosines: list[float] = []
    native_delta_norms: list[float] = []
    patched_delta_norms: list[float] = []
    native_reference_max_abs_difference = 0.0

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
            combined = base_hidden + lane_delta
            native_all_logits = model.base.lm_head(model.base.final_norm(combined))
            native_answer = native_all_logits[:, answer_position, :]
            memory_value = trace["memory_value"][:, answer_position, :].float()
            post_value = trace["memory_value"][:, 3, :].float()
            learned_gate = trace["read_gate"][:, answer_position].float()
            exact_gate = inputs[:, 2].eq(inputs[:, query_key_position]).float()
            patched_delta = model.readout(
                exact_gate.to(dtype=memory_value.dtype).unsqueeze(-1) * memory_value
            )
            patched_answer_hidden = base_hidden[:, answer_position, :] + patched_delta
            patched_answer = model.base.lm_head(
                model.base.final_norm(patched_answer_hidden)
            )
            if start == 0:
                reference = model(inputs).logits[:, answer_position, :].float()
                native_reference_max_abs_difference = max(
                    native_reference_max_abs_difference,
                    float((reference - native_answer.float()).abs().max().cpu()),
                )

        if not bool(exact_gate.eq(1.0).all()):
            raise RuntimeError("canonical held-out examples did not have exact key equality")
        native_logit_parts.append(_finite_tensor(native_answer.float().cpu(), "native logits"))
        patched_logit_parts.append(_finite_tensor(patched_answer.float().cpu(), "patched logits"))
        native_gates.extend(learned_gate.cpu().tolist())
        exact_gates.extend(exact_gate.cpu().tolist())
        value_norms.extend(memory_value.norm(dim=-1).cpu().tolist())
        value_cosines.extend(torch.nn.functional.cosine_similarity(memory_value, post_value, dim=-1).cpu().tolist())
        native_delta_norms.extend(trace["lane_delta"][:, answer_position, :].float().norm(dim=-1).cpu().tolist())
        patched_delta_norms.extend(patched_delta.float().norm(dim=-1).cpu().tolist())
        for example in batch:
            value_targets.append(int(example.target) - VALUE_BASE)
            target_tokens.append(int(example.target))

    if native_reference_max_abs_difference > 1e-3:
        raise RuntimeError(
            f"functional native logits diverged from model forward: "
            f"{native_reference_max_abs_difference}"
        )

    native_logits = torch.cat(native_logit_parts, dim=0)
    patched_logits = torch.cat(patched_logit_parts, dim=0)
    value_target_tensor = torch.tensor(value_targets, dtype=torch.long)
    target_token_tensor = torch.tensor(target_tokens, dtype=torch.long)
    native_metrics = _output_metrics(native_logits, value_target_tensor, target_token_tensor)
    patched_metrics = _output_metrics(patched_logits, value_target_tensor, target_token_tensor)
    native_target = native_logits.gather(1, target_token_tensor[:, None]).squeeze(1)
    patched_target = patched_logits.gather(1, target_token_tensor[:, None]).squeeze(1)
    native_rank = 1 + (native_logits > native_target[:, None]).sum(dim=1)
    patched_rank = 1 + (patched_logits > patched_target[:, None]).sum(dim=1)
    after_digest = _parameter_digest(model)
    if before_digest != after_digest:
        raise RuntimeError(f"inference intervention changed parameters: {metadata['run_name']}")
    result = {
        **metadata,
        "heldout_examples": len(examples),
        "answer_position": answer_position,
        "query_key_position": query_key_position,
        "native_output": native_metrics,
        "exact_address_output": patched_metrics,
        "deltas_exact_minus_native": {
            "value_cross_entropy": patched_metrics["value_candidate_range"]["cross_entropy"]
            - native_metrics["value_candidate_range"]["cross_entropy"],
            "value_top1_accuracy": patched_metrics["value_candidate_range"]["top1_accuracy"]
            - native_metrics["value_candidate_range"]["top1_accuracy"],
            "value_mean_target_rank": patched_metrics["value_candidate_range"]["mean_target_rank"]
            - native_metrics["value_candidate_range"]["mean_target_rank"],
            "full_mean_target_rank": _finite(
                (patched_rank.float() - native_rank.float()).mean(),
                "patched-native full rank delta",
            ),
            "mean_target_logit": _finite(
                (patched_target - native_target).mean(), "patched-native target logit delta"
            ),
        },
        "gate_and_state": {
            "mean_native_learned_gate": _finite(sum(native_gates) / len(native_gates), "native gate"),
            "mean_exact_address_gate": _finite(sum(exact_gates) / len(exact_gates), "exact gate"),
            "exact_gate_all_one": all(value == 1.0 for value in exact_gates),
            "mean_memory_value_norm": _finite(sum(value_norms) / len(value_norms), "memory norm"),
            "mean_memory_value_cosine_to_post_value": _finite(
                sum(value_cosines) / len(value_cosines), "memory cosine"
            ),
            "mean_native_lane_delta_norm": _finite(
                sum(native_delta_norms) / len(native_delta_norms), "native lane delta norm"
            ),
            "mean_exact_address_delta_norm": _finite(
                sum(patched_delta_norms) / len(patched_delta_norms), "patched delta norm"
            ),
        },
        "native_reference_max_abs_difference": native_reference_max_abs_difference,
        "parameter_digest_before": before_digest,
        "parameter_digest_after": after_digest,
        "weights_unchanged": True,
        "checkpoint_written": False,
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _verify_sources() -> dict[str, Any]:
    if sha256_file(REPO_ROOT / PROTOCOL_PATH) != PROTOCOL_SHA256:
        raise ValueError("exact-address protocol hash drifted")
    if sha256_file(REPO_ROOT / SOURCE_AGGREGATE_PATH) != SOURCE_AGGREGATE_SHA256:
        raise ValueError("protected-lane source aggregate hash drifted")
    if sha256_file(REPO_ROOT / LOCALIZATION_PATH) != LOCALIZATION_SHA256:
        raise ValueError("localization result hash drifted")
    if sha256_file(REPO_ROOT / RUNNER_PATH) != RUNNER_SHA256:
        raise ValueError("synthetic runner hash drifted")
    if sha256_file(REPO_ROOT / MODEL_PATH) != MODEL_SHA256:
        raise ValueError("protected-lane model source hash drifted")
    aggregate = _load_json(REPO_ROOT / SOURCE_AGGREGATE_PATH)
    if aggregate.get("status") != "completed" or aggregate.get("completed_arm_count") != 8:
        raise ValueError("protected-lane source aggregate is incomplete")
    return aggregate


def _report(result: dict[str, Any]) -> str:
    lines = [
        "# A1 protected-lane exact-address intervention result",
        "",
        "Status: **complete; inference-only frozen-checkpoint intervention**.",
        "",
        f"Protocol: `{PROTOCOL_PATH}` (sha256 `{result['protocol_sha256']}`).",
        "",
        "The exact-address path replaced only the learned key-similarity/read gate at the answer marker. It retained each checkpoint's original stored value state and trained readout.",
        "",
        "| distance | mode | native value CE | exact-address value CE | native value rank | exact-address value rank | native top-1 | exact top-1 | CE delta | rank delta | target-logit delta |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in result["runs"]:
        native = run["native_output"]["value_candidate_range"]
        patched = run["exact_address_output"]["value_candidate_range"]
        delta = run["deltas_exact_minus_native"]
        lines.append(
            f"| {run['distance']} | {run['mode']} | {native['cross_entropy']:.4f} | {patched['cross_entropy']:.4f} | "
            f"{native['mean_target_rank']:.1f} | {patched['mean_target_rank']:.1f} | {native['top1_accuracy']:.4f} | "
            f"{patched['top1_accuracy']:.4f} | {delta['value_cross_entropy']:.4f} | "
            f"{delta['value_mean_target_rank']:.1f} | {delta['mean_target_logit']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Integrity",
            "",
            f"All {len(result['runs'])} final checkpoints and sidecars were verified. Canonical exact-address gates were one for every held-out example. Native functional logits matched the model forward within the recorded tolerance, all tensors and metrics were finite, parameter digests were unchanged, and no checkpoint was written.",
            "",
            "Interpretation is governed by the frozen protocol: this is a mechanistic oracle-like address isolation, not a production architecture or a training result.",
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
        default=REPO_ROOT / "results/a1_protected_lane_exact_address_intervention.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "docs/A1_PROTECTED_LANE_EXACT_ADDRESS_INTERVENTION_RESULT.md",
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    aggregate = _verify_sources()
    arms = sorted(
        aggregate["arms"],
        key=lambda arm: (int(arm["distance"]), MODES.index(str(arm["mode"]))),
    )
    expected = [(distance, mode) for distance in DISTANCES for mode in MODES]
    actual = [(int(arm["distance"]), arm["mode"]) for arm in arms]
    if actual != expected:
        raise ValueError(f"source arm mismatch: {actual}")
    runs = [_run_arm(arm, device) for arm in arms]
    result = {
        "version": OUTPUT_VERSION,
        "status": "complete",
        "protocol_path": PROTOCOL_PATH,
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "analysis_script": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
        "analysis_script_sha256": sha256_file(Path(__file__)),
        "source_aggregate_path": SOURCE_AGGREGATE_PATH,
        "source_aggregate_sha256": SOURCE_AGGREGATE_SHA256,
        "localization_path": LOCALIZATION_PATH,
        "localization_sha256": LOCALIZATION_SHA256,
        "runner_sha256": RUNNER_SHA256,
        "protected_lane_model_sha256": MODEL_SHA256,
        "seed": SEED,
        "value_count": VALUE_COUNT,
        "distances": list(DISTANCES),
        "modes": list(MODES),
        "runs": runs,
        "integrity": {
            "source_verified": True,
            "all_final_checkpoints_verified": True,
            "all_exact_gates_one": True,
            "all_native_reference_checks_finite": True,
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
    print(json.dumps(result["runs"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
