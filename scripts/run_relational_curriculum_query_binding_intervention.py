#!/usr/bin/env python3
"""Run the preregistered A1 query-conditioned binding intervention."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from braidformer.models import A1RecurrentStateSpaceLM, CausalLMOutput
from braidformer.training.integrity import sha256_file, verify_checkpoint
from scripts import analyze_relational_curriculum_representation_probes as curriculum
from scripts import run_relational_curriculum_experiment as curriculum_runner
from scripts import run_relational_curriculum_value_trace_intervention as value_trace_runner


PLAN_PATH = "docs/A0_A1_RELATIONAL_CURRICULUM_QUERY_BINDING_INTERVENTION_PROTOCOL.md"
SOURCE_RESULT_PATH = "results/a0_a1_relational_curriculum_v1.json"
SOURCE_RESULT_SHA256 = "66f68cfb3a2f84f5e2193dfcdd606c9c5ef9e8e111aee3e99254ce5c975d8d71"
VALUE_TRACE_RESULT_PATH = "results/a0_a1_relational_curriculum_value_trace_intervention.json"
VALUE_TRACE_RESULT_SHA256 = "13dbace59f46a16ff6fc30e00532500b582f5a1ad5a8df8210297604e2532112"
VALUE_TRACE_COMMIT = "fdcd032d640053084b84c204b8e737fe72242464"
CONFIG_PATH = "configs/braidnano_a1_relational_curriculum_v1.json"
QUERY_MARKER = 30_002
VALUE_BASE = 8_000
VALUE_COUNT = 8_000
AUX_WEIGHT = 0.1
AUX_TEMPERATURE = 0.1
EXPECTED_PARAMS = 41_870_336
ARMS = ("staged", "shuffled")
OUTPUT_VERSION = "relational_curriculum_query_binding_intervention_v1"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _assert_finite(value: Any, label: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"non-finite value at {label}: {value!r}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite(item, f"{label}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite(item, f"{label}.{key}")
        return
    raise TypeError(f"unsupported value at {label}: {type(value).__name__}")


def _finite(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite {label}: {value!r}")
    return number


def _current_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def query_conditioned_binding_loss(
    states: torch.Tensor,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    token_embedding_weight: torch.Tensor,
    *,
    value_base: int = VALUE_BASE,
    value_count: int = VALUE_COUNT,
    query_marker: int = QUERY_MARKER,
    temperature: float = AUX_TEMPERATURE,
) -> torch.Tensor:
    """Score the target value from the query-key-conditioned layer-7 state.

    This is intentionally parameter-free beyond the existing tied embedding
    table. The elementwise product is the fixed joint key/state operation; it
    prevents this objective from collapsing to the prior unconditioned value
    trace, which scored only the later answer-position state.
    """

    if states.ndim != 3 or input_ids.ndim != 2 or labels.ndim != 2:
        raise ValueError("states/input_ids/labels have unexpected rank")
    if states.shape[:2] != input_ids.shape or labels.shape != input_ids.shape:
        raise ValueError("states/input_ids/labels shapes do not agree")
    batch_size, sequence_length = input_ids.shape

    marker_positions = input_ids.eq(query_marker).nonzero(as_tuple=False)
    marker_counts = torch.bincount(marker_positions[:, 0], minlength=batch_size)
    if marker_positions.shape[0] != batch_size or not bool(torch.all(marker_counts == 1)):
        raise ValueError("each training example must contain exactly one query marker")
    query_positions = torch.empty(batch_size, dtype=torch.long, device=input_ids.device)
    query_positions[marker_positions[:, 0]] = marker_positions[:, 1] + 1
    if bool(torch.any(query_positions >= sequence_length)):
        raise ValueError("query marker has no following query key")

    target_mask = (labels >= value_base) & (labels < value_base + value_count)
    target_positions = target_mask.nonzero(as_tuple=False)
    target_counts = torch.bincount(target_positions[:, 0], minlength=batch_size)
    if target_positions.shape[0] != batch_size or not bool(torch.all(target_counts == 1)):
        raise ValueError("each training example must contain exactly one value target")
    target_indices = torch.empty(batch_size, dtype=torch.long, device=labels.device)
    target_indices[target_positions[:, 0]] = target_positions[:, 1]

    rows = torch.arange(batch_size, device=input_ids.device)
    query_key_ids = input_ids[rows, query_positions]
    query_states = states[rows, query_positions].float()
    target_ids = labels[rows, target_indices].long() - value_base
    key_vectors = token_embedding_weight[query_key_ids].float()
    value_vectors = token_embedding_weight[value_base : value_base + value_count].float()
    joint = F.normalize(query_states, dim=-1) * F.normalize(key_vectors, dim=-1)
    logits = (joint @ F.normalize(value_vectors, dim=-1).T) / temperature
    loss = F.cross_entropy(logits, target_ids)
    if not bool(torch.isfinite(loss).item()):
        raise RuntimeError("non-finite query-conditioned binding auxiliary loss")
    return loss


class QueryBindingA1(A1RecurrentStateSpaceLM):
    """Original A1 with a training-only, parameter-free binding objective."""

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> CausalLMOutput:
        captured: list[torch.Tensor] = []

        def state_hook(
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
            captured.append(states)

        handle = None
        if self.training and labels is not None:
            handle = self.blocks[7].state_space.register_forward_hook(state_hook)
        try:
            native = super().forward(input_ids, labels=labels)
        finally:
            if handle is not None:
                handle.remove()

        if not self.training or labels is None or native.loss is None:
            return native
        if len(captured) != 1:
            raise RuntimeError(f"expected one layer-7 state capture, got {len(captured)}")
        auxiliary_loss = query_conditioned_binding_loss(
            captured[0],
            input_ids,
            labels,
            self.token_embedding.weight,
        )
        return CausalLMOutput(
            logits=native.logits,
            loss=native.loss + AUX_WEIGHT * auxiliary_loss,
        )


def _verify_source(path: Path) -> dict[str, Any]:
    if sha256_file(path) != SOURCE_RESULT_SHA256:
        raise ValueError(f"source aggregate hash mismatch: {path}")
    source = _load_json(path)
    if source.get("status") != "PASS":
        raise ValueError("source curriculum aggregate is not PASS")
    for architecture, arm in (("A1", "staged"), ("A1", "shuffled")):
        curriculum._verify_summary(architecture, arm, source)
    return source


def _verify_value_trace(path: Path) -> dict[str, Any]:
    if sha256_file(path) != VALUE_TRACE_RESULT_SHA256:
        raise ValueError(f"value-trace aggregate hash mismatch: {path}")
    result = _load_json(path)
    if result.get("status") != "complete":
        raise ValueError("value-trace aggregate is not complete")
    if result.get("analysis_commit") != VALUE_TRACE_COMMIT:
        raise ValueError("value-trace aggregate protocol commit mismatch")
    arms = {run.get("arm") for run in result.get("runs", [])}
    if arms != set(ARMS):
        raise ValueError(f"value-trace aggregate arms mismatch: {arms!r}")
    for run in result["runs"]:
        verified = value_trace_runner._verify_intervention_summary(
            run["summary"], run["arm"], VALUE_TRACE_COMMIT
        )
        if verified["checkpoint_sha256"] != run.get("checkpoint_sha256"):
            raise ValueError("value-trace aggregate/checkpoint mismatch")
    return result


def _augment_summary(summary: dict[str, Any], arm: str, run_dir: Path) -> dict[str, Any]:
    summary = dict(summary)
    summary["training_intervention"] = {
        "name": "layer7_query_conditioned_binding_auxiliary_objective",
        "aux_weight": AUX_WEIGHT,
        "aux_temperature": AUX_TEMPERATURE,
        "query_marker": QUERY_MARKER,
        "value_base": VALUE_BASE,
        "value_count": VALUE_COUNT,
        "joint_operation": "elementwise_product(normalize(query_state), normalize(query_key_embedding))",
        "query_state_site": "layer7_state_at_query_key_position",
        "training_only": True,
        "new_parameters": False,
        "arm": arm,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def _verify_intervention_summary(
    summary: dict[str, Any],
    arm: str,
    protocol_commit: str,
) -> dict[str, Any]:
    config_path = REPO_ROOT / CONFIG_PATH
    expected = {
        "status": "completed",
        "architecture": "A1",
        "arm": arm,
        "seed": 17,
        "parameter_count": EXPECTED_PARAMS,
        "config_path": CONFIG_PATH,
        "config_sha256": sha256_file(config_path),
        "protocol_commit": protocol_commit,
        "optimizer_steps": 2_000,
        "tokens_processed": 928_000,
        "tokens_per_step": 464,
        "order_independent_exposure_sha256": curriculum.EXPOSURE_SHA256,
    }
    for field, expected_value in expected.items():
        if summary.get(field) != expected_value:
            raise ValueError(
                f"intervention summary mismatch {field}: "
                f"{summary.get(field)!r} != {expected_value!r}"
            )
    checkpoint = Path(summary["checkpoint"])
    record = verify_checkpoint(checkpoint)
    if summary.get("checkpoint_sha256") != record.get("checkpoint_sha256"):
        raise ValueError("intervention summary/checkpoint digest mismatch")
    for field, expected_value in {
        "config_sha256": expected["config_sha256"],
        "seed": 17,
        "step": 2_000,
        "tokens_processed": 928_000,
        "parameter_count": EXPECTED_PARAMS,
        "protocol_commit": protocol_commit,
    }.items():
        if record.get(field) != expected_value:
            raise ValueError(f"intervention checkpoint sidecar mismatch: {field}")
    _assert_finite(summary, f"intervention summary {arm}")
    return {
        "arm": arm,
        "run_dir": summary["run_dir"],
        "summary_path": str(Path(summary["run_dir"]) / "summary.json"),
        "summary": summary,
        "checkpoint": summary["checkpoint"],
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "cuda_training_seconds": summary["cuda_training_seconds"],
        "training_wall_seconds": summary["training_wall_seconds"],
    }


def _deltas(control: dict[str, Any], observed: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, control_metrics in control.items():
        actual = observed[name]
        result[name] = {
            "control": control_metrics,
            "intervention": actual,
            "delta_target_loss": _finite(
                actual["target_loss"] - control_metrics["target_loss"],
                f"{name} loss delta",
            ),
            "delta_greedy_exact_match": _finite(
                actual["greedy_exact_match"] - control_metrics["greedy_exact_match"],
                f"{name} exact-match delta",
            ),
            "delta_greedy_answer_token_accuracy": _finite(
                actual["greedy_answer_token_accuracy"]
                - control_metrics["greedy_answer_token_accuracy"],
                f"{name} token-accuracy delta",
            ),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-approved", action="store_true")
    parser.add_argument("--analysis-commit", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results/a0_a1_relational_curriculum_query_binding_intervention.json",
    )
    args = parser.parse_args()
    if not args.execute_approved:
        raise SystemExit("training is gated; pass --execute-approved after the frozen protocol")
    current_commit = _current_commit()
    if current_commit != args.analysis_commit:
        raise RuntimeError(
            f"analysis commit mismatch: current {current_commit}, expected {args.analysis_commit}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    source = _verify_source(REPO_ROOT / SOURCE_RESULT_PATH)
    value_trace = _verify_value_trace(REPO_ROOT / VALUE_TRACE_RESULT_PATH)
    plan_path = REPO_ROOT / PLAN_PATH
    if not plan_path.is_file():
        raise FileNotFoundError(plan_path)

    original_builder = curriculum_runner._build_model

    def build_intervention_model(config: Any) -> torch.nn.Module:
        if config.stage != "A1":
            raise RuntimeError("query-conditioned binding intervention is A1-only")
        model = QueryBindingA1(config, use_parallel_scan=True)
        if model.parameter_count() != EXPECTED_PARAMS:
            raise RuntimeError("query-conditioned model parameter count changed")
        return model

    curriculum_runner._build_model = build_intervention_model
    summaries: list[dict[str, Any]] = []
    try:
        for arm in ARMS:
            run_dir = (
                REPO_ROOT
                / "results"
                / "runs"
                / f"relational_curriculum_v1_a1_query_binding_{arm}"
            )
            summary = curriculum_runner.run(
                config_path=REPO_ROOT / CONFIG_PATH,
                run_dir=run_dir,
                seed=17,
                arm=arm,
                protocol_commit=args.analysis_commit,
            )
            summary = _augment_summary(summary, arm, run_dir)
            summaries.append(_verify_intervention_summary(summary, arm, args.analysis_commit))
            torch.cuda.empty_cache()
    finally:
        curriculum_runner._build_model = original_builder

    value_trace_by_arm = {run["arm"]: run for run in value_trace["runs"]}
    runs = []
    for intervention in summaries:
        arm = intervention["arm"]
        observed = intervention["summary"]["final_evaluation"]
        runs.append(
            {
                "arm": arm,
                "run_dir": intervention["run_dir"],
                "summary_path": intervention["summary_path"],
                "checkpoint": intervention["checkpoint"],
                "checkpoint_sha256": intervention["checkpoint_sha256"],
                "cuda_training_seconds": intervention["cuda_training_seconds"],
                "training_wall_seconds": intervention["training_wall_seconds"],
                "final_evaluation_deltas_vs_original_a1": _deltas(
                    source["arms"][f"a1_{arm}"]["final_evaluation"], observed
                ),
                "final_evaluation_deltas_vs_value_trace": _deltas(
                    value_trace_by_arm[arm]["summary"]["final_evaluation"], observed
                ),
                "summary": intervention["summary"],
            }
        )

    result: dict[str, Any] = {
        "status": "complete",
        "version": OUTPUT_VERSION,
        "analysis_commit": args.analysis_commit,
        "analysis_plan_path": PLAN_PATH,
        "analysis_plan_sha256": sha256_file(plan_path),
        "training_runner_path": "scripts/run_relational_curriculum_experiment.py",
        "training_runner_sha256": sha256_file(
            REPO_ROOT / "scripts/run_relational_curriculum_experiment.py"
        ),
        "source_result_path": SOURCE_RESULT_PATH,
        "source_result_sha256": SOURCE_RESULT_SHA256,
        "value_trace_result_path": VALUE_TRACE_RESULT_PATH,
        "value_trace_result_sha256": VALUE_TRACE_RESULT_SHA256,
        "value_trace_protocol_commit": VALUE_TRACE_COMMIT,
        "order_independent_exposure_sha256": curriculum.EXPOSURE_SHA256,
        "architecture": "A1",
        "seed": 17,
        "arms": list(ARMS),
        "target_lengths": [2, 4, 8],
        "optimizer_steps": 2_000,
        "tokens_processed": 928_000,
        "parameter_count": EXPECTED_PARAMS,
        "intervention": {
            "name": "layer7_query_conditioned_binding_auxiliary_objective",
            "aux_weight": AUX_WEIGHT,
            "aux_temperature": AUX_TEMPERATURE,
            "query_marker": QUERY_MARKER,
            "value_base": VALUE_BASE,
            "value_count": VALUE_COUNT,
            "joint_operation": "elementwise_product(normalize(query_state), normalize(query_key_embedding))",
            "query_state_site": "layer7_state_at_query_key_position",
            "training_only": True,
            "new_parameters": False,
            "architecture_definition_unchanged": True,
        },
        "untouched_controls": {
            "a1_staged_checkpoint_sha256": source["arms"]["a1_staged"]["checkpoint_sha256"],
            "a1_shuffled_checkpoint_sha256": source["arms"]["a1_shuffled"]["checkpoint_sha256"],
            "a0_staged_checkpoint_sha256": source["arms"]["a0_staged"]["checkpoint_sha256"],
            "a0_shuffled_checkpoint_sha256": source["arms"]["a0_shuffled"]["checkpoint_sha256"],
            "value_trace_staged_checkpoint_sha256": value_trace_by_arm["staged"][
                "checkpoint_sha256"
            ],
            "value_trace_shuffled_checkpoint_sha256": value_trace_by_arm["shuffled"][
                "checkpoint_sha256"
            ],
        },
        "runs": runs,
    }
    _assert_finite(result, "result")
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "runs": len(runs), "status": result["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
