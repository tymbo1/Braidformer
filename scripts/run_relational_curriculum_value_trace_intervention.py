#!/usr/bin/env python3
"""Run the one authorized A1 value-trace auxiliary-loss intervention."""

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
from scripts import run_relational_curriculum_experiment as curriculum_runner
from scripts import analyze_relational_curriculum_representation_probes as curriculum


PLAN_PATH = "docs/A0_A1_RELATIONAL_CURRICULUM_VALUE_TRACE_INTERVENTION_PLAN.md"
SOURCE_RESULT_PATH = "results/a0_a1_relational_curriculum_v1.json"
SOURCE_RESULT_SHA256 = "66f68cfb3a2f84f5e2193dfcdd606c9c5ef9e8e111aee3e99254ce5c975d8d71"
ADJUDICATION_PATH = "results/a0_a1_relational_curriculum_mechanism_adjudication.json"
ADJUDICATION_SHA256 = "93da144642e29a7d943d0604e307599eaa7fff1f4d4d9c108b47478890b14cdf"
CONFIG_PATH = "configs/braidnano_a1_relational_curriculum_v1.json"
TARGET_LENGTHS = (2, 4, 8)
ARMS = ("staged", "shuffled")
VALUE_BASE = 8_000
VALUE_COUNT = 8_000
AUX_WEIGHT = 0.1
AUX_TEMPERATURE = 0.1
EXPECTED_PARAMS = 41_870_336
OUTPUT_VERSION = "relational_curriculum_value_trace_intervention_v1"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _finite(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite {label}: {value!r}")
    return number


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


def _current_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


class ValueTraceA1(A1RecurrentStateSpaceLM):
    """A1 with a training-only loss; it introduces no trainable parameters."""

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

        target_labels = labels[:, 1:]
        target_mask = (target_labels >= VALUE_BASE) & (target_labels < VALUE_BASE + VALUE_COUNT)
        if not bool(target_mask.any().item()):
            raise RuntimeError("curriculum batch contained no value targets for auxiliary loss")
        prediction_states = captured[0][:, :-1, :][target_mask].float()
        targets = (target_labels[target_mask] - VALUE_BASE).long()
        value_embeddings = self.token_embedding.weight[VALUE_BASE : VALUE_BASE + VALUE_COUNT].float()
        auxiliary_logits = (
            F.normalize(prediction_states, dim=-1)
            @ F.normalize(value_embeddings, dim=-1).T
        ) / AUX_TEMPERATURE
        auxiliary_loss = F.cross_entropy(auxiliary_logits, targets)
        if not bool(torch.isfinite(auxiliary_loss).item()):
            raise RuntimeError("non-finite value-trace auxiliary loss")
        return CausalLMOutput(
            logits=native.logits,
            loss=native.loss + AUX_WEIGHT * auxiliary_loss,
        )


def _verify_source(source_path: Path) -> dict[str, Any]:
    if sha256_file(source_path) != SOURCE_RESULT_SHA256:
        raise ValueError(f"source aggregate hash mismatch: {source_path}")
    source = _load_json(source_path)
    if source.get("status") != "PASS":
        raise ValueError("source curriculum aggregate is not PASS")
    for architecture, arm in (("A1", "staged"), ("A1", "shuffled")):
        curriculum._verify_summary(architecture, arm, source)
    return source


def _augment_summary(summary: dict[str, Any], arm: str, run_dir: Path) -> dict[str, Any]:
    summary = dict(summary)
    summary["training_intervention"] = {
        "name": "layer7_value_trace_auxiliary_objective",
        "aux_weight": AUX_WEIGHT,
        "aux_temperature": AUX_TEMPERATURE,
        "value_base": VALUE_BASE,
        "value_count": VALUE_COUNT,
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
    expected = {
        "status": "completed",
        "architecture": "A1",
        "arm": arm,
        "seed": 17,
        "parameter_count": EXPECTED_PARAMS,
        "config_path": CONFIG_PATH,
        "config_sha256": sha256_file(REPO_ROOT / CONFIG_PATH),
        "protocol_commit": protocol_commit,
        "optimizer_steps": 2_000,
        "tokens_processed": 928_000,
        "tokens_per_step": 464,
        "order_independent_exposure_sha256": curriculum.EXPOSURE_SHA256,
    }
    for field, expected_value in expected.items():
        if summary.get(field) != expected_value:
            raise ValueError(
                f"intervention summary mismatch {field}: {summary.get(field)!r} != {expected_value!r}"
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


def _deltas(
    source: dict[str, Any],
    intervention: dict[str, Any],
) -> dict[str, Any]:
    control = source["arms"][f"a1_{intervention['arm']}"]["final_evaluation"]
    observed = intervention["summary"]["final_evaluation"]
    result: dict[str, Any] = {}
    for name, control_metrics in control.items():
        actual = observed[name]
        result[name] = {
            "control": control_metrics,
            "intervention": actual,
            "delta_target_loss": _finite(actual["target_loss"] - control_metrics["target_loss"], f"{name} loss delta"),
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
        default=REPO_ROOT / "results/a0_a1_relational_curriculum_value_trace_intervention.json",
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
    adjudication_path = REPO_ROOT / ADJUDICATION_PATH
    if sha256_file(adjudication_path) != ADJUDICATION_SHA256:
        raise ValueError("mechanism adjudication hash mismatch")
    adjudication = _load_json(adjudication_path)
    if adjudication.get("status") != "complete":
        raise ValueError("mechanism adjudication is not complete")
    if adjudication.get("unlocked_intervention", {}).get("status") != "unlocked":
        raise ValueError("value-trace intervention was not explicitly unlocked")
    plan_path = REPO_ROOT / PLAN_PATH
    if not plan_path.is_file():
        raise FileNotFoundError(plan_path)

    original_builder = curriculum_runner._build_model

    def build_intervention_model(config: Any) -> torch.nn.Module:
        if config.stage != "A1":
            raise RuntimeError("value-trace intervention is A1-only")
        model = ValueTraceA1(config, use_parallel_scan=True)
        if model.parameter_count() != EXPECTED_PARAMS:
            raise RuntimeError("value-trace model parameter count changed")
        return model

    curriculum_runner._build_model = build_intervention_model
    summaries: list[dict[str, Any]] = []
    try:
        for arm in ARMS:
            run_dir = REPO_ROOT / "results" / "runs" / f"relational_curriculum_v1_a1_value_trace_{arm}"
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

    runs = []
    for intervention in summaries:
        runs.append(
            {
                "arm": intervention["arm"],
                "run_dir": intervention["run_dir"],
                "summary_path": intervention["summary_path"],
                "checkpoint": intervention["checkpoint"],
                "checkpoint_sha256": intervention["checkpoint_sha256"],
                "cuda_training_seconds": intervention["cuda_training_seconds"],
                "training_wall_seconds": intervention["training_wall_seconds"],
                "final_evaluation_deltas_vs_control": _deltas(source, intervention),
                "summary": intervention["summary"],
            }
        )

    result: dict[str, Any] = {
        "status": "complete",
        "version": OUTPUT_VERSION,
        "read_only": True,
        "analysis_commit": args.analysis_commit,
        "analysis_plan_path": PLAN_PATH,
        "analysis_plan_sha256": sha256_file(plan_path),
        "training_runner_path": "scripts/run_relational_curriculum_experiment.py",
        "training_runner_sha256": sha256_file(
            REPO_ROOT / "scripts/run_relational_curriculum_experiment.py"
        ),
        "source_result_path": SOURCE_RESULT_PATH,
        "source_result_sha256": SOURCE_RESULT_SHA256,
        "adjudication_path": ADJUDICATION_PATH,
        "adjudication_sha256": sha256_file(adjudication_path),
        "source_protocol_commit": curriculum.SOURCE_PROTOCOL_COMMIT,
        "source_protocol_sha256": curriculum.SOURCE_PROTOCOL_SHA256,
        "order_independent_exposure_sha256": curriculum.EXPOSURE_SHA256,
        "architecture": "A1",
        "seed": 17,
        "arms": list(ARMS),
        "target_lengths": list(TARGET_LENGTHS),
        "optimizer_steps": 2_000,
        "tokens_processed": 928_000,
        "parameter_count": EXPECTED_PARAMS,
        "intervention": {
            "name": "layer7_value_trace_auxiliary_objective",
            "aux_weight": AUX_WEIGHT,
            "aux_temperature": AUX_TEMPERATURE,
            "value_base": VALUE_BASE,
            "value_count": VALUE_COUNT,
            "training_only": True,
            "new_parameters": False,
            "architecture_definition_unchanged": True,
        },
        "untouched_controls": {
            "a1_staged_checkpoint_sha256": source["arms"]["a1_staged"]["checkpoint_sha256"],
            "a1_shuffled_checkpoint_sha256": source["arms"]["a1_shuffled"]["checkpoint_sha256"],
            "a0_staged_checkpoint_sha256": source["arms"]["a0_staged"]["checkpoint_sha256"],
            "a0_shuffled_checkpoint_sha256": source["arms"]["a0_shuffled"]["checkpoint_sha256"],
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
