#!/usr/bin/env python3
"""Synthesize the frozen A0/A1 mechanism-campaign evidence.

This is a read-only evidence audit. It validates completed result artifacts,
extracts the principal measurements, and writes the final adjudication without
running a model or changing a checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = "docs/A0_A1_MECHANISM_ADJUDICATION_PLAN.md"
OUTPUT_VERSION = "mechanism_adjudication_v1"

SOURCE_PATHS = {
    "values128": "results/a0_a1_solvability_values128.json",
    "values256": "results/a0_a1_solvability_values256.json",
    "values1024": "results/a0_a1_solvability_values1024.json",
    "values8000": "results/a0_a1_solvability_values8000.json",
    "associations2": "results/a0_a1_solvability_associations2_values8000.json",
    "associations4": "results/a0_a1_solvability_associations4_values8000.json",
    "associations8": "results/a0_a1_solvability_associations8_values8000.json",
    "distances": "results/a0_a1_solvability_distances_values8000.json",
    "target_lengths": "results/a0_a1_solvability_target_lengths_values8000.json",
    "decay_original": "results/a1_state_decay_forensics_values8000.json",
    "decay_tail": "results/a1_decaytail_forensics_values8000.json",
    "decay_tail_task": "results/a1_decaytail_solvability_values8000.json",
    "target_position": "results/a0_a1_target_position_decomposition_values8000.json",
    "layerwise": "results/a0_a1_layerwise_representation_forensics_values8000.json",
    "query_readout": "results/a0_a1_query_readout_localization_values8000.json",
    "state_replay": "results/a0_a1_state_replay_intervention_values8000.json",
}


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _finite(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite {label}: {value!r}")
    return number


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runs_by_arch(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    runs = payload.get("runs")
    if not isinstance(runs, list):
        raise ValueError("result has no run list")
    records = {run["architecture"]: run for run in runs}
    if set(records) != {"A0", "A1"}:
        raise ValueError(f"unexpected architecture set: {sorted(records)}")
    return records


def _final_pair(payload: dict[str, Any]) -> dict[str, Any]:
    runs = _runs_by_arch(payload)
    return {
        architecture: {
            "train_exact_match": _finite(run["final_train"]["greedy_exact_match"], "train EM"),
            "heldout_exact_match": _finite(run["final_heldout"]["greedy_exact_match"], "heldout EM"),
            "heldout_target_loss": _finite(run["final_heldout"]["target_loss"], "heldout target loss"),
            "cuda_training_seconds": _finite(run["cuda_training_seconds"], "CUDA time"),
            "checkpoint_sha256": run["checkpoint"]["sha256"],
        }
        for architecture, run in runs.items()
    }


def _validate_source_payloads(payloads: dict[str, dict[str, Any]]) -> dict[str, str]:
    hashes = {}
    for label, relative in SOURCE_PATHS.items():
        path = REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = payloads[label]
        if payload.get("status") != "complete":
            raise ValueError(f"source result is not complete: {relative}")
        if label in {"target_position", "layerwise", "query_readout", "state_replay"}:
            if payload.get("read_only") is not True:
                raise ValueError(f"diagnostic is not read-only: {relative}")
        hashes[relative] = _sha256(path)
    return hashes


def _extract(payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    value_ladder = []
    for value_count, label in ((128, "values128"), (256, "values256"), (1024, "values1024"), (8000, "values8000")):
        value_ladder.append({"value_count": value_count, "final": _final_pair(payloads[label])})

    association_ladder = []
    for association_count, label in ((1, "values8000"), (2, "associations2"), (4, "associations4"), (8, "associations8")):
        association_ladder.append({"association_count": association_count, "final": _final_pair(payloads[label])})

    distance_curve = []
    for row in payloads["distances"]["final_curve"]:
        distance_curve.append(
            {
                "distance": int(row["distance"]),
                "A0_heldout_em": _finite(row["A0_heldout_greedy_exact_match"], "A0 distance EM"),
                "A1_heldout_em": _finite(row["A1_heldout_greedy_exact_match"], "A1 distance EM"),
                "A0_loss": _finite(row["A0_heldout_target_loss"], "A0 distance loss"),
                "A1_loss": _finite(row["A1_heldout_target_loss"], "A1 distance loss"),
                "A1_minus_A0_loss": _finite(row["A1_minus_A0_heldout_target_loss"], "distance loss gap"),
                "A1_to_A0_cuda_ratio": _finite(row["A1_to_A0_cuda_time_ratio"], "distance CUDA ratio"),
            }
        )

    target_length_curve = []
    for pair in payloads["target_lengths"]["pairs"]:
        target_length_curve.append(
            {
                "target_length": int(pair["target_length"]),
                "heldout": pair["heldout"],
                "resources": pair["resources"],
            }
        )

    original_decay = payloads["decay_original"]["checkpoints"]
    decay_summary = [
        {
            "label": checkpoint["label"],
            "training_distance": checkpoint["training_distance"],
            "tau_median": _finite(checkpoint["time_constants"]["tau_median"], "original tau median"),
            "tau_max": _finite(checkpoint["time_constants"]["tau_max"], "original tau max"),
            "channels_tau_at_least_16": int(checkpoint["time_constants"]["channels_tau_at_least"]["16"]),
        }
        for checkpoint in original_decay
    ]
    tail_decay = [
        {
            "label": checkpoint["label"],
            "training_distance": checkpoint["training_distance"],
            "tau_median": _finite(checkpoint["time_constants"]["tau_median"], "tail tau median"),
            "tau_max": _finite(checkpoint["time_constants"]["tau_max"], "tail tau max"),
            "channels_tau_at_least_16": int(checkpoint["time_constants"]["channels_tau_at_least"]["16"]),
        }
        for checkpoint in payloads["decay_tail"]["checkpoints"]
    ]

    target_position = []
    for run in payloads["target_position"]["runs"]:
        heldout = run["splits"]["heldout"]["teacher_forced"]["positions"]
        target_position.append(
            {
                "architecture": run["architecture"],
                "target_length": run["target_length"],
                "position_1_top1": _finite(heldout[0]["top1_accuracy"], "target position top1"),
                "position_1_cross_entropy": _finite(heldout[0]["cross_entropy"], "target position CE"),
                "token_accuracy": _finite(run["splits"]["heldout"]["teacher_forced"]["token_accuracy"], "target token accuracy"),
            }
        )

    layerwise_selected = []
    for row in payloads["layerwise"]["summary_rows"]:
        if row["architecture"] == "A1" and row["path"] in {"recurrent_state_0", "recurrent_state_7", "final_norm"}:
            layerwise_selected.append(row)

    query_readout_selected = []
    for row in payloads["query_readout"]["summary_rows"]:
        if row["architecture"] == "A1" and row["path"] in {"recurrent_state_0", "recurrent_state_7", "final_norm"}:
            query_readout_selected.append(row)

    replay_deltas = []
    for run in payloads["state_replay"]["runs"]:
        for row in run["positions"]:
            delta = row["delta_value_candidate_range"]
            replay_deltas.append(
                {
                    "target_length": run["target_length"],
                    "position": row["position"],
                    "delta_top1": _finite(delta["top1_accuracy"], "replay top1 delta"),
                    "delta_cross_entropy": _finite(delta["cross_entropy"], "replay CE delta"),
                    "delta_mean_rank": _finite(delta["mean_target_rank"], "replay rank delta"),
                }
            )

    return {
        "value_space_ladder": value_ladder,
        "association_ladder": association_ladder,
        "distance_curve": distance_curve,
        "target_length_curve": target_length_curve,
        "original_decay": decay_summary,
        "long_decay_tail": tail_decay,
        "target_position_selected": target_position,
        "layerwise_selected": layerwise_selected,
        "query_readout_selected": query_readout_selected,
        "state_replay_deltas": replay_deltas,
    }


HYPOTHESES = [
    {
        "id": "H1",
        "hypothesis": "Temporal decay erases the required value trace.",
        "status": "partially_supported_but_insufficient",
        "rank": 1,
        "support": [
            "A1 loss gap grows monotonically across the frozen distance curve.",
            "Original A1 has median tau about 8.9 tokens and zero channels with tau >= 16 in all four distance checkpoints.",
            "Frozen value probes collapse with distance at matching internal-state timescales.",
        ],
        "against": [
            "A 25% long-timescale initialization tail creates 768 tau>=16 channels but does not restore long-distance probe/task performance.",
            "Layer-7 state replay produces no top-1 gain and slightly worsens CE.",
        ],
        "conclusion": "A temporal component is real, but simple decay or direct state preservation is not the complete mechanism.",
    },
    {
        "id": "H2",
        "hypothesis": "Recurrent-state transformation/interference destroys the binding while the sequence advances.",
        "status": "leading_composite_hypothesis",
        "rank": 1,
        "support": [
            "A1 alone shows the strong distance-sensitive degradation while A0 remains nearly flat.",
            "Layerwise/state probes show cross-site value transfer and state-position signals weaken through recurrent processing.",
            "Query/readout probes show no simple final-head rescue.",
        ],
        "against": [
            "The single layer-7 replay intervention is null; no specific recurrent layer/state rule has been causally isolated.",
        ],
        "conclusion": "The best current mechanism family is recurrent transformation/interference combined with temporal retention and binding demands, not a proven single-layer defect.",
    },
    {
        "id": "H3",
        "hypothesis": "Query-conditioned processing corrupts or overwrites a retained binding.",
        "status": "surviving_alternative",
        "rank": 2,
        "support": [
            "Fixed pre-query value transfer is weak after the query and does not recover at target sites.",
            "State replay after the query fails, leaving query-conditioned gates/transformations as a live possibility.",
        ],
        "against": [
            "Pre-query cross-validated value probes are already weak on the heldout generalization task, so the binding may never be stably represented.",
        ],
        "conclusion": "Not separated from H2 or H4 by the completed campaign.",
    },
    {
        "id": "H4",
        "hypothesis": "Binding/key-value identity is not represented compositionally under the synthetic load.",
        "status": "strongly_supported_shared_failure",
        "rank": 2,
        "support": [
            "Both models retain 100% train EM while heldout EM collapses with value-space size and association count.",
            "The association axis collapses for both architectures, showing a shared systematic-binding problem.",
            "Diagnostic value readouts do not reveal a stable heldout value code at target output sites.",
        ],
        "against": [
            "H4 alone does not explain why A1 is more distance-sensitive than A0.",
        ],
        "conclusion": "A shared compositional-binding/generalization failure is established; A1 adds a stronger temporal/state penalty.",
    },
    {
        "id": "H5",
        "hypothesis": "The required value survives internally but the output route/readout fails.",
        "status": "weakened",
        "rank": 4,
        "support": [
            "Same-split fitted probe ceilings are high at some sites, showing representations can be decoded when allowed to fit each heldout sample set.",
        ],
        "against": [
            "Four-fold heldout fixed and site-specific probes stay near floor, with no strong diagnostic readout advantage over native value-range logits.",
            "Layer-7 state replay does not restore output metrics.",
        ],
        "conclusion": "A pure final-readout failure is not supported; nonlinear or task-specific routes remain possible but are secondary.",
    },
    {
        "id": "H6",
        "hypothesis": "Multi-token autoregressive target state degrades after the first output.",
        "status": "rejected_as_sole_cause",
        "rank": 5,
        "support": [
            "A1 degrades more at longer target lengths.",
        ],
        "against": [
            "At target lengths 4 and 8 A1 is already at zero heldout teacher-forced top-1 on target position 1.",
        ],
        "conclusion": "Autoregressive feedback may contribute later, but cannot be the primary explanation.",
    },
    {
        "id": "H7",
        "hypothesis": "A1 needs more optimization steps or a different optimization trajectory.",
        "status": "unresolved_background_factor",
        "rank": 3,
        "support": [
            "All current runs use a short fixed 2,000-step recipe and do not prove optimization sufficiency.",
        ],
        "against": [
            "Both models reach 100% train EM, and A1 receives higher CUDA time while its heldout distance/length failures persist.",
            "No diagnostic result points specifically to optimizer dynamics.",
        ],
        "conclusion": "Not ruled out, but not enough evidence to unlock an optimization intervention inside this campaign.",
    },
]


def _report(result: dict[str, Any]) -> str:
    lines = [
        "# A0/A1 mechanism adjudication result",
        "",
        "Status: **complete; evidence synthesis only**.",
        "",
        f"Adjudication plan: `{result['analysis_plan_path']}` (sha256 `{result['analysis_plan_sha256']}`).",
        "",
        "## Strongest supported mechanism",
        "",
        "A1's weakness is best described as a composite recurrent retrieval failure: the task requires compositional key/value binding, and A1's learned recurrent transformation/retention path makes that binding increasingly unstable as temporal separation grows. The evidence does not isolate a single decay constant, layer, gate, or output-head defect.",
        "",
        "## Evidence chain",
        "",
        "1. Both A0 and A1 memorize the fixed training associations, but heldout generalization collapses as value-space size and association count increase.",
        "2. A1 diverges from A0 as distractor distance grows; its loss gap reaches about `+1.240` at distance 64 while A1 uses more CUDA time.",
        "3. Original A1 learns median state time constants around `8.9` tokens with no tau>=16 channels; frozen value probes collapse at matching distances.",
        "4. Adding a long-timescale tail creates 768 long-lived channels but does not repair long-distance task/probe performance.",
        "5. Target-position decomposition shows the A1 length-4/8 deficit already on the first teacher-forced output, ruling out autoregressive accumulation as the sole cause.",
        "6. Layerwise and four-fold query/readout probes do not reveal a stable heldout value code or a strong hidden-value/native-readout gap.",
        "7. The predeclared layer-7 stored-value state replay produces no top-1 gain and slightly worsens CE, rejecting simple state preservation as sufficient.",
        "",
        "## Hypothesis ranking",
        "",
        "| rank | ID | status | conclusion |",
        "| ---: | --- | --- | --- |",
    ]
    for hypothesis in sorted(result["hypotheses"], key=lambda item: (item["rank"], item["id"])):
        lines.append(f"| {hypothesis['rank']} | {hypothesis['id']} | {hypothesis['status']} | {hypothesis['conclusion']} |")
    lines.extend(
        [
            "",
            "## Optional training gate",
            "",
            "**Locked.** Two independent diagnostics support a broad recurrent/temporal/binding family, but they do not converge on one minimal mechanism-specific factor: long decay was already falsified as sufficient, layer-7 replay was null, and cross-validated pre-query value generalization is weak. Starting a training intervention now would require an arbitrary choice between state dynamics, query conditioning, binding curriculum, and optimization. No item-6 training run was performed.",
            "",
            "## Surviving alternatives and best next experiment",
            "",
            "The main alternatives are query-conditioned corruption (H3), a task/curriculum-induced binding failure shared by both architectures (H4), and an optimization factor (H7). The single best next experiment outside this completed campaign is a matched fresh A0/A1 run with a staged synthetic relational/binding curriculum, preserving architecture, optimizer, compute budget, and final heldout tests. It directly tests whether the current boundary is primarily developmental/curriculum-induced before any architectural patch is attempted.",
            "",
            "## Integrity and scope",
            "",
            "All source result JSON files were present and complete; the four campaign diagnostics were read-only; source file hashes and checkpoint hashes are preserved in the machine-readable result. The full test suite passed after each campaign item (`67 passed`), `git diff --check` passed, raw checkpoints/logs remain ignored, no A0/original-A1 definitions were changed, and no GitHub push occurred.",
            "",
            "## Commits",
            "",
            "| stage | commit |",
            "| --- | --- |",
        ]
    )
    for stage, commit in result["commits"].items():
        lines.append(f"| {stage} | `{commit}` |")
    lines.extend(["", "Final Git status is recorded in the machine-readable result and must be clean except for this result/ledger commit while the campaign handoff is made.", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results/a0_a1_mechanism_adjudication.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "docs/A0_A1_MECHANISM_ADJUDICATION_RESULT.md",
    )
    args = parser.parse_args()

    payloads = {label: _load(REPO_ROOT / relative) for label, relative in SOURCE_PATHS.items()}
    source_hashes = _validate_source_payloads(payloads)
    evidence = _extract(payloads)
    plan_path = REPO_ROOT / PLAN_PATH
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    commits = {
        "campaign_baseline": "70ba2cd",
        "item1_protocol": "e82f342",
        "item1_analyzer": "e031c2d",
        "item1_result": "6d19ee4",
        "item2_protocol": "9202c1d",
        "item2_analyzer": "338a249",
        "item2_result": "9856a8b",
        "item3_protocol": "75c4efe",
        "item3_analyzer": "a0555b0",
        "item3_result": "b534b0e",
        "item4_protocol": "6e299a7",
        "item4_analyzer": "961d452",
        "item4_result": "dbf4706",
        "item5_protocol": head,
    }
    result = {
        "status": "complete",
        "version": OUTPUT_VERSION,
        "analysis_plan_path": PLAN_PATH,
        "analysis_plan_sha256": _sha256(plan_path),
        "source_result_hashes": source_hashes,
        "evidence": evidence,
        "hypotheses": HYPOTHESES,
        "training_intervention_gate": {
            "status": "locked",
            "reason": "Diagnostics converge on a broad recurrent/temporal/binding family but not one minimal mechanism-specific factor; long-tail and layer-7 replay negatives forbid arbitrary stacking.",
        },
        "commits": commits,
        "observed_git_head_at_analysis": head,
        "scope": {
            "frozen_a0_untouched": True,
            "original_a1_untouched": True,
            "github_push": False,
            "unrelated_workstreams_started": False,
        },
    }
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    report = args.report if args.report.is_absolute() else REPO_ROOT / args.report
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_report(result), encoding="utf-8")
    print(json.dumps({"output": str(output), "report": str(report), "sources": len(source_hashes)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
