#!/usr/bin/env python3
"""Verify the A0 model definition without running training.

The command constructs the model, counts unique parameters, and emits a
machine-readable report.  It intentionally performs no forward pass, optimizer
step, dataset access, checkpoint access, or network operation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import torch

from braidformer.config import load_config
from braidformer.models import A0TransformerLM


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "braidnano_phase0_planning.json"


def _parameter_breakdown(model: A0TransformerLM) -> dict[str, int]:
    breakdown: dict[str, int] = {}
    for name, parameter in model.named_parameters():
        group = name.split(".", 1)[0]
        breakdown[group] = breakdown.get(group, 0) + parameter.numel()
    return breakdown


def build_report(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    torch.manual_seed(config.seed)
    model = A0TransformerLM(config)
    parameter_count = model.parameter_count()
    if not config.target_parameters_min <= parameter_count <= config.target_parameters_max:
        raise ValueError(
            "A0 parameter count is outside the configured target range: "
            f"{parameter_count} not in [{config.target_parameters_min}, "
            f"{config.target_parameters_max}]"
        )

    return {
        "verification_schema_version": 1,
        "config_path": str(config_path.relative_to(REPO_ROOT)),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "verification_scope": "architecture_and_construction_only",
        "training_started": False,
        "seed": config.seed,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
            "cuda_available": bool(torch.cuda.is_available()),
        },
        "architecture": {
            "stage": config.stage,
            "model_type": config.model_type,
            "vocab_size": config.vocab_size,
            "context_length": config.context_length,
            "hidden_size": config.hidden_size,
            "num_layers": config.num_layers,
            "num_attention_heads": config.num_attention_heads,
            "head_dimension": config.hidden_size // config.num_attention_heads,
            "intermediate_size": config.intermediate_size,
            "activation": "GELU",
            "normalization": "pre_layer_norm",
            "attention": "causal_scaled_dot_product_attention",
            "dropout": config.dropout,
            "layer_norm_eps": config.layer_norm_eps,
            "use_bias": config.use_bias,
            "tie_word_embeddings": config.tie_word_embeddings,
            "recurrent_path": False,
            "selective_attention": False,
            "experts": 0,
            "conditional_depth": False,
            "compute_controller": False,
        },
        "parameters": {
            "unique_trainable": parameter_count,
            "lm_head_weight_is_tied": model.lm_head.weight is model.token_embedding.weight,
            "breakdown_by_module": _parameter_breakdown(model),
        },
        "proposed_training": {
            "planned_only": config.planned_only,
            "optimizer": config.optimizer,
            "learning_rate": config.learning_rate,
            "optimizer_beta1": config.optimizer_beta1,
            "optimizer_beta2": config.optimizer_beta2,
            "optimizer_epsilon": config.optimizer_epsilon,
            "weight_decay": config.weight_decay,
            "lr_schedule": config.lr_schedule,
            "warmup_ratio": config.warmup_ratio,
            "gradient_clip_norm": config.gradient_clip_norm,
            "micro_batch_size": config.micro_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "precision": config.precision,
            "gradient_checkpointing": config.gradient_checkpointing,
            "max_steps": config.max_steps,
            "eval_interval_steps": config.eval_interval_steps,
            "checkpoint_interval_steps": config.checkpoint_interval_steps,
            "training_token_budget": config.training_token_budget,
            "evaluation_token_budget": config.evaluation_token_budget,
            "data_and_tokenizer": config.data_and_tokenizer,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="A0 JSON config to verify (default: configs/braidnano_phase0_planning.json)",
    )
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    print(json.dumps(build_report(config_path), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
