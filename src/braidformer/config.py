"""Small, dependency-free experiment configuration validation."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when an experiment configuration is invalid."""


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration shared by the controlled BraidNano experiments."""

    name: str = "braidnano_phase0_planning"
    stage: str = "A0"
    model_type: str = "transformer"
    recurrent_path: str = "sequential_reference"
    seed: int = 17
    context_length: int = 512
    vocab_size: int = 32_000
    hidden_size: int = 512
    num_layers: int = 8
    num_attention_heads: int = 8
    intermediate_size: int = 2_048
    micro_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    precision: str = "fp16"
    dropout: float = 0.0
    layer_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True
    use_bias: bool = True
    gradient_checkpointing: bool = True
    optimizer: str = "AdamW"
    learning_rate: float = 3e-4
    optimizer_beta1: float = 0.9
    optimizer_beta2: float = 0.999
    optimizer_epsilon: float = 1e-8
    weight_decay: float = 0.1
    lr_schedule: str = "cosine"
    lr_progress_reference: str = "optimizer_steps"
    warmup_ratio: float = 0.01
    schedule_target_cuda_seconds: float = 0.0
    schedule_step_cuda_seconds: float = 0.0
    gradient_clip_norm: float = 1.0
    target_parameters_min: int = 40_000_000
    target_parameters_max: int = 60_000_000
    training_token_budget: int = 0
    evaluation_token_budget: int = 0
    eval_interval_steps: int = 100
    checkpoint_interval_steps: int = 1_000
    data_and_tokenizer: str = "unselected"
    max_steps: int = 0
    planned_only: bool = True

    def validate(self) -> None:
        """Raise :class:`ConfigError` if the configuration is internally invalid."""

        errors: list[str] = []
        if not self.name.strip():
            errors.append("name must not be empty")
        if not self.stage.strip():
            errors.append("stage must not be empty")
        if not self.model_type.strip():
            errors.append("model_type must not be empty")
        if self.recurrent_path not in {"sequential_reference", "parallel_scan"}:
            errors.append(
                "recurrent_path must be sequential_reference or parallel_scan"
            )
        if self.stage == "A0" and self.recurrent_path != "sequential_reference":
            errors.append("A0 must use recurrent_path=sequential_reference")
        if self.seed < 0:
            errors.append("seed must be non-negative")
        for field_name in (
            "context_length",
            "vocab_size",
            "hidden_size",
            "num_layers",
            "num_attention_heads",
            "intermediate_size",
            "micro_batch_size",
            "gradient_accumulation_steps",
        ):
            if getattr(self, field_name) <= 0:
                errors.append(f"{field_name} must be positive")
        if self.hidden_size % self.num_attention_heads:
            errors.append("hidden_size must be divisible by num_attention_heads")
        if self.learning_rate <= 0:
            errors.append("learning_rate must be positive")
        if not 0 <= self.optimizer_beta1 < 1:
            errors.append("optimizer_beta1 must be in the range [0, 1)")
        if not 0 <= self.optimizer_beta2 < 1:
            errors.append("optimizer_beta2 must be in the range [0, 1)")
        if self.optimizer_epsilon <= 0:
            errors.append("optimizer_epsilon must be positive")
        if self.weight_decay < 0:
            errors.append("weight_decay must be non-negative")
        if not self.lr_schedule.strip():
            errors.append("lr_schedule must not be empty")
        if self.lr_progress_reference not in {"optimizer_steps", "cuda_seconds"}:
            errors.append("lr_progress_reference must be optimizer_steps or cuda_seconds")
        if not 0 <= self.warmup_ratio < 1:
            errors.append("warmup_ratio must be in the range [0, 1)")
        if not math.isfinite(self.schedule_target_cuda_seconds) or self.schedule_target_cuda_seconds < 0:
            errors.append("schedule_target_cuda_seconds must be finite and non-negative")
        if not math.isfinite(self.schedule_step_cuda_seconds) or self.schedule_step_cuda_seconds < 0:
            errors.append("schedule_step_cuda_seconds must be finite and non-negative")
        if self.lr_progress_reference == "cuda_seconds":
            if self.schedule_target_cuda_seconds <= 0:
                errors.append(
                    "cuda_seconds schedules require schedule_target_cuda_seconds > 0"
                )
            if self.schedule_step_cuda_seconds <= 0:
                errors.append(
                    "cuda_seconds schedules require schedule_step_cuda_seconds > 0"
                )
        elif self.schedule_target_cuda_seconds or self.schedule_step_cuda_seconds:
            errors.append(
                "CUDA schedule calibration fields require lr_progress_reference=cuda_seconds"
            )
        if self.gradient_clip_norm <= 0:
            errors.append("gradient_clip_norm must be positive")
        if not 0 <= self.dropout < 1:
            errors.append("dropout must be in the range [0, 1)")
        if self.layer_norm_eps <= 0:
            errors.append("layer_norm_eps must be positive")
        if self.target_parameters_min <= 0 or self.target_parameters_max <= 0:
            errors.append("parameter targets must be positive")
        if self.target_parameters_min > self.target_parameters_max:
            errors.append("target_parameters_min must not exceed target_parameters_max")
        if self.training_token_budget < 0:
            errors.append("training_token_budget must be non-negative")
        if self.evaluation_token_budget < 0:
            errors.append("evaluation_token_budget must be non-negative")
        if self.eval_interval_steps <= 0:
            errors.append("eval_interval_steps must be positive")
        if self.checkpoint_interval_steps <= 0:
            errors.append("checkpoint_interval_steps must be positive")
        if not self.data_and_tokenizer.strip():
            errors.append("data_and_tokenizer must not be empty")
        if self.max_steps < 0:
            errors.append("max_steps must be non-negative")
        if self.max_steps == 0 and not self.planned_only:
            errors.append("a zero-step configuration must be marked planned_only")
        if not self.planned_only and self.training_token_budget == 0:
            errors.append("a runnable configuration must specify training_token_budget")
        if not self.planned_only and self.evaluation_token_budget == 0:
            errors.append("a runnable configuration must specify evaluation_token_budget")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            errors.append("precision must be one of fp32, fp16, or bf16")
        if not self.optimizer.strip():
            errors.append("optimizer must not be empty")
        if errors:
            raise ConfigError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation after validation."""

        self.validate()
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        """Write a deterministic, human-readable JSON configuration."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an :class:`ExperimentConfig` from JSON."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ConfigError("configuration root must be a JSON object")
    try:
        config = ExperimentConfig(**payload)
    except TypeError as exc:
        raise ConfigError(f"unsupported or missing configuration field: {exc}") from exc
    config.validate()
    return config
