"""Small, dependency-free experiment configuration validation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when an experiment configuration is invalid."""


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration fields that are safe to validate before model code exists."""

    name: str = "braidnano_phase0_planning"
    stage: str = "A0"
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
    gradient_checkpointing: bool = True
    optimizer: str = "AdamW"
    learning_rate: float = 3e-4
    target_parameters_min: int = 40_000_000
    target_parameters_max: int = 60_000_000
    max_steps: int = 0
    planned_only: bool = True

    def validate(self) -> None:
        """Raise :class:`ConfigError` if the configuration is internally invalid."""

        errors: list[str] = []
        if not self.name.strip():
            errors.append("name must not be empty")
        if not self.stage.strip():
            errors.append("stage must not be empty")
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
        if self.target_parameters_min <= 0 or self.target_parameters_max <= 0:
            errors.append("parameter targets must be positive")
        if self.target_parameters_min > self.target_parameters_max:
            errors.append("target_parameters_min must not exceed target_parameters_max")
        if self.max_steps < 0:
            errors.append("max_steps must be non-negative")
        if self.max_steps == 0 and not self.planned_only:
            errors.append("a zero-step configuration must be marked planned_only")
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
