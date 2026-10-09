from pathlib import Path

import pytest

from braidformer.config import ConfigError, ExperimentConfig, load_config


ROOT = Path(__file__).resolve().parents[1]


def test_phase0_config_loads_and_validates() -> None:
    config = load_config(ROOT / "configs" / "braidnano_phase0_planning.json")

    assert config.stage == "A0"
    assert config.target_parameters_min == 40_000_000
    assert config.target_parameters_max == 60_000_000
    assert config.planned_only is True
    assert config.max_steps == 0


def test_invalid_attention_shape_is_rejected() -> None:
    config = ExperimentConfig(hidden_size=513, num_attention_heads=8)

    with pytest.raises(ConfigError, match="divisible"):
        config.validate()


def test_zero_step_run_must_be_explicitly_planning_only() -> None:
    config = ExperimentConfig(max_steps=0, planned_only=False)

    with pytest.raises(ConfigError, match="planned_only"):
        config.validate()
