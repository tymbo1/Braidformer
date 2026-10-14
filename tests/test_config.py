from pathlib import Path

import pytest

from braidformer.config import ConfigError, ExperimentConfig, load_config


ROOT = Path(__file__).resolve().parents[1]


def test_phase0_config_loads_and_validates() -> None:
    config = load_config(ROOT / "configs" / "braidnano_phase0_planning.json")

    assert config.stage == "A0"
    assert config.model_type == "transformer"
    assert config.recurrent_path == "sequential_reference"
    assert config.compile_model is False
    assert config.target_parameters_min == 40_000_000
    assert config.target_parameters_max == 60_000_000
    assert config.planned_only is True
    assert config.max_steps == 0


def test_a0_baseline_proposal_is_explicitly_planned_only() -> None:
    config = load_config(ROOT / "configs" / "braidnano_a0_baseline_proposal.json")

    assert config.planned_only is True
    assert config.max_steps == 1_000
    assert config.training_token_budget == 8_192_000
    assert config.evaluation_token_budget == 131_072
    assert config.data_and_tokenizer == "configs/braidnano_a0_data.json"


def test_a1_proposal_selects_the_verified_parallel_scan() -> None:
    config = load_config(
        ROOT / "configs" / "braidnano_a1_recurrent_state_space_proposal.json"
    )

    assert config.stage == "A1"
    assert config.recurrent_path == "parallel_scan"
    assert config.compile_model is False


def test_compile_model_is_an_explicit_a1_parallel_scan_opt_in() -> None:
    config = ExperimentConfig(
        stage="A1",
        model_type="recurrent_state_space",
        recurrent_path="parallel_scan",
        compile_model=True,
    )
    config.validate()

    with pytest.raises(ConfigError, match="compile_model"):
        ExperimentConfig(compile_model=True).validate()

    with pytest.raises(ConfigError, match="compile_model"):
        ExperimentConfig(
            stage="A1",
            model_type="recurrent_state_space",
            recurrent_path="sequential_reference",
            compile_model=True,
        ).validate()


def test_cuda_referenced_schedule_requires_calibration() -> None:
    config = ExperimentConfig(
        lr_progress_reference="cuda_seconds",
        schedule_target_cuda_seconds=100.0,
        schedule_step_cuda_seconds=0.2,
    )
    config.validate()

    with pytest.raises(ConfigError, match="schedule_target_cuda_seconds"):
        ExperimentConfig(lr_progress_reference="cuda_seconds").validate()


def test_invalid_attention_shape_is_rejected() -> None:
    config = ExperimentConfig(hidden_size=513, num_attention_heads=8)

    with pytest.raises(ConfigError, match="divisible"):
        config.validate()


def test_zero_step_run_must_be_explicitly_planning_only() -> None:
    config = ExperimentConfig(max_steps=0, planned_only=False)

    with pytest.raises(ConfigError, match="planned_only"):
        config.validate()


def test_model_controls_are_explicitly_validated() -> None:
    with pytest.raises(ConfigError, match="dropout"):
        ExperimentConfig(dropout=1.0).validate()

    with pytest.raises(ConfigError, match="layer_norm_eps"):
        ExperimentConfig(layer_norm_eps=0.0).validate()
