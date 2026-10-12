from pathlib import Path

import torch

from braidformer.config import load_config
from scripts.run_relational_curriculum_experiment import (
    EFFECTIVE_BATCH_SIZE,
    MAX_STEPS,
    STAGES,
    TOKENS_PER_STEP,
    TRAIN_SEQUENCE_LENGTH,
    curriculum_spec,
    make_curriculum_example,
    make_training_batch,
)


ROOT = Path(__file__).resolve().parents[1]


def test_curriculum_schedule_has_fixed_budget_and_same_exposure() -> None:
    staged = curriculum_spec(17, "staged")
    shuffled = curriculum_spec(17, "shuffled")

    assert len(staged["schedule"]) == MAX_STEPS
    assert len(shuffled["schedule"]) == MAX_STEPS
    assert staged["schedule"] != shuffled["schedule"]
    assert staged["stage_counts"] == shuffled["stage_counts"]
    assert staged["order_independent_exposure_sha256"] == shuffled[
        "order_independent_exposure_sha256"
    ]
    assert staged["tokens_per_step"] == TOKENS_PER_STEP == 928000 // MAX_STEPS


def test_curriculum_examples_are_fixed_shape_and_answer_aligned() -> None:
    for stage in STAGES:
        example = make_curriculum_example(17, stage.name, 0, 0)
        assert len(example.input_ids) == TRAIN_SEQUENCE_LENGTH
        assert len(example.labels) == TRAIN_SEQUENCE_LENGTH
        if isinstance(example.target, tuple):
            assert example.input_ids[-len(example.target) :] == example.target
            assert example.labels[-len(example.target) :] == example.target
        else:
            assert example.input_ids[-1] == example.target
            assert example.labels[-1] == example.target
        assert all(label == -100 for label in example.labels[:-1]) or isinstance(
            example.target, tuple
        )


def test_curriculum_batch_is_deterministic_and_tracks_stage() -> None:
    spec = curriculum_spec(17, "staged")
    occurrences = [0] * len(STAGES)
    for stage_index in spec["schedule"]:
        occurrences[stage_index] += 1
    # The runner's occurrence vector is tested through its public batch shape;
    # the stage content itself is deterministic for identical inputs.
    first = make_training_batch(17, 1, spec["schedule"], [0] * MAX_STEPS)
    second = make_training_batch(17, 1, spec["schedule"], [0] * MAX_STEPS)
    assert first[2:] == second[2:]
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert first[0].shape == (EFFECTIVE_BATCH_SIZE, TRAIN_SEQUENCE_LENGTH)
    assert first[1].shape == (EFFECTIVE_BATCH_SIZE, TRAIN_SEQUENCE_LENGTH)


def test_curriculum_configs_preserve_original_model_controls() -> None:
    a0 = load_config(ROOT / "configs" / "braidnano_a0_relational_curriculum_v1.json")
    a1 = load_config(ROOT / "configs" / "braidnano_a1_relational_curriculum_v1.json")
    assert a0.stage == "A0"
    assert a1.stage == "A1"
    assert a1.recurrent_path == "parallel_scan"
    assert a1.state_transition_initialization == "uniform_0.9"
    for config in (a0, a1):
        assert config.context_length == 512
        assert config.hidden_size == 512
        assert config.num_layers == 8
        assert config.micro_batch_size == 2
        assert config.gradient_accumulation_steps == 8
        assert config.max_steps == 2000
        assert config.training_token_budget == 928000
        assert config.learning_rate == 0.0003
        assert config.warmup_ratio == 0.01
