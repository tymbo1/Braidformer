from pathlib import Path

import torch

from braidformer.config import load_config
from scripts.run_synthetic_recall_battery import (
    CONTEXT_LENGTH,
    EFFECTIVE_BATCH_SIZE,
    MARKERS,
    TASKS,
    make_example,
    make_training_batch,
    synthetic_data_fingerprint,
)


ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_examples_are_fixed_length_and_task_shaped() -> None:
    expected_target_lengths = {"copy": 8, "associative": 4, "selective": 4}

    for task in TASKS:
        example = make_example(17, "validation", task, 0)

        assert len(example.input_ids) == CONTEXT_LENGTH
        assert len(example.labels) == CONTEXT_LENGTH
        assert len(example.target) == expected_target_lengths[task]
        assert example.input_ids[-len(example.target) :] == example.target
        assert example.labels[-len(example.target) :] == example.target
        assert all(label == -100 for label in example.labels[:-len(example.target)])
        assert MARKERS[f"{task}_task"] in example.input_ids


def test_synthetic_generation_is_deterministic_and_split_separated() -> None:
    first = make_example(23, "train", "associative", 11)
    second = make_example(23, "train", "associative", 11)
    validation = make_example(23, "validation", "associative", 11)

    assert first == second
    assert first != validation
    assert synthetic_data_fingerprint(23) == synthetic_data_fingerprint(23)
    assert synthetic_data_fingerprint(23) != synthetic_data_fingerprint(29)


def test_training_batch_has_frozen_effective_batch_geometry() -> None:
    input_ids, labels = make_training_batch(17, 1)

    assert input_ids.shape == (EFFECTIVE_BATCH_SIZE, CONTEXT_LENGTH)
    assert labels.shape == (EFFECTIVE_BATCH_SIZE, CONTEXT_LENGTH)
    assert input_ids.dtype == torch.long
    assert labels.dtype == torch.long
    assert int((labels != -100).sum().item()) > 0


def test_synthetic_configs_preserve_a0_a1_control_settings() -> None:
    configs = {
        "A0": load_config(ROOT / "configs" / "braidnano_a0_synthetic_recall_battery.json"),
        "A1": load_config(ROOT / "configs" / "braidnano_a1_synthetic_recall_battery.json"),
    }

    assert configs["A0"].model_type == "transformer"
    assert configs["A1"].model_type == "recurrent_state_space"
    assert configs["A1"].recurrent_path == "parallel_scan"
    for config in configs.values():
        assert config.context_length == 512
        assert config.micro_batch_size == 2
        assert config.gradient_accumulation_steps == 8
        assert config.max_steps == 1_000
        assert config.training_token_budget == 8_192_000
        assert config.learning_rate == 3e-4
        assert config.lr_schedule == "cosine"
        assert config.warmup_ratio == 0.01
