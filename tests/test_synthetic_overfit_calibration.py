from pathlib import Path

import torch

from braidformer.config import load_config
from scripts.run_synthetic_overfit_calibration import (
    EFFECTIVE_BATCH_SIZE,
    FIXED_EXAMPLES,
    SEQUENCE_LENGTH,
    make_example,
    make_training_batch,
)


ROOT = Path(__file__).resolve().parents[1]


def test_calibration_example_is_one_association_without_padding() -> None:
    example = make_example(17, "train", 0)

    assert len(example.input_ids) == SEQUENCE_LENGTH == 8
    assert example.input_ids[2] == example.input_ids[5]
    assert example.input_ids[3] == example.input_ids[7] == example.target
    assert example.labels[:-1] == (-100,) * 7
    assert example.labels[-1] == example.target


def test_calibration_examples_are_deterministic_and_split_keys_disjoint() -> None:
    first = make_example(17, "train", 7)
    second = make_example(17, "train", 7)
    heldout = make_example(17, "heldout", 7)

    assert first == second
    assert first.input_ids[2] != heldout.input_ids[2]
    assert 0 <= first.target - 8_000 < 64


def test_calibration_batch_repeats_fixed_training_set() -> None:
    first_inputs, first_labels = make_training_batch(17, 1)
    ninth_inputs, ninth_labels = make_training_batch(17, 9)

    assert first_inputs.shape == (EFFECTIVE_BATCH_SIZE, SEQUENCE_LENGTH)
    assert first_labels.shape == (EFFECTIVE_BATCH_SIZE, SEQUENCE_LENGTH)
    assert torch.equal(first_inputs, ninth_inputs)
    assert torch.equal(first_labels, ninth_labels)
    assert FIXED_EXAMPLES == 128


def test_calibration_configs_keep_frozen_architecture_and_budget() -> None:
    a0 = load_config(ROOT / "configs" / "braidnano_a0_synthetic_overfit_calibration.json")
    a1 = load_config(ROOT / "configs" / "braidnano_a1_synthetic_overfit_calibration.json")

    assert a0.stage == "A0"
    assert a1.stage == "A1"
    assert a1.recurrent_path == "parallel_scan"
    for config in (a0, a1):
        assert config.context_length == 512
        assert config.vocab_size == 32_000
        assert config.micro_batch_size == 2
        assert config.gradient_accumulation_steps == 8
        assert config.max_steps == 2_000
        assert config.training_token_budget == 256_000
        assert config.warmup_ratio == 0.01
