from pathlib import Path

import torch

from braidformer.config import load_config
from scripts.run_synthetic_overfit_calibration import (
    EFFECTIVE_BATCH_SIZE,
    FIXED_EXAMPLES,
    SEQUENCE_LENGTH,
    generator_spec,
    generator_version,
    make_example,
    make_training_batch,
    sequence_length,
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


def test_values128_rung_changes_only_the_reserved_answer_pool_identity() -> None:
    example = make_example(17, "train", 7, value_count=128)
    spec = generator_spec(17, value_count=128)

    assert generator_version(128) == "synthetic_solvability_values128_v1"
    assert spec["version"] == "synthetic_solvability_values128_v1"
    assert spec["value_count"] == 128
    assert len(example.input_ids) == 8
    assert example.input_ids[2] == example.input_ids[5]
    assert example.input_ids[3] == example.input_ids[7] == example.target
    assert 0 <= example.target - 8_000 < 128
    assert make_example(17, "train", 7, value_count=128) == example


def test_values256_rung_changes_only_the_reserved_answer_pool_identity() -> None:
    example = make_example(17, "train", 7, value_count=256)
    spec = generator_spec(17, value_count=256)

    assert generator_version(256) == "synthetic_solvability_values256_v1"
    assert spec["version"] == "synthetic_solvability_values256_v1"
    assert spec["value_count"] == 256
    assert len(example.input_ids) == 8
    assert example.input_ids[2] == example.input_ids[5]
    assert example.input_ids[3] == example.input_ids[7] == example.target
    assert 0 <= example.target - 8_000 < 256
    assert make_example(17, "train", 7, value_count=256) == example


def test_values1024_rung_changes_only_the_reserved_answer_pool_identity() -> None:
    example = make_example(17, "train", 7, value_count=1024)
    spec = generator_spec(17, value_count=1024)

    assert generator_version(1024) == "synthetic_solvability_values1024_v1"
    assert spec["version"] == "synthetic_solvability_values1024_v1"
    assert spec["value_count"] == 1024
    assert len(example.input_ids) == 8
    assert example.input_ids[2] == example.input_ids[5]
    assert example.input_ids[3] == example.input_ids[7] == example.target
    assert 0 <= example.target - 8_000 < 1024
    assert make_example(17, "train", 7, value_count=1024) == example


def test_values8000_rung_changes_only_the_reserved_answer_pool_identity() -> None:
    example = make_example(17, "train", 7, value_count=8000)
    spec = generator_spec(17, value_count=8000)

    assert generator_version(8000) == "synthetic_solvability_values8000_v1"
    assert spec["version"] == "synthetic_solvability_values8000_v1"
    assert spec["value_count"] == 8000
    assert len(example.input_ids) == 8
    assert example.input_ids[2] == example.input_ids[5]
    assert example.input_ids[3] == example.input_ids[7] == example.target
    assert 0 <= example.target - 8_000 < 8000
    assert make_example(17, "train", 7, value_count=8000) == example


def test_associations2_rung_has_two_bindings_and_one_queried_answer() -> None:
    example = make_example(17, "train", 7, value_count=8000, association_count=2)
    spec = generator_spec(17, value_count=8000, association_count=2)

    assert generator_version(8000, 2) == "synthetic_solvability_associations2_values8000_v1"
    assert spec["version"] == "synthetic_solvability_associations2_values8000_v1"
    assert spec["association_count"] == 2
    assert spec["sequence_length"] == sequence_length(2) == 11
    assert len(example.input_ids) == 11
    assert example.input_ids[0] == 30_000
    assert example.input_ids[1] == example.input_ids[4] == 30_001
    assert example.input_ids[7] == 30_002
    assert example.input_ids[9] == 30_003
    keys = (example.input_ids[2], example.input_ids[5])
    values = (example.input_ids[3], example.input_ids[6])
    assert example.input_ids[8] in keys
    assert example.target == example.input_ids[10] == values[keys.index(example.input_ids[8])]
    assert example.labels[:-1] == (-100,) * 10
    assert example.labels[-1] == example.target
    assert make_example(17, "train", 7, value_count=8000, association_count=2) == example


def test_associations4_rung_has_four_bindings_and_one_queried_answer() -> None:
    example = make_example(17, "train", 7, value_count=8000, association_count=4)
    spec = generator_spec(17, value_count=8000, association_count=4)

    assert generator_version(8000, 4) == "synthetic_solvability_associations4_values8000_v1"
    assert spec["version"] == "synthetic_solvability_associations4_values8000_v1"
    assert spec["association_count"] == 4
    assert spec["sequence_length"] == sequence_length(4) == 17
    assert len(example.input_ids) == 17
    assert example.input_ids[0] == 30_000
    assert example.input_ids[1] == example.input_ids[4] == example.input_ids[7] == example.input_ids[10] == 30_001
    assert example.input_ids[13] == 30_002
    assert example.input_ids[15] == 30_003
    keys = tuple(example.input_ids[offset] for offset in (2, 5, 8, 11))
    values = tuple(example.input_ids[offset] for offset in (3, 6, 9, 12))
    assert example.input_ids[14] in keys
    assert example.target == example.input_ids[16] == values[keys.index(example.input_ids[14])]
    assert example.labels[:-1] == (-100,) * 16
    assert example.labels[-1] == example.target
    assert make_example(17, "train", 7, value_count=8000, association_count=4) == example


def test_associations8_rung_has_eight_bindings_and_one_queried_answer() -> None:
    example = make_example(17, "train", 7, value_count=8000, association_count=8)
    spec = generator_spec(17, value_count=8000, association_count=8)

    assert generator_version(8000, 8) == "synthetic_solvability_associations8_values8000_v1"
    assert spec["version"] == "synthetic_solvability_associations8_values8000_v1"
    assert spec["association_count"] == 8
    assert spec["sequence_length"] == sequence_length(8) == 29
    assert len(example.input_ids) == 29
    assert example.input_ids[0] == 30_000
    assert all(example.input_ids[offset] == 30_001 for offset in range(1, 23, 3))
    assert example.input_ids[25] == 30_002
    assert example.input_ids[27] == 30_003
    key_offsets = tuple(range(2, 24, 3))
    value_offsets = tuple(range(3, 25, 3))
    keys = tuple(example.input_ids[offset] for offset in key_offsets)
    values = tuple(example.input_ids[offset] for offset in value_offsets)
    assert example.input_ids[26] in keys
    assert example.target == example.input_ids[28] == values[keys.index(example.input_ids[26])]
    assert example.labels[:-1] == (-100,) * 28
    assert example.labels[-1] == example.target
    assert make_example(17, "train", 7, value_count=8000, association_count=8) == example


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


def test_values128_configs_keep_controls_and_use_recovery_checkpoint_cadence() -> None:
    a0 = load_config(ROOT / "configs" / "braidnano_a0_solvability_values128.json")
    a1 = load_config(ROOT / "configs" / "braidnano_a1_solvability_values128.json")

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
        assert config.eval_interval_steps == 100
        assert config.checkpoint_interval_steps == 100
        assert config.data_and_tokenizer == "synthetic_solvability_values128_v1"
        assert config.planned_only is True


def test_values256_configs_keep_controls_and_use_recovery_checkpoint_cadence() -> None:
    a0 = load_config(ROOT / "configs" / "braidnano_a0_solvability_values256.json")
    a1 = load_config(ROOT / "configs" / "braidnano_a1_solvability_values256.json")

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
        assert config.eval_interval_steps == 100
        assert config.checkpoint_interval_steps == 100
        assert config.data_and_tokenizer == "synthetic_solvability_values256_v1"
        assert config.planned_only is True


def test_values1024_configs_keep_controls_and_use_recovery_checkpoint_cadence() -> None:
    a0 = load_config(ROOT / "configs" / "braidnano_a0_solvability_values1024.json")
    a1 = load_config(ROOT / "configs" / "braidnano_a1_solvability_values1024.json")

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
        assert config.eval_interval_steps == 100
        assert config.checkpoint_interval_steps == 100
        assert config.data_and_tokenizer == "synthetic_solvability_values1024_v1"
        assert config.planned_only is True


def test_values8000_configs_keep_controls_and_use_recovery_checkpoint_cadence() -> None:
    a0 = load_config(ROOT / "configs" / "braidnano_a0_solvability_values8000.json")
    a1 = load_config(ROOT / "configs" / "braidnano_a1_solvability_values8000.json")

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
        assert config.eval_interval_steps == 100
        assert config.checkpoint_interval_steps == 100
        assert config.data_and_tokenizer == "synthetic_solvability_values8000_v1"
        assert config.planned_only is True


def test_associations2_configs_keep_controls_and_use_expanded_sequence_budget() -> None:
    a0 = load_config(ROOT / "configs" / "braidnano_a0_solvability_associations2_values8000.json")
    a1 = load_config(ROOT / "configs" / "braidnano_a1_solvability_associations2_values8000.json")

    assert a0.stage == "A0"
    assert a1.stage == "A1"
    assert a1.recurrent_path == "parallel_scan"
    for config in (a0, a1):
        assert config.context_length == 512
        assert config.vocab_size == 32_000
        assert config.micro_batch_size == 2
        assert config.gradient_accumulation_steps == 8
        assert config.max_steps == 2_000
        assert config.training_token_budget == 352_000
        assert config.eval_interval_steps == 100
        assert config.checkpoint_interval_steps == 100
        assert config.data_and_tokenizer == "synthetic_solvability_associations2_values8000_v1"
        assert config.planned_only is True


def test_associations4_configs_keep_controls_and_use_expanded_sequence_budget() -> None:
    a0 = load_config(ROOT / "configs" / "braidnano_a0_solvability_associations4_values8000.json")
    a1 = load_config(ROOT / "configs" / "braidnano_a1_solvability_associations4_values8000.json")

    assert a0.stage == "A0"
    assert a1.stage == "A1"
    assert a1.recurrent_path == "parallel_scan"
    for config in (a0, a1):
        assert config.context_length == 512
        assert config.vocab_size == 32_000
        assert config.micro_batch_size == 2
        assert config.gradient_accumulation_steps == 8
        assert config.max_steps == 2_000
        assert config.training_token_budget == 544_000
        assert config.eval_interval_steps == 100
        assert config.checkpoint_interval_steps == 100
        assert config.data_and_tokenizer == "synthetic_solvability_associations4_values8000_v1"
        assert config.planned_only is True


def test_associations8_configs_keep_controls_and_use_expanded_sequence_budget() -> None:
    a0 = load_config(ROOT / "configs" / "braidnano_a0_solvability_associations8_values8000.json")
    a1 = load_config(ROOT / "configs" / "braidnano_a1_solvability_associations8_values8000.json")

    assert a0.stage == "A0"
    assert a1.stage == "A1"
    assert a1.recurrent_path == "parallel_scan"
    for config in (a0, a1):
        assert config.context_length == 512
        assert config.vocab_size == 32_000
        assert config.micro_batch_size == 2
        assert config.gradient_accumulation_steps == 8
        assert config.max_steps == 2_000
        assert config.training_token_budget == 928_000
        assert config.eval_interval_steps == 100
        assert config.checkpoint_interval_steps == 100
        assert config.data_and_tokenizer == "synthetic_solvability_associations8_values8000_v1"
        assert config.planned_only is True
