import pytest

torch = pytest.importorskip("torch")

from braidformer.config import ExperimentConfig
from braidformer.models import (
    A1AssociativeLaneLM,
    EXPECTED_LANE_PARAMETERS,
)


def tiny_lane_config(**overrides: object) -> ExperimentConfig:
    values: dict[str, object] = {
        "name": "a1_lane_test",
        "stage": "A1P",
        "model_type": "recurrent_state_space_associative_lane",
        "recurrent_path": "parallel_scan",
        "seed": 3,
        "context_length": 32,
        "vocab_size": 32_000,
        "hidden_size": 32,
        "num_layers": 2,
        "num_attention_heads": 4,
        "intermediate_size": 64,
        "dropout": 0.0,
        "target_parameters_min": 1,
        "target_parameters_max": 10_000_000,
    }
    values.update(overrides)
    return ExperimentConfig(**values)


def test_lane_parameter_count_and_mode_initialization_match() -> None:
    config = tiny_lane_config()
    torch.manual_seed(11)
    protected = A1AssociativeLaneLM(config, mode="protected")
    torch.manual_seed(11)
    ordinary = A1AssociativeLaneLM(config, mode="ordinary")

    assert protected.parameter_count() == ordinary.parameter_count()
    assert protected.parameter_count() > protected.base.parameter_count()
    assert protected.parameter_count() - protected.base.parameter_count() == 6_370
    for name, parameter in protected.state_dict().items():
        torch.testing.assert_close(parameter, ordinary.state_dict()[name])


def test_lane_forward_backward_returns_finite_logits_and_loss() -> None:
    torch.manual_seed(13)
    model = A1AssociativeLaneLM(tiny_lane_config(), mode="protected")
    input_ids = torch.tensor(
        [[30_000, 30_001, 100, 8_100, 30_004, 30_002, 100, 30_003, 8_100]],
        dtype=torch.long,
    )
    output = model(input_ids, labels=input_ids)
    assert output.logits.shape == (1, input_ids.shape[1], 32_000)
    assert output.loss is not None
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert model.key_projection.weight.grad is not None
    assert torch.isfinite(model.key_projection.weight.grad).all()


def test_protected_memory_holds_after_value_until_query() -> None:
    torch.manual_seed(17)
    config = tiny_lane_config()
    model = A1AssociativeLaneLM(config, mode="protected")
    model.eval()
    input_ids = torch.tensor(
        [[30_000, 30_001, 100, 8_100, 30_004, 30_004, 30_002, 100, 30_003, 8_100]],
        dtype=torch.long,
    )
    _output, trace = model.forward_with_lane_trace(input_ids)
    assert trace["value_event"][0, 3]
    assert trace["answer_event"][0, 8]
    value_memory = trace["memory_value"][0, 3]
    distractor_memory = trace["memory_value"][0, 5]
    torch.testing.assert_close(value_memory, distractor_memory, atol=1e-6, rtol=0.0)


def test_ordinary_memory_is_allowed_to_update_after_value() -> None:
    torch.manual_seed(17)
    config = tiny_lane_config()
    model = A1AssociativeLaneLM(config, mode="ordinary")
    model.eval()
    input_ids = torch.tensor(
        [[30_000, 30_001, 100, 8_100, 30_004, 30_004, 30_002, 100, 30_003, 8_100]],
        dtype=torch.long,
    )
    _output, trace = model.forward_with_lane_trace(input_ids)
    value_memory = trace["memory_value"][0, 3]
    distractor_memory = trace["memory_value"][0, 5]
    assert not torch.allclose(value_memory, distractor_memory, atol=1e-6, rtol=0.0)
