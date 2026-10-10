import pytest

torch = pytest.importorskip("torch")

from braidformer.config import ExperimentConfig, load_config
from braidformer.models import (
    A0TransformerLM,
    A1RecurrentStateSpaceLM,
    GatedDiagonalStateSpaceMixer,
    ModelConfigError,
)


def tiny_a1_config(**overrides: object) -> ExperimentConfig:
    values: dict[str, object] = {
        "name": "a1_test",
        "stage": "A1",
        "model_type": "recurrent_state_space",
        "seed": 3,
        "context_length": 16,
        "vocab_size": 31,
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


def test_a1_forward_backward_returns_expected_shapes() -> None:
    torch.manual_seed(3)
    model = A1RecurrentStateSpaceLM(tiny_a1_config())
    input_ids = torch.randint(0, 31, (2, 8), dtype=torch.long)

    output = model(input_ids, labels=input_ids)

    assert output.logits.shape == (2, 8, 31)
    assert output.loss is not None
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert model.token_embedding.weight.grad is not None
    assert torch.isfinite(model.token_embedding.weight.grad).all()


def test_a1_recurrence_is_causal() -> None:
    torch.manual_seed(7)
    model = A1RecurrentStateSpaceLM(tiny_a1_config())
    model.eval()
    prefix = torch.tensor([[1, 2, 3, 4]])
    first_suffix = torch.tensor([[5, 6]])
    second_suffix = torch.tensor([[19, 20]])

    first = model(torch.cat((prefix, first_suffix), dim=1)).logits
    second = model(torch.cat((prefix, second_suffix), dim=1)).logits

    assert torch.allclose(first[:, :4], second[:, :4], atol=1e-6, rtol=0.0)


def test_a1_state_is_reset_for_each_forward_sequence() -> None:
    torch.manual_seed(13)
    model = A1RecurrentStateSpaceLM(tiny_a1_config())
    model.eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])

    first = model(input_ids).logits
    second = model(input_ids).logits

    assert torch.allclose(first, second, atol=0.0, rtol=0.0)


def test_a1_has_recurrent_mixer_and_no_attention_module() -> None:
    model = A1RecurrentStateSpaceLM(tiny_a1_config())

    assert all(isinstance(block.state_space, GatedDiagonalStateSpaceMixer) for block in model.blocks)
    assert not any(module.__class__.__name__ == "CausalSelfAttention" for module in model.modules())


def test_a1_rejects_a0_or_non_recurrent_configs() -> None:
    with pytest.raises(ModelConfigError, match="stage 'A1'"):
        A1RecurrentStateSpaceLM(tiny_a1_config(stage="A0"))
    with pytest.raises(ModelConfigError, match="recurrent_state_space"):
        A1RecurrentStateSpaceLM(tiny_a1_config(model_type="transformer"))
    with pytest.raises(ModelConfigError, match="stage 'A0'"):
        A0TransformerLM(tiny_a1_config())


def test_a1_full_proposal_parameter_count_is_close_to_a0() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    a1_config = load_config(root / "configs" / "braidnano_a1_recurrent_state_space_proposal.json")
    a0_config = load_config(root / "configs" / "braidnano_a0_baseline_proposal.json")
    a1_model = A1RecurrentStateSpaceLM(a1_config)
    a0_model = A0TransformerLM(a0_config)

    assert a1_model.parameter_count() == 41_870_336
    assert a0_model.parameter_count() == 41_866_240
    difference = abs(a1_model.parameter_count() - a0_model.parameter_count())
    assert difference == 4_096
    assert difference / a0_model.parameter_count() < 0.0001


def test_a1_context_length_is_enforced() -> None:
    model = A1RecurrentStateSpaceLM(tiny_a1_config(context_length=4))
    input_ids = torch.zeros((1, 5), dtype=torch.long)

    with pytest.raises(ValueError, match="context_length"):
        model(input_ids)
