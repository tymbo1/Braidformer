import pytest

torch = pytest.importorskip("torch")

from braidformer.config import ExperimentConfig
from braidformer.models.transformer import A0TransformerLM, ModelConfigError


def tiny_config(**overrides: object) -> ExperimentConfig:
    values: dict[str, object] = {
        "name": "a0_test",
        "stage": "A0",
        "model_type": "transformer",
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


def test_a0_forward_backward_returns_expected_shapes() -> None:
    torch.manual_seed(3)
    model = A0TransformerLM(tiny_config())
    input_ids = torch.randint(0, 31, (2, 8), dtype=torch.long)

    output = model(input_ids, labels=input_ids)

    assert output.logits.shape == (2, 8, 31)
    assert output.loss is not None
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert model.token_embedding.weight.grad is not None


def test_a0_loss_uses_next_token_shift_and_ignores_first_label() -> None:
    torch.manual_seed(11)
    model = A0TransformerLM(tiny_config())
    model.eval()
    input_ids = torch.tensor([[1, 4, 7, 10, 13, 16]])
    labels = input_ids.clone()

    output = model(input_ids, labels=labels)
    expected = torch.nn.functional.cross_entropy(
        output.logits[:, :-1, :].reshape(-1, 31),
        labels[:, 1:].reshape(-1),
    )

    changed_first_label = labels.clone()
    changed_first_label[:, 0] = 30
    changed_first_label_output = model(input_ids, labels=changed_first_label)

    assert output.loss is not None
    assert changed_first_label_output.loss is not None
    assert torch.allclose(output.loss, expected)
    assert torch.allclose(output.loss, changed_first_label_output.loss)


def test_a0_attention_is_causal() -> None:
    torch.manual_seed(7)
    model = A0TransformerLM(tiny_config())
    model.eval()
    prefix = torch.tensor([[1, 2, 3, 4]])
    first_suffix = torch.tensor([[5, 6]])
    second_suffix = torch.tensor([[19, 20]])

    first = model(torch.cat((prefix, first_suffix), dim=1)).logits
    second = model(torch.cat((prefix, second_suffix), dim=1)).logits

    assert torch.allclose(first[:, :4], second[:, :4], atol=1e-6, rtol=0.0)


def test_a0_rejects_later_stage_configs() -> None:
    with pytest.raises(ModelConfigError, match="stage 'A0'"):
        A0TransformerLM(tiny_config(stage="A1"))


def test_planning_config_is_in_target_parameter_range() -> None:
    from pathlib import Path

    from braidformer.config import load_config

    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "braidnano_phase0_planning.json")
    model = A0TransformerLM(config)

    assert config.target_parameters_min <= model.parameter_count()
    assert model.parameter_count() <= config.target_parameters_max
    assert model.lm_head.weight is model.token_embedding.weight


def test_context_length_is_enforced() -> None:
    model = A0TransformerLM(tiny_config(context_length=4))
    input_ids = torch.zeros((1, 5), dtype=torch.long)

    with pytest.raises(ValueError, match="context_length"):
        model(input_ids)
