import pytest

torch = pytest.importorskip("torch")

from braidformer.config import ExperimentConfig, load_config
from braidformer.models import (
    A0TransformerLM,
    A1RecurrentStateSpaceLM,
    GatedDiagonalStateSpaceMixer,
    ModelConfigError,
)
from braidformer.models.recurrent import initial_state_transition_decays


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


@pytest.mark.parametrize("use_parallel_scan", [False, True])
def test_a1_recurrence_is_causal(use_parallel_scan: bool) -> None:
    torch.manual_seed(7)
    model = A1RecurrentStateSpaceLM(
        tiny_a1_config(),
        use_parallel_scan=use_parallel_scan,
    )
    model.eval()
    prefix = torch.tensor([[1, 2, 3, 4]])
    first_suffix = torch.tensor([[5, 6]])
    second_suffix = torch.tensor([[19, 20]])

    first = model(torch.cat((prefix, first_suffix), dim=1)).logits
    second = model(torch.cat((prefix, second_suffix), dim=1)).logits

    assert torch.allclose(first[:, :4], second[:, :4], atol=1e-6, rtol=0.0)


def test_a1_parallel_scan_matches_reference_forward_and_gradients() -> None:
    torch.manual_seed(23)
    config = tiny_a1_config()
    reference = A1RecurrentStateSpaceLM(config, use_parallel_scan=False)
    accelerated = A1RecurrentStateSpaceLM(config, use_parallel_scan=True)
    accelerated.load_state_dict(reference.state_dict())
    input_ids = torch.randint(0, 31, (2, 12), dtype=torch.long)

    reference_output = reference(input_ids, labels=input_ids)
    accelerated_output = accelerated(input_ids, labels=input_ids)

    assert reference_output.loss is not None
    assert accelerated_output.loss is not None
    torch.testing.assert_close(
        accelerated_output.logits,
        reference_output.logits,
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        accelerated_output.loss,
        reference_output.loss,
        atol=1e-6,
        rtol=1e-6,
    )

    reference_output.loss.backward()
    accelerated_output.loss.backward()
    for reference_parameter, accelerated_parameter in zip(
        reference.parameters(), accelerated.parameters()
    ):
        assert reference_parameter.grad is not None
        assert accelerated_parameter.grad is not None
        torch.testing.assert_close(
            accelerated_parameter.grad,
            reference_parameter.grad,
            atol=2e-5,
            rtol=2e-5,
        )


@pytest.mark.parametrize("use_parallel_scan", [False, True])
def test_a1_state_is_reset_for_each_forward_sequence(use_parallel_scan: bool) -> None:
    torch.manual_seed(13)
    model = A1RecurrentStateSpaceLM(
        tiny_a1_config(),
        use_parallel_scan=use_parallel_scan,
    )
    model.eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])

    first = model(input_ids).logits
    second = model(input_ids).logits

    assert torch.allclose(first, second, atol=0.0, rtol=0.0)


def test_a1_parallel_scan_matches_reference_at_context_512() -> None:
    torch.manual_seed(31)
    config = tiny_a1_config(context_length=512)
    reference = A1RecurrentStateSpaceLM(config, use_parallel_scan=False)
    accelerated = A1RecurrentStateSpaceLM(config, use_parallel_scan=True)
    accelerated.load_state_dict(reference.state_dict())
    input_ids = torch.randint(0, 31, (1, 512), dtype=torch.long)

    reference_logits = reference(input_ids).logits
    accelerated_logits = accelerated(input_ids).logits

    torch.testing.assert_close(
        accelerated_logits,
        reference_logits,
        atol=1e-5,
        rtol=1e-5,
    )


def test_a1_has_recurrent_mixer_and_no_attention_module() -> None:
    model = A1RecurrentStateSpaceLM(tiny_a1_config())

    assert all(isinstance(block.state_space, GatedDiagonalStateSpaceMixer) for block in model.blocks)
    assert not any(module.__class__.__name__ == "CausalSelfAttention" for module in model.modules())


def test_decay_tail_initialization_has_expected_time_scale_tail() -> None:
    decays = initial_state_transition_decays(
        512,
        "tail_75_original_25_long_v1",
    )
    taus = -1.0 / torch.log(decays)

    assert decays.shape == (512,)
    assert torch.allclose(decays[:384], torch.full((384,), 0.9))
    for start, expected_tau in zip((384, 416, 448, 480), (16.0, 32.0, 64.0, 128.0)):
        assert torch.allclose(
            taus[start : start + 32],
            torch.full((32,), expected_tau),
            atol=1e-3,
            rtol=0.0,
        )


def test_decay_tail_preserves_a1_parameter_count() -> None:
    model = A1RecurrentStateSpaceLM(
        tiny_a1_config(state_transition_initialization="tail_75_original_25_long_v1")
    )

    assert model.parameter_count() == A1RecurrentStateSpaceLM(tiny_a1_config()).parameter_count()


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
    assert A1RecurrentStateSpaceLM(a1_config, use_parallel_scan=True).parameter_count() == a1_model.parameter_count()


def test_a1_context_length_is_enforced() -> None:
    model = A1RecurrentStateSpaceLM(tiny_a1_config(context_length=4))
    input_ids = torch.zeros((1, 5), dtype=torch.long)

    with pytest.raises(ValueError, match="context_length"):
        model(input_ids)
