from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from braidformer.config import ExperimentConfig
from braidformer.models import A0TransformerLM, A1RecurrentStateSpaceLM
from braidformer.training.runner import (
    PackedTokenStream,
    TrainingConfigurationError,
    build_forward_model,
    build_model,
    cosine_warmup_learning_rate,
    cosine_warmup_learning_rate_at_progress,
)


def _model_config(
    stage: str,
    model_type: str,
    recurrent_path: str = "sequential_reference",
) -> ExperimentConfig:
    return ExperimentConfig(
        name=f"{stage.lower()}_runner_test",
        stage=stage,
        model_type=model_type,
        recurrent_path=recurrent_path,
        context_length=8,
        vocab_size=17,
        hidden_size=16,
        num_layers=1,
        num_attention_heads=4,
        intermediate_size=32,
        compile_model=False,
    )


def test_forward_model_compilation_is_default_off_and_uses_default_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _model_config("A1", "recurrent_state_space", "parallel_scan")
    model = build_model(config)
    calls: list[tuple[object, str]] = []
    compiled_sentinel = object()

    def fake_compile(module: object, *, mode: str) -> object:
        calls.append((module, mode))
        return compiled_sentinel

    monkeypatch.setattr(torch, "compile", fake_compile)

    assert build_forward_model(model, compile_model=False) is model
    assert calls == []

    compiled = build_forward_model(
        model,
        compile_model=True,
    )
    assert compiled is compiled_sentinel
    assert calls == [(model, "default")]


def test_model_factory_selects_only_explicit_a0_or_a1() -> None:
    a0 = build_model(_model_config("A0", "transformer"))
    a1 = build_model(_model_config("A1", "recurrent_state_space"))

    assert isinstance(a0, A0TransformerLM)
    assert isinstance(a1, A1RecurrentStateSpaceLM)
    assert a1.use_parallel_scan is False

    a1_parallel = build_model(
        _model_config("A1", "recurrent_state_space", "parallel_scan")
    )
    assert isinstance(a1_parallel, A1RecurrentStateSpaceLM)
    assert a1_parallel.use_parallel_scan is True

    with pytest.raises(TrainingConfigurationError, match="unsupported model selection"):
        build_model(_model_config("A2", "transformer"))


def test_cosine_warmup_schedule_is_bounded_and_reaches_zero() -> None:
    values = [cosine_warmup_learning_rate(3e-4, step, 10, 100) for step in (1, 10, 55, 100)]

    assert values[0] == pytest.approx(3e-5)
    assert values[1] == pytest.approx(3e-4)
    assert 0.0 < values[2] < 3e-4
    assert values[3] == pytest.approx(0.0)


def test_progress_referenced_cosine_schedule_is_bounded() -> None:
    values = [
        cosine_warmup_learning_rate_at_progress(3e-4, progress, 0.01)
        for progress in (0.0, 0.005, 0.01, 0.5, 1.0)
    ]

    assert values[0] == pytest.approx(0.0)
    assert values[1] == pytest.approx(1.5e-4)
    assert values[2] == pytest.approx(3e-4)
    assert 0.0 < values[3] < 3e-4
    assert values[4] == pytest.approx(0.0)


def test_packed_token_stream_reads_little_endian_windows(tmp_path: Path) -> None:
    path = tmp_path / "tokens.bin"
    path.write_bytes(bytes((1, 0, 2, 0, 255, 0, 0, 1)))

    with PackedTokenStream(path, expected_tokens=4) as stream:
        assert stream.read_tokens(1, 2).tolist() == [2, 255]


def test_packed_token_stream_rejects_out_of_range_window(tmp_path: Path) -> None:
    path = tmp_path / "tokens.bin"
    path.write_bytes(bytes((1, 0, 2, 0)))

    with PackedTokenStream(path) as stream:
        with pytest.raises(IndexError):
            stream.read_tokens(1, 2)
