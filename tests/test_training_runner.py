from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from braidformer.training.runner import PackedTokenStream, cosine_warmup_learning_rate


def test_cosine_warmup_schedule_is_bounded_and_reaches_zero() -> None:
    values = [cosine_warmup_learning_rate(3e-4, step, 10, 100) for step in (1, 10, 55, 100)]

    assert values[0] == pytest.approx(3e-5)
    assert values[1] == pytest.approx(3e-4)
    assert 0.0 < values[2] < 3e-4
    assert values[3] == pytest.approx(0.0)


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
