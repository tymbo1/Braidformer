import json

import pytest

torch = pytest.importorskip("torch")

from braidformer.training.integrity import (
    IntegrityError,
    IntegrityMonitor,
    save_atomic_checkpoint,
    set_reproducible_seed,
    verify_checkpoint,
)


def test_seed_report_and_monitor_record_verified_progress(tmp_path) -> None:
    report = set_reproducible_seed(17, deterministic=False)
    assert report["seed"] == 17
    assert report["torch"] is True

    monitor = IntegrityMonitor(
        run_dir=tmp_path / "run",
        metadata={"config_sha256": "config", "data_manifest_sha256": "data", "seed": 17},
        tokens_per_step=8192,
    )
    record = monitor.record_step(
        step=1,
        tokens_processed=8192,
        train_loss=4.0,
        validation_loss=4.2,
        gradient_norm=0.8,
        learning_rate=3e-4,
        tokens_per_second=1000.0,
        cuda_elapsed_seconds=0.2,
    )

    assert record["tokens_processed"] == 8192
    assert record["cuda_elapsed_seconds"] == pytest.approx(0.2)
    lines = (tmp_path / "run" / "steps.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["event"] == "step"


def test_monitor_stops_on_nonfinite_loss_and_progress_errors(tmp_path) -> None:
    monitor = IntegrityMonitor(
        run_dir=tmp_path / "run",
        metadata={},
        tokens_per_step=8192,
    )
    with pytest.raises(IntegrityError, match="non_finite_train_loss"):
        monitor.record_step(step=1, tokens_processed=8192, train_loss=float("nan"))

    with pytest.raises(IntegrityError, match="non_monotonic_step"):
        monitor.record_step(step=2, tokens_processed=16384, train_loss=4.0)


def test_checkpoint_sidecar_is_atomic_and_verifiable(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    metadata = {
        "step": 100,
        "tokens_processed": 819200,
        "seed": 17,
        "config_sha256": "config",
        "data_manifest_sha256": "data",
    }
    record = save_atomic_checkpoint({"weights": torch.ones(2)}, checkpoint, metadata)

    assert record["complete"] is True
    assert verify_checkpoint(checkpoint)["checkpoint_sha256"] == record["checkpoint_sha256"]

    checkpoint.write_bytes(checkpoint.read_bytes() + b"corruption")
    with pytest.raises(IntegrityError, match="hash mismatch"):
        verify_checkpoint(checkpoint)
