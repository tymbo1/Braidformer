"""Bounded training-integrity utilities and the explicit A0 pilot runner."""

from braidformer.training.integrity import (
    IntegrityError,
    IntegrityMonitor,
    save_atomic_checkpoint,
    set_reproducible_seed,
    sha256_file,
    sha256_json,
    verify_checkpoint,
)
from braidformer.training.runner import (
    A0TrainingRunner,
    PackedTokenStream,
    TrainingConfigurationError,
    cosine_warmup_learning_rate,
)

__all__ = [
    "IntegrityError",
    "IntegrityMonitor",
    "A0TrainingRunner",
    "PackedTokenStream",
    "TrainingConfigurationError",
    "cosine_warmup_learning_rate",
    "save_atomic_checkpoint",
    "set_reproducible_seed",
    "sha256_file",
    "sha256_json",
    "verify_checkpoint",
]
