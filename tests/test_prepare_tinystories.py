from scripts.prepare_tinystories import (
    SPECIAL_TOKENS,
    assign_validation_split,
    canonicalize_text,
    text_sha256,
)


def test_text_preprocessing_is_explicit_and_minimal() -> None:
    assert canonicalize_text("  Once upon a time.  ") == "Once upon a time."
    assert canonicalize_text("\n\t") is None
    assert canonicalize_text(None) is None


def test_validation_partition_is_hash_deterministic() -> None:
    assert assign_validation_split("0" * 64) == "validation"
    assert assign_validation_split("1" * 64) == "test"
    digest = text_sha256("same document")
    assert assign_validation_split(digest) == assign_validation_split(digest)


def test_special_token_order_is_stable() -> None:
    assert SPECIAL_TOKENS == ("<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>")
