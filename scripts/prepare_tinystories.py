#!/usr/bin/env python3
"""Prepare a pinned TinyStories corpus and 32k byte-level BPE tokenizer.

This script downloads only the pinned TinyStories parquet shards, trains a
tokenizer on the deduplicated source ``train`` split, creates deterministic
train/validation/test assignments, and writes token streams. It never imports
the model and never performs a training step.

The generated corpus and tokenizer stay under ``data/`` and are ignored by
git. The small JSON manifest is the reproducibility record for those local
artifacts; it contains source checksums, split counts, packing rules, and
library versions.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from array import array
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "tinystories_a0"
DATASET_REPO = "roneneldan/TinyStories"
DATASET_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
DATASET_LICENSE = "CDLA-Sharing-1.0"
DATASET_CARD_URL = "https://huggingface.co/datasets/roneneldan/TinyStories"
PAPER_URL = "https://arxiv.org/abs/2305.07759"
LICENSE_URL = "https://cdla.dev/sharing-1-0/"

TRAIN_FILES = (
    "data/train-00000-of-00004-2d5a1467fff1081b.parquet",
    "data/train-00001-of-00004-5852b56a2bd28fd9.parquet",
    "data/train-00002-of-00004-a26307300439e943.parquet",
    "data/train-00003-of-00004-d243063613e5a057.parquet",
)
VALIDATION_FILES = ("data/validation-00000-of-00001-869c898b519ad725.parquet",)
SOURCE_FILES: Mapping[str, tuple[str, ...]] = {
    "train": TRAIN_FILES,
    "validation": VALIDATION_FILES,
}

VOCAB_SIZE = 32_000
CONTEXT_LENGTH = 512
SPECIAL_TOKENS = ("<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>")
PAD_ID, BOS_ID, EOS_ID, UNK_ID = range(len(SPECIAL_TOKENS))


@dataclass(frozen=True)
class SourceDocument:
    """A source row with stable identity independent of local file paths."""

    source_split: str
    source_file: str
    row_index: int
    text: str

    @property
    def document_id(self) -> str:
        return f"{self.source_split}:{self.source_file}:{self.row_index}"


@dataclass
class PartitionStats:
    """Counters used to audit deterministic assignment and deduplication."""

    source_rows: dict[str, int] = field(default_factory=lambda: {"train": 0, "validation": 0})
    empty_rows: dict[str, int] = field(default_factory=lambda: {"train": 0, "validation": 0})
    duplicate_rows: dict[str, int] = field(default_factory=lambda: {"train": 0, "validation": 0})
    assigned_documents: dict[str, int] = field(
        default_factory=lambda: {"train": 0, "validation": 0, "test": 0}
    )


def canonicalize_text(value: object) -> str | None:
    """Apply the fixed text preprocessing rule used by this preparation."""

    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def assign_validation_split(digest: str) -> str:
    """Assign a source-validation document to validation or test by stable hash."""

    return "validation" if int(digest[:8], 16) % 2 == 0 else "test"


def _require_pyarrow() -> Any:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError(
            "data preparation requires pyarrow; install the project's data extras"
        ) from exc
    return parquet


def iter_parquet_documents(path: Path, source_split: str, source_file: str) -> Iterator[SourceDocument]:
    """Yield text rows in stable shard and row order."""

    parquet = _require_pyarrow()
    parquet_file = parquet.ParquetFile(path)
    try:
        text_index = parquet_file.schema_arrow.get_field_index("text")
    except AttributeError:  # pragma: no cover - older pyarrow compatibility
        text_index = parquet_file.schema_arrow.names.index("text")
    if text_index < 0:
        raise ValueError(f"source file has no text column: {path}")

    row_index = 0
    for batch in parquet_file.iter_batches(columns=["text"], batch_size=8192):
        for value in batch.column(text_index).to_pylist():
            text = canonicalize_text(value)
            if text is not None:
                yield SourceDocument(source_split, source_file, row_index, text)
            row_index += 1


def iter_source_documents(source_paths: Mapping[str, Sequence[tuple[str, Path]]]) -> Iterator[SourceDocument]:
    """Yield all source rows in the pinned order."""

    for source_split in ("train", "validation"):
        for source_file, path in source_paths[source_split]:
            yield from iter_parquet_documents(path, source_split, source_file)


def iter_partitioned_documents(
    source_paths: Mapping[str, Sequence[tuple[str, Path]]],
    stats: PartitionStats | None = None,
) -> Iterator[tuple[str, SourceDocument, str]]:
    """Yield deduplicated documents and their final deterministic split.

    The official source ``train`` split remains training data. The official
    source ``validation`` split is divided by text hash into validation/test.
    A first-seen hash wins, so duplicate documents cannot cross final splits.
    """

    audit = stats or PartitionStats()
    seen_hashes: set[str] = set()
    for source_split in ("train", "validation"):
        for source_file, path in source_paths[source_split]:
            parquet = _require_pyarrow()
            parquet_file = parquet.ParquetFile(path)
            try:
                text_index = parquet_file.schema_arrow.get_field_index("text")
            except AttributeError:  # pragma: no cover - older pyarrow compatibility
                text_index = parquet_file.schema_arrow.names.index("text")
            if text_index < 0:
                raise ValueError(f"source file has no text column: {path}")
            row_index = 0
            for batch in parquet_file.iter_batches(columns=["text"], batch_size=8192):
                for value in batch.column(text_index).to_pylist():
                    audit.source_rows[source_split] += 1
                    text = canonicalize_text(value)
                    if text is None:
                        audit.empty_rows[source_split] += 1
                        row_index += 1
                        continue
                    digest = text_sha256(text)
                    if digest in seen_hashes:
                        audit.duplicate_rows[source_split] += 1
                        row_index += 1
                        continue
                    seen_hashes.add(digest)
                    final_split = source_split if source_split == "train" else assign_validation_split(digest)
                    audit.assigned_documents[final_split] += 1
                    yield final_split, SourceDocument(source_split, source_file, row_index, text), digest
                    row_index += 1


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, logical_name: str) -> dict[str, Any]:
    return {
        "path": logical_name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def download_sources(data_root: Path, revision: str) -> dict[str, list[tuple[str, Path]]]:
    """Fetch the five pinned parquet files into the repository data directory."""

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError(
            "data preparation requires huggingface_hub; install the project's data extras"
        ) from exc

    source_paths: dict[str, list[tuple[str, Path]]] = {"train": [], "validation": []}
    for source_split, filenames in SOURCE_FILES.items():
        for filename in filenames:
            downloaded = hf_hub_download(
                repo_id=DATASET_REPO,
                filename=filename,
                repo_type="dataset",
                revision=revision,
                local_dir=str(data_root / "raw"),
            )
            source_paths[source_split].append((filename, Path(downloaded)))
    return source_paths


def write_split_manifest(
    source_paths: Mapping[str, Sequence[tuple[str, Path]]],
    destination: Path,
) -> tuple[PartitionStats, dict[str, Any]]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    stats = PartitionStats()
    with destination.open("w", encoding="utf-8") as handle:
        for final_split, document, digest in iter_partitioned_documents(source_paths, stats):
            record = {
                "document_id": document.document_id,
                "source_split": document.source_split,
                "source_file": document.source_file,
                "source_row_index": document.row_index,
                "text_sha256": digest,
                "final_split": final_split,
            }
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    return stats, {
        "source_rows": stats.source_rows,
        "empty_rows": stats.empty_rows,
        "duplicate_rows_removed": stats.duplicate_rows,
        "assigned_documents": stats.assigned_documents,
        "sha256": sha256_file(destination),
        "bytes": destination.stat().st_size,
    }


def _training_texts(
    source_paths: Mapping[str, Sequence[tuple[str, Path]]],
    expected_count: int,
) -> Iterator[str]:
    count = 0
    for final_split, document, _ in iter_partitioned_documents(source_paths):
        if final_split == "train":
            count += 1
            yield document.text
    if count != expected_count:
        raise RuntimeError(f"training document count changed during tokenizer pass: {count} != {expected_count}")


def train_tokenizer(source_paths: Mapping[str, Sequence[tuple[str, Path]]], output_path: Path, train_count: int) -> dict[str, Any]:
    """Train and save the fixed 32k byte-level BPE tokenizer."""

    try:
        from tokenizers import Tokenizer
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.trainers import BpeTrainer
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError(
            "data preparation requires tokenizers; install the project's data extras"
        ) from exc

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    tokenizer = Tokenizer(BPE(unk_token=SPECIAL_TOKENS[UNK_ID]))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=list(SPECIAL_TOKENS),
        show_progress=True,
    )
    tokenizer.train_from_iterator(
        _training_texts(source_paths, train_count),
        trainer=trainer,
        length=train_count,
    )
    actual_vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    if actual_vocab_size != VOCAB_SIZE:
        raise RuntimeError(f"trained tokenizer vocabulary is {actual_vocab_size}, expected {VOCAB_SIZE}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))
    return {
        "path": str(output_path.relative_to(REPO_ROOT)),
        "sha256": sha256_file(output_path),
        "bytes": output_path.stat().st_size,
        "type": "byte_level_bpe",
        "vocab_size": actual_vocab_size,
        "special_tokens": list(SPECIAL_TOKENS),
        "trained_on_source_split": "train",
        "min_frequency": 2,
    }


def _write_uint16(handle: Any, token_ids: Iterable[int]) -> int:
    values = list(token_ids)
    if any(token_id < 0 or token_id >= 2**16 for token_id in values):
        raise ValueError("token id does not fit in uint16")
    encoded = array("H", values)
    if sys.byteorder != "little":
        encoded.byteswap()
    handle.write(encoded.tobytes())
    return len(values)


def write_token_stream(
    tokenizer: Any,
    source_paths: Mapping[str, Sequence[tuple[str, Path]]],
    final_split: str,
    output_path: Path,
) -> dict[str, Any]:
    """Encode one split as BOS/text/EOS uint16 tokens in document order."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    documents = 0
    token_count = 0
    with output_path.open("wb") as handle:
        for assigned_split, document, _ in iter_partitioned_documents(source_paths):
            if assigned_split != final_split:
                continue
            encoding = tokenizer.encode(document.text, add_special_tokens=False)
            token_count += _write_uint16(handle, (BOS_ID, *encoding.ids, EOS_ID))
            documents += 1
    complete_sequences, remainder = divmod(token_count, CONTEXT_LENGTH)
    return {
        "path": str(output_path.relative_to(REPO_ROOT)),
        "sha256": sha256_file(output_path),
        "bytes": output_path.stat().st_size,
        "documents": documents,
        "tokens": token_count,
        "complete_context_sequences": complete_sequences,
        "discarded_tail_tokens": remainder,
    }


def package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def prepare(data_root: Path, revision: str) -> dict[str, Any]:
    source_paths = download_sources(data_root, revision)
    split_manifest_path = data_root / "split_manifest.jsonl"
    stats, split_manifest = write_split_manifest(source_paths, split_manifest_path)
    tokenizer_path = data_root / "tokenizer.json"
    tokenizer = train_tokenizer(source_paths, tokenizer_path, stats.assigned_documents["train"])

    from tokenizers import Tokenizer

    loaded_tokenizer = Tokenizer.from_file(str(tokenizer_path))
    streams = {
        split: write_token_stream(
            loaded_tokenizer,
            source_paths,
            split,
            data_root / f"tokens_{split}.bin",
        )
        for split in ("train", "validation", "test")
    }

    source_records = {
        split: [file_record(path, logical_name) for logical_name, path in source_paths[split]]
        for split in ("train", "validation")
    }
    manifest = {
        "schema_version": 1,
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "repo_id": DATASET_REPO,
            "revision": revision,
            "license": DATASET_LICENSE,
            "license_url": LICENSE_URL,
            "dataset_card_url": DATASET_CARD_URL,
            "paper_url": PAPER_URL,
            "source_files": source_records,
            "license_note": "Local computational use is recorded; redistribution of data or enhanced data must preserve CDLA-Sharing-1.0 terms and attribution.",
        },
        "preprocessing": {
            "text_column": "text",
            "rule": "strip leading and trailing Unicode whitespace; preserve interior text and case; discard null/empty rows",
        },
        "partition": {
            "assignment": "official train remains train; official validation is assigned by SHA-256(text) first 32 bits parity; first-seen text hash wins",
            "manifest": {
                "path": str(split_manifest_path.relative_to(REPO_ROOT)),
                "sha256": split_manifest["sha256"],
                "bytes": split_manifest["bytes"],
            },
            "source_rows": stats.source_rows,
            "empty_rows": stats.empty_rows,
            "duplicate_rows_removed": stats.duplicate_rows,
            "assigned_documents": stats.assigned_documents,
            "no_cross_split_duplicate_text_hashes": True,
        },
        "tokenizer": tokenizer,
        "special_token_ids": {name: index for index, name in enumerate(SPECIAL_TOKENS)},
        "packing": {
            "context_length": CONTEXT_LENGTH,
            "format": "uint16 little-endian token ids",
            "document_boundary": "prepend BOS and append EOS to every document",
            "streaming": "concatenate documents within each split in source order",
            "training_sequence_rule": "consume non-overlapping complete 512-token sequences and drop only the recorded split tail",
            "streams": streams,
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "huggingface_hub": package_version("huggingface_hub"),
            "pyarrow": package_version("pyarrow"),
            "tokenizers": package_version("tokenizers"),
        },
    }
    manifest_path = data_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--revision", default=DATASET_REVISION)
    args = parser.parse_args(argv)
    data_root = args.data_root if args.data_root.is_absolute() else REPO_ROOT / args.data_root
    manifest = prepare(data_root, args.revision)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print("Preparation complete. No model training was run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
