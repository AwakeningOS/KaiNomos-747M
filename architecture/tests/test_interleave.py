from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
from interleave import (
    DeterministicInterleaver,
    adapt_manifest,
)


def _write_shard(root: Path, name: str, documents: list[list[int]]) -> dict:
    values = np.asarray([token for doc in documents for token in doc], dtype=np.uint16)
    offsets = np.asarray(
        [0, *np.cumsum([len(document) for document in documents], dtype=np.uint64)],
        dtype=np.uint64,
    )
    values.tofile(root / f"{name}.bin")
    offsets.tofile(root / f"{name}.idx")
    return {
        "path": f"{name}.bin",
        "index": f"{name}.idx",
        "tokens": int(values.size),
        "documents": len(documents),
    }


def _manifest(tmp_path: Path, *, large: bool = False) -> dict:
    if large:
        # Long documents exercise bounded chunks while keeping the 1M-token
        # resume comparison fast (about one thousand chunks, not one million).
        alpha_docs = [
            ((np.arange(2048, dtype=np.uint16) + 10 + i) % 49_000).tolist()
            for i in range(128)
        ]
        beta_docs = [
            ((np.arange(2048, dtype=np.uint16) + 20_000 + i) % 49_000).tolist()
            for i in range(128)
        ]
    else:
        alpha_docs = [[10, 11, 12], [13, 14], [15, 16, 17, 18]]
        beta_docs = [[100, 101], [102, 103, 104], [105], [106, 107]]
    return {
        "interleave": {
            "sources": [
                {
                    "source_id": "alpha",
                    "weight": 0.7,
                    "major": True,
                    "shards": [_write_shard(tmp_path, "alpha-train-00000", alpha_docs)],
                },
                {
                    "source_id": "beta",
                    "weight": 0.3,
                    "major": True,
                    "shards": [_write_shard(tmp_path, "beta-train-00000", beta_docs)],
                },
            ]
        }
    }


def _signature(stream: DeterministicInterleaver, count: int) -> list[tuple]:
    result = []
    for _ in range(count):
        chunk = stream.next_document_or_chunk()
        result.append(
            (
                chunk.source_id,
                chunk.shard_index,
                chunk.document_index,
                chunk.token_offset,
                chunk.tokens.tobytes(),
            )
        )
    return result


def _next_tokens(stream: DeterministicInterleaver, count: int) -> np.ndarray:
    pieces = []
    remaining = count
    while remaining:
        chunk = stream.next_document_or_chunk().tokens
        take = min(remaining, chunk.size)
        pieces.append(chunk[:take])
        remaining -= take
    return np.concatenate(pieces)


def test_same_seed_has_identical_weighted_source_and_document_order(tmp_path):
    manifest = _manifest(tmp_path)
    first = DeterministicInterleaver.from_manifest(
        manifest, tmp_path, seed=11, max_chunk_tokens=2
    )
    second = DeterministicInterleaver.from_manifest(
        manifest, tmp_path, seed=11, max_chunk_tokens=2
    )

    assert _signature(first, 200) == _signature(second, 200)
    assert first.source_token_accounting() == second.source_token_accounting()
    assert all(
        values["tokens"] > 0
        for values in first.source_token_accounting().values()
    )


def test_manifest_weights_control_tokens_even_when_document_lengths_differ(tmp_path):
    short = [[10, 11] for _ in range(20)]
    long = [list(range(100, 120)) for _ in range(20)]
    manifest = {
        "interleave": {
            "sources": [
                {
                    "source_id": "short", "weight": 0.7, "major": True,
                    "shards": [_write_shard(tmp_path, "short-train-00000", short)],
                },
                {
                    "source_id": "long", "weight": 0.3, "major": True,
                    "shards": [_write_shard(tmp_path, "long-train-00000", long)],
                },
            ]
        }
    }
    stream = DeterministicInterleaver.from_manifest(
        manifest, tmp_path, seed=11, max_chunk_tokens=10
    )
    accounting = {"short": 0, "long": 0}
    for _ in range(2_000):
        chunk = stream.next_document_or_chunk()
        assert chunk.token_count == 10
        accounting[chunk.source_id] += chunk.token_count
    assert accounting["short"] / sum(accounting.values()) == pytest.approx(
        0.7, abs=0.03
    )


def test_resume_next_one_million_tokens_is_bitwise_identical(tmp_path):
    manifest = _manifest(tmp_path, large=True)
    uninterrupted = DeterministicInterleaver.from_manifest(
        manifest, tmp_path, seed=20260731, max_chunk_tokens=1024
    )
    _signature(uninterrupted, 37)
    checkpoint = copy.deepcopy(uninterrupted.state_dict())

    expected = _next_tokens(uninterrupted, 1_000_000)
    resumed = DeterministicInterleaver.from_manifest(
        manifest, tmp_path, seed=20260731, max_chunk_tokens=1024
    )
    resumed.load_state_dict(checkpoint)
    actual = _next_tokens(resumed, 1_000_000)

    assert expected.dtype == np.uint16
    assert np.array_equal(actual, expected)
    assert resumed.source_token_accounting() == uninterrupted.source_token_accounting()
    assert resumed.state_dict()["cursors"] == uninterrupted.state_dict()["cursors"]


def test_major_source_coverage_gate_and_step_source_log(tmp_path):
    manifest = _manifest(tmp_path)
    stream = DeterministicInterleaver.from_manifest(
        manifest, tmp_path, seed=11, max_chunk_tokens=4
    )
    alpha = stream.next_document_or_chunk()
    while alpha.source_id != "alpha":
        alpha = stream.next_document_or_chunk()
    beta = stream.next_document_or_chunk()
    while beta.source_id != "beta":
        beta = stream.next_document_or_chunk()
    record = stream.note_optimizer_step([alpha, beta])
    for _ in range(99):
        stream.note_optimizer_step(["alpha"])
    assert record["source_chunks"] == {"alpha": 1, "beta": 1}
    assert record["source_tokens"] == {
        "alpha": alpha.token_count,
        "beta": beta.token_count,
    }
    stream.assert_major_source_coverage()

    failing = DeterministicInterleaver.from_manifest(
        manifest, tmp_path, seed=11, max_chunk_tokens=4
    )
    for _ in range(99):
        failing.note_optimizer_step(["alpha"])
    with pytest.raises(RuntimeError, match="beta"):
        failing.note_optimizer_step(["alpha"])


def test_datamix_v2_schema_difference_uses_named_prefix_adapter(tmp_path):
    local = _write_shard(tmp_path, "local-train-00000", [[1, 2], [3]])
    jpnmix = _write_shard(tmp_path, "jpnmix-train-00000", [[4, 5, 6]])
    manifest = {
        "format": "document-indexed-uint16-shards",
        # These totals cannot identify the domain of an individual document.
        "splits": {
            "train": {
                "shards": [local, jpnmix],
                "domain_tokens": {"ja_web": 3, "code": 3},
            }
        },
    }

    adapted = adapt_manifest(manifest, tmp_path)
    assert adapted.adapter_id == "datamix_v2_filename_prefix_v1"
    assert [source.source_id for source in adapted.sources] == ["jpnmix", "local"]
    assert sum(source.weight for source in adapted.sources) == pytest.approx(1.0)
    assert "cannot identify each document" in adapted.note


def test_canonical_manifest_requires_explicit_major_flag(tmp_path):
    shard = _write_shard(tmp_path, "alpha-train-00000", [[1, 2, 3]])
    manifest = {
        "interleave": {
            "sources": [{"source_id": "alpha", "weight": 1.0, "shards": [shard]}]
        }
    }
    with pytest.raises(ValueError, match="explicitly declare major"):
        adapt_manifest(manifest, tmp_path)
