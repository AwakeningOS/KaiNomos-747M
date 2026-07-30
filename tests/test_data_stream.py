import json

import numpy as np
import pytest

from train import (
    EpochLimitReached,
    ShardedTokenStream,
    manifest_split_paths,
    manifest_vocab_size,
)


def write_tokens(path, values):
    np.asarray(values, dtype=np.uint16).tofile(path)


def test_sharded_stream_crosses_boundaries_and_resumes(tmp_path):
    first = tmp_path / "train-000.bin"
    second = tmp_path / "train-001.bin"
    write_tokens(first, range(5))
    write_tokens(second, range(5, 12))

    stream = ShardedTokenStream(
        [first, second], sequence_length=4, batch_size=2, prefetch=False
    )
    assert stream.next().tolist() == [list(range(4)), list(range(4, 8))]
    assert stream.position == 8
    stream.close()

    resumed = ShardedTokenStream(
        [first, second], sequence_length=4, batch_size=1, position=8,
        prefetch=False,
    )
    assert resumed.next().tolist() == [list(range(8, 12))]
    with pytest.raises(EpochLimitReached):
        resumed.next()
    resumed.close()


def test_manifest_paths_follow_declared_shard_order(tmp_path):
    write_tokens(tmp_path / "b.bin", [3, 4])
    write_tokens(tmp_path / "a.bin", [1, 2, 3])
    manifest = {
        "tokenizer": {"vocab_size": 49_152},
        "splits": {
            "train": {
                "tokens": 5,
                "shards": [{"path": "b.bin"}, {"path": "a.bin"}],
            }
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    assert manifest_vocab_size(manifest) == 49_152
    assert [path.name for path in manifest_split_paths(
        manifest, tmp_path, "train"
    )] == ["b.bin", "a.bin"]


def test_manifest_rejects_escape_and_wrong_token_count(tmp_path):
    outside = tmp_path.parent / "outside.bin"
    write_tokens(outside, [1, 2])
    with pytest.raises(ValueError, match="escapes"):
        manifest_split_paths(
            {"splits": {"train": {"shards": [{"path": "../outside.bin"}]}}},
            tmp_path,
            "train",
        )

    write_tokens(tmp_path / "train.bin", [1, 2])
    with pytest.raises(ValueError, match="declares 3"):
        manifest_split_paths(
            {"splits": {"train": {"tokens": 3}}},
            tmp_path,
            "train",
        )
