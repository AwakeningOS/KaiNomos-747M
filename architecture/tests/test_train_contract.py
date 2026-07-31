from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from config import KaiNomosConfig
from interleave import (
    DeterministicInterleaver,
    InterleavedSequenceStream,
    ShardSpec,
    SourceSpec,
)
from model import KaiNomosForCausalLM
from muon import Muon, muon_param_groups
from train import (
    TrainConfig,
    assert_stream_alignment,
    consumed_stream_tokens,
    learning_rate_at_step,
    parameter_sha256,
    save_checkpoint,
)


class _Interleaver:
    def __init__(self, tokens: int):
        self.tokens = tokens

    def source_token_accounting(self):
        return {"a": {"tokens": self.tokens}}


def _stream(*, read: int, pending: int):
    return SimpleNamespace(
        interleaver=_Interleaver(read),
        pending=[] if not pending else [("a", np.zeros(pending, dtype=np.uint16))],
    )


def test_stream_alignment_includes_pending_read_ahead():
    stream = _stream(read=160, pending=32)
    assert consumed_stream_tokens(stream) == 128
    assert_stream_alignment(stream, step=2, tokens_done=128, tokens_per_step=64)


def test_stream_alignment_rejects_step_and_cursor_mismatch():
    with pytest.raises(RuntimeError, match="step/token mismatch"):
        assert_stream_alignment(
            _stream(read=128, pending=0), 2, tokens_done=64, tokens_per_step=64
        )
    with pytest.raises(RuntimeError, match="data cursor mismatch"):
        assert_stream_alignment(
            _stream(read=127, pending=0), 2, tokens_done=128, tokens_per_step=64
        )


def test_frozen_token_budgets_align_and_resume_lr_uses_absolute_step():
    config = TrainConfig()
    assert config.tokens_per_step == 65_536
    assert config.target_tokens % config.tokens_per_step == 0
    assert config.schedule_tokens % config.tokens_per_step == 0
    assert learning_rate_at_step(1234, config) == learning_rate_at_step(1234, config)
    assert learning_rate_at_step(0, config) < learning_rate_at_step(1, config)


def test_initial_parameter_hash_proves_same_seed_same_weights():
    torch.manual_seed(11)
    first = KaiNomosForCausalLM(KaiNomosConfig.tiny())
    torch.manual_seed(11)
    second = KaiNomosForCausalLM(KaiNomosConfig.tiny())
    assert parameter_sha256(first) == parameter_sha256(second)
    with torch.no_grad():
        second.model.embed_tokens.weight[0, 0].add_(1)
    assert parameter_sha256(first) != parameter_sha256(second)


def test_atomic_checkpoint_roundtrip_preserves_next_batch(tmp_path):
    token_path = tmp_path / "tokens.bin"
    index_path = tmp_path / "tokens.idx"
    np.arange(256, dtype=np.uint16).tofile(token_path)
    np.asarray([0, 256], dtype=np.uint64).tofile(index_path)
    source = SourceSpec(
        "only", 1.0, True,
        (ShardSpec(token_path, index_path, 256, 1),),
    )
    stream = InterleavedSequenceStream(
        DeterministicInterleaver([source], seed=11, max_chunk_tokens=13), 8, 1
    )
    stream.next_batch()

    model_config = KaiNomosConfig.tiny()
    model = KaiNomosForCausalLM(model_config)
    optimizer = Muon(muon_param_groups(model, lr=3e-4), lr=3e-4)
    train_config = TrainConfig(
        sequence_length=8, micro_batch=1, grad_accum=1,
        target_tokens=8, schedule_tokens=8,
    )
    checkpoint = tmp_path / "step_00000001.pt"
    save_checkpoint(
        checkpoint, model, optimizer, stream, 1, 8,
        model_config, train_config, {"contract": "test"},
    )
    assert checkpoint.is_file()
    assert not checkpoint.with_suffix(".pt.tmp").exists()
    payload = torch.load(checkpoint, weights_only=False)

    resumed = InterleavedSequenceStream(
        DeterministicInterleaver([source], seed=11, max_chunk_tokens=13), 8, 1
    )
    resumed.load_state_dict(payload["stream"])
    expected, _ = stream.next_batch()
    actual, _ = resumed.next_batch()
    np.testing.assert_array_equal(actual, expected)
