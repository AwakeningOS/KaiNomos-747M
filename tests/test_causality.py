"""No layer may read a future token, a future layer, or a future block."""

import torch

from config import K3MiniPlusPlusPlusConfig as Config
from model import K3MiniPlusPlusPlusForCausalLM as Model


def test_no_future_token_leakage():
    torch.manual_seed(3)
    cfg = Config.tiny()
    m = Model(cfg).double().eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 12))
    pos = 7
    other = ids.clone()
    other[:, pos] = (other[:, pos] + 1) % cfg.vocab_size

    with torch.no_grad():
        a = m(ids).logits
        b = m(other).logits
    assert (a[:, :pos] - b[:, :pos]).abs().max().item() < 1e-10
    assert (a[:, pos] - b[:, pos]).abs().max().item() > 1e-8


def test_mudd_only_sees_its_own_and_earlier_depth_states():
    """Layer l mixes over l+1 sources: the embedding plus outputs 0..l-1."""
    cfg = Config.tiny()
    m = Model(cfg)
    for index, layer in enumerate(m.model.layers):
        assert layer.num_sources == index + 1
        assert layer.mudd.num_sources == index + 1


def test_delta_bank_never_exposes_an_unfinished_future_block():
    from delta_block import DeltaBank

    bank = DeltaBank()
    h0 = torch.randn(1, 3, 8)
    bank.start_block(h0)
    assert len(bank.completed) == 0
    # before any block closes, the only source is the current partial delta
    assert len(bank.sources(h0 + 1)) == 1

    bank.close_block(h0 + 2)
    assert len(bank.completed) == 1
    assert torch.allclose(bank.completed[0], torch.full_like(h0, 2.0))
    # a completed block plus the new partial
    assert len(bank.sources(h0 + 3)) == 2


def test_mtp_target_shift_predicts_the_token_after_next():
    from mtp import mtp_slices

    ids = torch.arange(8).view(1, 8)
    h_sl, e_sl, t_sl = mtp_slices(ids)
    hidden_pos = ids[:, h_sl]      # positions 0..5
    given_next = ids[:, e_sl]      # positions 1..6
    target = ids[:, t_sl]          # positions 2..7

    assert hidden_pos.shape == given_next.shape == target.shape
    # the head is told t+1 and must predict t+2, never the token it was handed
    assert torch.equal(given_next, hidden_pos + 1)
    assert torch.equal(target, hidden_pos + 2)
