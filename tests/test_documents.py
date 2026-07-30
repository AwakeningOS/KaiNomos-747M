"""Documents packed into one sequence must not leak into each other.

The pool concatenates documents and cuts every 1024 tokens, so a sequence
normally holds two or three unrelated documents. Four things have to hold:

1. the variable-length KDA path equals running each document separately,
   in the forward pass and in both sets of gradients
2. changing a token in an earlier document does not change anything after the
   boundary
3. exactly one prediction per boundary is excluded from the loss
4. QK normalisation leaves unit-RMS queries and keys, trains, and stays finite
"""

import pytest
import torch

from config import KaiNomosConfig as Config
from model import KaiNomosForCausalLM as Model
from segments import (
    cu_seqlens,
    document_mask,
    document_starts,
    mask_targets_at_boundaries,
    segment_ids,
)

EOD = 4


def dense_tiny():
    """A CPU-sized configuration with the production document semantics."""
    cfg = Config.tiny()
    cfg.kda_impl = "reference"
    return cfg


def packed_ids(cfg, lengths, seed=0):
    """One row holding `lengths` documents, each terminated by `<|eod|>`."""
    generator = torch.Generator().manual_seed(seed)
    rows = []
    for length in lengths:
        body = torch.randint(5, cfg.vocab_size, (length - 1,), generator=generator)
        rows.append(torch.cat([body, torch.tensor([EOD])]))
    return torch.cat(rows).unsqueeze(0)


def test_segment_ids_and_offsets_follow_the_separator():
    ids = packed_ids(Config.tiny(), [4, 3, 5])
    starts = document_starts(ids, EOD)
    assert starts[0].tolist() == [
        True, False, False, False,       # doc 0, ends with EOD at index 3
        True, False, False,              # doc 1
        True, False, False, False, False,
    ]
    assert segment_ids(ids, EOD)[0].tolist() == [1] * 4 + [2] * 3 + [3] * 5
    assert cu_seqlens(starts).tolist() == [0, 4, 7, 12]


def test_document_mask_is_block_diagonal_and_causal():
    ids = packed_ids(Config.tiny(), [3, 2])
    mask = document_mask(segment_ids(ids, EOD))[0, 0]
    assert mask.tolist() == [
        [1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 1, 1],
    ]


def test_kda_varlen_matches_running_each_document_alone():
    """Forward, input gradient and parameter gradient must all agree."""
    from kda import KDAttention

    cfg = Config.tiny()
    cfg.kda_impl = "reference"
    torch.manual_seed(3)
    attn = KDAttention(cfg).double()
    lengths = [5, 4, 3]
    ids = packed_ids(cfg, lengths)
    segments = segment_ids(ids, EOD)

    x = torch.randn(1, sum(lengths), cfg.hidden_size, dtype=torch.float64,
                    requires_grad=True)
    packed, _ = attn(x, segments=segments)
    packed.square().sum().backward()
    packed_grad = x.grad.clone()
    packed_params = {n: p.grad.clone() for n, p in attn.named_parameters()
                     if p.grad is not None}

    attn.zero_grad(set_to_none=True)
    separate = x.detach().clone().requires_grad_(True)
    outputs, start = [], 0
    for length in lengths:
        piece, _ = attn(separate[:, start:start + length])
        outputs.append(piece)
        start += length
    joined = torch.cat(outputs, dim=1)
    joined.square().sum().backward()

    assert torch.allclose(packed, joined, atol=1e-10), \
        float((packed - joined).abs().max())
    assert torch.allclose(packed_grad, separate.grad, atol=1e-10)
    for name, grad in packed_params.items():
        other = dict(attn.named_parameters())[name].grad
        assert torch.allclose(grad, other, atol=1e-10), name


def test_earlier_documents_cannot_change_later_ones():
    """The whole point: edit document 0, and nothing after the boundary moves."""
    cfg = dense_tiny()
    torch.manual_seed(11)
    model = Model(cfg).double().eval()

    lengths = [6, 7]
    ids = packed_ids(cfg, lengths, seed=1)
    edited = ids.clone()
    edited[0, :3] = torch.tensor([9, 10, 11])       # rewrite document 0's body
    assert not torch.equal(ids, edited)

    with torch.no_grad():
        a = model(ids).logits
        b = model(edited).logits

    boundary = lengths[0]
    after_a, after_b = a[:, boundary:], b[:, boundary:]
    assert torch.allclose(after_a, after_b, atol=1e-12), \
        float((after_a - after_b).abs().max())
    # and the first document really was affected, or the test proves nothing
    assert not torch.allclose(a[:, :boundary], b[:, :boundary])


def test_only_the_separator_position_is_dropped_from_the_loss():
    cfg = Config.tiny()
    ids = packed_ids(cfg, [4, 3])
    inputs, targets = ids[:, :-1], ids[:, 1:]
    masked = mask_targets_at_boundaries(targets, inputs, EOD)

    dropped = (masked == -100)
    assert int(dropped.sum()) == 1, masked
    # The dropped pair is the separator predicting the next document's first
    # token: input index 3 is `<|eod|>`, and its target is document 1's opener.
    assert inputs[dropped][0].item() == EOD
    assert int(dropped.nonzero()[0, 1]) == 3
    # Everything inside document 1 is still trained on, including the prediction
    # made *from* its first token -- that one does not cross anything.
    assert masked[0, 4].item() == targets[0, 4].item()
    assert masked[0, 5].item() == targets[0, 5].item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the FLA kernels")
def test_varlen_kernel_isolates_documents_on_gpu():
    """The Triton path, not the reference one, is what training actually runs.

    Comparing it numerically against per-document calls only bounds kernel
    precision -- the chunk decomposition differs, and the dots run in TF32, which
    leaves ~3e-04 relative even in float32.  The property that matters is
    insensitive to that: rewrite the first document and nothing after the boundary
    may move at all.  Without the handling the same edit moves the following
    documents by 1.13 against an output scale of about 1.5, so this is not a
    second-order effect.
    """
    from kda import KDAttention

    cfg = Config()
    torch.manual_seed(7)
    attn = KDAttention(cfg).cuda().eval()
    assert attn.impl == "fla"

    lengths = [300, 450, 274]
    boundary = lengths[0]
    ids = packed_ids(cfg, lengths, seed=2).cuda()
    segments = segment_ids(ids, EOD)
    offsets = cu_seqlens(document_starts(ids, EOD))

    x = torch.randn(1, sum(lengths), cfg.hidden_size, device="cuda")
    edited = x.clone()
    edited[:, :boundary - 1] = torch.randn_like(edited[:, :boundary - 1])

    with torch.no_grad():
        a, _ = attn(x, segments=segments, seq_offsets=offsets)
        b, _ = attn(edited, segments=segments, seq_offsets=offsets)
        leaky_a, _ = attn(x)
        leaky_b, _ = attn(edited)

    assert float((a[:, :boundary] - b[:, :boundary]).abs().max()) > 0.1
    assert torch.equal(a[:, boundary:], b[:, boundary:])
    # the handling is what does it, not the input happening not to matter
    assert float((leaky_a[:, boundary:] - leaky_b[:, boundary:]).abs().max()) > 0.1


def test_qk_norm_gives_unit_rms_and_trains():
    from mla import GatedMLA

    cfg = Config.tiny()
    torch.manual_seed(5)
    attn = GatedMLA(cfg)
    assert attn.q_norm.weight.shape == (cfg.mla.q_head_dim,)
    assert attn.k_norm.weight.shape == (cfg.mla.q_head_dim,)
    assert torch.equal(attn.q_norm.weight, torch.ones(cfg.mla.q_head_dim))

    x = torch.randn(2, 6, cfg.hidden_size)
    latent, shared = attn.project_latent(x)
    q = attn.q_norm(attn.project_q(x))
    k = attn.k_norm(attn.expand_kv(latent, shared)[0])
    for name, value in (("q", q), ("k", k)):
        rms = value.float().square().mean(-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3), (name, rms)

    out, _ = attn(x)
    assert torch.isfinite(out).all()
    out.square().sum().backward()
    for name in ("q_norm", "k_norm"):
        grad = getattr(attn, name).weight.grad
        assert grad is not None and torch.isfinite(grad).all()
        assert float(grad.abs().sum()) > 0, name
