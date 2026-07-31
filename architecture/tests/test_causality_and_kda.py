from __future__ import annotations

import copy

import torch
from config import KaiNomosConfig
from kda import KDAttention
from model import KaiNomosForCausalLM


def test_model_has_no_future_token_leakage():
    torch.manual_seed(31)
    config = KaiNomosConfig.tiny()
    model = KaiNomosForCausalLM(config).eval()
    first = torch.randint(5, config.vocab_size, (1, 12))
    second = first.clone()
    second[:, 7:] = torch.randint(5, config.vocab_size, second[:, 7:].shape)
    with torch.no_grad():
        left = model(first).logits[:, :7]
        right = model(second).logits[:, :7]
    torch.testing.assert_close(left, right)


def test_packed_document_does_not_change_after_boundary():
    torch.manual_seed(32)
    config = KaiNomosConfig.tiny()
    model = KaiNomosForCausalLM(config).eval()
    left = torch.randint(5, config.vocab_size, (1, 5))
    changed = torch.randint(5, config.vocab_size, (1, 5))
    suffix = torch.randint(5, config.vocab_size, (1, 6))
    eod = torch.tensor([[config.eod_token_id]])
    one = torch.cat((left, eod, suffix), 1)
    two = torch.cat((changed, eod, suffix), 1)
    with torch.no_grad():
        logits_one = model(one).logits[:, 6:]
        logits_two = model(two).logits[:, 6:]
    torch.testing.assert_close(logits_one, logits_two, atol=2e-5, rtol=2e-5)


def test_kda_reference_cache_matches_full_forward_and_gradients_are_finite():
    torch.manual_seed(33)
    config = KaiNomosConfig.tiny()
    module = KDAttention(config)
    cached_module = copy.deepcopy(module)
    x = torch.randn(1, 8, config.hidden_size, requires_grad=True)
    full, _ = module(x)
    pieces = []
    cache = None
    for position in range(x.shape[1]):
        value, cache = cached_module(
            x[:, position:position + 1], cache=cache, use_cache=True
        )
        pieces.append(value)
    cached = torch.cat(pieces, dim=1)
    torch.testing.assert_close(full, cached, atol=2e-5, rtol=2e-5)
    full.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in module.parameters()
    )
