from __future__ import annotations

import torch
from config import KaiNomosConfig
from generation import GenerationState, cached_forward
from mla import GatedMLA
from model import KaiNomosForCausalLM


def test_cached_tokenwise_logits_match_full_context():
    torch.manual_seed(22)
    config = KaiNomosConfig.tiny()
    model = KaiNomosForCausalLM(config).eval()
    ids = torch.randint(5, config.vocab_size, (1, 9))
    with torch.no_grad():
        full = model(ids, respect_documents=False).logits
        cached, state = cached_forward(model, ids)
    torch.testing.assert_close(full, cached, atol=3e-5, rtol=3e-5)
    assert state.cache.position == ids.shape[1]
    for layer in state.cache.layers:
        if layer.mla is not None:
            assert not hasattr(layer.mla, "key")
            assert not hasattr(layer.mla, "value")


def test_parallel_prefill_and_absorbed_decode_match_full_context():
    torch.manual_seed(23)
    config = KaiNomosConfig.tiny()
    model = KaiNomosForCausalLM(config).eval()
    prompt = torch.randint(5, config.vocab_size, (1, 9))
    continuation = torch.randint(5, config.vocab_size, (1, 1))
    for module in model.modules():
        if isinstance(module, GatedMLA):
            module.absorbed_decode_enabled = True
    with torch.no_grad():
        expected = model(
            torch.cat((prompt, continuation), dim=1),
            respect_documents=False,
        ).logits[:, -1:]
        _, state = cached_forward(model, prompt)
        actual, state = cached_forward(model, continuation, state)
    torch.testing.assert_close(expected, actual, atol=3e-5, rtol=3e-5)
    assert state.cache.position == prompt.shape[1] + 1


def test_prefill_uses_one_model_call_without_eod():
    torch.manual_seed(25)
    config = KaiNomosConfig.tiny()
    model = KaiNomosForCausalLM(config).eval()
    prompt = torch.randint(5, config.vocab_size, (1, 9))
    calls = []
    original = model.forward

    def counted_forward(*args, **kwargs):
        calls.append(args[0].shape[1])
        return original(*args, **kwargs)

    model.forward = counted_forward
    with torch.no_grad():
        cached_forward(model, prompt)
    assert calls == [prompt.shape[1]]


def test_eod_resets_all_temporal_caches():
    torch.manual_seed(24)
    config = KaiNomosConfig.tiny()
    model = KaiNomosForCausalLM(config).eval()
    prefix = torch.randint(5, config.vocab_size, (1, 5))
    eod = torch.tensor([[config.eod_token_id]])
    fresh = torch.randint(5, config.vocab_size, (1, 1))
    _, state = cached_forward(model, torch.cat((prefix, eod), dim=1))
    after_reset, _ = cached_forward(model, fresh, state)
    direct, _ = cached_forward(model, fresh, GenerationState())
    torch.testing.assert_close(after_reset, direct)
