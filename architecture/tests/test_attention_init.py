from __future__ import annotations

import torch
from config import KaiNomosConfig
from kda import KDAttention
from mla import GatedMLA, MLACache


def test_kda_official_compatible_initialization():
    module = KDAttention(KaiNomosConfig.tiny())
    assert torch.equal(module.A_log, torch.zeros_like(module.A_log))
    assert torch.isfinite(module.dt_bias).all()
    assert float(module.dt_bias.detach().std()) > 0
    for conv in (module.q_conv, module.k_conv, module.v_conv):
        assert torch.equal(conv.weight[..., :-1], torch.zeros_like(conv.weight[..., :-1]))
        assert torch.equal(conv.weight[..., -1], torch.ones_like(conv.weight[..., -1]))


def test_mla_is_strict_nope_and_cache_has_no_full_kv():
    config = KaiNomosConfig.tiny()
    module = GatedMLA(config).eval()
    assert not module.absorbed_decode_enabled
    assert config.mla.qk_shared_head_dim == 0
    x = torch.randn(1, 5, config.hidden_size)
    with torch.no_grad():
        full, _ = module(x)
        chunks = []
        cache = None
        for index in range(x.shape[1]):
            value, cache = module(x[:, index:index + 1], cache=cache, use_cache=True)
            chunks.append(value)
    torch.testing.assert_close(full, torch.cat(chunks, dim=1), atol=2e-5, rtol=2e-5)
    assert isinstance(cache, MLACache)
    assert not hasattr(cache, "key") and not hasattr(cache, "value")
    assert cache.latent.shape[-1] == config.mla.kv_lora_rank
