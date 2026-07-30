"""At initialisation MUDD must be the identity on the newest depth state.

If it were not, switching the module on would already change the model, and any
later gain could not be attributed to the mixing that training learned.
"""

import torch

from config import KaiNomosConfig as Config
from mudd_qkv import MuDDQKV


def test_identity_initialisation_returns_the_newest_source():
    """Exactly the newest source -- not a normalised version of it.

    This assertion used to compare against `input_norm(sources[-1])`, which made
    it pass while the module actually returned `RMSNorm(h)` instead of `h`.  The
    per-stream norms now live on the layer as a single `attn_norm`, so identity
    means bit-equality with the newest depth state.
    """
    torch.manual_seed(5)
    mudd = MuDDQKV(hidden_size=16, num_sources=4, mlp_hidden=8).double()
    sources = [torch.randn(2, 5, 16, dtype=torch.float64) for _ in range(4)]
    out = mudd(sources)

    for name in ("q", "k", "v"):
        assert torch.equal(out[name], sources[-1]), name


def test_mla_layers_mix_one_kv_stream():
    """MLA reads a single compressed KV latent, so it gets one KV stream."""
    from config import MuDDConfig

    mudd = MuDDConfig()
    assert mudd.streams_for("KDA") == ("q", "k", "v")
    assert mudd.streams_for("MLA") == ("q", "kv")

    mla = MuDDQKV(hidden_size=8, num_sources=3, mlp_hidden=4,
                  streams=mudd.streams_for("MLA"))
    sources = [torch.randn(2, 4, 8) for _ in range(3)]
    out = mla(sources)
    assert set(out) == {"q", "kv"}


def test_mla_rejects_a_separate_v_input():
    """The bug this guards: v_input was accepted and silently dropped."""
    import pytest

    from mla import GatedMLA

    cfg = Config.tiny()
    attn = GatedMLA(cfg)
    x = torch.randn(1, 4, cfg.hidden_size)
    with pytest.raises(ValueError, match="single KV input"):
        attn(q_input=x, k_input=x, v_input=x)


def test_static_bias_selects_only_the_newest_source():
    mudd = MuDDQKV(hidden_size=8, num_sources=5, mlp_hidden=4)
    assert torch.equal(mudd.static_bias[:, -1], torch.ones(3))
    assert torch.equal(mudd.static_bias[:, :-1], torch.zeros(3, 4))
    assert torch.equal(mudd.fc2.weight, torch.zeros_like(mudd.fc2.weight))


def test_coefficients_are_not_softmaxed():
    """Direct linear mixing, so a coefficient may be negative or exceed one."""
    torch.manual_seed(1)
    mudd = MuDDQKV(hidden_size=8, num_sources=3, mlp_hidden=4)
    torch.nn.init.normal_(mudd.fc2.weight, std=1.0)
    sources = [torch.randn(2, 4, 8) for _ in range(3)]
    summary = mudd.coefficient_summary(sources)
    assert summary.shape == (3, 3)
    # a softmax would force every row of raw coefficients onto the simplex
    coef = mudd.fc2(torch.nn.functional.gelu(mudd.fc1(mudd.norm(sources[-1]))))
    coef = coef.view(2, 4, 3, 3) + mudd.static_bias
    assert not torch.allclose(coef.sum(-1), torch.ones(2, 4, 3), atol=1e-3)


def test_model_with_identity_mudd_matches_direct_qkv_input():
    """Turning MUDD on must not move the logits at all.

    Previously this only checked that the per-stream norm weights were ones and
    that the logits were finite -- it never compared the two paths, so it could
    not have caught the norm that made MUDD a non-identity.  Now it runs the same
    weights with and without MUDD and demands equality.
    """
    from model import KaiNomosForCausalLM as Model

    torch.manual_seed(7)
    cfg = Config.tiny()
    m = Model(cfg).double().eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 10))
    with torch.no_grad():
        with_mudd = m(ids).logits
        for layer in m.model.layers:
            layer.mudd = None            # same weights, MUDD path removed
        without_mudd = m(ids).logits

    assert torch.isfinite(with_mudd).all()
    assert torch.equal(with_mudd, without_mudd)
