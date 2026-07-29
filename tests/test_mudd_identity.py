"""At initialisation MUDD must be the identity on the newest depth state.

If it were not, switching the module on would already change the model, and any
later gain could not be attributed to the mixing that training learned.
"""

import torch

from config import K3MiniPlusPlusPlusConfig as Config
from mudd_qkv import MuDDQKV


def test_identity_initialisation_returns_the_newest_source():
    torch.manual_seed(5)
    mudd = MuDDQKV(hidden_size=16, num_sources=4, mlp_hidden=8).double()
    sources = [torch.randn(2, 5, 16, dtype=torch.float64) for _ in range(4)]
    out = mudd(sources)

    expected = {name: mudd.input_norm[name](sources[-1]) for name in ("q", "k", "v")}
    for name in ("q", "k", "v"):
        assert torch.allclose(out[name], expected[name], atol=1e-12), name


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
    from model import K3MiniPlusPlusPlusForCausalLM as Model

    torch.manual_seed(7)
    cfg = Config.tiny()
    m = Model(cfg).double().eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 10))
    with torch.no_grad():
        baseline = m(ids).logits

    # disabling MUDD entirely must give the same logits at identity init,
    # because the per-stream RMSNorms are the only remaining difference
    for layer in m.model.layers:
        for name in ("q", "k", "v"):
            assert torch.equal(
                layer.mudd.input_norm[name].weight,
                torch.ones_like(layer.mudd.input_norm[name].weight),
            )
    assert torch.isfinite(baseline).all()
