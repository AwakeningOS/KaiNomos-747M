"""With the gate at zero the Delta Block must be exactly the identity.

That is what lets an 82M checkpoint be migrated in and keep behaving as it did:
the new mechanism starts contributing nothing and has to earn its output.
"""

import torch

from config import K3MiniPlusPlusPlusConfig as Config
from delta_block import DeltaRouter
from model import K3MiniPlusPlusPlusForCausalLM as Model


def test_zero_gate_returns_the_residual_untouched():
    torch.manual_seed(11)
    router = DeltaRouter(hidden_size=16, key_rank=8).double()
    h = torch.randn(2, 5, 16, dtype=torch.float64)
    keys = [torch.randn(2, 5, 8, dtype=torch.float64) for _ in range(3)]
    values = [torch.randn(2, 5, 16, dtype=torch.float64) for _ in range(3)]
    assert torch.equal(router(h, keys, values), h)


def test_all_gates_start_at_zero_in_a_fresh_model():
    m = Model(Config.tiny())
    for layer in m.model.layers:
        assert layer.delta_attn.gate.item() == 0.0
        assert layer.delta_ffn.gate.item() == 0.0


def test_delta_is_added_not_substituted():
    torch.manual_seed(13)
    router = DeltaRouter(hidden_size=16, key_rank=8).double()
    with torch.no_grad():
        router.gate.fill_(1.0)
    h = torch.randn(2, 5, 16, dtype=torch.float64)
    keys = [torch.randn(2, 5, 8, dtype=torch.float64)]
    values = [torch.zeros(2, 5, 16, dtype=torch.float64)]
    # a zero delta must leave h exactly alone even with the gate wide open,
    # which only holds for an additive formulation
    assert torch.allclose(router(h, keys, values), h, atol=1e-12)


def test_tier_zero_retrieves_nothing_and_tier_all_retrieves_everything():
    torch.manual_seed(17)
    router = DeltaRouter(hidden_size=8, key_rank=4).double()
    with torch.no_grad():
        router.gate.fill_(2.0)
    h = torch.randn(1, 3, 8, dtype=torch.float64)
    keys = [torch.randn(1, 3, 4, dtype=torch.float64) for _ in range(3)]
    values = [torch.randn(1, 3, 8, dtype=torch.float64) for _ in range(3)]
    tiers = (0, 1, 2, -1)

    def onehot(index):
        t = torch.zeros(1, 3, 4, dtype=torch.float64)
        t[..., index] = 1.0
        return t

    assert torch.allclose(router(h, keys, values, onehot(0), tiers), h, atol=1e-12)
    full = router(h, keys, values, onehot(3), tiers)
    assert not torch.allclose(full, h, atol=1e-9)
    assert torch.allclose(full, router(h, keys, values), atol=1e-12)


def test_low_rank_key_but_full_width_value():
    cfg = Config()
    assert cfg.delta.key_rank == 64
    m = Model(Config.tiny())
    proj = m.model.delta_keys.proj[0]
    assert proj.in_features == Config.tiny().hidden_size     # value stays full width
    assert proj.out_features == Config.tiny().delta.key_rank  # only the key shrinks
