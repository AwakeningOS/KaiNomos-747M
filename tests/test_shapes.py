"""Shapes, parameter budget, and finite loss/gradients."""

import torch

from config import K3MiniPlusPlusPlusConfig as Config
from model import K3MiniPlusPlusPlusForCausalLM as Model


def test_forward_shapes_and_finite_losses():
    cfg = Config.tiny()
    m = Model(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    out = m(ids, labels=ids)

    assert out.logits.shape == (2, 16, cfg.vocab_size)
    assert out.mtp_logits.shape == (2, 14, cfg.vocab_size)   # loses two positions
    for value in (out.loss, out.ntp_loss, out.mtp_loss, out.expected_cost):
        assert torch.isfinite(value).all()

    out.loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_all_sixteen_layers_are_always_executed():
    """No layer may be skipped; routing varies capacity inside a layer only."""
    cfg = Config()
    assert cfg.num_hidden_layers == 16
    assert cfg.layer_pattern.count("KDA") == 12
    assert cfg.layer_pattern.count("MLA") == 4
    m = Model(cfg)
    assert len(m.model.layers) == 16


def test_ffn_widths_are_nested_and_bracket_the_base_width():
    cfg = Config()
    tiers = cfg.joint_route.ffn_width_tiers
    assert tiers == (1024, 1408, 1792, 2176, 2432, 2816)
    assert tuple(sorted(set(tiers))) == tiers          # strictly nested prefixes
    assert cfg.dense_intermediate_size == 1792 == cfg.joint_route.fixed_ffn_width
    assert max(tiers) > cfg.dense_intermediate_size    # reinvestment room exists
    assert min(tiers) < cfg.dense_intermediate_size    # pruning room exists
    assert cfg.ffn_intermediate_size == 2816           # one matrix, not six FFNs

    m = Model(Config.tiny())
    widths = {layer.ffn.gate_proj.out_features for layer in m.model.layers}
    assert widths == {max(Config.tiny().joint_route.ffn_width_tiers)}


def test_parameter_budget_is_inside_the_target_range():
    report = Model(Config()).parameter_report()
    assert 109_000_000 <= report["total_params"] <= 112_000_000, report["total_params"]
    assert report["controller_params"] < 100_000
    assert report["mtp_only_params"] > 0
    assert (report["inference_backbone_params"]
            == report["total_params"] - report["mtp_only_params"])
