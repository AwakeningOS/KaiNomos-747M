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


def test_the_deployed_model_is_dense():
    """One FFN width, no controller, nothing to constrain.

    Grouped per-width dispatch measured slower than the dense path it was meant
    to beat at this scale, so the deployed policy is a single width.
    """
    cfg = Config()
    cfg.kda_impl = "reference"          # this test runs on CPU
    assert cfg.joint_route.enabled is False
    assert cfg.joint_route.ffn_width_tiers == (6144,)
    assert cfg.dense_intermediate_size == 6144 == cfg.joint_route.fixed_ffn_width
    assert cfg.ffn_intermediate_size == 6144

    m = Model(cfg)
    assert m.model.controller is None
    widths = {layer.ffn.gate_proj.out_features for layer in m.model.layers}
    assert widths == {6144}

    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    out = m(ids, labels=ids)
    assert out.expected_cost is None
    assert out.joint_decisions == []


def test_ffn_widths_stay_nested_prefixes_when_routing_is_on():
    """The supernet is one matrix used at a prefix width, not N separate FFNs."""
    cfg = Config.tiny()
    tiers = cfg.joint_route.ffn_width_tiers
    assert tuple(sorted(set(tiers))) == tiers          # strictly nested prefixes
    assert max(tiers) > cfg.dense_intermediate_size    # reinvestment room exists
    assert min(tiers) < cfg.dense_intermediate_size    # pruning room exists
    assert cfg.ffn_intermediate_size == max(tiers)

    m = Model(cfg)
    widths = {layer.ffn.gate_proj.out_features for layer in m.model.layers}
    assert widths == {max(tiers)}


def test_parameter_budget_is_inside_the_target_range():
    report = Model(Config()).parameter_report()
    # h1536 / L16 / FFN 6144 dense with Muon, sized against a measured 20.24 GB
    # peak and 1,797 tok/s at micro-batch 2 on a 24 GB RTX 3090.  This is the
    # ceiling twice over: h1664 needs 22.42 GB, over the limit, *and* falls to
    # D/N 19.0 against a 16B pool, under the compute-optimal ratio.
    assert 718_000_000 <= report["total_params"] <= 727_000_000, report["total_params"]
    assert report["controller_params"] == 0        # dense: no controller at all
    assert report["mtp_only_params"] > 0
    assert (report["inference_backbone_params"]
            == report["total_params"] - report["mtp_only_params"])


def test_every_parameter_receives_a_gradient():
    """A parameter no forward pass reaches is dead weight, not capacity.

    `delta_keys` held `num_blocks + 1` projections while the bank never offers
    more than `num_blocks` sources, so the last one was never indexed.
    """
    cfg = Config.tiny()
    m = Model(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    m(ids, labels=ids).loss.backward()
    missing = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, missing
