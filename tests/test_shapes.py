"""Shapes, parameter budget, and finite loss/gradients."""

import torch

from config import KaiNomosConfig as Config
from model import KaiNomosForCausalLM as Model


def test_forward_shapes_and_finite_losses():
    cfg = Config.tiny()
    m = Model(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    out = m(ids, labels=ids)

    assert out.logits.shape == (2, 16, cfg.vocab_size)
    assert out.mtp_logits.shape == (2, 14, cfg.vocab_size)   # loses two positions
    for value in (out.loss, out.ntp_loss, out.mtp_loss):
        assert torch.isfinite(value).all()

    out.loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_all_sixteen_layers_are_always_executed():
    """The fixed production stack always contains all sixteen dense layers."""
    cfg = Config()
    assert cfg.num_hidden_layers == 16
    assert cfg.layer_pattern.count("KDA") == 12
    assert cfg.layer_pattern.count("MLA") == 4
    m = Model(cfg)
    assert len(m.model.layers) == 16


def test_the_deployed_model_is_dense():
    """The production model has one FFN width and no routing controller."""
    cfg = Config()
    cfg.kda_impl = "reference"          # this test runs on CPU
    assert cfg.dense_intermediate_size == 6144
    assert cfg.ffn_intermediate_size == 6144

    m = Model(cfg)
    widths = {layer.ffn.gate_proj.out_features for layer in m.model.layers}
    assert widths == {6144}

    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    assert torch.isfinite(m(ids, labels=ids).loss)


def test_production_parameter_count_matches_public_architecture():
    report = Model(Config()).parameter_report()
    # h1536 / L16 / FFN 6144 dense with Muon. The complete production smoke path
    # measured 18.08 GB peak at micro-batch 1 on a 24 GB RTX 3090.
    assert report["total_params"] == 747_368_168
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
