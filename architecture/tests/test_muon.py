from __future__ import annotations

import math

import pytest
import torch
from config import KaiNomosConfig
from model import KaiNomosForCausalLM
from muon import Muon, muon_param_groups, orthogonalise, validate_shared_lr
from torch import nn


def _group_by_name(groups):
    return {name: group for group in groups for name in group["param_names"]}


def test_shared_lr_contract_rejects_legacy_mismatch():
    assert validate_shared_lr(3e-4, muon_lr=3e-4, adamw_lr=3e-4) == 3e-4
    with pytest.raises(ValueError, match="share one LR"):
        validate_shared_lr(3e-4, muon_lr=2e-2)

    first = nn.Parameter(torch.zeros(2, 2))
    second = nn.Parameter(torch.zeros(2))
    with pytest.raises(ValueError, match="share one LR"):
        Muon([
            {"params": [first], "use_muon": True, "muon_layout": "full_matrix",
             "lr": 3e-4},
            {"params": [second], "use_muon": False, "lr": 2e-4},
        ])


@pytest.mark.parametrize("mtp", [False, True])
def test_explicit_classification_is_complete_unique_and_shared(mtp):
    model = KaiNomosForCausalLM(KaiNomosConfig.tiny(mtp=mtp))
    groups = muon_param_groups(model, lr=3e-4)
    grouped = [param for group in groups for param in group["params"]]
    expected = [param for param in model.parameters() if param.requires_grad]

    assert len(grouped) == len(expected)
    assert len({id(param) for param in grouped}) == len(grouped)
    assert {id(param) for param in grouped} == {id(param) for param in expected}
    assert {group["lr"] for group in groups} == {3e-4}


def test_per_head_metadata_uses_config_dimensions_for_kda_and_mla():
    config = KaiNomosConfig.tiny(mtp=True)
    model = KaiNomosForCausalLM(config)
    by_name = _group_by_name(muon_param_groups(model, lr=3e-4))

    kda = by_name["model.stages.0.layers.0.attn.q_proj.weight"]
    assert (kda["muon_layout"], kda["num_blocks"], kda["block_rows"]) == (
        "row_blocks", config.kda.num_heads, config.kda.head_dim,
    )
    mla_q = by_name["model.stages.0.layers.3.attn.q_b_proj.weight"]
    assert (mla_q["num_blocks"], mla_q["block_rows"]) == (
        config.mla.num_heads, config.mla.q_head_dim,
    )
    mla_kv = by_name["model.stages.0.layers.3.attn.kv_b_proj.weight"]
    assert (mla_kv["num_blocks"], mla_kv["block_rows"]) == (
        config.mla.num_heads,
        config.mla.qk_nope_head_dim + config.mla.v_head_dim,
    )
    mtp_q = by_name["mtp.attn.q_proj.weight"]
    assert (mtp_q["num_blocks"], mtp_q["block_rows"]) == (
        config.kda.num_heads, config.kda.head_dim,
    )


def test_embedding_decays_but_norm_decay_conv_and_delta_query_do_not():
    model = KaiNomosForCausalLM(KaiNomosConfig.tiny(mtp=True))
    by_name = _group_by_name(muon_param_groups(model, lr=3e-4))

    assert by_name["model.embed_tokens.weight"]["group_name"] == "adamw_decay"
    no_decay = (
        "model.final_norm.weight",
        "model.stages.0.layers.0.attn.A_log",
        "model.stages.0.layers.0.attn.dt_bias",
        "model.stages.0.layers.0.attn.q_conv.weight",
        "model.stages.0.layers.0.delta_attn.query",
        "mtp.hidden_norm.weight",
    )
    for name in no_decay:
        assert by_name[name]["group_name"] == "adamw_no_decay"
        assert by_name[name]["weight_decay"] == 0.0


def test_each_head_block_is_orthogonalised_independently_against_reference():
    parameter = nn.Parameter(torch.arange(48, dtype=torch.float64).view(8, 6) / 17)
    original = parameter.detach().clone()
    gradient = torch.linspace(-1, 1, 48, dtype=torch.float64).view_as(parameter)
    parameter.grad = gradient.clone()
    lr = 0.01
    scale = 0.2 * math.sqrt(6)
    expected_blocks = torch.stack([
        orthogonalise(block, steps=2).to(torch.float64) * scale
        for block in gradient.view(2, 4, 6)
    ])
    expected = original - lr * expected_blocks.view_as(parameter)

    optimizer = Muon([{
        "params": [parameter], "param_names": ["test.q_proj.weight"],
        "group_name": "muon_per_head", "use_muon": True,
        "muon_layout": "row_blocks", "num_blocks": 2, "block_rows": 4,
        "lr": lr, "momentum": 0.0, "nesterov": False, "ns_steps": 2,
        "update_rms": 0.2, "weight_decay": 0.0,
    }], lr=lr)
    optimizer.step(collect_stats=True)

    torch.testing.assert_close(parameter, expected, rtol=1e-12, atol=1e-12)
    stats = optimizer.last_step_stats[0]
    assert len(stats["head_update_rms"]) == 2
    assert math.isfinite(stats["update_rms"])
    assert math.isfinite(stats["head_update_rms_cv"])


def test_unknown_trainable_parameter_is_rejected_instead_of_shape_classified():
    model = KaiNomosForCausalLM(KaiNomosConfig.tiny())
    model.unreviewed_matrix = nn.Parameter(torch.zeros(3, 3))
    with pytest.raises(ValueError, match="unclassified"):
        muon_param_groups(model, lr=3e-4)
