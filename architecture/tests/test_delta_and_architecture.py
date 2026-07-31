from __future__ import annotations

import copy

import torch
from config import KaiNomosConfig
from delta_block import DeltaRouter, DeltaState, visible_sources
from kda import KDAttention
from mla import GatedMLA
from model import KaiNomosForCausalLM


def test_first_attention_route_is_exact_identity():
    router = DeltaRouter(8)
    hidden = torch.randn(2, 5, 8)
    routed, stats = router(hidden, (), return_stats=True)
    assert torch.equal(routed, hidden)
    assert stats.source_count == 0


def test_source_rules_do_not_publish_fake_zero_partial():
    embedding = torch.randn(1, 4, 8)
    state = DeltaState(embedding)
    first = visible_sources(
        state, embedding, embedding,
        embedding_visible=False, include_partial=False,
    )
    assert first == ()
    attention_delta = torch.randn_like(embedding)
    ffn = visible_sources(
        state, embedding + attention_delta, embedding,
        embedding_visible=True, include_partial=True,
    )
    assert ffn[0] is embedding
    torch.testing.assert_close(ffn[1], attention_delta)
    completed = (torch.randn_like(embedding),)
    next_stage = visible_sources(
        DeltaState(embedding, completed), embedding, embedding,
        embedding_visible=True, include_partial=False,
    )
    assert len(next_stage) == 2
    assert next_stage[-1] is completed[0]


def test_tiny_structure_and_main_residual_forward_backward():
    config = KaiNomosConfig.tiny()
    model = KaiNomosForCausalLM(config)
    assert len(model.model.layers) == 8
    assert sum(isinstance(layer.attn, KDAttention) for layer in model.model.layers) == 6
    assert sum(isinstance(layer.attn, GatedMLA) for layer in model.model.layers) == 2
    assert isinstance(model.model.layers[-1].attn, GatedMLA)
    ids = torch.randint(0, config.vocab_size, (1, 16))
    output = model(ids, labels=ids, return_route_stats=True)
    assert output.logits.shape == (1, 16, config.vocab_size)
    assert len(output.route_stats) == 8
    output.loss.backward()
    assert torch.isfinite(output.loss)


def test_48_zero_initialized_routers_and_no_mudd_keys():
    model = KaiNomosForCausalLM(KaiNomosConfig())
    report = model.parameter_report()
    assert report["delta_router_queries"] == 48
    assert report["mudd_params"] == 0
    assert report["mudd_state_keys"] == 0
    queries = [
        value for name, value in model.named_parameters()
        if name.endswith(("delta_attn.query", "delta_ffn.query"))
    ]
    assert all(torch.equal(value, torch.zeros_like(value)) for value in queries)


def test_exact_parameter_budget_without_and_with_mtp():
    without = KaiNomosForCausalLM(KaiNomosConfig())
    assert without.parameter_report()["inference_backbone_params"] == 718_341_812
    del without
    config = KaiNomosConfig()
    config.mtp.enabled = True
    with_mtp = KaiNomosForCausalLM(config)
    report = with_mtp.parameter_report()
    assert report["inference_backbone_params"] == 718_341_812
    assert report["mtp_only_params"] == 31_491_978
    assert report["total_params"] == 749_833_790


def test_mtp_toggle_does_not_change_backbone_initialization():
    torch.manual_seed(123)
    off = KaiNomosForCausalLM(KaiNomosConfig.tiny(mtp=False))
    torch.manual_seed(123)
    on = KaiNomosForCausalLM(KaiNomosConfig.tiny(mtp=True))
    off_state = off.model.state_dict()
    on_state = on.model.state_dict()
    assert off_state.keys() == on_state.keys()
    for key in off_state:
        assert torch.equal(off_state[key], on_state[key]), key


def test_stage_checkpoint_forward_and_grad_match():
    torch.manual_seed(9)
    config = KaiNomosConfig.tiny()
    regular = KaiNomosForCausalLM(config)
    checked = copy.deepcopy(regular)
    checked.gradient_checkpointing_enable()
    ids = torch.randint(0, config.vocab_size, (1, 12))
    regular_loss = regular(ids, labels=ids).loss
    checked_loss = checked(ids, labels=ids).loss
    torch.testing.assert_close(regular_loss, checked_loss)
    regular_loss.backward()
    checked_loss.backward()
    for expected, actual in zip(regular.parameters(), checked.parameters(), strict=True):
        torch.testing.assert_close(expected.grad, actual.grad)
