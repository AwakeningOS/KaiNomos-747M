"""The routed arm must spend what the 16-layer fixed model spends.

Fixed reference: KDA FULL, MLA GLOBAL_READ, FFN 1792, Delta ALL.  MUDD-QKV and
the low-rank key projection are fixed costs on both sides.
"""

import torch

from config import K3MiniPlusPlusPlusConfig as Config
from cost_model import OrganCosts
from joint_router import RouteState
from model import K3MiniPlusPlusPlusForCausalLM as Model


def test_budget_target_is_the_fixed_policy_cost():
    cfg = Config()
    costs = OrganCosts(cfg, cfg.context_length_train)
    assert cfg.joint_route.budget_ratio == 1.0
    assert costs.budget_target == costs.full_policy_cost


def test_pruning_and_reinvestment_both_have_room():
    cfg = Config()
    costs = OrganCosts(cfg, cfg.context_length_train)
    widest = float(costs.ffn[-1] - costs.ffn[cfg.joint_route.fixed_ffn_index])
    richest = costs.full_policy_cost + widest * cfg.num_hidden_layers
    assert costs.min_policy_cost < costs.budget_target < richest, (
        costs.min_policy_cost, costs.budget_target, richest
    )


def test_forced_fixed_costs_exactly_the_target():
    torch.manual_seed(19)
    cfg = Config.tiny()
    m = Model(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    with torch.no_grad():
        out = m(ids, route_state=RouteState(force_fixed=True))
    assert abs(float(out.expected_cost) - out.cost_target) < 1e-6


def test_mudd_and_delta_keys_are_charged_to_both_arms():
    """They are unconditional, so they belong in the fixed share, not the
    variable one; charging them to the routed arm alone would understate its
    budget and hand it less compute than the reference."""
    cfg = Config()
    costs = OrganCosts(cfg, cfg.context_length_train)
    assert costs.fixed_share > 0
    assert costs.controller_share > 0
    assert costs.fixed_share > costs.controller_share


def test_cheaper_and_richer_policies_move_the_cost_the_right_way():
    torch.manual_seed(23)
    cfg = Config.tiny()
    m = Model(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 16))

    def cost_with(index_by_organ):
        with torch.no_grad():
            for organ, index in index_by_organ.items():
                head = m.model.controller.head[organ]
                head.weight.zero_()
                head.bias.zero_()
                head.bias[index] = 50.0
            return float(m(ids, route_state=RouteState(hard=True)).expected_cost)

    fixed_f = cfg.joint_route.fixed_ffn_index
    baseline = cost_with({"K": 1, "M": 1, "F": fixed_f, "R": 3})
    cheap = cost_with({"K": 0, "M": 0, "F": 0, "R": 0})
    rich = cost_with({"K": 1, "M": 1, "F": len(cfg.joint_route.ffn_width_tiers) - 1, "R": 3})
    assert cheap < baseline < rich
