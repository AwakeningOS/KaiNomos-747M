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


def test_hard_selection_is_straight_through():
    """Training and deployment must be the same policy.

    A bare argmax carries no gradient, so the controller could not learn; Gumbel
    sampling carries gradient but trains a policy that is not the one deployed.
    At 82M that gap was 5.3% of compute -- the sampled policy sat on budget while
    its argmax counterpart overspent, because argmax always takes the expensive
    side of a distribution sampling only visits sometimes.
    """
    torch.manual_seed(41)
    cfg = Config.tiny()
    m = Model(cfg)
    m.train()
    ids = torch.randint(0, cfg.vocab_size, (2, 16))

    out = m(ids, route_state=RouteState(hard=True))
    grad = torch.autograd.grad(
        out.expected_cost, m.model.controller.head["F"].weight, allow_unused=True
    )[0]
    assert grad is not None and grad.abs().sum() > 0, "hard selection lost its gradient"

    # and the forward pass is still a one-hot, not a softened stand-in
    onehot = out.joint_decisions[0].hard_modes["F"]
    assert torch.allclose(onehot.sum(-1), torch.ones_like(onehot.sum(-1)), atol=1e-6)
    assert torch.minimum(onehot.abs(), (onehot - 1).abs()).max() < 1e-6


def test_price_closes_the_budget_for_the_deployed_policy():
    """Solving the price must put the *argmax* cost on target, not a sampled one."""
    from route_eval import solve_batch_price, _cost_at_price

    torch.manual_seed(43)
    cfg = Config.tiny()
    m = Model(cfg)
    m.train()
    ids = torch.randint(0, cfg.vocab_size, (2, 32))
    costs = OrganCosts(cfg, 32)

    price = 0.0
    for _ in range(4):
        out = m(ids, route_state=RouteState(price=price, hard=True))
        price = solve_batch_price(out.joint_decisions, costs.fixed_share,
                                  float(out.cost_target), price)
    achieved = _cost_at_price(out.joint_decisions, costs.fixed_share, price, price)
    assert abs(achieved - out.cost_target) < 0.02, (achieved, out.cost_target)
