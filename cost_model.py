"""Analytical MAC model normalised to the Dense Fixed-Full backbone cost."""

from __future__ import annotations

import torch

from config import K3MiniPlusPlusPlusConfig as K3MiniConfig


class OrganCosts:
    """JointRoute costs with Dense Fixed-Full defined as exactly 1.0.

    Fixed-Full is the `fixed_ffn_width` policy, not the widest FFN tier: the
    tiers above it are reinvestment modes that cost *more* than Fixed.  The
    controller overhead is charged only to JointRoute, so its all-maximum policy
    sits above 1.0 while its budget target is `budget_ratio` (1.0 = spend
    exactly what Fixed-Full spends, redistributed).
    """

    def __init__(self, config: K3MiniConfig, context_length: int):
        self.config = config
        self.context_length = context_length
        h = config.hidden_size
        k = config.kda
        m = config.mla

        kda_extra = 3.0 * k.num_heads * k.head_dim * k.head_dim
        mla_read = float(
            m.num_heads * context_length
            * (2 * m.kv_lora_rank + m.qk_shared_head_dim)
        )
        ffn_costs = [
            float(3 * h * width) for width in config.joint_route.ffn_width_tiers
        ]

        unavoidable = float(h * config.vocab_size)
        variable_full = 0.0
        for index, kind in enumerate(config.layer_pattern):
            unavoidable += self._attention_fixed(config, kind)
            # Delta routing: the key projection and the rank-R score run for
            # every available source at both sublayer positions, whatever tier
            # is chosen; only the full-width value read is switchable.  The two
            # source counts below are different quantities and must not share a
            # name: MUDD sees every past *layer*, Delta sees every past *block*.
            delta_sources = index // config.block_size + 1
            mudd_sources = index + 1
            unavoidable += 2.0 * self._delta_scores(
                h, config.delta.key_rank, delta_sources
            )
            unavoidable += 4.0 * h
            # MUDD is always on for both arms: the coefficient MLP plus one mix
            # per stream over the layer outputs this layer can see.  KDA mixes
            # three streams, MLA two, so the count is per operator.
            streams = len(config.mudd.streams_for(kind))
            unavoidable += float(h * config.mudd.hidden
                                 + config.mudd.hidden * streams * mudd_sources
                                 + streams * mudd_sources * h)
            variable_full += kda_extra if kind == "KDA" else mla_read
            variable_full += ffn_costs[config.joint_route.fixed_ffn_index]
            # The delta value read is unconditional now: Delta Block always reads
            # every source, so it is a fixed cost rather than a routed one.
            unavoidable += 2.0 * self._delta_values(h, delta_sources + 1)

        self.normalizer = unavoidable + variable_full
        controller = self._controller_cost(config) * config.num_hidden_layers
        self.fixed_share = (unavoidable + controller) / self.normalizer
        self.controller_share = controller / self.normalizer
        self.kda = torch.tensor([0.0, kda_extra / self.normalizer])
        self.mla = torch.tensor([0.0, mla_read / self.normalizer])
        self.ffn = torch.tensor([value / self.normalizer for value in ffn_costs])
        self._h = h
        self._tiers = config.delta.tiers

        full_variable = variable_full / self.normalizer
        self.full_policy_cost = self.fixed_share + full_variable
        # The budget is a ratio of what the *Base arm actually spends*, not of a
        # notional 1.0.  Both arms run the controller -- Base simply overrides its
        # output -- so the controller overhead is present on both sides and must
        # be inside the target, otherwise JointRoute is held 0.7% below Base and
        # the comparison is no longer equal-FLOPs.
        self.budget_target = config.joint_route.budget_ratio * self.full_policy_cost
        self.min_policy_cost = self._minimum_cost(config)

    @staticmethod
    def _attention_fixed(config: K3MiniConfig, kind: str) -> float:
        h = config.hidden_size
        if kind == "KDA":
            k = config.kda
            projections = 5.0 * h * h
            decay = 2.0 * h * k.decay_rank
            beta = h * k.num_heads
            conv = 3.0 * h * k.short_conv_kernel_size
            decay_only = k.num_heads * k.head_dim * k.head_dim
            return projections + decay + beta + conv + decay_only
        m = config.mla
        return float(
            h * m.q_lora_rank
            + m.q_lora_rank * m.num_heads * m.q_head_dim
            + h * (m.kv_lora_rank + m.qk_shared_head_dim)
            + m.kv_lora_rank * m.num_heads
            * (m.qk_nope_head_dim + m.v_head_dim)
            + 2 * h * m.num_heads * m.v_head_dim
        )

    @staticmethod
    def _delta_scores(hidden: int, key_rank: int, sources: int) -> float:
        """Key projection plus the rank-R dot product, paid for every source."""
        return float(sources * (hidden * key_rank + key_rank))

    @staticmethod
    def _delta_values(hidden: int, selected: int) -> float:
        """Full-width value read and weighted sum over the retrieved sources."""
        return float(2 * hidden * selected)

    @staticmethod
    def _depth_past_values(hidden: int, pre_past: int, post_past: int) -> float:
        return float(2 * hidden * (pre_past + post_past))

    @staticmethod
    def _controller_cost(config: K3MiniConfig) -> float:
        h = config.hidden_size
        c = config.joint_route.controller_hidden
        # Shared trunk plus the active attention head (K or M), F head and R head.
        return float(h * c + c * c + c * (2 + 4 + 4))

    def attn_res(self, pre_past: int, post_past: int) -> torch.Tensor:
        """Per-tier delta value-read cost for a layer with `post_past` sources."""
        sources = post_past
        values = []
        for tier in self._tiers:
            used = sources if tier < 0 else min(tier, sources)
            values.append(2.0 * self._delta_values(self._h, used) / self.normalizer)
        return torch.tensor(values)

    def _minimum_cost(self, config: K3MiniConfig) -> float:
        total = self.fixed_share
        for index in range(config.num_hidden_layers):
            sources = index // config.block_size + 1
            total += float(self.ffn[0])
        return total


__all__ = ["OrganCosts"]
