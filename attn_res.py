"""Block Attention Residuals using the official prefix-sum semantics."""

from __future__ import annotations

import torch
import torch.nn as nn

from layers import RMSNorm


def _selects_all(tier: torch.Tensor) -> bool:
    """True when every token picked the last (retrieve-everything) tier."""
    index = tier.argmax(-1)
    return bool((index == tier.shape[-1] - 1).all())


class DepthAttention(nn.Module):
    """Select over completed block values plus the current block prefix."""

    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps)
        self.query = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self,
        completed_blocks: list[torch.Tensor],
        prefix_sum: torch.Tensor,
        tier: torch.Tensor | None = None,
        return_weights: bool = False,
    ):
        if not completed_blocks:
            if return_weights:
                weights = prefix_sum.new_ones(*prefix_sum.shape[:-1], 1)
                return prefix_sum, weights
            return prefix_sum
        values = torch.stack([*completed_blocks, prefix_sum], dim=-2)
        keys = self.norm(values).float()
        scores = self.query(keys).squeeze(-1).float()
        # An ALL tier keeps every candidate, so it must take the *same* code
        # path as the unrouted model rather than an equivalent-but-reordered one:
        # the tiered branch renormalises a masked exponential, which differs from
        # a plain softmax in the last bits and would leave the forced-Fixed
        # supernet merely close to the Standalone Base instead of identical.
        if tier is not None and _selects_all(tier):
            tier = None
        weights = scores.softmax(-1) if tier is None else self._tiered_softmax(
            scores, tier.float(), len(completed_blocks)
        )
        output = torch.einsum("btj,btjh->bth", weights, values.float()).to(prefix_sum.dtype)
        return (output, weights) if return_weights else output

    def _tiered_softmax(
        self, scores: torch.Tensor, tier: torch.Tensor, n_past: int
    ) -> torch.Tensor:
        tiers = self._joint_route_tiers
        past = scores[..., :n_past]
        order = past.argsort(-1, descending=True)
        ranks = torch.empty_like(order)
        ranks.scatter_(
            -1, order, torch.arange(n_past, device=scores.device).expand_as(order)
        )
        limits = torch.tensor(
            [n_past if value < 0 else min(value, n_past) for value in tiers],
            device=scores.device,
        )
        per_tier = ranks.unsqueeze(-1) < limits.view(1, 1, 1, -1)
        keep_past = (per_tier.to(tier.dtype) * tier.unsqueeze(-2)).sum(-1)
        keep = torch.cat([keep_past, torch.ones_like(scores[..., :1])], -1)
        values = keep * torch.exp(scores - scores.amax(-1, keepdim=True))
        return values / values.sum(-1, keepdim=True).clamp_min(1e-20)

    def set_joint_route_tiers(self, tiers: tuple[int, ...]) -> None:
        self._joint_route_tiers = tiers


__all__ = ["DepthAttention"]
