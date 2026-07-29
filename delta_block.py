"""Projected Low-Rank Delta Block.

What is re-used is the *change* each block made, not the accumulated hidden
state: `delta_b = block_end - block_start`.  Values are kept at full width so a
re-added delta loses nothing; only the routing key is projected to 64 dims,
which is where the cost actually is.

The result is added to the residual stream, never substituted for it:

    h_hat = h + tanh(gate) * sum_i alpha_i delta_i

so with the gate at zero the block is exactly the identity and a checkpoint
migrated into this model behaves as it did before.  A replacing formulation
would have no such starting point.

JointRoute chooses how many sources to retrieve (0 / 1 / 2 / ALL); the softmax
is renormalised over the retained set, so retrieving fewer sources is a real
restriction rather than a rescaling.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from layers import RMSNorm


class DeltaBank:
    """Completed block deltas plus the current partial delta, per forward pass."""

    def __init__(self) -> None:
        self.completed: list[torch.Tensor] = []
        self.block_start: torch.Tensor | None = None

    def start_block(self, hidden: torch.Tensor) -> None:
        self.block_start = hidden

    def close_block(self, hidden: torch.Tensor) -> None:
        if self.block_start is None:
            raise RuntimeError("close_block() without start_block()")
        self.completed.append(hidden - self.block_start)
        self.block_start = hidden

    def sources(self, hidden: torch.Tensor) -> list[torch.Tensor]:
        """Completed deltas plus the delta accumulated so far in this block."""
        partial = hidden - self.block_start if self.block_start is not None else None
        out = list(self.completed)
        if partial is not None:
            out.append(partial)
        return out


class DeltaRouter(nn.Module):
    """One (layer, sublayer) routing head over the delta bank."""

    def __init__(self, hidden_size: int, key_rank: int, eps: float = 1e-6):
        super().__init__()
        self.key_rank = key_rank
        self.query = nn.Parameter(torch.zeros(key_rank))
        # a scalar gate through tanh: exactly zero at init, bounded afterwards
        self.gate = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.query, std=0.02)

    def forward(
        self,
        hidden: torch.Tensor,
        keys: list[torch.Tensor],
        values: list[torch.Tensor],
        tier: torch.Tensor | None = None,
        tiers: tuple[int, ...] = (0, 1, 2, -1),
    ) -> torch.Tensor:
        if not values:
            return hidden
        key = torch.stack(keys, dim=-2).float()            # [B, T, S, R]
        value = torch.stack(values, dim=-2)                # [B, T, S, H]
        scores = torch.einsum("btsr,r->bts", key, self.query.float())
        scores = scores / (self.key_rank ** 0.5)

        if tier is None:
            weights = scores.softmax(-1)
        else:
            weights = self._tiered_softmax(scores, tier.float(), tiers)

        pooled = torch.einsum("bts,btsh->bth", weights.to(value.dtype), value)
        return hidden + torch.tanh(self.gate).to(hidden.dtype) * pooled

    @staticmethod
    def _tiered_softmax(
        scores: torch.Tensor, tier: torch.Tensor, tiers: tuple[int, ...]
    ) -> torch.Tensor:
        """Softmax over the top-r sources only, renormalised within that set."""
        n = scores.shape[-1]
        order = scores.argsort(dim=-1, descending=True)
        rank = torch.empty_like(order)
        rank.scatter_(-1, order,
                      torch.arange(n, device=scores.device).expand_as(order).contiguous())

        limits = torch.tensor([n if r < 0 else min(r, n) for r in tiers],
                              device=scores.device, dtype=scores.dtype)
        keep_per_tier = (rank.unsqueeze(-1) < limits.view(1, 1, 1, -1)).to(scores.dtype)
        keep = (keep_per_tier * tier.unsqueeze(-2)).sum(-1)

        num = keep * torch.exp(scores - scores.amax(-1, keepdim=True))
        total = num.sum(-1, keepdim=True)
        # r = 0 keeps nothing; the router then contributes exactly zero
        return torch.where(total > 0, num / total.clamp_min(1e-20),
                           torch.zeros_like(num))


class DeltaKeyProjection(nn.Module):
    """Full-width delta value -> low-rank routing key, one per block slot."""

    def __init__(self, hidden_size: int, key_rank: int, num_slots: int, eps: float = 1e-6):
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps)
        self.proj = nn.ModuleList(
            [nn.Linear(hidden_size, key_rank, bias=False) for _ in range(num_slots)]
        )

    def forward(self, values: list[torch.Tensor]) -> list[torch.Tensor]:
        keys = []
        for index, value in enumerate(values):
            slot = self.proj[min(index, len(self.proj) - 1)]
            keys.append(slot(self.norm(value)))
        return keys


__all__ = ["DeltaBank", "DeltaRouter", "DeltaKeyProjection"]
