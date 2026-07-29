"""Numerically stable norms and the K3 SiTU-GLU feed-forward function."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.float() if dtype != torch.float64 else x
        y = x32 * torch.rsqrt(x32.square().mean(-1, keepdim=True) + self.eps)
        return y.to(dtype) * self.weight


def situ_gate(x: torch.Tensor, beta: float = 4.0) -> torch.Tensor:
    return beta * torch.tanh(x / beta) * torch.sigmoid(x)


def softcap_up(x: torch.Tensor, beta: float = 25.0) -> torch.Tensor:
    return beta * torch.tanh(x / beta)


class SiTUMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        channel_mask: torch.Tensor | None = None,
        uniform_width: int | None = None,
    ) -> torch.Tensor:
        """`uniform_width` slices instead of masking when every token agrees.

        Masking multiplies the inactive channels by zero and still sums them in
        `down_proj`, which is mathematically identical but changes the summation
        order.  On the supernet that made the forced-Fixed policy differ from the
        Standalone Base by ~2e-6 -- harmless in size, but it would have left the
        equivalence claim resting on a tolerance instead of on identity.  When
        the whole batch selects one prefix, slicing restores bit-exactness (and
        is the cheaper path).
        """
        if uniform_width is not None and uniform_width < self.gate_proj.out_features:
            w = uniform_width
            gate = situ_gate(F.linear(x, self.gate_proj.weight[:w]))
            up = softcap_up(F.linear(x, self.up_proj.weight[:w]))
            return F.linear(gate * up, self.down_proj.weight[:, :w])

        gate = situ_gate(self.gate_proj(x))
        up = softcap_up(self.up_proj(x))
        hidden = gate * up
        if channel_mask is not None:
            hidden = hidden * channel_mask.to(hidden.dtype)
        return self.down_proj(hidden)


__all__ = ["RMSNorm", "SiTUMLP", "situ_gate", "softcap_up"]
