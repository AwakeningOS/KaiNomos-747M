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
    ) -> torch.Tensor:
        gate = situ_gate(self.gate_proj(x))
        up = softcap_up(self.up_proj(x))
        return self.down_proj(gate * up)


__all__ = ["RMSNorm", "SiTUMLP", "situ_gate", "softcap_up"]
