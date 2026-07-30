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
        tier_index: torch.Tensor | None = None,
        tier_widths: tuple[int, ...] | None = None,
        tier_gain: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run each token at the width it selected, not at the widest width.

        Three paths, in order of preference:

        `uniform_width` -- the whole batch agreed, so slice. Masking multiplies
        the inactive channels by zero and still sums them in `down_proj`, which is
        mathematically identical but changes the summation order; on the supernet
        that made the forced-Fixed policy differ from the Standalone Base by ~2e-6.
        Slicing restores bit-exactness and is the cheapest path. `<=` rather than
        `<` so that agreeing on the widest tier also avoids an all-ones mask.

        `tier_index` -- widths are mixed, so group the tokens by tier and run one
        matmul per group. This is the path that makes routing mean anything: the
        masked path below computes **every** token at the full 2816 width and then
        zeroes the channels it "pruned", so a token routed to 1024 cost exactly as
        much as one routed to 2816. The analytical cost model charged the narrow
        width while the GPU did the wide work, which is why the routed arm
        measured *slower* than the fixed arm (12,139 vs 14,242 tok/s) while
        claiming to spend less.

        `channel_mask` -- the original masked path, kept for callers that have no
        tier assignment (the unit tests exercise the FFN directly).
        """
        if uniform_width is not None and uniform_width <= self.gate_proj.out_features:
            w = uniform_width
            gate = situ_gate(F.linear(x, self.gate_proj.weight[:w]))
            up = softcap_up(F.linear(x, self.up_proj.weight[:w]))
            out = F.linear(gate * up, self.down_proj.weight[:, :w])
            return self._apply_gain(out, tier_gain)

        if tier_index is not None and tier_widths is not None:
            return self._grouped(x, tier_index, tier_widths, tier_gain)

        gate = situ_gate(self.gate_proj(x))
        up = softcap_up(self.up_proj(x))
        hidden = gate * up
        if channel_mask is not None:
            hidden = hidden * channel_mask.to(hidden.dtype)
        return self.down_proj(hidden)

    @staticmethod
    def _apply_gain(out: torch.Tensor, gain: torch.Tensor | None) -> torch.Tensor:
        """Multiply by the selected tier's straight-through weight.

        Dispatching on a hard index would cut the controller out of the graph
        entirely -- the masked path used to carry its gradient through the mask
        itself.  The selected entry of the one-hot is exactly 1.0 in the forward
        pass, so this is numerically a no-op, while in the backward pass it hands
        the router the gradient of the FFN output with respect to its own choice.
        """
        if gain is None:
            return out
        return out * gain.unsqueeze(-1).to(out.dtype)

    def _grouped(
        self,
        x: torch.Tensor,
        tier_index: torch.Tensor,
        tier_widths: tuple[int, ...],
        tier_gain: torch.Tensor | None,
    ) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        index = tier_index.reshape(-1)
        parts: list[torch.Tensor] = []
        positions: list[torch.Tensor] = []
        for tier, width in enumerate(tier_widths):
            selected = (index == tier).nonzero(as_tuple=True)[0]
            if selected.numel() == 0:
                continue
            rows = flat.index_select(0, selected)
            gate = situ_gate(F.linear(rows, self.gate_proj.weight[:width]))
            up = softcap_up(F.linear(rows, self.up_proj.weight[:width]))
            parts.append(F.linear(gate * up, self.down_proj.weight[:, :width]))
            positions.append(selected)

        joined = torch.cat(parts, dim=0)
        order = torch.cat(positions, dim=0)
        # `order` is a permutation of every row, so inverting it restores the
        # original token order in one gather instead of a scatter per tier.
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(order.numel(), device=order.device)
        out = joined.index_select(0, inverse).reshape(shape)
        return self._apply_gain(out, tier_gain)


__all__ = ["RMSNorm", "SiTUMLP", "situ_gate", "softcap_up"]
