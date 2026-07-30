"""MUDD-QKV: per-token, per-stream mixing over every past layer output.

Each layer builds its own Query, Key and Value inputs as a linear mix of all
depth states available to it -- the token embedding and every completed layer
output -- instead of reading only the layer below.  Coefficients come from a
small per-layer MLP and are used directly, not softmaxed, matching the reference
MUDDFormer implementation: a softmax would force the mixes onto a simplex and
forbid both negative coefficients and amplification.

Only the attention input streams are mixed here.  The residual direction is left
to the Delta Block, so exactly one mechanism writes to each path.

At initialisation the static bias selects the newest source and the dynamic MLP
outputs zero, so every stream returns `h_l` *exactly*.  Normalisation is not done
here: the layer owns a single shared `attn_norm`, which is applied to these
mixes by the caller. An earlier version normalised each
stream inside this module, which meant the "identity" initialisation actually
returned `RMSNorm(h)` rather than `h` -- a measured max deviation of 0.45 -- and
silently replaced the shared pre-attention norm with three fresh ones.

Stream sets differ by operator, because the operators differ in what they can
accept: KDA projects Q, K and V independently, while MLA compresses K and V into
one latent and therefore has exactly one KV input to mix.  Handing MLA a separate
V stream produced coefficients that received no gradient at all.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import RMSNorm


class MuDDQKV(nn.Module):
    """Dynamic dense mixing of past depth states into Q/K/V inputs."""

    def __init__(self, hidden_size: int, num_sources: int, mlp_hidden: int,
                 eps: float = 1e-6, streams: tuple[str, ...] = ("q", "k", "v")):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_sources = num_sources
        self.streams = streams

        self.norm = RMSNorm(hidden_size, eps)
        self.fc1 = nn.Linear(hidden_size, mlp_hidden, bias=False)
        self.fc2 = nn.Linear(mlp_hidden, len(streams) * num_sources, bias=False)

        # static component: pick the newest source; dynamic component: start at 0
        bias = torch.zeros(len(streams), num_sources)
        bias[:, -1] = 1.0
        self.static_bias = nn.Parameter(bias)
        self.reset_to_identity()

    def reset_to_identity(self) -> None:
        with torch.no_grad():
            nn.init.normal_(self.fc1.weight, std=0.02)
            self.fc2.weight.zero_()          # dynamic mixing starts silent
            self.static_bias.zero_()
            self.static_bias[:, -1] = 1.0

    def forward(self, sources: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        """`sources`: depth states usable by this layer, newest last.

        Returns one *unnormalised* mix per stream; the caller applies the layer's
        `attn_norm`.  The caller must never pass a state the layer cannot
        causally see.
        """
        if len(sources) != self.num_sources:
            raise ValueError(
                f"expected {self.num_sources} depth sources, got {len(sources)}"
            )
        stack = torch.stack(sources, dim=-2)          # [B, T, S, H]
        newest = sources[-1]

        coef = self.fc2(F.gelu(self.fc1(self.norm(newest))))
        coef = coef.view(*newest.shape[:-1], len(self.streams), self.num_sources)
        coef = coef + self.static_bias                # [B, T, streams, S]

        mixed = torch.einsum("btcs,btsh->btch", coef.to(stack.dtype), stack)
        return {name: mixed[..., i, :] for i, name in enumerate(self.streams)}

    def coefficient_summary(self, sources: list[torch.Tensor]) -> torch.Tensor:
        """Mean |coefficient| per (stream, source), for the depth-mixing log."""
        newest = sources[-1]
        coef = self.fc2(F.gelu(self.fc1(self.norm(newest))))
        coef = coef.view(*newest.shape[:-1], len(self.streams), self.num_sources)
        coef = coef + self.static_bias
        return coef.detach().abs().mean(dim=(0, 1))


__all__ = ["MuDDQKV"]
