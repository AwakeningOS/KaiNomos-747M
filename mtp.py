"""MTP-1: predict the token after next from a shared body.

The main head answers `h_t -> y_{t+1}`.  The MTP head answers
`(h_t, E(y_{t+1})) -> y_{t+2}`: it is told the next token and asked for the one
after, so the body has to carry information that a pure next-token objective can
leave implicit.

The extra block is deliberately plain -- one fixed KDA block without Delta or
MUDD depth mixing -- so the auxiliary objective shapes the *body* instead of
receiving a second copy of its depth mechanisms.
The final norm and the LM head are shared with the body; the MTP module is used
during training only and can be dropped at inference.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import RMSNorm, SiTUMLP


class MTPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        from kda import KDAttention

        h = config.hidden_size
        eps = config.rms_norm_eps
        self.hidden_norm = RMSNorm(h, eps)
        self.token_norm = RMSNorm(h, eps)
        self.fuse = nn.Linear(2 * h, h, bias=False)

        self.attn_norm = RMSNorm(h, eps)
        self.attn = KDAttention(config)
        self.ffn_norm = RMSNorm(h, eps)
        self.ffn = SiTUMLP(h, config.mtp.ffn_width)

    def forward(self, hidden: torch.Tensor, next_embeddings: torch.Tensor,
                segments: torch.Tensor | None = None) -> torch.Tensor:
        z = self.fuse(torch.cat(
            [self.hidden_norm(hidden), self.token_norm(next_embeddings)], dim=-1
        ))
        # The auxiliary block is recurrent too, so it needs the same document
        # boundaries the body used, re-derived from the sliced segment ids.
        offsets = None
        if segments is not None:
            from segments import cu_seqlens
            starts = torch.zeros_like(segments, dtype=torch.bool)
            starts[:, 1:] = segments[:, 1:] != segments[:, :-1]
            starts[:, 0] = True
            offsets = cu_seqlens(starts)
        attn_out, _ = self.attn(self.attn_norm(z), segments=segments,
                                seq_offsets=offsets)
        z = z + attn_out
        return z + self.ffn(self.ffn_norm(z))


def mtp_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        targets.reshape(-1),
        ignore_index=ignore_index,
    )


def mtp_slices(input_ids: torch.Tensor) -> tuple[slice, slice, slice]:
    """Index ranges for the (hidden, next-token, target) triple.

    For a length-T sequence the last usable position is T-3: it has both a next
    token (T-2) to condition on and a target (T-1) to predict.  Off-by-one here
    silently trains the model on the token it was just handed, which looks like a
    working loss curve and teaches nothing.
    """
    return slice(None, -2), slice(1, -1), slice(2, None)


__all__ = ["MTPHead", "mtp_loss", "mtp_slices"]
