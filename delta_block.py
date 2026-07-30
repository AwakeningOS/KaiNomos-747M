"""Delta Block Attention Residuals.

Delta Attention Residuals, Cheng Luo, Zefan Cai, Junjie Hu, arXiv:2605.18855.
This is the paper's *Delta Block* variant, adopted as published rather than
invented here.

What is routed is the *change* each block made, never the accumulated hidden
state:

    delta_b = h_{(b+1)B} - h_{bB}

and the routed result is *added* to the residual stream, never substituted for
it.  Routing cumulative state is what collapses in deeper layers -- every source
looks alike once it is dominated by the same accumulated signal -- and replacing
the stream would make the mechanism a substitute for the residual path rather
than a contribution to it.

    sources = [embedding, delta_0, ..., delta_{b-1}, partial_delta]
    K       = norm(V)
    logits  = w_l^T RMSNorm(source_i)
    alpha   = softmax(logits, over sources)
    routed  = h + sum_i alpha_i * source_i

The embedding is a permanent first source; the partial delta is the change the
current, still-open block has made so far and is zero immediately after a
boundary.

Following the paper there is no gate, no temperature and no entropy term.  The
query is zero-initialised, which means the mechanism does *not* start as the
identity: at zero logits the softmax is uniform, so the routed value is the mean
of the sources rather than nothing.  That is deliberate here -- this model is
pre-trained from scratch, so there is no trained checkpoint whose behaviour an
identity start would have to preserve.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from layers import RMSNorm


def _promote(tensor: torch.Tensor) -> torch.Tensor:
    """To float32 from anything narrower; float64 is left alone."""
    return tensor if tensor.dtype in (torch.float32, torch.float64) else tensor.float()


class DeltaBank:
    """The embedding, the completed block deltas, and the open block's delta."""

    def __init__(self, granularity: str = "block") -> None:
        if granularity not in ("block", "sublayer"):
            raise ValueError(f"unknown delta granularity {granularity!r}")
        self.granularity = granularity
        self.embedding: torch.Tensor | None = None
        self.completed: list[torch.Tensor] = []
        self.block_start: torch.Tensor | None = None
        # "sublayer" granularity: every sublayer output is its own source, so the
        # set grows to 2L rather than L/B.  Kept separate from `completed` because
        # the two are different quantities and a mode that silently reused the
        # other's list would look like it worked.
        self.sublayer: list[torch.Tensor] = []

    def start(self, embedding: torch.Tensor) -> None:
        """Record the embedding as the permanent first source."""
        self.embedding = embedding
        self.block_start = embedding

    def close_block(self, hidden: torch.Tensor) -> None:
        if self.block_start is None:
            raise RuntimeError("close_block() without start()")
        self.completed.append(hidden - self.block_start)
        self.block_start = hidden

    def record_sublayer(self, delta: torch.Tensor) -> None:
        """A sublayer's own output, i.e. its contribution to the stream."""
        if self.granularity == "sublayer":
            self.sublayer.append(delta)

    def sources(self, hidden: torch.Tensor) -> list[torch.Tensor]:
        if self.embedding is None or self.block_start is None:
            raise RuntimeError("sources() before start()")
        if self.granularity == "sublayer":
            return [self.embedding, *self.sublayer]
        return [self.embedding, *self.completed, hidden - self.block_start]


class DeltaRouter(nn.Module):
    """One depth router, owned by a single (layer, sublayer) position.

    The norm and the query are per sublayer and are never shared between the
    attention and feed-forward positions of a layer: they are asking different
    questions of the same sources.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps)
        # `w_l`, zero-initialised as published.  No bias: the paper's query is a
        # vector, not an affine map.
        self.query = nn.Parameter(torch.zeros(hidden_size))

    def forward(
        self,
        hidden: torch.Tensor,
        sources: list[torch.Tensor],
    ) -> torch.Tensor:
        if not sources:
            return hidden
        # Deliberately not `torch.stack(sources)`.  Stacking materialises a second
        # full copy of every source as one [B, T, S, H] tensor, and autograd holds
        # it for the backward pass; at 32 router positions that measured +0.9 GB on
        # the production config, enough to push it past its VRAM budget.  The
        # sources already exist as tensors in the graph, so scoring and pooling them
        # one at a time is the same arithmetic without the copy.
        # Logits are scored at float32 or better: under bf16 autocast a softmax
        # over raw bf16 dot products loses resolution between close sources.
        # Promoting rather than casting keeps a float64 run exact, which is what
        # makes the double-precision tests worth running.
        scores = torch.stack(
            [
                torch.einsum("bth,h->bt", _promote(self.norm(source)),
                             _promote(self.query))
                for source in sources
            ],
            dim=-1,
        )                                                    # [B, T, S]
        weights = scores.softmax(-1)
        pooled = hidden
        for index, source in enumerate(sources):
            pooled = pooled + weights[..., index, None].to(source.dtype) * source
        return pooled


__all__ = ["DeltaBank", "DeltaRouter"]
