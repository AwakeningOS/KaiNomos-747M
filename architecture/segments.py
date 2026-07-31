"""Document boundaries inside a packed training sequence.

The pool is a flat token stream: the DataMix-v2 packer concatenates documents and
terminates each one with `<|eod|>` (id 4), and the loader cuts it every 1024
tokens.  So a training sequence normally holds two or three unrelated documents
-- the median document is 455 tokens -- and without the masking here every
mechanism reads across them:

* MLA attends to the previous document. It can learn to down-weight it, but it
  spends capacity doing so.
* KDA is worse. Its recurrent state is *carried* into the next document, so the
  new document does not start from zero; it starts from whatever the previous one
  left behind. No attention weight can undo that.
* KDA's short depthwise convolution mixes the three preceding positions, so even
  the first tokens of a document see the tail of the one before.
* The next-token loss at an `<|eod|>` position asks the model to predict the
  first token of an unrelated document.

Boundaries are derived from the tokens themselves rather than from `train.idx`,
so the boundary information cannot drift out of step with the stream it
describes, and validation and test get the same treatment for free. This is
checked: over the first 40M tokens of the pool, id 4 occurs 29,329 times and
every one of them is a document end, with no occurrences anywhere else.

Not done here: documents are still *truncated* at sequence boundaries. Best-fit
packing (Fewer Truncations Improve Language Modeling, arXiv 2404.10830) would
reduce that, but it needs the pool rebuilt and is a separate change.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def document_starts(input_ids: torch.Tensor, eod_id: int) -> torch.Tensor:
    """`[B, T]` bool, true where a new document begins.

    A document begins at the position *after* an `<|eod|>`: the separator belongs
    to the document it terminates.
    """
    starts = torch.zeros_like(input_ids, dtype=torch.bool)
    starts[:, 1:] = input_ids[:, :-1] == eod_id
    starts[:, 0] = True                 # every row begins a fresh document
    return starts


def segment_ids(input_ids: torch.Tensor, eod_id: int) -> torch.Tensor:
    """`[B, T]` index of the document each position belongs to, per row."""
    return document_starts(input_ids, eod_id).cumsum(1)


def cu_seqlens(starts: torch.Tensor) -> torch.Tensor:
    """Segment offsets for FLA's variable-length kernels.

    Those kernels require the batch flattened to `[1, B*T, ...]`, so the offsets
    are into the flattened axis and the row boundaries are segment boundaries too
    -- which `document_starts` already guarantees by marking column 0.
    """
    rows, length = starts.shape
    offsets = starts.reshape(-1).nonzero().flatten()
    total = offsets.new_tensor([rows * length])
    return torch.cat([offsets, total]).to(torch.int32)


def document_mask(segments: torch.Tensor) -> torch.Tensor:
    """`[B, 1, T, T]` bool: attend only within the same document, and causally."""
    same = segments[:, :, None] == segments[:, None, :]
    length = segments.shape[1]
    causal = torch.ones(length, length, dtype=torch.bool,
                        device=segments.device).tril()
    return (same & causal).unsqueeze(1)


def masked_lagged_sum(
    x: torch.Tensor, weight: torch.Tensor, segments: torch.Tensor
) -> torch.Tensor:
    """Causal depthwise convolution that does not read across a boundary.

    `weight` is `[C, 1, K]` as `nn.Conv1d` stores it.  Each lag is applied
    separately and zeroed where it would reach into another document, which is
    exactly what running the convolution per document with zero left padding
    gives -- without having to actually split the batch.
    """
    _, _, kernel = weight.shape
    out = x * weight[:, 0, kernel - 1]
    for lag in range(1, kernel):
        shifted = F.pad(x[:, :-lag], (0, 0, lag, 0))
        previous = F.pad(segments[:, :-lag], (lag, 0), value=-1)
        same = (previous == segments).unsqueeze(-1).to(x.dtype)
        out = out + shifted * same * weight[:, 0, kernel - 1 - lag]
    return out


def mask_targets_at_boundaries(
    targets: torch.Tensor, inputs: torch.Tensor, eod_id: int,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Drop the one prediction that crosses a boundary.

    With the standard shift, position `i` predicts token `i+1`.  The only
    crossing pair is the one whose *input* is `<|eod|>`, so that is the position
    to silence -- not the first position of the new document, which is a perfectly
    ordinary prediction made from a document-initial state.
    """
    out = targets.clone()
    out[inputs == eod_id] = ignore_index
    return out


__all__ = [
    "cu_seqlens",
    "document_mask",
    "document_starts",
    "mask_targets_at_boundaries",
    "masked_lagged_sum",
    "segment_ids",
]
