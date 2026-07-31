# KaiNomos-750M pre-experiment research boundary

Reviewed before authorizing a new architecture experiment on 2026-07-31.

## Primary sources checked

- Kimi K3 technical report, arXiv:2607.24653: KDA/MLA layer pattern,
  Per-Head Muon and large-scale training design.
- Delta Attention Residuals, arXiv:2605.18855: depth-wise routing over layer
  deltas and the reported block-granularity trade-off.
- Muon is Scalable for LLM Training, arXiv:2502.16982: RMS-matched Muon update
  scale and the shared learning-rate interpretation used here.
- QK-Normed Multi-head Latent Attention, arXiv:2606.16310: explicit versus
  absorbed QK normalization and latent-cache considerations.

Links:

- https://arxiv.org/abs/2607.24653
- https://arxiv.org/abs/2605.18855
- https://arxiv.org/abs/2502.16982
- https://arxiv.org/abs/2606.16310

## What the literature already answers

- KDA, MLA, Delta residual routing, Muon and MTP are existing mechanisms; their
  mere combination is not a novel scientific result.
- Block-level Delta routing is the appropriate memory-conscious candidate for
  this single-RTX-3090 implementation.
- Muon and AdamW must not retain the obsolete `0.02` versus `0.0003` learning
  rate split under the selected RMS-matched formulation.
- MTP loss improving does not establish that backbone next-token prediction
  improved.

## Remaining empirical question

The bounded architecture question is only:

> With the same 24-layer KDA/MLA backbone, initialization, tokenizer, data
> order, optimizer and token budget, does Delta Block improve held-out NTP NLL
> over ordinary residuals?

This is tested by Arm A (`depth_routing=none`) versus Arm B
(`depth_routing=delta_block`) at 67,108,864 tokens. Promotion to 255,983,616
tokens requires the frozen held-out comparison. MTP OFF versus ON is a separate
later experiment after architecture selection.

No 32.55B-token production run is authorized by implementation tests alone.
