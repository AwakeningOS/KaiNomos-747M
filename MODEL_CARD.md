---
license: apache-2.0
library_name: pytorch
tags:
  - research
  - architecture-preview
  - consumer-gpu
  - japanese
  - kda
  - mla
  - muon
---

# KaiNomos-750M

**Architecture and training code release.** No trained checkpoint is published,
so the repository currently runs from random initialization. No model-quality
claim has been established.

KaiNomos-750M is a 24-layer, hidden-1,280 decoder with a dense 5,120-wide SiTU-GLU
FFN. It repeats three KDA layers and one strict-NoPE Gated MLA layer six times.
The 718,341,812-parameter deployment backbone uses tied 49,152-token embeddings
and 10 heads of dimension 128. Optional MTP raises the training total to
749,833,790 parameters.

MuDD has been removed. Additive block-level Delta routing that preserves the
main residual is the adopted architecture. The previously planned
ordinary-residual comparison was cancelled by user decision; no causal claim
about Delta's quality advantage is made. MTP remains off.

The implementation supports packed-document isolation, deterministic
source-balanced streaming, atomic exact-resume checkpoints, latent-only MLA
cache, EOD cache reset, Per-Head Muon and opt-in architecture observations.

The included RTX 3090 runtime measured 3,587.86 tok/s at 21.957 GiB peak
reserved VRAM while preserving the model, objective, and data order. See
`EFFICIENCY.md` for the selected settings and rejected alternatives.

## Intended use

- research on consumer-GPU language-model pre-training;
- Japanese-centred base-model studies;
- controlled KDA/MLA and depth-residual ablations;
- reproducible training, recovery and architecture monitoring.

The practical full-model path requires a CUDA GPU and FLA kernels. The reference
runtime was measured on a 24 GB RTX 3090.

## Current status and limitations

- CPU correctness and CUDA BF16/FLA parity gates passed;
- the adopted runtime measured 3,587.86 tok/s at 21.957 GiB peak reserved;
- the compatible 49,152-piece tokenizer is included; trained weights are not;
- the ordinary-residual A/B was cancelled and MTP remains disabled;
- no benchmark, safety, factuality or generation-quality results exist;
- the legacy packed manifest exposes `local`/`jpnmix`, not per-document recovery
  of every original domain.

Start with `examples/quickstart.py` for the full CUDA instantiation. The README
explains the architecture and setup.
