---
license: apache-2.0
library_name: pytorch
tags:
  - research
  - architecture-preview
  - consumer-gpu
  - language-model
  - kda
  - mla
---

# KaiNomos-110M

> **Architecture preview — not a pretrained model yet.**
>
> No trained weights or tokenizer are currently published in this repository.
> This page documents a tested 111M-parameter implementation and a
> pre-registered Base-versus-Adaptive experiment. It is not ready for inference.

KaiNomos-110M asks whether selected architecture ideas from frontier-scale
language models remain useful when reduced to a model that an individual can
train on a single consumer GPU.

- **Source and tests:** [AwakeningOS/KaiNomos-110M on GitHub](https://github.com/AwakeningOS/KaiNomos-110M)
- **Reference architecture:** [Kimi K3 technical report](https://arxiv.org/abs/2607.24653)
- **KDA reference:** [Kimi Linear](https://arxiv.org/abs/2510.26692)
- **Status:** implementation tested; full pre-training pending

This is an independent, text-only, dense research implementation. It is not an
official Moonshot AI project, a conversion of Kimi K3 weights, or a reproduction
of Kimi K3's capabilities.

## Model details

| Field | Value |
| --- | --- |
| Architecture | Decoder-only causal language model |
| Total training parameters | 111,042,670 |
| Inference backbone, excluding MTP | 106,409,510 |
| Layers | 16 |
| Layer pattern | 12 KDA + 4 Gated MLA, repeating 3:1 |
| Hidden size | 512 |
| Attention heads | 8 |
| Vocabulary size | 32,768 (KaiNomos-110M-specific SentencePiece Unigram) |
| Embedding / LM head | Weight tied |
| Training context | 1,024 tokens |
| FFN | SiTU-GLU with nested routed widths |
| Modalities | Text only |
| Framework | Custom PyTorch research implementation |
| License for original code | Apache-2.0 |

The public architecture configuration is available in
[`kainomos-110m-architecture.json`](kainomos-110m-architecture.json).

## Architecture

The K3-inspired core uses:

- Kimi Delta Attention with smooth lower-bounded decay;
- NoPE Gated Multi-head Latent Attention;
- a repeating three-KDA/one-MLA layer pattern; and
- SiTU-GLU feed-forward layers.

The experiment adds:

- **MUDD-QKV:** separate dynamic depth mixtures for Q, K, and V inputs;
- **Projected Low-Rank Delta Blocks:** route accumulated block deltas into
  attention and FFN inputs;
- **JointRoute:** jointly select KDA update mode, MLA read mode, nested FFN
  width, and delta-source tier under one compute price;
- **Compute reinvestment:** allow saved compute to fund FFN widths above the
  fixed Base width; and
- **MTP-1:** an auxiliary one-future-token training objective.

These mechanisms are hypotheses. They are not presented as improvements until
the controlled experiment is complete.

## Registered comparison

Both arms use the same supernet and initialization.

| Arm | Policy |
| --- | --- |
| **Base** | Fixed full KDA updates, global MLA reads, FFN width 1,792, and all available delta sources |
| **Adaptive** | Learned input-dependent capacity routing under a matched-compute constraint |

The comparison is intended to hold the following constant:

- initialization;
- tokenizer and dataset manifest;
- training tokens;
- optimizer and schedule;
- seed;
- supernet parameters; and
- cumulative executed analytical compute.

Final claims require held-out evaluation and a compute-budget error within the
registered tolerance.

## Current validation

The source repository currently records 28 passing tests covering:

- causal behavior;
- finite forward, loss, gradient, and backward paths;
- parameter-budget and tensor-shape checks;
- execution of all 16 layers;
- nested FFN tiers;
- compute accounting;
- identity initialization for added mechanisms;
- checkpoint migration; and
- fixed-policy and checkpoint-reload invariants.

A short development smoke run recorded:

| Measurement | Observed |
| --- | ---: |
| Sequence length | 1,024 |
| Micro-batch | 2 |
| Gradient accumulation | 32 |
| Peak allocated VRAM | 7.72 GB |
| Throughput | 3,571 tokens/s |
| Optimizer-step time | 18.353 s |
| Checkpoint reload | passed |

The hardware identifier was not stored in the smoke-test JSON. These values are
implementation diagnostics, not a quality benchmark.

## Training data

Training uses `KaiNomos-DataMix-v1`, a fixed-ratio
Japanese/English/code/math pool containing exactly **1,988,270,624 tokens** in
the KaiNomos tokenizer. Its train, validation, and test counts, source
revisions, tokenizer hash, split seed, document counts, filters, and artifact
hashes are recorded in `data/pool/manifest.json`.

Raw training data will not be copied into this model repository.

## Evaluation

No validation NLL, held-out test NLL, or downstream benchmark result is
published yet. Blank results are intentional: random-initialization loss and a
smoke-test throughput number are not evidence of model quality.

The eventual release will include:

- Base and Adaptive validation/test NLL;
- matched-compute audit;
- throughput and peak VRAM;
- per-seed results or an explicit single-seed limitation;
- route-dependence controls;
- complete training logs;
- checkpoint and artifact SHA-256 hashes; and
- adopted, rejected, or inconclusive verdict.

## Intended use

After trained weights are released, the intended uses will be:

- research on small language-model architecture;
- reproducible consumer-GPU pre-training experiments;
- ablation and scaling studies; and
- education about controlled model-development workflows.

The current architecture preview is intended for code review and experiment
design only.

## Limitations and out-of-scope uses

- There are currently no pretrained weights, so the repository cannot generate
  meaningful text.
- Results at 110M parameters must not be assumed to transfer to larger models.
- The model is text-only and does not reproduce Kimi K3's MoE, multimodal,
  one-million-token, or post-training systems.
- No safety, bias, factuality, multilingual, or downstream capability
  evaluation has been completed.
- This is not intended for production, high-stakes decisions, or deployment as
  a general assistant.

Additional limitations will be documented from observed evidence after
training, including failed experiments.

## AI-assisted development

Frontier AI coding assistants, including GPT-5.6, were used for literature
navigation, architecture discussion, implementation, test construction, bug
finding, and experimental-design review.

AI assistance is part of the research process, not evidence that the resulting
architecture works. Only controlled and reproducible measurements can support
that conclusion.

## Release checklist

- [x] 110M-class architecture implementation
- [x] CPU correctness and invariance tests
- [x] single-GPU forward/backward/checkpoint smoke test
- [x] public architecture configuration
- [x] tokenizer and tokenization pipeline
- [ ] matched-compute Base pre-training
- [ ] matched-compute Adaptive pre-training
- [ ] held-out evaluation and compute audit
- [ ] safetensors checkpoints and tokenizer
- [ ] training logs, manifests, hashes, and final model card

## Attribution

“Kimi” and “Kimi K3” refer to Moonshot AI's work. KaiNomos-110M is independent
and is not sponsored, endorsed, or maintained by Moonshot AI. Third-party
papers, datasets, kernels, weights, and trademarks remain subject to their own
terms.
