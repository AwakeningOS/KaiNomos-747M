---
license: apache-2.0
library_name: pytorch
tags:
  - research
  - architecture-preview
  - consumer-gpu
  - language-model
  - japanese
  - kda
  - mla
  - muon
---

# KaiNomos-747M

> **Untrained architecture preview.**
>
> No trained weights or tokenizer are currently published in this model
> repository. It cannot yet generate meaningful text.

KaiNomos-747M is a Japanese-centred, decoder-only research language model
designed for full pre-training on a single NVIDIA RTX 3090 24 GB GPU.

- **Source:** [AwakeningOS/KaiNomos-747M on GitHub](https://github.com/AwakeningOS/KaiNomos-747M)
- **Status:** implementation tested; data preparation and hardware validation in progress

## Model details

| Field | Value |
| --- | --- |
| Architecture | Decoder-only causal language model |
| Total training parameters | 747,368,168 |
| Inference backbone, excluding MTP | 702,134,416 |
| Training-only MTP head | 45,233,752 |
| Layers | 16 |
| Layer pattern | 12 KDA + 4 Gated MLA, repeating 3:1 |
| Hidden size | 1,536 |
| Attention heads | 24 |
| Head dimension | 64 |
| Dense FFN width | 6,144 |
| Vocabulary size | 49,152 |
| Embedding / LM head | Weight tied |
| Training context | 1,024 tokens |
| Document boundary token | `<|eod|>` ID 4 |
| Modalities | Text only |
| Framework | Custom PyTorch research implementation |
| Optimizer plan | Muon + AdamW parameter groups |
| Original-code license | Apache-2.0 |

The public machine-readable configuration is
[`kainomos-747m-architecture.json`](kainomos-747m-architecture.json).

## Architecture summary

- Kimi Delta Attention and NoPE Gated MLA in a `KKKM` × 4 pattern;
- full-head QK normalization in MLA;
- MUDD depth mixtures for Q/K/V or Q/KV attention inputs;
- block-granularity Delta Attention Residual retrieval;
- a dense 6,144-wide SiTU-GLU FFN in every layer;
- a training-only MTP-1 auxiliary head; and
- strict packed-document isolation across attention, recurrent state,
  convolution and loss.

The production model trains one dense configuration.

## Training data

The planned corpus is **DoubleDragon-DataMix-v2**, using a dedicated
49,152-piece SentencePiece Unigram tokenizer. Its registered fixed base mix
contains at least 16 billion unique tokens, with up to 1 billion additional
post-deduplication tokens retained only where quality and domain availability
permit.

The final release will record exact source revisions, licenses, split seeds,
filters, deduplication and contamination reports, tokenizer hashes, packed-shard
hashes and the final executed-token count.

Raw training data will not be copied into this model repository.

## Current validation

The implementation has tests for causal behavior, finite gradients, parameter
shapes, document isolation, initialization invariants, dense execution, Muon
parameter grouping, checkpoint resume and observation snapshots.

These are implementation checks, not evidence of language-model quality.
Full pre-training has not started, so there are currently no valid validation
NLL, held-out benchmark, safety, bias, factuality or downstream capability
results.

## Intended use

After trained weights are released, intended uses include:

- research on consumer-GPU language-model pre-training;
- Japanese-centred base-model and tokenizer studies;
- KDA/MLA, depth-residual and optimizer ablations;
- reproducible training and checkpoint-resume experiments; and
- education about model-development and data-curation workflows.

The current architecture preview is intended for code review and experiment
design only.

## Limitations

- No trained weights are available.
- The model is text-only.
- Training and evaluation are incomplete.
- No production, high-stakes or general-assistant use is supported.
- Single-GPU design constraints may limit throughput and experimental breadth.

Observed limitations and failed experiments will be added after training rather
than inferred from architecture alone.

## Release checklist

- [x] 747M dense architecture implementation
- [x] CPU correctness and invariance tests
- [x] single-GPU forward/backward/checkpoint smoke path
- [x] document-boundary isolation
- [x] Muon training and resume path
- [x] public architecture configuration
- [x] 49,152-piece tokenizer selected and validated
- [ ] final deduplicated and decontaminated token pool
- [ ] full pre-training
- [ ] held-out evaluation
- [ ] safetensors checkpoint and tokenizer release
- [ ] training logs, manifests and artifact hashes

## Attribution

KaiNomos-747M is an independent research project inspired by selected published
mechanisms. It is not sponsored, endorsed or maintained by Moonshot AI.
Third-party papers, datasets, kernels, weights and trademarks remain subject to
their own terms.
