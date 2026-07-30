# KaiNomos-747M

> **Reforming the laws of compute allocation.**
> 計算資源配分の法則を作り変える。

KaiNomos-747M is a Japanese-centred, decoder-only research language model
designed for full pre-training on a single 24 GB consumer GPU.

The production design is a **747,368,168-parameter dense model**, selected from
implementation audits and measurements on the target GPU.

> **Status: implementation and smoke tests are complete; full pre-training has
> not started. No trained weights are published yet.**

## Current design

| Field | Production value |
| --- | ---: |
| Total training parameters | 747,368,168 |
| Inference backbone, excluding MTP | 702,134,416 |
| MTP training head | 45,233,752 |
| Layers | 16 |
| Hidden size | 1,536 |
| Attention heads | 24 × 64 |
| Layer pattern | 12 KDA + 4 Gated MLA (`KKKM` × 4) |
| Dense FFN width | 6,144 |
| Vocabulary | 49,152, tied embedding / LM head |
| Training context | 1,024 tokens |
| Optimizer | Muon for matrix parameters, AdamW for the remaining parameters |
| Target hardware | NVIDIA RTX 3090 24 GB |

The machine-readable configuration is
[`kainomos-747m-architecture.json`](kainomos-747m-architecture.json).

## Architecture

**Dense execution.** Every layer and the full 6,144-wide FFN execute for every
token.

**KDA and Gated MLA.** The 16-layer body repeats three Kimi Delta Attention
layers followed by one NoPE Gated Multi-head Latent Attention layer. MLA uses
QK normalization over each complete 64-dimensional query/key head.

**MUDD-QKV.** Attention inputs are learned mixtures of visible depth states.
KDA has separate Q, K and V streams; MLA has Q and shared-KV streams.

**Delta Block Attention Residuals.** Each four-layer block exposes its change,
rather than another accumulated hidden state, as a retrieval source for later
sublayers. The production model uses block granularity; the per-sublayer
variant does not fit the selected micro-batch on the target GPU.

**MTP-1.** A training-only auxiliary head predicts one additional future token
with loss weight 0.30. It is excluded from the inference backbone.

**Document isolation.** Packed documents are separated with `<|eod|>` token ID
4. MLA attention, KDA recurrent state, short convolution, next-token loss and
MTP loss all respect those boundaries so adjacent packed documents cannot leak
context into one another.

## Training data and plan

Training uses **DoubleDragon-DataMix-v2**, with a dedicated 49,152-piece
SentencePiece Unigram tokenizer. The registered base mix contains at least
16 billion unique tokens; up to 1 billion additional post-deduplication tokens
may be retained when source quality and domain headroom permit it.

The data pipeline performs:

1. immutable validation/test split selection;
2. exact and cross-source MinHash deduplication;
3. benchmark-contamination filtering;
4. fixed-ratio source selection;
5. deterministic document-aware shuffling and token packing; and
6. full EOD and artifact-hash verification.

Full pre-training is planned as one dense Muon run. Checkpoints are resumable,
and weight-only observation snapshots are scheduled at logarithmic token
milestones to record how the model changes during training.

Raw training corpora and unpublished checkpoints are not stored in this source
repository.

## Validation

The source includes CPU invariance tests and GPU checks for:

- causal forward and backward behavior;
- KDA/MLA tensor shapes and finite gradients;
- document-boundary isolation;
- MUDD and Delta initialization behavior;
- Muon parameter grouping and checkpoint resume;
- dense execution accounting; and
- observation-snapshot generation.

Run the suite with an environment containing the dependencies in
`requirements.txt`:

```bash
PYTHONPATH=. python -m pytest tests/ -q
```

These checks establish implementation consistency, not model quality. No
validation NLL, benchmark score, safety evaluation or generation-quality claim
will be published until trained checkpoints exist.

## Attribution

This is an independent research implementation inspired by selected mechanisms
described in the Kimi K3, Kimi Linear, Muon and Delta Attention Residuals
reports. It is not affiliated with, endorsed by, or derived from Moonshot AI,
and it is not a conversion, distillation or reduced release of another model.

See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for mechanism,
dependency and dataset provenance.

## License

Original source code is released under the Apache License 2.0. Third-party
datasets, papers, kernels, trademarks and future model artifacts remain subject
to their own terms. The current repository is for research and reproducibility;
it does not contain a trained model suitable for production use.
