# KaiNomos-750M

[日本語](README.md) | **English**

KaiNomos-750M is a Japanese-centered, decoder-only language model designed for
pre-training on a single 24 GB consumer GPU. Rather than simply making a
Transformer smaller, the architecture focuses on how to combine sequential
memory with periodic global attention under a limited compute budget.

This repository currently contains the untrained architecture and its training
infrastructure. No trained weights or performance benchmarks have been
published yet.

## How the model reads a sequence

The backbone has 24 layers. Six four-layer stages repeat the same pattern:

```text
(KDA → KDA → KDA → MLA) × 6
```

The first three layers in each stage use Kimi Delta Attention (KDA). Instead of
keeping every past key and value in a growing KV cache, KDA writes information
into a recurrent state as the sequence progresses. This lets it track the flow
of the text without making cache size grow in proportion to context length.

The fourth layer uses Multi-head Latent Attention (MLA) to revisit information
across the stage with global attention. Keys and values are represented through
a 256-dimensional latent state rather than stored in full, reducing cache size
relative to conventional attention. This implementation is strictly NoPE and
normalizes queries and keys per head.

In practical terms, KDA carries information forward step by step, while MLA
periodically lets the model reorganize information across the available
context. The final layer is also MLA, so every output passes through global
attention before prediction.

## Using information from earlier depths

A conventional Transformer repeatedly adds each layer's output to the residual
stream. From a later layer's point of view, this makes it difficult to separate
what an earlier layer changed from everything that had already accumulated.

KaiNomos-750M records the change produced by each four-layer stage as
`stage_output - stage_input`. Later layers can use a Delta Block to retrieve the
embedding and selected stage-level changes. The main residual stream remains
intact; routed information is added as a separate contribution.

The first attention layer has no earlier change to retrieve, so its depth route
is an exact identity. MuDD, which previously overlapped with this role, has been
removed. Delta Block is now the model's only depth-routing mechanism.

## FFN and training-only components

Every layer contains a dense SiTU-GLU feed-forward network. Its gate uses
`SiLU(x) × tanh(softplus(x))`. There is no sparse expert routing: every token
passes through the same FFN.

Multi-Token Prediction (MTP) is implemented but disabled by default. It will be
used in full training only if a controlled comparison shows that it improves
the backbone's held-out next-token NLL. A lower MTP loss on its own is not
enough to justify enabling it.

## Model size

| Component | Value |
| --- | ---: |
| Layers | 24 |
| Hidden size | 1,280 |
| Dense FFN width | 5,120 |
| Attention heads | 10 |
| Head dimension | 128 |
| Vocabulary | 49,152 |
| Training context | 1,024 tokens |
| Deployment backbone | 718,341,812 parameters |
| Training-only MTP | 31,491,978 parameters |
| Training model with MTP | 749,833,790 parameters |

The token embedding and language-model head share weights. The backbone
contains 18 KDA layers and 6 MLA layers.

## Optimizer

Muon is used for large matrix parameters. The Q, K, and V projections in KDA
and MLA are updated independently per head. Embeddings, normalization
parameters, biases, decay parameters, and other ineligible tensors are handled
by AdamW.

After RMS alignment, both Muon and AdamW use the same learning rate of
`0.0003`. Parameter ownership is explicit: training stops if a newly introduced
parameter has not been assigned to exactly one optimizer group.

## Document boundaries and data delivery

Multiple documents can be packed into one sequence without allowing one
document to leak into the next. The EOD token (ID 4) simultaneously separates:

- MLA attention;
- the KDA recurrent state;
- the short KDA convolution state;
- next-token loss; and
- MTP loss.

Each data source has its own cursor. Fixed-token chunks are mixed with a seeded,
deterministic schedule. Checkpoints preserve every source cursor, unconsumed
read-ahead data, optimizer state, and the Python, NumPy, PyTorch, and CUDA random
states. A resumed run therefore continues from the same token sequence rather
than merely restarting from similar model weights.

## Validation strategy

Delta Block is evaluated against an ordinary-residual baseline under matched
conditions: identical initialization, tokenizer, data order, optimizer, and
token budget.

- Arm A: ordinary residuals only
- Arm B: KaiNomos-750M with Delta Block

The decision is based on next-token NLL over a fixed held-out split, not on
training loss. MTP OFF versus ON is tested separately after the backbone
architecture has been selected.

## Repository layout

```text
architecture/
├── model.py          # 24-layer KDA/MLA backbone
├── kda.py            # recurrent KDA and FLA path
├── mla.py            # NoPE MLA and latent cache
├── delta_block.py    # depth-wise Delta routing
├── muon.py           # Per-Head Muon and AdamW ownership
├── interleave.py     # source mixing and exact resume
├── train.py          # training, checkpoints, and validation
├── observe.py        # architecture and optimizer observations
└── tests/            # CPU correctness tests
```

## Tests

```bash
cd architecture
PYTHONPATH=. python -m pytest --rootdir=. --confcutdir=. -q tests
python -m ruff check .
```

The CPU suite checks parameter counts, causality, document isolation, cached
generation equivalence, Delta sources, optimizer ownership, deterministic data
resume, and checkpoint consistency. CUDA BF16 and FLA kernel acceptance are
recorded separately from CPU correctness and model-quality evaluation.

## Status

- Architecture implementation: complete
- CPU correctness tests: passed
- Deterministic interleave and exact resume: passed
- CUDA BF16 and FLA acceptance: pending
- Trained weights: not published
- Downstream benchmarks: not run

Passing implementation tests does not establish language-model quality.

## License and attribution

Original project code is licensed under the Apache License 2.0. KDA, MLA,
Delta Block, Muon, and MTP are mechanisms described in published research. This
project does not claim that combining them is itself a novel research result.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for details.
