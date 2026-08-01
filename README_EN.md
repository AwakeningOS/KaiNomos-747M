# KaiNomos-750M

[日本語](README.md) | **English**

KaiNomos-750M is a Japanese-centered, decoder-only language model designed for
pre-training on a single 24 GB consumer GPU. Rather than simply making a
Transformer smaller, the architecture focuses on how to combine sequential
memory with periodic global attention under a limited compute budget.

This repository contains the model implementation, exact-resume training
infrastructure, and an RTX 3090 runtime. It also includes the matching
49,152-piece tokenizer, data packer, training/resume entry point, and checkpoint
chat CLI. Trained weights are not published yet.

> **A CUDA GPU is required.** The public quick start runs the full 718M-parameter
> backbone on GPU. The measured reference system uses a 24 GB RTX 3090.

## Quick start

```bash
git clone https://github.com/AwakeningOS/KaiNomos-750M.git
cd KaiNomos-750M
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-cuda.txt
python examples/quickstart.py
```

The final command runs the full backbone on CUDA and prints the logits shape,
parameter count, and number of Delta-routing records. It verifies that the
implementation imports and runs on GPU; it is not text generation from a
trained model. The measured environment is recorded in
[ENVIRONMENT_LOCK.md](ENVIRONMENT_LOCK.md).

## Train on your data and chat

The matching tokenizer is included. Supply train/validation data and run:

```bash
python tools/prepare_data.py --input corpus/train.jsonl --output data/myrun --split train --source-id mydata
python tools/prepare_data.py --input corpus/validation.jsonl --output data/myrun --split validation --source-id mydata
python scripts/run_kainomos_runtime_tuned.py --runtime-activation-checkpointing on --runtime-micro-batch 16 --runtime-checkpoint-every-steps 200 --architecture kainomos_750m_v1 --data-dir data/myrun --run-dir runs/myrun --device cuda --allow-gpu --optimizer muon --depth-routing delta_block --mtp off --target-tokens 65536
```

Run the same training command again to resume from `latest.json`. Then:

```bash
python examples/chat.py --checkpoint runs/myrun/step_00000001.pt
```

See [TRAINING.md](TRAINING.md) for input formats and source mixing.
Generation prefills each EOD-delimited prompt segment in one call, then passes
only the new token while reusing KDA state and the MLA latent cache. Measurements
and the rejected absorbed-MLA decode candidate are in
[INFERENCE.md](INFERENCE.md).

## How the model reads a sequence

The backbone has 24 layers. Six four-layer stages repeat the same pattern:

```text
(KDA → KDA → KDA → MLA) × 6
```

Conceptually, each stage follows this data flow:

```text
h = token_embedding
completed_deltas = []

for stage in 6 stages:
    stage_input = h
    for attention in [KDA, KDA, KDA, MLA]:
        context = DeltaRoute(h, token_embedding, completed_deltas)
        h = h + attention(RMSNorm(context))

        context = DeltaRoute(h, token_embedding, completed_deltas)
        h = h + SiTU_GLU(RMSNorm(context))

    completed_deltas.append(h - stage_input)
```

DeltaRoute constructs read-only context; it does not replace the main residual
stream `h`.

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

## RTX 3090 runtime efficiency

The runtime was optimized under a hard `22.0 GiB` peak-reserved VRAM limit while
keeping the model, optimizer, data order, and 65,536 tokens per optimizer step
unchanged.

| Configuration | Steady tok/s | Peak reserved |
| --- | ---: | ---: |
| mb8 stage-2 bracket baseline | 2,773.20 | 19.840 GiB |
| mb8 all-FLA RMSNorm plus Delta score | 3,518.97 | 17.285 GiB |
| selected mb16, ten-step confirmation | **3,587.86** | **21.957 GiB** |

The selected configuration uses activation checkpointing, micro-batch 16 with
accumulation 4, 32-token chunked CE, BF16 variable-length Flash MLA, FLA BF16
RMSNorm for every norm, fused Delta scoring, and
`expandable_segments:True`. It is 29.38% faster than the bracketed baseline and
reduces time and GPU energy per token by 22.71% at equal average power.

See [EFFICIENCY.md](EFFICIENCY.md) for the comparison contract, successful and
rejected candidates, the VRAM decision rule, and the lower-memory fallback.
These are runtime results, not evidence of improved language-model quality.

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

## Default training configuration

The public configuration enables Delta Block and disables MTP. The compatible
tokenizer is included as `tokenizer/kainomos-49152.model`. Trained weights are
not published.

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
scripts/
├── run_kainomos_runtime_tuned.py       # exact resume with the selected runtime
├── kainomos_optimization_runtime.py    # state-dict-preserving runtime patches
├── benchmark_kainomos_runtime_candidate.py
├── benchmark_kainomos_generation.py    # prefill/decode A/B
└── validate_optimization_cuda.py       # CUDA parity gates
examples/
├── quickstart.py       # minimal full CUDA forward pass
└── chat.py             # checkpoint chat CLI
tokenizer/
└── kainomos-49152.model # included SentencePiece tokenizer
tools/
└── prepare_data.py     # JSONL/text to training shards
```

## Tests

```bash
cd architecture
PYTHONPATH=. python -m pytest --rootdir=. --confcutdir=. -q tests
python -m ruff check .
```

The CPU suite checks parameter counts, causality, document isolation, cached
generation equivalence, Delta sources, optimizer ownership, deterministic data
resume, and checkpoint consistency. CUDA BF16/FLA parity and the 22 GiB runtime
limit have also been verified. Model-quality evaluation remains separate.

## Status

- Architecture implementation: complete
- CPU correctness tests: passed
- Deterministic interleave and exact resume: passed
- CUDA BF16 and FLA runtime parity: passed
- Selected runtime: 3,587.86 tok/s, 21.957 GiB peak reserved (10-step check)
- Trained weights: not published
- Downstream benchmarks: not run

See [EFFICIENCY.md](EFFICIENCY.md) for runtime usage and the optimization
findings.

Passing implementation tests does not establish language-model quality.

## License and attribution

Original project code is licensed under the Apache License 2.0. KDA, MLA,
Delta Block, Muon, and MTP are mechanisms described in published research. This
project does not claim that combining them is itself a novel research result.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for details.
