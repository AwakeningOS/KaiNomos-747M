# K3Mini-110M

**Can ideas from a frontier-scale architecture survive when compressed into a
language model that one person can train on a home GPU?**

K3Mini-110M is an independent, AI-assisted research implementation inspired by
selected architectural ideas described in the
[Kimi K3 technical report](https://arxiv.org/abs/2607.24653). It reduces the
problem to a 111M-parameter, text-only model designed for controlled experiments
on a single consumer GPU.

This is not an official Moonshot AI project, a conversion of Kimi K3 weights, or
a claim to reproduce Kimi K3's capabilities. Kimi K3 is a 2.8T-parameter
multimodal MoE system with a one-million-token context. K3Mini is a small dense
research model with a 1,024-token training context.

> **Current status: architecture preview.** The implementation and tests are
> public-ready. The full Base-versus-Adaptive pre-training comparison is not finished,
> so this repository does not yet claim a quality improvement. Trained Base and
> Adaptive checkpoints, complete logs, and held-out results will be released when the
> matched-compute experiment is complete.

## The experiment

The core question is deliberately simple:

1. Build the smallest practical model that preserves a recognizable subset of
   the K3 language architecture.
2. Make it trainable at home-GPU scale.
3. Add experimental mechanisms.
4. Compare Base and Adaptive under the same data, initialization, token count, and
   measured analytical compute budget.
5. Publish the result even if the modification fails.

The two training arms use the same supernet and initialization:

| Arm | Policy |
| --- | --- |
| **Base** | Fixed full KDA updates, global MLA reads, FFN width 1,792, and all available delta sources |
| **Adaptive** | Learned per-input capacity routing under a matched-compute constraint, including widths below and above the Base width |

Holding the supernet constant makes `force_fixed` an inclusion test: differences
should come from the learned policy rather than unrelated parameter
initialization.

## Architecture

### K3-inspired core

- 16 decoder layers: 12 KDA and 4 Gated MLA in a repeating 3:1 pattern
- 512 hidden size and 8 attention heads
- Kimi Delta Attention with smooth lower-bounded decay
- NoPE Gated Multi-head Latent Attention
- SiTU-GLU feed-forward layers
- 16,384-token vocabulary
- 1,024-token training context

### Experimental additions

- **MUDD-QKV:** separate dynamic depth mixtures for Q, K, and V inputs
- **Projected Low-Rank Delta Blocks:** route accumulated block deltas back into
  attention and FFN inputs
- **JointRoute:** choose KDA update mode, MLA read mode, nested FFN width, and
  delta-source tier under one compute price
- **Compute reinvestment:** saved compute can be spent on FFN tiers wider than
  the Base instead of only pruning the model
- **MTP-1:** one auxiliary future-token prediction head used during training

These additions are research hypotheses, not established improvements.

| Measurement | Value |
| --- | ---: |
| Total training parameters | 111,042,670 |
| Inference backbone parameters, excluding MTP | 106,409,510 |
| Controller parameters | 39,694 |
| MUDD parameters | 583,576 |
| Delta-routing parameters | 166,432 |

## What has been verified

The repository currently has 26 passing tests covering:

- forward shapes, finite losses, and finite gradients
- causal behavior
- all 16 layers executing in every policy
- the 109M–112M parameter target
- nested FFN tiers
- compute-budget accounting
- MUDD and delta identity initialization
- checkpoint migration
- fixed-policy equivalence and checkpoint reload

A development GPU smoke run recorded:

| Setup | Observed |
| --- | ---: |
| Sequence length | 1,024 |
| Micro-batch | 2 |
| Gradient accumulation | 32 |
| Peak allocated VRAM | 7.72 GB |
| Throughput | 3,571 tokens/s |
| Optimizer-step time | 18.353 s |
| Checkpoint reload | passed |

These numbers come from a short implementation smoke test, not full
pre-training, and should not be read as a final performance benchmark.

## Quick start

Python 3.11 or 3.12 and a recent PyTorch build are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

The pure PyTorch KDA reference path is intentionally slow and exists for tests.
Practical GPU training requires the compatible
[`fla-core`](https://github.com/fla-org/flash-linear-attention) kernels:

```bash
pip install fla-core
```

Inspect the planned pre-training data mixture:

```bash
python data_mix.py plan
```

The collector downloads source datasets from Hugging Face. Review every source
dataset's license and terms before redistribution; this repository does not
redistribute those datasets.

Train either arm after producing `train.bin` and its matching `manifest.json`:

```bash
python train.py \
  --data-dir /path/to/tokenized-data \
  --run-dir runs/base \
  --arm base \
  --device cuda \
  --allow-gpu

python train.py \
  --data-dir /path/to/tokenized-data \
  --run-dir runs/adaptive \
  --arm adaptive \
  --device cuda \
  --allow-gpu
```

GPU work never starts unless `--allow-gpu` is supplied explicitly.

## Release plan

- [x] 110M-class architecture implementation
- [x] CPU correctness and invariance tests
- [x] single-GPU forward/backward/checkpoint smoke test
- [ ] tokenizer and tokenization pipeline release
- [ ] matched-compute Base pre-training
- [ ] matched-compute Adaptive pre-training
- [ ] validation and held-out test evaluation
- [ ] checkpoints, tokenizer, manifests, logs, and hashes
- [ ] technical article with successes, failures, and limitations

Checkpoints are intentionally excluded from ordinary Git history. Final weights
will be published through a release system suitable for large model artifacts,
with hashes linked from this repository.

## Built with AI as a research collaborator

Frontier AI coding assistants, including GPT-5.6, were used throughout the
project for literature navigation, architecture discussion, implementation,
test construction, bug finding, and experimental-design review.

That is part of the experiment. The goal is not merely to ask an AI to generate
a model file. The goal is to test whether a person with a consumer GPU and a
strong AI research collaborator can:

- read and connect current architecture research,
- implement a coherent small model,
- design controls that can falsify the idea,
- audit subtle implementation and compute-accounting bugs, and
- publish enough evidence for strangers to reproduce the conclusion.

AI assistance is not evidence that an architecture works. Only controlled,
reproducible results can establish that.

## From one model to a distributed home-GPU lab

K3Mini is intended as a seed for a broader English-language community:
**Homebrew LLM Lab**.

The idea is to make small-model architecture research accessible to people
outside large laboratories. Participants can use AI tools to design, implement,
train, challenge, and extend models small enough to run on personal hardware.
The competition is cooperative: fork an experiment, improve it, disprove it, or
explain why it failed.

The community's most valuable output should not be a leaderboard. It should be
the complete research trail:

```text
hypothesis → prior work → implementation → bugs → results → verdict → next test
```

Every experiment should report at least:

```yaml
experiment_id: EXP-0001
base_model: K3Mini-Base-110M
change:
  description: What changed?
prior_work:
  papers: []
claim: What should improve, and why?

training:
  gpu:
  seed:
  tokens:
  compute_flops:
  peak_vram:
  wall_time:

results:
  validation_nll:
  test_nll:
  tokens_per_second:

verdict: adopted | rejected | inconclusive
failures: []
```

Code, configs, seeds, tokenizer identity, data manifests, logs, checkpoint
hashes, negative results, and discovered bugs all matter. Important decisions
belong in GitHub Discussions or experiment reports—not only in ephemeral chat.
See [CONTRIBUTING.md](CONTRIBUTING.md) to propose an experiment.

If many people can run rigorous small experiments in parallel, the community
can eliminate weak ideas earlier, identify mechanisms worth scaling, and create
better training data for future AI research agents. Results at 110M parameters
do not automatically transfer to billion-parameter models; promising findings
must be reproduced across scales.

## Attribution and scope

K3Mini is inspired by the published Kimi K3 architecture and Kimi Delta
Attention research:

- [Kimi K3: Open Frontier Intelligence](https://arxiv.org/abs/2607.24653)
- [Official MoonshotAI/Kimi-K3 repository](https://github.com/MoonshotAI/Kimi-K3)
- [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)

“Kimi” and “Kimi K3” refer to Moonshot AI's work. This independent project is
not sponsored, endorsed, or maintained by Moonshot AI.

## License

The original code in this repository is released under the
[Apache License 2.0](LICENSE). Third-party papers, datasets, kernels, model
weights, and trademarks remain subject to their own terms.
