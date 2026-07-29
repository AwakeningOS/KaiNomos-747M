# KaiNomos-110M

> **Reforming the laws of compute allocation.**
> 計算資源配分の法則を作り変える。

A 110M-parameter research language model built to answer one question:

> Given a fixed compute budget, does deciding *per token and per layer* how much
> of that budget to spend beat spending it uniformly?

Two arms are trained on identical data, from identical weights, for an identical
number of tokens, at an identical average analytical FLOPs budget. They differ in
one thing only.

| Arm | Behaviour |
|---|---|
| **Base** | every layer runs at its standard capacity |
| **Adaptive** | a shared controller varies each layer's capacity per token, under the same average budget |

The verdict is a single number: **validation and test NLL at equal compute.**

## Attribution

This is an **independent small-scale research implementation inspired by selected
mechanisms published in the Kimi K3 and Kimi Linear reports.** It is not
affiliated with, endorsed by, or derived from Moonshot AI, and it is not a
distillation, conversion or reduced version of any released model.

It differs from those systems in every dimension that defines them: 110M dense
parameters rather than a large mixture of experts, text-only rather than
multimodal, 1,024-token context rather than 1M, and with mechanisms — MUDD-QKV,
the Delta Block, adaptive execution routing — that are this project's own.

`THIRD_PARTY_NOTICES.md` records the provenance of every mechanism, dependency
and dataset, and which parts came from a paper, from external code, or from here.

## Architecture

16 layers, d_model 512, `KKKM` × 4 — 12 recurrent-memory layers and 4
latent-attention layers. **No layer is ever skipped.** Routing varies capacity
*inside* a layer, so even an easy token keeps updating its representation all the
way to the last layer.

**Nested dense FFN.** One SiTU-GLU matrix used at a prefix width chosen per
token: 1024 / 1408 / 1792 / 2176 / 2432 / 2816. 1792 is the standard capacity;
the narrower tiers prune and the wider tiers are funded by that pruning, so the
average holds.

**MUDD-QKV.** Each layer builds its Query, Key and Value inputs as a learned
mixture of *every* past layer output, not just the one below it. Initialised to
select the newest state, so it starts as a no-op and any mixing is learned rather
than imposed.

**Projected Low-Rank Delta Block.** Re-uses the *change* each 4-layer block made
rather than the accumulated state. Values stay at full width; only the
64-dimensional routing key is projected. The gate starts at zero, so the block
begins as exactly the identity.

**MTP-1.** An auxiliary head predicts the token *after* next, from the hidden
state plus the next token's embedding, at loss weight 0.30. Training only — it
can be dropped at inference.

**Adaptive routing.** One shared controller, one common price, four axes
(recurrent update, latent read, FFN width, delta retrieval). The price is solved
per batch so that the executed cost equals the Base cost, and training selects by
the same rule that inference uses — otherwise the policy that is measured is not
the policy that was trained.

| | |
|---|---|
| Total parameters | 111,042,670 |
| Inference (excluding MTP) | 106,409,510 |
| Controller | 39,694 (0.036%) |

## Training data

`KaiNomos-DataMix-v1` — a fixed-ratio pool of 1,988,270,624 tokens built
for this model. The ratios below are exact; the pool stops where the scarcest
source runs out rather than drifting off the mixture to reach a round number.

| Share | Source |
|---|---|
| 35% | Japanese organic web |
| 20% | Japanese paraphrase |
| 10% | Japanese document-grounded instruction |
| 10% | Japanese Wikipedia reference |
| 10% | English educational web |
| 10% | Educational code, permissively licensed only |
| 5% | Mathematics and reasoning |

Ratios are defined over tokens *in this project's tokenizer*: a corpus counted
with someone else's tokenizer says nothing about how much of it this model will
actually read.

A dedicated 32,768-piece SentencePiece Unigram tokenizer is trained on the same
mixture. The predecessor's 16,384-piece English BPE encoded Japanese at 2.56
tokens per character — decomposing it into raw UTF-8 bytes — which would have
spent most of the training budget re-deriving the character encoding.

## Status

Implementation complete and tested; full training pending.

```bash
python -m pytest tests/ -q      # 28 tests
python gpu_smoke.py             # forward, backward, step, eval, save, reload
```

## Licence and use

Research code. Before any commercial use or public model release, consult a
qualified IP professional; this repository makes no warranty of patent or
trademark clearance.
