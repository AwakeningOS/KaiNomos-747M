# Third-party notices — KaiNomos-110M

KaiNomos-110M is an **independent research implementation**. It is not
affiliated with, endorsed by, or derived from Moonshot AI or any other model
provider, and it is not a distillation, conversion or reduced version of any
released model.

No published model's weights, configuration files or source code were copied
into this repository. The mechanisms below were implemented from the equations
and descriptions in public reports.

## Provenance of each mechanism

| Mechanism | Origin | What this repository contributes |
|---|---|---|
| Recurrent-memory attention with delta-rule updates and address-wise decay | Kimi Linear report (arXiv:2510.26692) | independent implementation from the published recurrence; the DECAY_ONLY execution mode is this project's |
| Latent attention with compressed KV, NoPE | Kimi K3 report | independent implementation; the BYPASS execution mode is this project's |
| Block attention residuals over depth | Attention Residuals (arXiv:2603.15031) | superseded here by the Delta Block, which re-uses per-block *changes* rather than accumulated state |
| SiTU-GLU activation | Kimi K3 report | independent implementation |
| Multiway dynamic dense connections | MUDDFormer (PMLR v267) | **restricted to Q/K/V only** in this project; the residual direction is left to the Delta Block so exactly one mechanism writes each path |
| Delta attention residuals, low-rank routing | arXiv:2605.18855, arXiv:2607.09694 | the Projected Low-Rank Delta Block, its zero-gate identity initialisation and its tiered retrieval are this project's |
| Multi-token prediction | arXiv:2404.19737 | independent single-extra-token implementation |
| Joint budgeted execution routing | related to TriRoute (arXiv:2607.06601), Mixture-of-Depths (arXiv:2404.02258) | the four-axis controller, the per-batch price solve, and the equal-cost reinvestment formulation are this project's |

**Original to this project:** the combination itself; the nested-FFN
reinvestment tiers above the standard width; the per-batch price solve that
closes the budget against the *deployed* policy; the identity-initialisation
discipline applied to every added mechanism.

## Runtime dependencies

| Package | Licence |
|---|---|
| PyTorch | BSD-3-Clause |
| Triton | MIT |
| fla-core (flash-linear-attention) | MIT — © 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li and contributors |
| SentencePiece | Apache-2.0 |
| datasets, huggingface_hub, tokenizers, pyarrow | Apache-2.0 |
| numpy, pandas, boto3 | BSD-3-Clause / Apache-2.0 |
| xxhash | BSD-2-Clause |

`fla.ops.kda.chunk_kda` and `fla.ops.kda.fused_recurrent_kda` are used as the
fast path for the recurrent-memory layers. No fla-core source is vendored; it is
an ordinary dependency.

## Training data

| Source | Dataset | Licence / terms |
|---|---|---|
| Japanese web, paraphrase, instruction | `llm-jp/scaling-data-constrained-llms` | see dataset card |
| Japanese reference | `wikimedia/wikipedia` (20231101.ja) | CC-BY-SA 4.0 / GFDL |
| English educational web | `HuggingFaceTB/dclm-edu` | see dataset card |
| Educational code | `HuggingFaceTB/stack-edu` metadata + Software Heritage content | per-file; see below |
| Mathematics | `HuggingFaceTB/finemath` (finemath-4plus) | see dataset card |

Exact revision SHAs, per-source document and byte counts, filters, and the
stream range consumed are recorded in `data/mix/manifest.json`, so the pool can
be rebuilt or audited.

### Code licensing

Stack-Edu ships metadata only; file contents are fetched from the Software
Heritage archive by blob id. **Only files carrying a permissive licence are
used** — MIT, Apache-2.0, BSD-2/3-Clause, ISC, Unlicense, CC0, 0BSD, Zlib and
similar. Files with no licence grant, or with an ambiguous `LicenseRef-*`
marker, are rejected; in sampling, roughly 81% of candidate rows carried no
grant and were discarded.

For every kept file, `data/mix/code_*/provenance.jsonl` records the repository,
path, SWHID and detected licence, so the corpus can be audited and an individual
file withdrawn without rebuilding the pool.

## Naming

"KaiNomos" is this project's own name. Other model and company names appear in
this repository only as citations of published work, never as a claim of origin,
affiliation or endorsement.

## No warranty of clearance

This document records provenance. It is not a legal opinion and it is not a
patent or trademark clearance. Consult a qualified IP professional before
commercial use or public model release.
