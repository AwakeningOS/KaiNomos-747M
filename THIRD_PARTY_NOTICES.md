# Third-party notices — KaiNomos-747M

KaiNomos-747M is an **independent research implementation**. It is not
affiliated with, endorsed by, or derived from Moonshot AI or any other model
provider, and it is not a distillation, conversion or reduced version of any
released model.

No published model's weights, configuration files or source code were copied
into this repository. The mechanisms below were implemented from the equations
and descriptions in public reports.

## Provenance of each mechanism

| Mechanism | Origin | What this repository contributes |
|---|---|---|
| Recurrent-memory attention with delta-rule updates and address-wise decay | Kimi Linear report (arXiv:2510.26692) | independent implementation from the published recurrence |
| Latent attention with compressed KV, NoPE | Kimi K3 report | independent implementation with full-head QK normalization |
| SiTU-GLU activation | Kimi K3 report | independent implementation |
| Multiway dynamic dense connections | MUDDFormer (PMLR v267) | **restricted to Q/K/V only** in this project; the residual direction is left to the Delta Block so exactly one mechanism writes each path |
| Delta Block Attention Residuals | Delta Attention Residuals, Cheng Luo, Zefan Cai, Junjie Hu, [arXiv:2605.18855](https://arxiv.org/abs/2605.18855) | **adopted as published**, not a mechanism of this project. Sources are the embedding, the completed block deltas and the open block's partial delta; keys are `norm(V)` at full width; the query is `w_l` in R^d, zero-initialised; the softmax is over the source axis; the routed value is added, never substituted. No gate, temperature or entropy term, as published. Written from the paper rather than ported from the reference implementation ([wdlctc/delta-attention-residuals-code](https://github.com/wdlctc/delta-attention-residuals-code), MIT), because that repository also carries gated, V-separated, null-source and entropy-regularised variants that the paper does not require and this project does not want. The paper's per-sublayer variant is implemented as `delta.granularity = "sublayer"` and is **not used** — measured OOM at micro-batch 2 and half the throughput at micro-batch 1 on this model's config; the Block variant is production. |
| Multi-token prediction | arXiv:2404.19737 | independent single-extra-token implementation |

**Original to this project:** the selected combination, its integration and the
identity-initialisation discipline applied to added mechanisms.

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

Exact revisions, counts, filters and artifact hashes are recorded in the
external DoubleDragon-DataMix-v2 manifests; raw corpora are not copied here.

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
