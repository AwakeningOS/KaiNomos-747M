# Third-party notices — KaiNomos-750M

KaiNomos-750M is an independent research implementation. It is not affiliated
with, endorsed by, or derived from Moonshot AI or another model provider, and
it is not a distillation, conversion or reduced release of another model.

No published model weights are included. The mechanisms below were implemented
from public technical descriptions.

| Mechanism | Primary reference | Project usage |
| --- | --- | --- |
| Kimi Delta Attention | Kimi K3, arXiv:2607.24653; Kimi Linear | recurrent-memory attention with address-wise decay |
| NoPE Multi-head Latent Attention | Kimi K3; QK-Normed MLA, arXiv:2606.16310 | compressed latent cache with explicit QK normalization |
| SiTU-GLU | Kimi K3 | dense FFN activation |
| Delta Block residual routing | Delta Attention Residuals, arXiv:2605.18855 | additive routing over full hidden deltas; no MuDD |
| Per-Head Muon | Kimi K3; Muon is Scalable, arXiv:2502.16982 | shared-LR RMS-matched optimizer path |
| Multi-token prediction | arXiv:2404.19737 | optional single-extra-token auxiliary objective |

Combining known mechanisms is not presented as a new scientific result. The
remaining project-specific question is whether Delta Block improves held-out
next-token NLL over the same KDA/MLA backbone with ordinary residuals.

## Runtime dependencies

| Package | Licence |
| --- | --- |
| PyTorch | BSD-3-Clause |
| Triton | MIT |
| fla-core | MIT |
| SentencePiece | Apache-2.0 |
| datasets, huggingface_hub, tokenizers, pyarrow | Apache-2.0 |
| numpy, boto3 | BSD-3-Clause / Apache-2.0 |

No fla-core source is vendored. `fla.ops.kda` is an ordinary runtime dependency.

## Training data

The implementation expects an external document-indexed uint16 pool and does
not include raw corpora. The development manifest combines a locally prepared
multisource pool with `AdaMLLab/JpnMix/minhash_deduped`. Exact revisions,
licences, filters, counts and artifact hashes belong in the external data
manifest. Users must independently comply with every source dataset's terms.

"KaiNomos" is this project's own name. Other names appear only as citations.
This notice records provenance and is not legal advice or a patent/trademark
clearance opinion.
