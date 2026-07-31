---
license: apache-2.0
library_name: pytorch
tags:
  - research
  - architecture-preview
  - consumer-gpu
  - japanese
  - kda
  - mla
  - muon
---

# KaiNomos-750M

**Untrained architecture candidate.** No trained checkpoint is published and
no model-quality claim has been established.

KaiNomos-750M is a 24-layer, hidden-1,280 decoder with a dense 5,120-wide SiTU-GLU
FFN. It repeats three KDA layers and one strict-NoPE Gated MLA layer six times.
The 718,341,812-parameter deployment backbone uses tied 49,152-token embeddings
and 10 heads of dimension 128. Optional MTP raises the training total to
749,833,790 parameters.

MuDD has been removed. The candidate depth mechanism is additive block-level
Delta routing that preserves the main residual. Its value is not assumed: an
equal-condition, MTP-off comparison against ordinary residuals must decide it
using held-out next-token NLL.

The implementation supports packed-document isolation, deterministic
source-balanced streaming, atomic exact-resume checkpoints, latent-only MLA
cache, EOD cache reset, Per-Head Muon and opt-in architecture observations.

## Intended use

- research on consumer-GPU language-model pre-training;
- Japanese-centred base-model studies;
- controlled KDA/MLA and depth-residual ablations;
- reproducible training, recovery and architecture monitoring.

## Current limitations

- CUDA BF16 and FLA production gates are pending;
- the short startup check and monitored 90-minute run are pending;
- Arm A/B and MTP selection have not been completed;
- no benchmark, safety, factuality or generation-quality results exist;
- the legacy packed manifest exposes `local`/`jpnmix`, not per-document recovery
  of every original domain.

The old KaiNomos-747M step-650 checkpoint is retained only as historical
evidence and is incompatible with the current architecture.
