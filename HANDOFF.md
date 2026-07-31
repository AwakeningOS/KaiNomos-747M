# KaiNomos-750M handoff

The canonical implementation is `architecture/`.
Do not resume the old step-650 checkpoint or copy any of its tensors into the
current model.

## Fixed architecture

- ID `kainomos_750m_v1`
- 24 layers, hidden 1,280, FFN 5,120
- `(KDA, KDA, KDA, MLA) × 6`, strict NoPE MLA, head dimension 128
- Delta Block is the only depth mechanism; no MuDD
- backbone 718,341,812 parameters
- optional MTP 31,491,978; total 749,833,790
- shared LR 0.0003 across Per-Head/full-matrix Muon and AdamW
- MTP defaults off

The executable configuration is `architecture/config.py`. Read
`architecture/RESEARCH.md` before changing the experiment.

## Data

Use a local SSD copy of DoubleDragon-DataMix-v2. The verified development
manifest declares 32,552,055,906 train tokens; the model consumes the aligned
frozen schedule of 32,551,993,344. The legacy adapter exposes `local` and
`jpnmix` as the two recoverable major sources and records that limitation in
checkpoint metadata.

## Current verified state

- CPU architecture suite: passed
- deterministic source interleave and resume: passed
- old checkpoint import: prohibited
- CUDA BF16 / FLA equivalence: pending
- short three-step startup check: pending
- monitored approximately 90-minute training segment: pending
- 67,108,864-token Arm A/B screen: pending

Do not skip those gates or start the 32.55B production run directly.

The old 200-step and 2,000-step gates were superseded on 2026-07-31. The active
protocol is recorded in `architecture/RUN_PROTOCOL_AMENDMENT_2026-07-31.md`.

## Runtime status

No trained checkpoint is published. CUDA acceptance and the monitored
90-minute development run remain pending and must be reported separately from
the architecture implementation checks.
