# KaiNomos-750M implementation

This directory contains the current `kainomos_750m_v1` implementation. It does
not load or partially migrate the old KaiNomos step-650 checkpoint.

## Frozen shape

- 24 layers, hidden size 1,280, SiTU-GLU FFN 5,120
- `(KDA, KDA, KDA, MLA) × 6`; final layer is MLA
- 10 heads × 128 dimensions
- Delta Block only; MuDD is absent
- 718,341,812 deployment-backbone parameters
- optional 31,491,978-parameter MTP module
- 749,833,790 parameters with MTP enabled
- tied 49,152-token embedding/head; EOD ID 4
- shared LR 0.0003 for Per-Head/full-matrix Muon and AdamW groups

## Verification

From this directory:

```bash
PYTHONPATH=. python -m pytest --rootdir=. --confcutdir=. -q tests
```

CPU tests cover the exact parameter budget, Delta source semantics, document
isolation, checkpoint-on/off gradients, MTP initialization invariance, Muon
classification/update behavior, deterministic interleave and exact resume,
cached generation, observation output and training cursor alignment.

CUDA BF16 and FLA equivalence remain separate gates. The current operating
protocol is a short three-step start check followed, if clean, by one monitored
approximately 90-minute training segment. The superseded 200-step/2,000-step
gates are no longer part of the active acceptance protocol.

## Architecture screen

Arm A uses `--depth-routing none --mtp off`; Arm B uses
`--depth-routing delta_block --mtp off`. Both use seed 11 and exactly
67,108,864 tokens. See `RESEARCH.md` for the frozen research question.

The current legacy DataMix-v2 manifest can recover two packed sources,
`local` and `jpnmix`, from shard names. It cannot reconstruct the original
nine domains per document. The loader records this adapter ID explicitly and
mixes both recoverable major sources from the beginning.
