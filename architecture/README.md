# KaiNomos-750M implementation

This directory contains the adopted `kainomos_750m_v1` implementation.

For installation, training, and inference, start with the root `README.md`,
`TRAINING.md`, and `examples/quickstart.py`.

## Frozen shape

- 24 layers, hidden size 1,280, dense SiTU-GLU FFN 5,120
- `(KDA, KDA, KDA, MLA) × 6`; final layer is MLA
- 10 heads × 128 dimensions
- additive block-level Delta routing; no MuDD
- 718,341,812 deployment-backbone parameters
- optional 31,491,978-parameter MTP module; MTP is currently off
- tied 49,152-token embedding/head; EOD ID 4
- shared LR 0.0003 for Per-Head/full-matrix Muon and AdamW groups

## Verification

From this directory:

```bash
PYTHONPATH=. ../.venv/bin/python -m pytest --rootdir=. --confcutdir=. -q tests
../.venv/bin/python -m ruff check .
```

CPU tests cover parameter budgets, Delta semantics, document isolation,
checkpoint gradients, MTP initialization invariance, optimizer ownership,
deterministic interleave, exact resume, cached generation, observations, and
training cursor alignment. CUDA BF16/FLA parity gates for the adopted runtime
also passed.

## Selected runtime

The selected runtime is checkpoint-on, mb16/acc4, chunk32, BF16 varlen Flash
MLA, all-FLA RMSNorm, fused Delta scoring, and the expandable CUDA allocator.

Runtime settings, measurements, and the lower-memory fallback are documented in
the root `EFFICIENCY.md`.
