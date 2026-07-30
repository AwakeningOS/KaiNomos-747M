# Environment lock

The exact environment every measured number in this repository was produced on.
`requirements.txt` states the floors that the code needs; this file states what
was actually installed, so a result can be reproduced rather than approximated.

Captured 2026-07-30.

## Host

| | |
|---|---|
| OS | Linux 7.0.0-28-generic |
| GPU | NVIDIA GeForce RTX 3090, 24 GB |
| NVIDIA driver | 580.173.02 |
| CPU | Intel i5-12600 (65 W, non-K) |

## Python

Interpreter: **3.12.13** at `~/デスクトップ/mini_kimi_organism/.venv/bin/python`
(shared with the predecessor project; `run_overnight.sh` points at it directly).

| Package | Version |
|---|---|
| torch | 2.13.0+cu130 |
| CUDA (torch) | 13.0 |
| triton | 3.7.1 |
| fla-core | 0.5.2 |
| numpy | 2.5.1 |
| sentencepiece | 0.2.2 |
| tokenizers | 0.23.1 |
| datasets | 5.0.1 |
| huggingface_hub | 1.25.1 |
| xxhash | 3.8.1 |
| boto3 | 1.43.58 |
| pytest | 9.1.1 |

## Reproducing

```bash
PY=~/デスクトップ/mini_kimi_organism/.venv/bin/python
PYTHONPATH=. $PY -m pytest tests/ -q      # 31 tests
PYTHONPATH=. $PY gpu_smoke.py             # forward, backward, step, eval, save, reload
```

`trust_remote_code=True` is never used; dataset loads are restricted to parquet
branches.

## Note on the KDA kernel

`kda_impl="auto"` prefers the FLA Triton kernels and falls back to the sequential
PyTorch reference when `fla-core` is missing. The fallback is numerically the
reference path used by the CPU tests but roughly 80x slower per KDA layer, and
nothing in the training log distinguishes the two. If throughput is far below the
figures in `HANDOFF.md`, check that `fla.ops.kda` imported.
