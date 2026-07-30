# Environment lock

Captured 2026-07-30 on the target machine.

## Host

| Component | Version |
| --- | --- |
| OS kernel | Linux 7.0.0-28-generic |
| GPU | NVIDIA GeForce RTX 3090, 24 GB |
| NVIDIA driver | 580.173.02 |
| CPU | Intel Core i5-12600 |

## Python environment

Interpreter: Python 3.12.13 at
`/home/youthk/デスクトップ/DoubleDragon-110M/.venv/bin/python`.

| Package | Version |
| --- | --- |
| torch | 2.13.0+cu130 |
| CUDA reported by torch | 13.0 |
| triton | 3.7.1 |
| fla-core | 0.5.2 |
| transformers | 5.14.1 |
| huggingface_hub | 1.25.1 |
| numpy | 2.5.1 |
| sentencepiece | 0.2.2 |
| tokenizers | 0.23.1 |
| datasets | 5.0.1 |
| pytest | 9.1.1 |

## Verification

```bash
PY=/home/youthk/デスクトップ/DoubleDragon-110M/.venv/bin/python
PYTHONPATH=. "$PY" -m pytest tests/ -q
PYTHONPATH=. "$PY" gpu_smoke.py
```

`kda_impl="auto"` must resolve to the FLA Triton implementation on CUDA.
Falling back to the sequential reference path is valid for CPU correctness
tests but is not usable for production throughput.

The complete 747M Muon smoke path at sequence length 1,024 and micro-batch 1
measured 18.08 GB peak allocated VRAM and 3,139 tokens/s. Micro-batch 2 exceeded
the 24 GB card after accounting for optimizer state, so production uses
micro-batch 1 with gradient accumulation 64.
