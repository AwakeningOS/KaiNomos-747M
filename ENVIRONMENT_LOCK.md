# Environment lock

Captured 2026-07-31 on the development machine. This records the implementation
test environment, not a promise of broad compatibility.

## Host

| Component | Version |
| --- | --- |
| OS kernel | Linux 7.0.0-28-generic |
| GPU | NVIDIA GeForce RTX 3090, 24 GB |
| CPU | Intel Core i5-12600 |

## Python environment

| Package | Version |
| --- | --- |
| Python | 3.12.13 |
| torch | 2.13.0+cu130 |
| CUDA reported by torch | 13.0 |
| triton | 3.7.1 |
| fla-core | 0.5.2 |
| numpy | 2.5.1 |
| sentencepiece | 0.2.2 |
| tokenizers | 0.23.1 |
| pytest | 9.1.1 |

## Verification

```bash
cd architecture
PYTHONPATH=. python -m pytest --rootdir=. --confcutdir=. -q tests
python -m ruff check .
```

`kda_impl="auto"` must resolve to the FLA implementation on CUDA. The sequential
reference implementation is intended for correctness testing, not production
throughput. CUDA BF16/FLA equivalence and the measured peak-VRAM result remain
pending for KaiNomos-750M and must not be inferred from the superseded 747M run.
