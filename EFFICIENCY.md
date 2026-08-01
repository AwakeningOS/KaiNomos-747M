# KaiNomos-750M runtime efficiency under a 22 GiB limit

This document records both successful and rejected training-runtime
optimizations for KaiNomos-750M on one RTX 3090. The goal was to reduce
wall-clock time and energy per token without changing the model, optimizer
objective, token order, or learning-rate schedule.

The selected runtime is implemented by `scripts/run_kainomos_runtime_tuned.py`.
It patches runtime behavior without changing parameter objects or state-dict
keys, so the existing exact-resume checkpoint remains compatible.

## Result

| Configuration | Steady tok/s | Peak reserved | Verdict |
| --- | ---: | ---: | --- |
| Stage-2 baseline A | 2,771.87 | 19.840 GiB | bracket baseline |
| Stage-2 baseline B | 2,774.52 | 19.840 GiB | bracket baseline |
| mb8, all FLA RMSNorm and fused Delta score | 3,518.97 | 17.285 GiB | fallback |
| mb16 without allocator tuning | 3,589.20 | 22.381 GiB | rejected: over limit |
| selected mb16, ten-step confirmation | **3,587.86** | **21.957 GiB** | adopted |

The selected runtime improved throughput by 29.38% over the bracket mean of
2,773.20 tok/s. At equal average power, this reduces time and GPU energy per
token by 22.71%.

From the private step-610 checkpoint at 39,976,960 tokens to a 16B-token target,
a constant-throughput estimate falls from 66.61 days to 51.49 days, a reduction
of 15.12 days. At a constant 260 W GPU board-power assumption, the difference is
approximately 94.38 kWh. These estimates exclude validation, checkpoint I/O,
stops, host power, cooling, and clock variation.

## Comparison contract

The optimization study kept these conditions fixed:

- identical starting checkpoint for every compared candidate;
- architecture and parameter values;
- optimizer and training objective;
- tokenizer, data order, and stream position;
- learning-rate schedule;
- sequence length 1,024;
- 65,536 tokens per optimizer step;
- MTP disabled;
- no writes to the production run during benchmarks.

The hard acceptance rule was:

```text
maximize steady tokens/second
subject to peak_reserved_gib <= 22.0
and finite loss and gradients
```

Peak allocated memory was recorded, but peak reserved memory controlled the
decision. A candidate that completed once but exceeded 22.0 GiB was rejected.

Short screens discarded the first warm-up step. The final candidate was rerun
for ten isolated steps, and every loss and gradient-norm record was finite.

## Selected configuration

```text
activation checkpointing: on
micro batch: 16
gradient accumulation: 4
tokens per optimizer step: 65,536
LM cross-entropy chunk: 32
MLA attention: BF16 variable-length Flash Attention
MLA output gate: eager FP32
KDA final state during training: off
KDA disable_recompute: false
RMSNorm: FLA BF16 for all norms, including Delta sources
Delta score: FLA fused RMSNorm plus scalar linear
torch.compile: off
PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True
```

Ten-step confirmation:

- steady median: 3,587.86 tok/s;
- steady mean: 3,578.55 tok/s;
- peak allocated: 21.356 GiB;
- peak reserved: 21.957 GiB;
- headroom to the hard limit: 0.043 GiB, approximately 44 MiB.

The small headroom is intentional but must be watched during the first long
production segment.

## What worked

### Larger micro-batches

Activation checkpointing was enabled and total tokens per optimizer step stayed
fixed while the micro-batch grew:

| Micro-batch | Accumulation | Steady tok/s | Peak reserved |
| ---: | ---: | ---: | ---: |
| 2 | 32 | 2,165.51 | 12.303 GiB |
| 4 | 16 | 2,503.30 | 14.953 GiB |
| 8 | 8 | 2,664.86 | 19.820 GiB |

The mb2-to-mb4 increase was 15.60%, followed by another 6.45% from mb4 to mb8.
Reducing the number of forward/backward micro-iterations per optimizer step was
more valuable than keeping VRAM empty.

### Variable-length Flash MLA

The BF16 variable-length Flash Attention path preserved packed-document
boundaries and passed its CUDA parity gate:

- output cosine: 0.9999996;
- input-gradient cosine: 0.9999876;
- relative loss error: 0.00000335.

An early screen reached 2,977.15 tok/s, but a repeat produced 2,749.78 tok/s and
the later bracket settled near 2,773 tok/s. The direction was useful, but the
first maximum was not treated as a reproducible result.

### FLA RMSNorm, especially on Delta sources

The largest stage-2 improvement came from avoiding repeated transient FP32
activations in normalization paths.

| Change | Steady tok/s | Peak reserved | Baseline change |
| --- | ---: | ---: | ---: |
| bracket baseline | 2,773.20 | 19.840 GiB | — |
| FLA RMSNorm excluding Delta sources | 2,864.13 | 19.191 GiB | +3.28% |
| KDA switches plus partial FLA RMSNorm | 2,970.27 | 20.211 GiB | +7.11% |
| FLA RMSNorm including Delta sources | 3,509.91 | 18.156 GiB | +26.56% |
| plus fused Delta score | 3,518.97 | 17.285 GiB | +26.90% |

Normalization modules contain few parameters, but they process full sequence
activations many times. Delta routing also repeats normalization for multiple
depth sources. Parameter size therefore hid the true runtime cost.

### Fused Delta scoring

Fusing source RMSNorm with the scalar query projection improved mb8 throughput
by only about 0.26% after all-RMSNorm fusion. Its larger value was memory: the
reduced activation footprint helped make the later mb16 search possible.

An optimization can therefore be valuable even when its isolated throughput
gain is small, if it unlocks a better global configuration.

### KDA training-only switches

Suppressing an unused final recurrent state and testing KDA backward recompute
both passed output/gradient parity. Their isolated gains were small:

| KDA change | Steady tok/s | Peak reserved | Baseline change |
| --- | ---: | ---: | ---: |
| no training final state | 2,802.99 | 19.918 GiB | +1.07% |
| disable backward recompute | 2,809.03 | 20.863 GiB | +1.29% |
| both | 2,824.93 | 20.738 GiB | +1.87% |

The selected mb16 runtime suppresses the unused final state but keeps
recomputation enabled because disabling it consumed too much activation memory.

### Expandable CUDA allocator segments

Without allocator tuning, the otherwise-fast mb16 candidate reserved
22.381 GiB and failed the hard gate. Setting
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before importing Torch brought
the ten-step confirmation down to 21.957 GiB.

Raw throughput stayed almost unchanged. The allocator setting did not make the
math faster; it reduced fragmentation enough to make mb16 admissible.

## What did not work

### Activation-checkpointing OFF by itself

A diagnostic run disabled activation checkpointing but kept micro-batch 1 and
accumulation 64. It reached only about 2,072.78 tok/s. Removing recomputation
freed memory, but the free memory was not reinvested into fewer, larger
micro-iterations.

The mature conclusion is not “checkpointing OFF is faster.” The useful question
is which checkpointing policy permits the fastest complete optimizer step under
the memory limit. The final winner uses checkpointing ON.

### Larger CE chunks

With math FP32 MLA and mb8/acc8:

| CE chunk | Steady tok/s | Result versus chunk 32 |
| ---: | ---: | ---: |
| 32 | 2,672.89 | baseline |
| 64 | 2,699.31 | +0.99% |
| 128 | 2,651.29 | -0.81% |
| 256 | 2,619.96 | -1.98% |

Larger chunks reduced launch count but increased temporary logits and memory
traffic. The best chunk also changed with the batch envelope: mb8 preferred 64,
while the final mb16 configuration required 32.

### Liger fused linear cross-entropy

Liger passed its gradient and loss parity gate but measured 2,585.27 tok/s,
3.28% below the same-stage chunk32 baseline. A fused implementation or a known
fast library is not evidence of end-to-end speed on a particular model.

### Compiled MLA output gate

The compiled gate passed output and gradient parity but reached 3,362.32 tok/s,
4.45% below the comparable eager path. “Compiles successfully” and “runs
faster” are separate claims.

### Partial removal of activation checkpointing

Skipping checkpointing for selected final stages either reached approximately
22.75–22.79 GiB or failed to outperform the full-checkpoint configuration while
staying under the limit. Reducing recomputation locally was not useful when it
forced a worse global batch envelope.

### mb16 without allocator tuning

The candidate reached 3,589.20 tok/s, but 22.381 GiB peak reserved violated the
predefined limit. It was rejected even though it completed once.

## Lower-memory fallback

If the 44 MiB production headroom proves unstable, stop after a verified durable
checkpoint and use:

```text
activation checkpointing: on
micro batch / accumulation: 8 / 8
LM CE chunk: 64
BF16 variable-length Flash MLA
KDA training final state: off
KDA disable_recompute: true
FLA BF16 RMSNorm for all norms
fused Delta score
```

This configuration measured approximately 3,518.97 tok/s at 17.285 GiB peak
reserved. Do not silently change runtime settings within a live segment; record
the fallback in the next checkpoint metadata and handoff.

## Wall-clock efficiency beyond kernels

Kernel throughput is only one part of time-to-target.

- A 60-minute train / 5-minute rest cycle has a maximum 92.31% training duty
  cycle before counting checkpoint and restart overhead.
- Moving a roughly 6 GB full checkpoint cadence from every 50 steps to every 200
  steps reduces steady-state checkpoint writes by 75%, at the cost of a larger
  recomputation window after failure.
- Reloading a full model and optimizer between short segments adds avoidable
  overhead.
- Token targets must align to optimizer-step boundaries to guarantee the final
  durable save.

The requested 16,000,000,000-token target is not divisible by 65,536. The tuned
runner uses a ceiling and records both values:

```text
requested target: 16,000,000,000
final optimizer step: 244,141
exact durable target: 16,000,024,576
```

## Reusable optimization procedure

1. Freeze the checkpoint, model, optimizer, data order, schedule, and tokens per
   optimizer step.
2. Keep candidate benchmarks isolated from production checkpoints.
3. Run output and gradient parity before throughput tests for kernel, dtype, or
   fusion changes.
4. Increase the micro-batch geometrically while preserving total tokens per
   step.
5. Profile activation dtype and repeated per-source operations, not only large
   parameter modules.
6. Record both peak allocated and peak reserved memory.
7. Bracket candidates with repeated baselines to expose warm-up and clock drift.
8. Reject memory-limit violations without changing the gate after seeing the
   result.
9. Confirm the winner over more steps with finite loss and gradient checks.
10. Optimize total wall-clock-to-target, including rest periods, checkpoint I/O,
    and restart time.

## Scope and limitations

The measurements were made with an RTX 3090, context length 1,024, 65,536 tokens
per optimizer step, Torch 2.13.0+cu130, and fla-core 0.5.2. Different GPUs,
drivers, contexts, vocabularies, and models may have different optima.

The ten-step confirmation establishes bounded runtime behavior, not long-run
VRAM stability. The selected configuration's 44 MiB headroom is narrow and must
be monitored during the first production segment.

These results do not establish held-out NLL, downstream quality, safety,
factuality, generation quality, or a causal benefit from Delta Block. They only
show that the fixed training computation can be executed more efficiently on
the tested hardware.

The generated raw benchmark JSON files are intentionally excluded from the
public repository together with training runs and checkpoints. The compact
adoption registry is `scripts/optimization_candidates_stage2.json`; the
reusable measurements and conclusions are fixed in this document.
