# Inference runtime

KaiNomos generation uses a hybrid temporal cache:

- each KDA layer keeps its recurrent matrix and short-convolution windows;
- each MLA layer keeps the 256-dimensional KV latent and one inverse-RMS
  scalar per token/head;
- the prompt is evaluated in one model call per EOD-delimited segment;
- subsequent decode calls pass only the new token.

The model therefore does not rerun the whole prefix through all 24 layers for
each generated token.

## Selected prompt prefill

The original generation helper invoked the full model once per prompt token.
The selected path evaluates the prompt as a causal block, returns the same cache
types, and keeps EOD cache resets by splitting only at document boundaries.
KDA uses its recurrent FLA kernel for both prefill and decode; MLA uses causal
attention during prefill.

The following batch-one BF16 measurements used the same trained checkpoint and
fixed token sequences on one RTX 3090:

| Prompt | Tokenwise prefill | Selected prefill | Speedup | Final cache |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 28.08 tok/s | 329.09 tok/s | 11.7× | 11.77 MiB |
| 256 | 27.19 tok/s | 4,523.68 tok/s | 166× | 13.40 MiB |
| 768 | 24.42 tok/s | 7,214.33 tok/s | 295× | 16.52 MiB |

At prompt 256 plus 32 fixed decode tokens, parallel versus tokenwise execution
had maximum absolute logit difference `0.09375` and mean absolute difference
`0.00659`. The prompt-final argmax and every measured decode-position argmax
matched. This is BF16 kernel-order agreement, not bitwise identity.

## Absorbed MLA decode candidate

An algebraically equivalent NoPE MLA path was implemented from the QK-Normed
MLA formulation. It moves the static key RMSNorm weight and key up-projection to
the query side, attends directly over cached latents, applies the cached
per-token inverse-RMS scalars, and moves value up-projection after the weighted
latent sum. It changes no parameters or checkpoint keys.

The candidate remains disabled by default because it did not provide a stable
RTX 3090 speedup with the current PyTorch kernels. At prompt 256, five
alternating A/B repetitions gave these median decode results:

| Decode path | Median throughput |
| --- | ---: |
| Existing explicit expansion | **26.57 tok/s** |
| Absorbed latent path | 26.50 tok/s |

The absorbed versus explicit maximum absolute logit difference was `0.03125`,
with identical argmax at every measured position. A fused CUDA/Triton kernel
could change the performance result; the supplied benchmark keeps the candidate
available for that work without making it the production default.

## Reproduce

```bash
python scripts/benchmark_kainomos_generation.py \
  --checkpoint /path/to/checkpoint.pt \
  --prompt-length 256 \
  --decode-tokens 32 \
  --decode-ab-repeats 5
```

The benchmark compares:

1. parallel prefill plus absorbed MLA decode;
2. parallel prefill plus the selected explicit MLA decode;
3. the former tokenwise prefill plus explicit MLA decode.

All paths use identical weights and fixed prompt/continuation token IDs.

## Sources

- [DeepSeek-V2 technical report](https://arxiv.org/abs/2405.04434)
- [QK-Normed MLA](https://arxiv.org/abs/2606.16310)
- [DeepSeek-AI FlashMLA](https://github.com/deepseek-ai/FlashMLA)
