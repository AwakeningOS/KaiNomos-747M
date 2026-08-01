"""Compare the legacy tokenwise/expanded-MLA path with optimized generation."""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURE = ROOT / "architecture"
sys.path.insert(0, str(ARCHITECTURE))

from config import KaiNomosConfig
from generation import GenerationState, cached_forward
from mla import GatedMLA
from model import KaiNomosForCausalLM


def set_absorbed_decode(model: KaiNomosForCausalLM, enabled: bool) -> None:
    for module in model.modules():
        if isinstance(module, GatedMLA):
            module.absorbed_decode_enabled = enabled


def tensor_bytes(value: object) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if dataclasses.is_dataclass(value):
        return sum(
            tensor_bytes(getattr(value, field.name))
            for field in dataclasses.fields(value)
        )
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(item) for item in value)
    return 0


def run_path(
    model: KaiNomosForCausalLM,
    prompt: torch.Tensor,
    continuation: torch.Tensor,
    *,
    absorbed_decode: bool,
    parallel_prefill: bool,
) -> tuple[dict, torch.Tensor]:
    set_absorbed_decode(model, absorbed_decode)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    state = GenerationState()
    outputs = []
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        if parallel_prefill:
            logits, state = cached_forward(model, prompt, state)
        else:
            prompt_outputs = []
            for position in range(prompt.shape[1]):
                logits, state = cached_forward(
                    model, prompt[:, position : position + 1], state
                )
                prompt_outputs.append(logits)
            logits = torch.cat(prompt_outputs, dim=1)
    torch.cuda.synchronize()
    prefill_seconds = time.perf_counter() - started
    outputs.append(logits.float().cpu())

    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for position in range(continuation.shape[1]):
            logits, state = cached_forward(
                model, continuation[:, position : position + 1], state
            )
            outputs.append(logits.float().cpu())
    torch.cuda.synchronize()
    decode_seconds = time.perf_counter() - started
    result = {
        "prefill_seconds": prefill_seconds,
        "prefill_tokens_per_second": prompt.shape[1] / prefill_seconds,
        "decode_seconds": decode_seconds,
        "decode_tokens_per_second": continuation.shape[1] / decode_seconds,
        "cache_mib": tensor_bytes(state.cache) / 1024**2,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
    }
    return result, torch.cat(outputs, dim=1)


def median_result(values: list[dict]) -> dict:
    return {
        key: statistics.median(float(value[key]) for value in values)
        for key in values[0]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-length", type=int, default=64)
    parser.add_argument("--decode-tokens", type=int, default=16)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--decode-ab-repeats", type=int, default=1)
    args = parser.parse_args()
    if args.prompt_length < 1 or args.decode_tokens < 1:
        raise SystemExit("token counts must be positive")
    if args.decode_ab_repeats < 1:
        raise SystemExit("--decode-ab-repeats must be positive")

    torch.manual_seed(20260801)
    checkpoint = None
    if args.checkpoint is not None:
        checkpoint = torch.load(
            args.checkpoint, map_location="cpu", weights_only=False, mmap=True
        )
        config = KaiNomosConfig.from_dict(checkpoint["model_config"])
    else:
        config = KaiNomosConfig.tiny() if args.tiny else KaiNomosConfig()
    model = KaiNomosForCausalLM(config).to("cuda").eval()
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"], strict=True)
        del checkpoint
        model = model.to("cuda").eval()
    prompt = torch.randint(
        5, config.vocab_size, (1, args.prompt_length), device="cuda"
    )
    continuation = torch.randint(
        5, config.vocab_size, (1, args.decode_tokens), device="cuda"
    )

    warm_prompt = prompt[:, : min(4, prompt.shape[1])]
    warm_continuation = continuation[:, :1]
    run_path(
        model,
        warm_prompt,
        warm_continuation,
        absorbed_decode=True,
        parallel_prefill=True,
    )
    run_path(
        model,
        warm_prompt,
        warm_continuation,
        absorbed_decode=False,
        parallel_prefill=True,
    )
    run_path(
        model,
        warm_prompt,
        warm_continuation,
        absorbed_decode=False,
        parallel_prefill=False,
    )

    optimized_runs = []
    explicit_runs = []
    optimized_logits = explicit_logits = None
    for repeat in range(args.decode_ab_repeats):
        order = (True, False) if repeat % 2 == 0 else (False, True)
        for absorbed in order:
            measured, logits = run_path(
                model,
                prompt,
                continuation,
                absorbed_decode=absorbed,
                parallel_prefill=True,
            )
            if absorbed:
                optimized_runs.append(measured)
                optimized_logits = logits
            else:
                explicit_runs.append(measured)
                explicit_logits = logits
    optimized = median_result(optimized_runs)
    explicit = median_result(explicit_runs)
    assert optimized_logits is not None and explicit_logits is not None
    legacy, legacy_logits = run_path(
        model,
        prompt,
        continuation,
        absorbed_decode=False,
        parallel_prefill=False,
    )
    absorbed_difference = (optimized_logits - explicit_logits).abs()
    parallel_difference = (explicit_logits - legacy_logits).abs()
    prompt_last = args.prompt_length - 1
    decode_slice = slice(args.prompt_length, None)
    report = {
        "configuration": "tiny" if args.tiny else "full",
        "checkpoint": None if args.checkpoint is None else str(args.checkpoint),
        "prompt_length": args.prompt_length,
        "decode_tokens": args.decode_tokens,
        "optimized": optimized,
        "parallel_prefill_explicit_decode": explicit,
        "legacy": legacy,
        "speedup": {
            "prefill": (
                optimized["prefill_tokens_per_second"]
                / legacy["prefill_tokens_per_second"]
            ),
            "decode": (
                optimized["decode_tokens_per_second"]
                / explicit["decode_tokens_per_second"]
            ),
        },
        "logit_difference": {
            "absorbed_vs_explicit_maximum_absolute": (
                absorbed_difference.max().item()
            ),
            "absorbed_vs_explicit_mean_absolute": absorbed_difference.mean().item(),
            "parallel_vs_tokenwise_maximum_absolute": parallel_difference.max().item(),
            "parallel_vs_tokenwise_mean_absolute": parallel_difference.mean().item(),
            "absorbed_vs_explicit_argmax_agreement": (
                (optimized_logits.argmax(-1) == explicit_logits.argmax(-1))
                .float()
                .mean()
                .item()
            ),
            "parallel_vs_tokenwise_argmax_agreement": (
                (explicit_logits.argmax(-1) == legacy_logits.argmax(-1))
                .float()
                .mean()
                .item()
            ),
            "parallel_vs_tokenwise_prompt_last_argmax_equal": bool(
                explicit_logits[:, prompt_last].argmax(-1).item()
                == legacy_logits[:, prompt_last].argmax(-1).item()
            ),
            "parallel_vs_tokenwise_decode_argmax_agreement": (
                (
                    explicit_logits[:, decode_slice].argmax(-1)
                    == legacy_logits[:, decode_slice].argmax(-1)
                )
                .float()
                .mean()
                .item()
            ),
        },
    }
    rendered = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
