"""Apply runtime-only training controls without changing architecture source hash."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

# The selected mb16 runtime needs the lower-fragmentation allocator.  This must
# be configured before kainomos_optimization_runtime imports torch.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from kainomos_optimization_runtime import (
    OptimizationOptions,
    apply_runtime_optimizations,
)

ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURE = ROOT / "architecture"
sys.path.insert(0, str(ARCHITECTURE))

canonical_train = importlib.import_module("train")


def parse_runtime_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--runtime-activation-checkpointing",
        choices=("on", "off"),
        required=True,
    )
    parser.add_argument("--runtime-micro-batch", type=int, default=16)
    parser.add_argument("--runtime-checkpoint-every-steps", type=int, default=200)
    parser.add_argument("--runtime-lm-chunk-tokens", type=int, default=32)
    parser.add_argument(
        "--runtime-mla-attention",
        choices=("math_fp32", "varlen_flash_bf16"),
        default="varlen_flash_bf16",
    )
    parser.add_argument(
        "--runtime-mla-gate",
        choices=("eager-fp32", "compiled-fp32"),
        default="eager-fp32",
    )
    parser.add_argument(
        "--runtime-kda-final-state",
        choices=("canonical", "training-off"),
        default="training-off",
    )
    parser.add_argument("--runtime-kda-disable-recompute", action="store_true")
    parser.add_argument(
        "--runtime-rms-norm",
        choices=("canonical", "fla-bf16", "fla-bf16-all"),
        default="fla-bf16-all",
    )
    parser.add_argument(
        "--runtime-delta-score",
        choices=("canonical", "fla-rms-linear"),
        default="fla-rms-linear",
    )
    parser.add_argument(
        "--runtime-compile-mode",
        choices=("off", "pointwise-default"),
        default="off",
    )
    parser.add_argument(
        "--runtime-checkpoint-policy",
        choices=("full",),
        default="full",
    )
    parser.add_argument("--runtime-dry-run", action="store_true")
    return parser.parse_known_args(argv)


def option_value(
    arguments: list[str], option: str, default: str | None = None
) -> str | None:
    for index, value in enumerate(arguments):
        if value == option:
            return arguments[index + 1]
        if value.startswith(option + "="):
            return value.split("=", 1)[1]
    return default


def replace_option(arguments: list[str], option: str, replacement: str) -> list[str]:
    """Return a copy with one canonical value for an argparse option."""
    rewritten: list[str] = []
    skip = False
    found = False
    for value in arguments:
        if skip:
            skip = False
            continue
        if value == option:
            rewritten.extend((option, replacement))
            skip = True
            found = True
        elif value.startswith(option + "="):
            rewritten.extend((option, replacement))
            found = True
        else:
            rewritten.append(value)
    if not found:
        rewritten.extend((option, replacement))
    return rewritten


def target_step_for_tokens(target_tokens: int, tokens_per_step: int) -> int:
    if target_tokens < 1 or tokens_per_step < 1:
        raise ValueError("target tokens and tokens per step must be positive")
    return (target_tokens + tokens_per_step - 1) // tokens_per_step


def main() -> None:
    runtime, train_arguments = parse_runtime_args(sys.argv[1:])
    if runtime.runtime_checkpoint_every_steps < 1:
        raise SystemExit("runtime checkpoint cadence must be positive")
    runtime_micro_batch = runtime.runtime_micro_batch
    if runtime_micro_batch is not None and (
        runtime_micro_batch < 1 or 64 % runtime_micro_batch
    ):
        raise SystemExit("runtime micro batch must be a positive divisor of 64")
    optimization = OptimizationOptions(
        lm_chunk_tokens=runtime.runtime_lm_chunk_tokens,
        mla_attention=runtime.runtime_mla_attention,
        mla_gate=runtime.runtime_mla_gate,
        compile_mode=runtime.runtime_compile_mode,
        checkpoint_policy=runtime.runtime_checkpoint_policy,
        kda_final_state=runtime.runtime_kda_final_state,
        kda_disable_recompute=runtime.runtime_kda_disable_recompute,
        rms_norm=runtime.runtime_rms_norm,
        delta_score=runtime.runtime_delta_score,
    )
    optimization_record = apply_runtime_optimizations(canonical_train, optimization)
    optimization_record["pytorch_cuda_alloc_conf"] = os.environ.get(
        "PYTORCH_CUDA_ALLOC_CONF"
    )
    run_dir_value = option_value(train_arguments, "--run-dir")
    if run_dir_value is None:
        raise SystemExit("--run-dir is required")
    run_dir = Path(run_dir_value).resolve()
    latest_path = run_dir / "latest.json"
    start_step = 0
    if latest_path.is_file():
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        start_step = int(latest["step"])
    stop_after = option_value(train_arguments, "--stop-after-steps")
    target_tokens = int(option_value(train_arguments, "--target-tokens", "67108864"))
    sequence_length = int(option_value(train_arguments, "--sequence-length", "1024"))
    if runtime_micro_batch is None:
        micro_batch = int(option_value(train_arguments, "--micro-batch", "1"))
        grad_accum = int(option_value(train_arguments, "--grad-accum", "64"))
    else:
        micro_batch = runtime_micro_batch
        grad_accum = 64 // micro_batch
    tokens_per_step = sequence_length * micro_batch * grad_accum
    target_step = target_step_for_tokens(target_tokens, tokens_per_step)
    aligned_target_tokens = target_step * tokens_per_step
    optimization_record["requested_target_tokens"] = target_tokens
    optimization_record["aligned_target_tokens"] = aligned_target_tokens
    optimization_record["target_step"] = target_step
    final_step = (
        min(start_step + int(stop_after), target_step) if stop_after else target_step
    )

    original_config = canonical_train.TrainConfig

    def runtime_config(*args, **kwargs):
        kwargs["activation_checkpointing"] = (
            runtime.runtime_activation_checkpointing == "on"
        )
        kwargs["checkpoint_every_steps"] = runtime.runtime_checkpoint_every_steps
        kwargs["micro_batch"] = micro_batch
        kwargs["grad_accum"] = grad_accum
        return original_config(*args, **kwargs)

    if runtime_micro_batch is not None:
        original_stream_load = canonical_train.InterleavedSequenceStream.load_state_dict

        def rebatched_stream_load(stream, state):
            migrated = dict(state)
            migrated["batch_size"] = stream.batch_size
            return original_stream_load(stream, migrated)

        canonical_train.InterleavedSequenceStream.load_state_dict = (
            rebatched_stream_load
        )

    original_save = canonical_train.save_checkpoint

    def cadence_filtered_save(
        path,
        model,
        optimizer,
        stream,
        step,
        tokens_done,
        model_config,
        train_config,
        metadata,
    ):
        is_diagnostic = "diagnostic" in path.name
        is_due = step % runtime.runtime_checkpoint_every_steps == 0
        is_final = step == final_step
        if not (is_diagnostic or is_due or is_final or step == 0):
            return
        recorded_metadata = dict(metadata)
        recorded_metadata["runtime_optimization"] = optimization_record
        return original_save(
            path,
            model,
            optimizer,
            stream,
            step,
            tokens_done,
            model_config,
            train_config,
            recorded_metadata,
        )

    canonical_train.TrainConfig = runtime_config
    canonical_train.save_checkpoint = cadence_filtered_save
    # The canonical parser still requires 50. The runtime factory records and
    # applies the requested cadence, while the save wrapper enforces it.
    filtered = replace_option(train_arguments, "--checkpoint-every-steps", "50")
    filtered = replace_option(filtered, "--micro-batch", str(micro_batch))
    filtered = replace_option(filtered, "--grad-accum", str(grad_accum))
    filtered = replace_option(
        filtered,
        "--target-tokens",
        str(aligned_target_tokens),
    )

    if runtime.runtime_dry_run:
        preview = runtime_config(
            target_tokens=aligned_target_tokens,
            sequence_length=sequence_length,
        )
        print(
            json.dumps(
                {
                    "architecture_source_sha256": canonical_train.source_sha256(
                        ARCHITECTURE
                    ),
                    "start_step": start_step,
                    "requested_target_tokens": target_tokens,
                    "aligned_target_tokens": aligned_target_tokens,
                    "expected_final_step": final_step,
                    "runtime_train_config": asdict(preview),
                    "runtime_optimization": optimization_record,
                    "canonical_arguments": filtered,
                },
                indent=2,
            )
        )
        return
    sys.argv = [sys.argv[0], *filtered]
    canonical_train.main()


if __name__ == "__main__":
    main()
