"""Benchmark one runtime configuration without writing to the production run."""

from __future__ import annotations

import argparse
import contextlib
import gc
import importlib
import json
import os
import statistics
import sys
import tempfile
from pathlib import Path

import torch
from kainomos_optimization_runtime import (
    OptimizationOptions,
    apply_runtime_optimizations,
)

ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURE = ROOT / "architecture"
sys.path.insert(0, str(ARCHITECTURE))

canonical_train = importlib.import_module("train")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--activation-checkpointing", choices=("on", "off"), required=True
    )
    parser.add_argument("--micro-batch", type=int, required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--peak-reserved-limit-gib", type=float, default=22.0)
    parser.add_argument("--lm-chunk-tokens", type=int, default=32)
    parser.add_argument(
        "--compile-mode",
        choices=(
            "off",
            "regional-default",
            "regional-max-autotune",
            "pointwise-default",
        ),
        default="off",
    )
    parser.add_argument(
        "--checkpoint-policy",
        choices=(
            "full",
            "selective-matmul",
            "skip-last-stage",
            "skip-last-2-stages",
            "skip-last-3-stages",
            "skip-last-4-stages",
        ),
        default="full",
    )
    parser.add_argument(
        "--mla-attention",
        choices=("math_fp32", "varlen_flash_bf16"),
        default="math_fp32",
    )
    parser.add_argument(
        "--mla-gate",
        choices=("eager-fp32", "compiled-fp32"),
        default="eager-fp32",
    )
    parser.add_argument(
        "--kda-final-state",
        choices=("canonical", "training-off"),
        default="canonical",
    )
    parser.add_argument("--kda-disable-recompute", action="store_true")
    parser.add_argument(
        "--rms-norm",
        choices=("canonical", "fla-bf16", "fla-bf16-all"),
        default="canonical",
    )
    parser.add_argument(
        "--delta-score",
        choices=("canonical", "fla-rms-linear"),
        default="canonical",
    )
    parser.add_argument(
        "--profile-trace",
        type=Path,
        help=(
            "write a one-step CPU/CUDA trace after one warmup step; "
            "profiled throughput is not a selection benchmark"
        ),
    )
    args = parser.parse_args()
    if args.micro_batch < 1 or 64 % args.micro_batch:
        raise SystemExit("micro batch must be a positive divisor of 64")
    if args.steps < 2:
        raise SystemExit("at least two steps are required")
    if args.peak_reserved_limit_gib <= 0:
        raise SystemExit("peak reserved limit must be positive")

    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    saved_train = dict(checkpoint["train_config"])
    saved_metadata = dict(checkpoint["metadata"])
    saved_step = int(checkpoint["step"])
    saved_tokens = int(checkpoint["tokens_done"])
    del checkpoint
    gc.collect()

    optimization = OptimizationOptions(
        lm_chunk_tokens=args.lm_chunk_tokens,
        compile_mode=args.compile_mode,
        checkpoint_policy=args.checkpoint_policy,
        mla_attention=args.mla_attention,
        mla_gate=args.mla_gate,
        kda_final_state=args.kda_final_state,
        kda_disable_recompute=args.kda_disable_recompute,
        rms_norm=args.rms_norm,
        delta_score=args.delta_score,
    )
    optimization_record = apply_runtime_optimizations(canonical_train, optimization)

    original_config = canonical_train.TrainConfig

    def benchmark_config(*config_args, **kwargs):
        kwargs["activation_checkpointing"] = args.activation_checkpointing == "on"
        return original_config(*config_args, **kwargs)

    original_stream_load = canonical_train.InterleavedSequenceStream.load_state_dict

    def rebatched_stream_load(stream, state):
        migrated = dict(state)
        migrated["batch_size"] = stream.batch_size
        return original_stream_load(stream, migrated)

    canonical_train.TrainConfig = benchmark_config
    canonical_train.InterleavedSequenceStream.load_state_dict = rebatched_stream_load
    canonical_train.source_sha256 = lambda _root: saved_metadata["source_sha256"]
    canonical_train.save_checkpoint = lambda *unused_args, **unused_kwargs: None
    canonical_train.retain_latest = lambda *unused_args, **unused_kwargs: None

    profiler = None
    original_durable_json_line = canonical_train.durable_json_line
    if args.profile_trace is not None:
        trace_path = args.profile_trace.resolve()
        trace_path.parent.mkdir(parents=True, exist_ok=True)

        def export_trace(active_profiler):
            active_profiler.export_chrome_trace(str(trace_path))

        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1),
            on_trace_ready=export_trace,
            record_shapes=False,
            profile_memory=True,
            with_stack=False,
        )

        def profiled_durable_json_line(path, value):
            original_durable_json_line(path, value)
            if Path(path).name == "train.jsonl":
                profiler.step()

        canonical_train.durable_json_line = profiled_durable_json_line

    micro_batch = args.micro_batch
    grad_accum = 64 // micro_batch
    with tempfile.TemporaryDirectory(prefix="kainomos-speed-candidate-") as temp_name:
        run_dir = Path(temp_name)
        (run_dir / checkpoint_path.name).symlink_to(checkpoint_path)
        namespace = argparse.Namespace(
            architecture=saved_train["architecture"],
            data_dir=str(args.data_dir.resolve()),
            run_dir=str(run_dir),
            device="cuda",
            allow_gpu=True,
            optimizer="muon",
            lr=saved_train["lr"],
            muon_lr=None,
            min_lr=saved_train["min_lr"],
            weight_decay=saved_train["weight_decay"],
            warmup_steps=saved_train["warmup_steps"],
            sequence_length=saved_train["sequence_length"],
            micro_batch=micro_batch,
            grad_accum=grad_accum,
            target_tokens=saved_train["target_tokens"],
            schedule_tokens=saved_train["schedule_tokens"],
            depth_routing=saved_train["depth_routing"],
            mtp=saved_train["mtp"],
            mtp_loss_weight=saved_train["mtp_loss_weight"],
            seed=saved_train["seed"],
            checkpoint_every_steps=50,
            validate_every_steps=0,
            max_chunk_tokens=saved_train["max_chunk_tokens"],
            stop_after_steps=args.steps,
            stop_after_minutes=None,
        )
        profile_context = profiler if profiler is not None else contextlib.nullcontext()
        with profile_context:
            canonical_train.train(namespace)
        records = [
            json.loads(line)
            for line in (run_dir / "train.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]

    steady = records[1:]
    peak_reserved_gib = max(
        record["vram"]["peak_reserved"] for record in records
    ) / 1024**3
    result = {
        "checkpoint": str(checkpoint_path),
        "start_step": saved_step,
        "start_tokens": saved_tokens,
        "activation_checkpointing": args.activation_checkpointing == "on",
        "micro_batch": micro_batch,
        "grad_accum": grad_accum,
        "tokens_per_optimizer_step": saved_train["sequence_length"]
        * micro_batch
        * grad_accum,
        "steps": [record["step"] for record in records],
        "tokens_per_second": [record["tokens_per_second"] for record in records],
        "steady_median_tokens_per_second": statistics.median(
            record["tokens_per_second"] for record in steady
        ),
        "steady_mean_tokens_per_second": statistics.fmean(
            record["tokens_per_second"] for record in steady
        ),
        "peak_allocated_gib": max(
            record["vram"]["peak_allocated"] for record in records
        )
        / 1024**3,
        "peak_reserved_gib": peak_reserved_gib,
        "peak_reserved_limit_gib": args.peak_reserved_limit_gib,
        "vram_headroom_gib": args.peak_reserved_limit_gib - peak_reserved_gib,
        "within_peak_reserved_limit": peak_reserved_gib
        <= args.peak_reserved_limit_gib,
        "all_finite": all(
            torch.isfinite(torch.tensor(record["loss"]))
            and torch.isfinite(torch.tensor(record["grad_norm_pre_clip"]))
            for record in records
        ),
        "optimization": optimization_record,
        "profiled": profiler is not None,
        "profile_trace": (
            str(args.profile_trace.resolve()) if args.profile_trace is not None else None
        ),
        "throughput_valid_for_selection": profiler is None,
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["all_finite"] or not result["within_peak_reserved_limit"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
