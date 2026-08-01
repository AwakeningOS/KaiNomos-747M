#!/usr/bin/env python
"""Resumable KaiNomos-750M architecture-screen training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from config import ARCHITECTURE_ID, KaiNomosConfig
from interleave import (
    DATA_ORDER_CONTRACT,
    DeterministicInterleaver,
    InterleavedSequenceStream,
)
from model import KaiNomosForCausalLM
from muon import Muon, muon_param_groups
from segments import mask_targets_at_boundaries

OPTIMIZER_CONTRACT = "shared_lr_per_head_muon_v1"


@dataclass
class TrainConfig:
    architecture: str = ARCHITECTURE_ID
    depth_routing: str = "delta_block"
    mtp: str = "off"
    mtp_loss_weight: float = 0.1
    seed: int = 11
    target_tokens: int = 67_108_864
    schedule_tokens: int = 32_551_993_344
    sequence_length: int = 1_024
    micro_batch: int = 1
    grad_accum: int = 64
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 4_968
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    precision: str = "bf16"
    checkpoint_every_steps: int = 50
    validate_every_steps: int = 256
    max_chunk_tokens: int = 8_192
    activation_checkpointing: bool = True

    @property
    def tokens_per_step(self) -> int:
        return self.sequence_length * self.micro_batch * self.grad_accum

    @property
    def schedule_steps(self) -> int:
        return self.schedule_tokens // self.tokens_per_step


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def source_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


@torch.no_grad()
def parameter_sha256(model: torch.nn.Module) -> str:
    """Hash unique initial parameters before they are moved to the GPU."""
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(memoryview(value.numpy()).cast("B"))
    return digest.hexdigest()


def learning_rate_at_step(step: int, config: TrainConfig) -> float:
    if step < config.warmup_steps:
        return config.lr * (step + 1) / config.warmup_steps
    denominator = max(config.schedule_steps - config.warmup_steps, 1)
    progress = min(max((step - config.warmup_steps) / denominator, 0.0), 1.0)
    return config.min_lr + 0.5 * (config.lr - config.min_lr) * (
        1 + math.cos(math.pi * progress)
    )


def fsync_path(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def durable_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fsync_path(path)


def durable_json_line(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def checkpoint_paths(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("step_*.pt"))


def consumed_stream_tokens(stream: InterleavedSequenceStream) -> int:
    """Return tokens handed to training, excluding interleaver read-ahead."""
    read = sum(
        row["tokens"]
        for row in stream.interleaver.source_token_accounting().values()
    )
    pending = sum(int(values.size) for _, values in stream.pending)
    return read - pending


def assert_stream_alignment(
    stream: InterleavedSequenceStream,
    step: int,
    tokens_done: int,
    tokens_per_step: int,
) -> None:
    expected = step * tokens_per_step
    if tokens_done != expected:
        raise RuntimeError(
            f"checkpoint step/token mismatch: {tokens_done} != {expected}"
        )
    consumed = consumed_stream_tokens(stream)
    if consumed != tokens_done:
        raise RuntimeError(
            f"checkpoint data cursor mismatch: consumed={consumed}, "
            f"tokens_done={tokens_done}"
        )


def save_checkpoint(
    path: Path,
    model: KaiNomosForCausalLM,
    optimizer: Muon,
    stream: InterleavedSequenceStream,
    step: int,
    tokens_done: int,
    model_config: KaiNomosConfig,
    train_config: TrainConfig,
    metadata: dict,
) -> None:
    assert_stream_alignment(
        stream, step, tokens_done, train_config.tokens_per_step
    )
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "stream": stream.state_dict(),
        "step": step,
        "tokens_done": tokens_done,
        "model_config": model_config.to_dict(),
        "train_config": asdict(train_config),
        "metadata": metadata,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": (
            torch.cuda.get_rng_state_all()
            if next(model.parameters()).device.type == "cuda" else None
        ),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    fsync_path(temporary)
    os.replace(temporary, path)
    fsync_path(path)
    durable_json(path.parent / "latest.json", {
        "checkpoint": path.name,
        "step": step,
        "tokens_done": tokens_done,
    })


def retain_latest(run_dir: Path, keep: int = 2) -> None:
    paths = [
        path for path in checkpoint_paths(run_dir)
        if path.name != "step_00000000.pt"
    ]
    for path in paths[:-keep]:
        path.unlink()


def controlled_nonfinite_stop(
    *,
    reason: str,
    run_dir: Path,
    model: KaiNomosForCausalLM,
    optimizer: Muon,
    stream: InterleavedSequenceStream,
    step: int,
    tokens_done: int,
    model_config: KaiNomosConfig,
    train_config: TrainConfig,
    metadata: dict,
    recent_records: list[dict],
    stream_state: dict,
    python_rng,
    numpy_rng,
    torch_rng: torch.Tensor,
    cuda_rng: list[torch.Tensor] | None,
) -> None:
    """Rollback an uncommitted step and save a valid diagnostic checkpoint."""
    stream.load_state_dict(stream_state)
    random.setstate(python_rng)
    np.random.set_state(numpy_rng)
    torch.set_rng_state(torch_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)
    optimizer.zero_grad(set_to_none=True)
    diagnostic = run_dir / f"step_{step:08d}_diagnostic.pt"
    save_checkpoint(
        diagnostic, model, optimizer, stream, step, tokens_done,
        model_config, train_config, metadata,
    )
    durable_json(run_dir / "abnormal_stop.json", {
        "status": "controlled_nonfinite_stop",
        "reason": reason,
        "step": step,
        "tokens_done": tokens_done,
        "resumable_checkpoint": diagnostic.name,
        "recent_successful_steps": recent_records,
    })


@torch.no_grad()
def validation_nll(
    model: KaiNomosForCausalLM,
    manifest: dict,
    data_dir: Path,
    max_tokens: int = 2_000_000,
) -> dict:
    declared = manifest.get("splits", {}).get("validation", {})
    shards = declared.get("shards", [])
    if not shards:
        return {
            "weighted_nll": None,
            "per_source_nll": {},
            "source_weights": {},
            "validation_tokens": 0,
        }
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    sums: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    sequence = model.config.context_length_train
    for shard in shards:
        source = str(
            shard.get("source_id", Path(shard["path"]).name.split("-", 1)[0])
        )
        values = np.memmap(data_dir / shard["path"], dtype=np.uint16, mode="r")
        for start in range(0, len(values) - sequence, sequence):
            if sum(counts.values()) >= max_tokens:
                break
            ids = torch.from_numpy(
                np.asarray(values[start:start + sequence], dtype=np.int64).copy()
            ).unsqueeze(0).to(device)
            output = model(ids)
            targets = mask_targets_at_boundaries(
                ids[:, 1:], ids[:, :-1], model.config.eod_token_id
            )
            loss = F.cross_entropy(
                output.logits[:, :-1].reshape(-1, output.logits.shape[-1]).float(),
                targets.reshape(-1), ignore_index=-100, reduction="sum",
            )
            valid = int((targets != -100).sum())
            sums[source] += float(loss.cpu())
            counts[source] += valid
        if sum(counts.values()) >= max_tokens:
            break
    model.train(was_training)
    total_tokens = sum(counts.values())
    per_source = {
        source: sums[source] / counts[source] for source in sorted(counts)
    }
    declared_weights = manifest.get("validation_source_weights", {})
    if declared_weights:
        active_weights = {
            source: float(declared_weights[source])
            for source in per_source
            if source in declared_weights
        }
        if set(active_weights) != set(per_source):
            missing = sorted(set(per_source) - set(active_weights))
            raise ValueError(
                "validation weights missing for sources: " + ", ".join(missing)
            )
    else:
        active_weights = {
            source: counts[source] / max(total_tokens, 1)
            for source in per_source
        }
    weight_total = sum(active_weights.values())
    return {
        "weighted_nll": sum(
            active_weights[source] * per_source[source] for source in per_source
        ) / max(weight_total, 1e-30),
        "per_source_nll": per_source,
        "source_weights": active_weights,
        "validation_tokens": total_tokens,
    }


def train(args: argparse.Namespace) -> None:
    if args.device.startswith("cuda") and not args.allow_gpu:
        raise SystemExit("GPU use requires --allow-gpu")
    if args.architecture != ARCHITECTURE_ID:
        raise ValueError(f"--architecture must be {ARCHITECTURE_ID}")
    if args.optimizer != "muon":
        raise ValueError("KaiNomos-750M v1 requires the frozen Muon contract")
    if args.muon_lr is not None and args.muon_lr != args.lr:
        raise ValueError("RMS-matched Muon and AdamW must share one LR")
    if args.checkpoint_every_steps != 50:
        raise ValueError(
            "KaiNomos-750M v1 checkpoint cadence is frozen at 50 steps before step "
            "1000 and 200 steps thereafter; use --checkpoint-every-steps 50"
        )
    train_config = TrainConfig(
        architecture=args.architecture,
        depth_routing=args.depth_routing,
        mtp=args.mtp,
        mtp_loss_weight=args.mtp_loss_weight,
        seed=args.seed,
        target_tokens=args.target_tokens,
        schedule_tokens=args.schedule_tokens,
        sequence_length=args.sequence_length,
        micro_batch=args.micro_batch,
        grad_accum=args.grad_accum,
        lr=args.lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        checkpoint_every_steps=args.checkpoint_every_steps,
        validate_every_steps=args.validate_every_steps,
        max_chunk_tokens=args.max_chunk_tokens,
    )
    if train_config.target_tokens % train_config.tokens_per_step:
        raise ValueError("target tokens must align exactly to optimizer steps")
    if train_config.schedule_tokens % train_config.tokens_per_step:
        raise ValueError("schedule tokens must align exactly to optimizer steps")
    model_config = KaiNomosConfig(depth_routing=args.depth_routing)
    model_config.mtp.enabled = args.mtp == "on"
    model_config.mtp.loss_weight = args.mtp_loss_weight
    if model_config.context_length_train != train_config.sequence_length:
        raise ValueError("sequence length differs from the frozen model context")

    data_dir = Path(args.data_dir)
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tokenizer = manifest.get("tokenizer", {})
    if int(tokenizer.get("vocab_size", -1)) != model_config.vocab_size:
        raise ValueError("tokenizer vocabulary mismatch")
    if int(manifest.get("eod_token_id", -1)) != model_config.eod_token_id:
        raise ValueError("EOD token mismatch")
    interleaver = DeterministicInterleaver.from_manifest(
        manifest_path,
        seed=train_config.seed,
        max_chunk_tokens=train_config.max_chunk_tokens,
    )
    stream = InterleavedSequenceStream(
        interleaver, train_config.sequence_length, train_config.micro_batch
    )
    source_root = Path(__file__).resolve().parent
    metadata = {
        "architecture_id": ARCHITECTURE_ID,
        "source_sha256": source_sha256(source_root),
        "config_sha256": json_sha256(model_config.to_dict()),
        "tokenizer_sha256": tokenizer.get("sha256", ""),
        "data_manifest_sha256": sha256(manifest_path),
        "optimizer_contract": OPTIMIZER_CONTRACT,
        "data_order_contract": DATA_ORDER_CONTRACT,
        "manifest_adapter_id": getattr(interleaver, "manifest_adapter_id", ""),
        "command_line": sys.argv,
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    }

    random.seed(train_config.seed)
    np.random.seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    device = torch.device(args.device)
    model = KaiNomosForCausalLM(model_config)
    metadata["initial_parameter_sha256"] = parameter_sha256(model)
    model = model.to(device)
    if train_config.activation_checkpointing:
        model.gradient_checkpointing_enable()
    groups = muon_param_groups(
        model,
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
    )
    optimizer = Muon(
        groups,
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
        momentum=model_config.optimizer.muon_momentum,
        nesterov=model_config.optimizer.muon_nesterov,
        ns_steps=model_config.optimizer.muon_ns_steps,
        adamw_betas=model_config.optimizer.adamw_betas,
        adamw_eps=model_config.optimizer.adamw_eps,
        update_rms=model_config.optimizer.muon_update_rms,
    )
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    step = tokens_done = 0
    resume = checkpoint_paths(run_dir)
    if resume:
        checkpoint_blob = torch.load(resume[-1], map_location="cpu", weights_only=False)
        for contract_key in (
            "architecture_id", "source_sha256", "config_sha256",
            "tokenizer_sha256", "data_manifest_sha256",
            "optimizer_contract", "data_order_contract",
            "initial_parameter_sha256",
        ):
            if checkpoint_blob["metadata"].get(contract_key) != metadata.get(contract_key):
                raise RuntimeError(
                    f"checkpoint contract metadata mismatch: {contract_key}"
                )
        if checkpoint_blob["model_config"] != model_config.to_dict():
            raise RuntimeError("checkpoint model config mismatch")
        saved_train = checkpoint_blob["train_config"]
        for frozen in ("architecture", "depth_routing", "mtp", "seed", "schedule_tokens"):
            if saved_train[frozen] != asdict(train_config)[frozen]:
                raise RuntimeError(f"checkpoint training contract mismatch: {frozen}")
        model.load_state_dict(checkpoint_blob["model"])
        optimizer.load_state_dict(checkpoint_blob["optimizer"])
        stream.load_state_dict(checkpoint_blob["stream"])
        step = int(checkpoint_blob["step"])
        tokens_done = int(checkpoint_blob["tokens_done"])
        assert_stream_alignment(
            stream, step, tokens_done, train_config.tokens_per_step
        )
        random.setstate(checkpoint_blob["python_rng"])
        np.random.set_state(checkpoint_blob["numpy_rng"])
        torch.set_rng_state(checkpoint_blob["torch_rng"])
        if device.type == "cuda" and checkpoint_blob["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(checkpoint_blob["cuda_rng"])
    else:
        save_checkpoint(
            run_dir / "step_00000000.pt", model, optimizer, stream,
            0, 0, model_config, train_config, metadata,
        )

    dtype = torch.bfloat16 if train_config.precision == "bf16" else torch.float32
    log_path = run_dir / "train.jsonl"
    start_step = step
    segment_started = time.monotonic()
    recent_records: list[dict] = []
    model.train()
    while tokens_done < train_config.target_tokens:
        if args.stop_after_steps is not None and step - start_step >= args.stop_after_steps:
            break
        if (
            args.stop_after_minutes is not None
            and time.monotonic() - segment_started >= args.stop_after_minutes * 60
        ):
            break
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = time.monotonic()
        stream_before_step = stream.state_dict()
        random_before_step = random.getstate()
        numpy_before_step = np.random.get_state()
        torch_before_step = torch.get_rng_state()
        cuda_before_step = (
            torch.cuda.get_rng_state_all() if device.type == "cuda" else None
        )

        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        ntp_loss = 0.0
        mtp_loss_total = 0.0
        source_tokens: Counter[str] = Counter()
        last_values = None
        for _ in range(train_config.grad_accum):
            values, composition = stream.next_batch()
            last_values = values
            source_tokens.update(composition)
            ids = torch.from_numpy(values.astype(np.int64, copy=False)).to(
                device, non_blocking=True
            )
            with torch.autocast(
                device_type=device.type,
                dtype=dtype,
                enabled=device.type == "cuda" and dtype != torch.float32,
            ):
                output = model(ids, labels=ids)
                loss = output.loss / train_config.grad_accum
            if not torch.isfinite(loss):
                controlled_nonfinite_stop(
                    reason=f"non-finite loss at attempted step {step + 1}",
                    run_dir=run_dir, model=model, optimizer=optimizer,
                    stream=stream, step=step, tokens_done=tokens_done,
                    model_config=model_config, train_config=train_config,
                    metadata=metadata, recent_records=recent_records,
                    stream_state=stream_before_step,
                    python_rng=random_before_step, numpy_rng=numpy_before_step,
                    torch_rng=torch_before_step, cuda_rng=cuda_before_step,
                )
                return
            loss.backward()
            total_loss += float(loss.detach())
            ntp_loss += float(output.ntp_loss.detach()) / train_config.grad_accum
            if output.mtp_loss is not None:
                mtp_loss_total += float(output.mtp_loss.detach()) / train_config.grad_accum
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), train_config.max_grad_norm
        )
        if not torch.isfinite(grad_norm):
            controlled_nonfinite_stop(
                reason=f"non-finite gradient at attempted step {step + 1}",
                run_dir=run_dir, model=model, optimizer=optimizer,
                stream=stream, step=step, tokens_done=tokens_done,
                model_config=model_config, train_config=train_config,
                metadata=metadata, recent_records=recent_records,
                stream_state=stream_before_step,
                python_rng=random_before_step, numpy_rng=numpy_before_step,
                torch_rng=torch_before_step, cuda_rng=cuda_before_step,
            )
            return
        rate = learning_rate_at_step(step, train_config)
        for group in optimizer.param_groups:
            group["lr"] = rate
        collect_optimizer_stats = (step + 1) % 50 == 0
        optimizer.step(collect_stats=collect_optimizer_stats)
        step += 1
        tokens_done += train_config.tokens_per_step
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = max(time.monotonic() - started, 1e-9)
        source_record = interleaver.note_optimizer_step(source_tokens.keys())
        source_record["source_tokens"] = dict(sorted(source_tokens.items()))
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "step": step,
            "tokens_done": tokens_done,
            "loss": total_loss,
            "ntp_loss": ntp_loss,
            "mtp_loss": mtp_loss_total,
            "learning_rate": rate,
            "grad_norm_pre_clip": float(grad_norm),
            "tokens_per_second": train_config.tokens_per_step / seconds,
            "step_seconds": seconds,
            "source_composition": source_record,
            "source_accounting": interleaver.source_token_accounting(),
        }
        if device.type == "cuda":
            record["vram"] = {
                "allocated": torch.cuda.memory_allocated(device),
                "reserved": torch.cuda.memory_reserved(device),
                "peak_allocated": torch.cuda.max_memory_allocated(device),
                "peak_reserved": torch.cuda.max_memory_reserved(device),
            }
        if step % 50 == 0:
            record["optimizer_stats"] = getattr(optimizer, "last_step_stats", [])
            probe_ids = torch.from_numpy(
                last_values[:, :128].astype(np.int64, copy=False)
            ).to(device)
            was_training = model.training
            model.eval()
            with torch.no_grad():
                diagnostic = model(probe_ids, return_route_stats=True)
            model.train(was_training)
            record["delta_route_stats"] = diagnostic.route_stats
        durable_json_line(log_path, record)
        recent_records.append(record)
        del recent_records[:-100]
        print(
            f"step={step} tokens={tokens_done:,} nll={ntp_loss:.4f} "
            f"tok/s={record['tokens_per_second']:.0f}",
            flush=True,
        )
        cadence = 50 if step < 1_000 else 200
        if step % cadence == 0:
            save_checkpoint(
                run_dir / f"step_{step:08d}.pt", model, optimizer, stream,
                step, tokens_done, model_config, train_config, metadata,
            )
            retain_latest(run_dir)
        if train_config.validate_every_steps and step % train_config.validate_every_steps == 0:
            full = step % 10_000 == 0
            validation = validation_nll(
                model, manifest, data_dir,
                max_tokens=32_000_000 if full else 2_000_000,
            )
            durable_json(
                run_dir / f"validation_step{step:08d}.json",
                {
                    "step": step,
                    "tokens_done": tokens_done,
                    "scope": "full" if full else "quick",
                    **validation,
                },
            )

    final_path = run_dir / f"step_{step:08d}.pt"
    save_checkpoint(
        final_path, model, optimizer, stream, step, tokens_done,
        model_config, train_config, metadata,
    )
    retain_latest(run_dir)
    if tokens_done >= train_config.target_tokens:
        validation = validation_nll(
            model, manifest, data_dir, max_tokens=32_000_000
        )
        durable_json(
            run_dir / f"validation_final_step{step:08d}.json",
            {
                "step": step,
                "tokens_done": tokens_done,
                "scope": "full",
                **validation,
            },
        )
    durable_json(run_dir / "run_summary.json", {
        "status": "complete" if tokens_done >= train_config.target_tokens else "segment_stopped",
        "step": step,
        "tokens_done": tokens_done,
        "target_tokens": train_config.target_tokens,
        "model_config": model_config.to_dict(),
        "train_config": asdict(train_config),
        "metadata": metadata,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", default=ARCHITECTURE_ID)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--optimizer", default="muon")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--muon-lr", type=float)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=4_968)
    parser.add_argument("--sequence-length", type=int, default=1_024)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=64)
    parser.add_argument("--target-tokens", type=int, default=67_108_864)
    parser.add_argument("--schedule-tokens", type=int, default=32_551_993_344)
    parser.add_argument("--depth-routing", choices=["none", "delta_block"], default="delta_block")
    parser.add_argument("--mtp", choices=["off", "on"], default="off")
    parser.add_argument("--mtp-loss-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--checkpoint-every-steps", type=int, default=50)
    parser.add_argument("--validate-every-steps", type=int, default=256)
    parser.add_argument("--max-chunk-tokens", type=int, default=8_192)
    parser.add_argument("--stop-after-steps", type=int)
    parser.add_argument("--stop-after-minutes", type=float)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
