"""Resumable pre-training CLI. It never starts GPU work without --allow-gpu."""

from __future__ import annotations

import argparse
import bisect
import concurrent.futures
import hashlib
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from config import KaiNomosConfig
from model import KaiNomosForCausalLM


@dataclass
class TrainConfig:
    seed: int = 11
    target_tokens: int = 100_000_000
    sequence_length: int = 1024
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 64
    learning_rate: float = 3e-4
    # Muon's rate is not comparable to AdamW's even after the update-RMS
    # matching; it governs an orthogonalised step, so it lives separately.
    muon_learning_rate: float = 2e-2
    optimizer: str = "adamw"
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    # The schedule is a function of tokens consumed, not of steps taken, so an
    # overnight segment that resumes mid-run lands on the same learning rate it
    # would have had in one continuous run.  `schedule_tokens` is the *whole*
    # planned horizon: if it were the segment target, every segment would restart
    # the cosine and the run would never actually decay.
    schedule_tokens: int = 0            # 0 == use target_tokens
    warmup_fraction: float = 0.01
    min_lr_ratio: float = 0.1
    checkpoint_every_steps: int = 50
    # Fractions of the token budget that get a checkpoint exempt from rotation,
    # so no later analysis ever needs a retraining run.
    milestone_fractions: tuple[float, ...] = (0.5, 1.0)
    keep_latest_checkpoints: int = 2
    # Absolute token counts at which to keep a weights-only snapshot, roughly
    # log-spaced so early growth -- where everything changes fastest -- is sampled
    # as densely as the long tail.
    #
    # Absolute, not fractions of the target: a segmented run changes its target
    # every night, so fractions would move the observation points around and the
    # ladder would not line up across segments.
    #
    # Weights only, no optimizer state and no RNG: 2.0 GB per snapshot at 500M
    # instead of 5.7 GB, which is what makes a 13-rung ladder affordable.  These
    # are for reading the model, not for resuming it.
    observation_tokens: tuple[int, ...] = (
        50_000_000, 100_000_000, 200_000_000, 400_000_000, 800_000_000,
        1_500_000_000, 2_500_000_000, 4_000_000_000, 6_000_000_000,
        8_000_000_000, 12_000_000_000, 16_000_000_000, 20_000_000_000,
    )
    precision: str = "bf16"

    @property
    def tokens_per_step(self) -> int:
        return self.sequence_length * self.micro_batch_size * self.gradient_accumulation_steps


class ShardedTokenStream:
    """Deterministic uint16 shard stream whose global position is checkpointed."""

    def __init__(
        self,
        paths: list[Path],
        sequence_length: int,
        batch_size: int,
        position: int = 0,
        max_epochs: int | None = 1,
        epochs: int = 0,
        prefetch: bool = True,
    ):
        if not paths:
            raise ValueError("at least one train shard is required")
        self.paths = [Path(path) for path in paths]
        itemsize = np.dtype(np.uint16).itemsize
        self.lengths = []
        for path in self.paths:
            if path.stat().st_size % itemsize:
                raise ValueError(f"token shard has an invalid byte length: {path}")
            self.lengths.append(path.stat().st_size // itemsize)
        self.ends = np.cumsum(self.lengths, dtype=np.int64).tolist()
        self.total_tokens = int(self.ends[-1])
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.epochs = int(epochs)
        self.prefetch = prefetch
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._future = None
        self._future_index = None
        self._current = None
        self._current_index = None
        self._position = 0
        if self.total_tokens <= sequence_length:
            raise ValueError("train shards are shorter than one sequence")
        self.position = position

    @property
    def position(self) -> int:
        return self._position

    @position.setter
    def position(self, value: int) -> None:
        value = int(value)
        if value < 0 or value > self.total_tokens:
            raise ValueError("stream position is outside the train shards")
        self._position = value
        self._current = None
        self._current_index = None
        if self._future is not None:
            self._future.cancel()
        self._future = None
        self._future_index = None

    def _load(self, index: int) -> np.ndarray:
        values = np.fromfile(self.paths[index], dtype=np.uint16)
        if len(values) != self.lengths[index]:
            raise RuntimeError(f"short read from token shard: {self.paths[index]}")
        return values

    def _schedule(self, index: int) -> None:
        if self.prefetch and index < len(self.paths) and self._future_index != index:
            self._future = self._executor.submit(self._load, index)
            self._future_index = index

    def _ensure_current(self) -> tuple[np.ndarray, int]:
        if self._position == self.total_tokens:
            self.epochs += 1
            if self.max_epochs is not None and self.epochs >= self.max_epochs:
                raise EpochLimitReached(
                    f"reached the end of the train shards after {self.epochs} "
                    "epoch(s); raise --max-epochs to repeat the pool"
                )
            self.position = 0
        index = bisect.bisect_right(self.ends, self._position)
        if self._current_index != index:
            if self._future_index == index and self._future is not None:
                self._current = self._future.result()
            else:
                self._current = self._load(index)
            self._current_index = index
            self._future = None
            self._future_index = None
            self._schedule(index + 1)
        start = 0 if index == 0 else self.ends[index - 1]
        return self._current, self._position - start

    def _take(self, count: int) -> np.ndarray:
        pieces = []
        remaining = count
        while remaining:
            current, offset = self._ensure_current()
            take = min(remaining, len(current) - offset)
            pieces.append(current[offset:offset + take])
            self._position += take
            remaining -= take
        return pieces[0] if len(pieces) == 1 else np.concatenate(pieces)

    def next(self) -> torch.Tensor:
        values = self._take(self.sequence_length * self.batch_size)
        return torch.from_numpy(
            np.asarray(values, dtype=np.int64).reshape(
                self.batch_size, self.sequence_length
            )
        )

    def close(self) -> None:
        if self._future is not None:
            self._future.cancel()
        self._executor.shutdown(wait=True, cancel_futures=True)

    def __del__(self) -> None:
        executor = getattr(self, "_executor", None)
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)


class EpochLimitReached(RuntimeError):
    """The token pool ran out and repeating it was not explicitly allowed."""


def parameter_groups(model: KaiNomosForCausalLM, weight_decay: float) -> list[dict]:
    """Split the parameters that must not be shrunk from the matrices that may be.

    Weight decay on a matrix is a prior towards a smaller map. On the tensors
    below it is simply wrong:

    * `A_log` and `dt_bias` set the KDA decay timescale. Decaying them drags the
      forgetting rate towards a fixed point that has nothing to do with the loss.
    * RMSNorm gains should not be pulled away from their normalization role.
    * `static_bias` is MUDD's identity selector, exactly 1.0 on the newest source.
      Nothing in the loss defends it, so decay alone would erode it towards 0 and
      dissolve the identity initialisation the mechanism is built on.
    * the Delta gate is a scalar at 0 whose whole purpose is to start as identity.
    * KDA's short convolutions start as delta filters, i.e. as pass-through.

    Everything here is 0- or 1-dimensional except `static_bias` and the depthwise
    filters, which are named explicitly.
    """
    from muon import _is_conv_filter

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith("static_bias") or _is_conv_filter(name):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_optimizer(model: KaiNomosForCausalLM, train_cfg: TrainConfig, kind: str):
    """AdamW over everything, or Muon over the matrices with AdamW beside it.

    Each group records its own peak rate under `base_lr`, because a Muon rate and
    an AdamW rate are different quantities even after Muon's update-RMS matching:
    the schedule scales both by the same factor rather than assigning one value to
    every group.
    """
    if kind == "adamw":
        optimizer = torch.optim.AdamW(
            parameter_groups(model, train_cfg.weight_decay),
            lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay,
        )
        for group in optimizer.param_groups:
            group["base_lr"] = train_cfg.learning_rate
        return optimizer
    if kind != "muon":
        raise ValueError(f"unknown optimizer {kind!r}")

    from muon import Muon, muon_param_groups

    groups = muon_param_groups(
        model, train_cfg.weight_decay, train_cfg.muon_learning_rate,
        train_cfg.learning_rate,
    )
    optimizer = Muon(
        groups, lr=train_cfg.muon_learning_rate, adamw_lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )
    for group in optimizer.param_groups:
        group["base_lr"] = (
            train_cfg.muon_learning_rate if group.get("use_muon")
            else train_cfg.learning_rate
        )
    return optimizer


def apply_learning_rate(optimizer, rate: float, train_cfg: TrainConfig) -> None:
    """Scale every group by where the schedule currently is.

    Muon reads `lr`, the auxiliary AdamW groups read `adamw_lr`, so setting a
    single key would silently leave one half of the model on its initial rate for
    the whole run.
    """
    scale = rate / train_cfg.learning_rate
    for group in optimizer.param_groups:
        base = group.get("base_lr", train_cfg.learning_rate)
        key = "lr" if group.get("use_muon") or "adamw_lr" not in group else "adamw_lr"
        group[key] = base * scale


def learning_rate_at(tokens_done: int, train_cfg: TrainConfig) -> float:
    """Linear warmup then cosine decay, evaluated at a token count.

    Deriving the rate from `tokens_done` rather than from a stateful scheduler
    means there is no scheduler state to checkpoint and none to desynchronise on
    resume: the same token position always gives the same rate.
    """
    horizon = train_cfg.schedule_tokens or train_cfg.target_tokens
    peak = train_cfg.learning_rate
    floor = peak * train_cfg.min_lr_ratio
    warmup = int(horizon * train_cfg.warmup_fraction)
    if warmup > 0 and tokens_done < warmup:
        # start at one step's worth rather than exactly zero
        return peak * max(tokens_done, 1) / warmup
    span = max(horizon - warmup, 1)
    progress = min(max(tokens_done - warmup, 0) / span, 1.0)
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * progress))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_vocab_size(manifest: dict) -> int:
    value = manifest.get("tokenizer", {}).get(
        "vocab_size", manifest.get("vocab_size")
    )
    if value is None:
        raise ValueError("manifest must declare tokenizer vocabulary size")
    return int(value)


def manifest_split_paths(manifest: dict, data_dir: Path, split: str) -> list[Path]:
    declared = manifest.get("splits", {}).get(split, {})
    shards = declared.get("shards")
    names = (
        [str(item["path"] if isinstance(item, dict) else item) for item in shards]
        if shards else [f"{split}.bin"]
    )
    root = data_dir.resolve()
    paths = []
    for name in names:
        path = (root / name).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"{split} shard escapes data-dir: {name}") from error
        if not path.is_file():
            raise FileNotFoundError(f"missing {split} shard: {path}")
        paths.append(path)
    measured = sum(path.stat().st_size for path in paths) // np.dtype(np.uint16).itemsize
    expected = declared.get("tokens")
    if expected is not None and int(expected) != measured:
        raise ValueError(
            f"manifest declares {expected} {split} tokens but shards contain {measured}"
        )
    return paths


def checkpoint_paths(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("step_*.pt"))


def retain_latest(run_dir: Path, keep: int) -> None:
    paths = checkpoint_paths(run_dir)
    for path in paths[:-keep]:
        path.unlink()


def save_checkpoint(
    path: Path,
    model: KaiNomosForCausalLM,
    optimizer: torch.optim.Optimizer,
    step: int,
    tokens_done: int,
    data_position: int,
    model_config: KaiNomosConfig,
    train_config: TrainConfig,
    manifest_hash: str,
    training_state: dict | None = None,
) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "tokens_done": tokens_done,
        "data_position": data_position,
        "model_config": model_config.to_dict(),
        "train_config": asdict(train_config),
        "manifest_sha256": manifest_hash,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "training_state": training_state or {},
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_observation(
    run_dir: Path, model: KaiNomosForCausalLM, model_config: KaiNomosConfig,
    step: int, tokens_done: int, mark: int,
) -> Path:
    """A weights-only snapshot for reading the model as it grows.

    Deliberately not resumable.  Carrying the optimizer state would nearly triple
    the size for something that will only ever be loaded to measure NLL, sample
    text, or fit a throwaway adapter -- none of which continue training.
    """
    directory = run_dir / "observations"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"obs_{mark // 1_000_000:06d}M.pt"
    payload = {
        "model": {k: v.detach().to(torch.float32).cpu()
                  for k, v in model.state_dict().items()},
        "config": model_config.to_dict(),
        "step": step,
        "tokens_done": tokens_done,
        "observation_mark": mark,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return path


def latest_checkpoint(run_dir: Path) -> Path | None:
    paths = checkpoint_paths(run_dir)
    return paths[-1] if paths else None


def train(args: argparse.Namespace) -> None:
    if args.device.startswith("cuda") and not args.allow_gpu:
        raise SystemExit("GPU use requires the explicit --allow-gpu flag")
    device = torch.device(args.device)
    model_cfg = KaiNomosConfig()
    train_cfg = TrainConfig()
    data_dir = Path(args.data_dir)
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    # The vocabulary is a property of the pool, not of the model defaults.
    model_cfg.vocab_size = manifest_vocab_size(manifest)
    model_cfg.tie_word_embeddings = True

    if getattr(args, "seed", None) is not None:
        train_cfg.seed = args.seed
    if getattr(args, "target_tokens", None):
        train_cfg.target_tokens = args.target_tokens
    if getattr(args, "schedule_tokens", None):
        train_cfg.schedule_tokens = args.schedule_tokens
    train_cfg.optimizer = getattr(args, "optimizer", "muon")
    if getattr(args, "muon_lr", None):
        train_cfg.muon_learning_rate = args.muon_lr
    if getattr(args, "warmup_fraction", None):
        train_cfg.warmup_fraction = args.warmup_fraction
    if getattr(args, "max_epochs", 1) == 0:
        args.max_epochs = None
    if getattr(args, "micro_batch", None):
        effective = train_cfg.tokens_per_step        # read before mutating
        train_cfg.micro_batch_size = args.micro_batch
        train_cfg.gradient_accumulation_steps = max(
            1, effective // (args.micro_batch * train_cfg.sequence_length)
        )
    if train_cfg.sequence_length != model_cfg.context_length_train:
        raise ValueError("training sequence length must match model context_length_train")

    manifest_hash = sha256(manifest_path)
    train_paths = manifest_split_paths(manifest, data_dir, "train")
    if int(manifest.get("eod_token_id", model_cfg.eod_token_id)) != model_cfg.eod_token_id:
        raise ValueError("dataset EOD token does not match model configuration")

    random.seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)
    torch.manual_seed(train_cfg.seed)
    model = KaiNomosForCausalLM(model_cfg).to(device)
    optimizer = build_optimizer(model, train_cfg, train_cfg.optimizer)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    step = tokens_done = data_position = 0
    epochs_done = 0

    resume = latest_checkpoint(run_dir)
    if resume is not None:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        if checkpoint["manifest_sha256"] != manifest_hash:
            raise RuntimeError("dataset manifest changed since the checkpoint")
        if checkpoint["model_config"] != model_cfg.to_dict():
            raise RuntimeError("model configuration changed since the checkpoint")
        saved_optimizer = (checkpoint.get("train_config") or {}).get("optimizer", "adamw")
        if saved_optimizer != train_cfg.optimizer:
            raise RuntimeError(
                f"checkpoint was trained with {saved_optimizer!r} but this run asks "
                f"for {train_cfg.optimizer!r}; the saved optimizer state does not "
                f"transfer between them"
            )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        step = int(checkpoint["step"])
        tokens_done = int(checkpoint["tokens_done"])
        data_position = int(checkpoint["data_position"])
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if checkpoint["cuda_rng"] is not None and device.type == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        training_state = checkpoint.get("training_state", {})
        epochs_done = int(training_state.get("epochs_done", 0))
        saved_train = checkpoint.get("train_config") or {}
        # `--target-tokens` is a cumulative destination, so re-running a finished
        # segment with the same value trains nothing at all: `tokens_done` is
        # already there and the loop exits immediately.  `--additional-tokens`
        # asks for that much *more* than the checkpoint has.
        if getattr(args, "additional_tokens", None):
            train_cfg.target_tokens = tokens_done + args.additional_tokens
        # The cosine schedule must span the whole planned run, not this segment,
        # or every segment restarts the decay.  Inherit the original horizon
        # unless this invocation states a new one.
        if not getattr(args, "schedule_tokens", None):
            train_cfg.schedule_tokens = int(
                saved_train.get("schedule_tokens") or 0
            ) or train_cfg.schedule_tokens

    if tokens_done >= train_cfg.target_tokens:
        raise SystemExit(
            f"nothing to do: the checkpoint already holds {tokens_done:,} tokens "
            f"and the target is {train_cfg.target_tokens:,}.  Use "
            f"--additional-tokens N to train N more tokens from here."
        )

    stream = ShardedTokenStream(
        train_paths, train_cfg.sequence_length, train_cfg.micro_batch_size, data_position,
        max_epochs=args.max_epochs, epochs=epochs_done,
    )
    remaining = (stream.total_tokens - data_position) + (
        (args.max_epochs - 1 - epochs_done) * stream.total_tokens
        if args.max_epochs is not None else 0
    )
    wanted = train_cfg.target_tokens - tokens_done
    if args.max_epochs is not None and wanted > remaining:
        raise SystemExit(
            f"the target needs {wanted:,} more tokens but only {remaining:,} remain "
            f"within --max-epochs {args.max_epochs} of a {stream.total_tokens:,}-token "
            f"pool.  Raise --max-epochs to reuse the pool deliberately."
        )
    dtype = torch.bfloat16 if train_cfg.precision == "bf16" else torch.float32
    log_path = run_dir / "train.jsonl"

    model.train()
    while tokens_done < train_cfg.target_tokens:
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        ntp_sum = 0.0
        mtp_sum = 0.0
        for _ in range(train_cfg.gradient_accumulation_steps):
            ids = stream.next().to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=dtype,
                enabled=(dtype != torch.float32 and device.type != "cpu"),
            ):
                output = model(ids, labels=ids)
                loss = output.loss / train_cfg.gradient_accumulation_steps
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step + 1}")
            loss.backward()
            loss_sum += float(loss.detach())
            if output.ntp_loss is not None:
                ntp_sum += float(output.ntp_loss.detach()) / train_cfg.gradient_accumulation_steps
            if output.mtp_loss is not None:
                mtp_sum += float(output.mtp_loss.detach()) / train_cfg.gradient_accumulation_steps

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite gradient norm at step {step + 1}")
        # The rate for the tokens this step is about to consume, not the ones
        # already consumed: at `tokens_done` the very first step would run at
        # 1/warmup of the peak, which is 7.6e-12 -- a step thrown away.
        learning_rate = learning_rate_at(tokens_done + train_cfg.tokens_per_step,
                                         train_cfg)
        apply_learning_rate(optimizer, learning_rate, train_cfg)
        optimizer.step()
        step += 1
        tokens_done += train_cfg.tokens_per_step

        record = {
            "step": step,
            "tokens_done": tokens_done,
            "lm_loss": loss_sum,
            "ntp_loss": ntp_sum,
            "mtp_loss": mtp_sum,
            "grad_norm": float(grad_norm),
            "learning_rate": learning_rate,
            "data_position": stream.position,
            "epochs_done": stream.epochs,
            "optimizer": train_cfg.optimizer,
            "ffn_width": model_cfg.dense_intermediate_size,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

        # Observation ladder: every rung the step just crossed.  A single step can
        # cross more than one early rung when the batch is large.
        for mark in train_cfg.observation_tokens:
            if tokens_done < mark or tokens_done - train_cfg.tokens_per_step >= mark:
                continue
            path = save_observation(
                run_dir, model, model_cfg, step, tokens_done, mark
            )
            print(f"[observe] {mark/1e9:.3f}B tokens -> {path.name}", flush=True)

        for frac in train_cfg.milestone_fractions:
            mark = int(frac * train_cfg.target_tokens)
            tag = f"{frac:g}"
            if tokens_done >= mark and not any(run_dir.glob(f"milestone_{tag}_*.pt")):
                save_checkpoint(
                    run_dir / f"milestone_{tag}_step{step:08d}.pt", model, optimizer,
                    step, tokens_done, stream.position, model_cfg, train_cfg,
                    manifest_hash,
                    {"epochs_done": stream.epochs},
                )
                print(f"[milestone] {tag} at {tokens_done:,} tokens", flush=True)

        if step % train_cfg.checkpoint_every_steps == 0:
            path = run_dir / f"step_{step:08d}.pt"
            save_checkpoint(
                path, model, optimizer, step, tokens_done, stream.position,
                model_cfg, train_cfg, manifest_hash,
                {"epochs_done": stream.epochs},
            )
            retain_latest(run_dir, train_cfg.keep_latest_checkpoints)

    final = run_dir / f"step_{step:08d}.pt"
    save_checkpoint(
        final, model, optimizer, step, tokens_done, stream.position,
        model_cfg, train_cfg, manifest_hash,
        {"epochs_done": stream.epochs},
    )
    retain_latest(run_dir, train_cfg.keep_latest_checkpoints)
    summary = {
        "tokens_done": tokens_done,
        "steps": step,
        "optimizer": train_cfg.optimizer,
        "epochs_done": stream.epochs,
        "final_lm_loss": loss_sum,
        "final_ntp_loss": ntp_sum,
        "final_mtp_loss": mtp_sum,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[complete] {tokens_done:,} tokens, {step:,} steps", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--target-tokens", type=int, default=None,
        help="cumulative destination: the run stops once this many tokens have "
             "been seen in total, counting everything the checkpoint already has",
    )
    parser.add_argument(
        "--additional-tokens", type=int, default=None,
        help="train this many tokens *beyond* the resumed checkpoint.  Use this "
             "for overnight segments: repeating the same --target-tokens trains "
             "nothing, because the target has already been reached",
    )
    parser.add_argument(
        "--schedule-tokens", type=int, default=None,
        help="horizon the warmup/cosine schedule spans, normally the whole "
             "planned run.  Inherited from the checkpoint when omitted, so "
             "segments continue one schedule instead of restarting it",
    )
    parser.add_argument(
        "--max-epochs", type=int, default=1,
        help="how many times the token pool may be read.  The stream used to "
             "wrap silently at the end of the pool; pass a larger value to "
             "repeat it deliberately, or 0 for no limit",
    )
    parser.add_argument(
        "--optimizer", choices=["adamw", "muon"], default="muon",
        help="production default `muon` puts 2-D hidden matrices on Muon and "
             "everything else on an auxiliary AdamW",
    )
    parser.add_argument("--muon-lr", type=float, default=None)
    parser.add_argument(
        "--warmup-fraction", type=float, default=None,
        help="fraction of --schedule-tokens spent warming up.  The 1%% default is "
             "sized for a full run; a short diagnostic may need a larger value",
    )
    parser.add_argument("--micro-batch", type=int, default=None)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
