"""Resumable pre-training CLI. It never starts GPU work without --allow-gpu."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from config import K3MiniPlusPlusPlusConfig as K3MiniConfig
from joint_router import RouteState
from cost_model import OrganCosts
from route_eval import merge_decisions, solve_batch_price

# The pre-registered budget is a *ratio to Fixed-Full*; a run whose cumulative
# executed cost lands outside this band did not train at the compute it claims,
# so its NLL comparison is void.
BUDGET_TOLERANCE = 0.001
from model import K3MiniPlusPlusPlusForCausalLM as K3MiniForCausalLM


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
    checkpoint_every_steps: int = 100
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
    budget_rho: float = 10.0
    dual_learning_rate: float = 0.05
    price_offset_gain: float = 0.5
    cost_ema_decay: float = 0.99

    @property
    def tokens_per_step(self) -> int:
        return self.sequence_length * self.micro_batch_size * self.gradient_accumulation_steps


class SequentialTokenStream:
    """Deterministic uint16 token stream whose position is checkpointed."""

    def __init__(self, path: Path, sequence_length: int, batch_size: int, position: int = 0,
                 max_epochs: int | None = 1, epochs: int = 0):
        self.path = path
        self.tokens = np.memmap(path, mode="r", dtype=np.uint16)
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.position = int(position)
        # Wrapping past the end of the pool means seeing the same tokens again.
        # It used to happen silently, so a run could slide into a second epoch --
        # and into memorisation -- with nothing in the log to say so.
        self.max_epochs = max_epochs
        self.epochs = int(epochs)
        if len(self.tokens) <= sequence_length:
            raise ValueError("token file is shorter than one sequence")

    def next(self) -> torch.Tensor:
        width = self.sequence_length
        rows = []
        for _ in range(self.batch_size):
            if self.position + width > len(self.tokens):
                self.epochs += 1
                if self.max_epochs is not None and self.epochs >= self.max_epochs:
                    raise EpochLimitReached(
                        f"reached the end of {self.path.name} after "
                        f"{self.epochs} epoch(s); pass --max-epochs to allow "
                        f"repeating the pool"
                    )
                self.position = 0
            rows.append(np.asarray(self.tokens[self.position:self.position + width], dtype=np.int64))
            self.position += width
        return torch.from_numpy(np.stack(rows))


class EpochLimitReached(RuntimeError):
    """The token pool ran out and repeating it was not explicitly allowed."""


def collect_route_counts(decisions, totals: dict[str, list[float]]) -> None:
    for layer_index, decision in enumerate(decisions):
        for organ, onehot in decision.hard_modes.items():
            counts = onehot.detach().float().sum(
                dim=tuple(range(onehot.ndim - 1))
            ).cpu().tolist()
            for key in (organ, f"{organ}@{layer_index}"):
                if key not in totals:
                    totals[key] = [0.0] * len(counts)
                totals[key] = [
                    a + float(b) for a, b in zip(totals[key], counts)
                ]


def normalised_route_usage(counts: dict[str, list[float]]) -> dict[str, list[float]]:
    result = {}
    for organ, values in counts.items():
        total = sum(values)
        result[organ] = [value / total if total else 0.0 for value in values]
    return result


def build_static_patterns(
    layer_mode_counts: dict[str, list[float]], pattern_length: int = 256
) -> dict[str, tuple[int, ...]]:
    """Convert validation mode ratios to deterministic content-free patterns."""
    if pattern_length < 1:
        raise ValueError("pattern_length must be positive")
    patterns = {}
    for key, values in layer_mode_counts.items():
        if "@" not in key:
            continue
        total = sum(values)
        if total <= 0:
            continue
        targets = [value / total for value in values]
        used = [0] * len(values)
        pattern = []
        for position in range(pattern_length):
            # Largest deficit creates a deterministic, approximately even
            # distribution instead of clumping each mode into one interval.
            mode = max(
                range(len(values)),
                key=lambda index: targets[index] * (position + 1) - used[index],
            )
            pattern.append(mode)
            used[mode] += 1
        patterns[key] = tuple(pattern)
    return patterns


def ffn_gradient_band_norms(
    model: K3MiniForCausalLM, tiers: tuple[int, ...]
) -> dict[str, float]:
    result = {}
    start = 0
    for end in tiers:
        squared = 0.0
        for layer_index, layer in enumerate(model.model.layers):
            layer_squared = 0.0
            for grad in (
                layer.ffn.gate_proj.weight.grad,
                layer.ffn.up_proj.weight.grad,
            ):
                if grad is not None:
                    layer_squared += float(grad[start:end].float().square().sum())
            grad = layer.ffn.down_proj.weight.grad
            if grad is not None:
                layer_squared += float(grad[:, start:end].float().square().sum())
            squared += layer_squared
            result[f"layer_{layer_index}/{start}:{end}"] = layer_squared ** 0.5
        result[f"all/{start}:{end}"] = squared ** 0.5
        start = end
    return result


def parameter_groups(model: K3MiniForCausalLM, weight_decay: float) -> list[dict]:
    """Split the parameters that must not be shrunk from the matrices that may be.

    Weight decay on a matrix is a prior towards a smaller map. On the tensors
    below it is simply wrong:

    * `A_log` and `dt_bias` set the KDA decay timescale. Decaying them drags the
      forgetting rate towards a fixed point that has nothing to do with the loss.
    * RMSNorm weights carry the migrated 82M scales (layer 0's mean is 0.9024).
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


def build_optimizer(model: K3MiniForCausalLM, train_cfg: TrainConfig, kind: str):
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


def checkpoint_paths(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("step_*.pt"))


def retain_latest(run_dir: Path, keep: int) -> None:
    paths = checkpoint_paths(run_dir)
    for path in paths[:-keep]:
        path.unlink()


def save_checkpoint(
    path: Path,
    model: K3MiniForCausalLM,
    optimizer: torch.optim.Optimizer,
    step: int,
    tokens_done: int,
    data_position: int,
    model_config: K3MiniConfig,
    train_config: TrainConfig,
    manifest_hash: str,
    joint_route_state: dict | None = None,
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
        "joint_route_state": joint_route_state or {"price": 0.0, "cost_ema": None},
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_observation(
    run_dir: Path, model: K3MiniForCausalLM, model_config: K3MiniConfig,
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
    model_cfg = K3MiniConfig()
    train_cfg = TrainConfig()
    # The vocabulary is a property of the pool, not of the model defaults: the
    # tokenizer changed with the data and the two must not drift apart.
    model_cfg.vocab_size = json.loads(
        (Path(args.data_dir) / "manifest.json").read_text())["vocab_size"]
    model_cfg.tie_word_embeddings = True

    # `dense` builds no controller at all.  `base` and `adaptive` keep the
    # supernet and differ only in force_fixed, for reproducing the routed runs.
    arm = getattr(args, "arm", "dense")
    model_cfg.joint_route.enabled = arm != "dense"
    model_cfg.joint_route.force_fixed = arm == "base"
    if getattr(args, "seed", None) is not None:
        train_cfg.seed = args.seed
    if getattr(args, "target_tokens", None):
        train_cfg.target_tokens = args.target_tokens
    if getattr(args, "schedule_tokens", None):
        train_cfg.schedule_tokens = args.schedule_tokens
    train_cfg.optimizer = getattr(args, "optimizer", "adamw")
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

    data_dir = Path(args.data_dir)
    manifest_path = data_dir / "manifest.json"
    train_path = data_dir / "train.bin"
    manifest_hash = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest["tokenizer"]["vocab_size"] != model_cfg.vocab_size:
        raise ValueError("dataset tokenizer vocabulary does not match model vocabulary")

    random.seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)
    torch.manual_seed(train_cfg.seed)
    model = K3MiniForCausalLM(model_cfg).to(device)
    if getattr(args, "init_from", None):
        # A 110M init produced by migrate_82m_to_110m.py.  Both arms load the
        # same file, so they start from identical weights and differ only in
        # force_fixed -- the widening, cloning and damping already happened
        # during migration and must not be redone per arm.
        blob = torch.load(args.init_from, map_location="cpu", weights_only=False)
        state = blob.get("model", blob) if isinstance(blob, dict) else blob
        missing, unexpected = model.load_state_dict(state, strict=False)
        model.to(device)
        print(f"[init] loaded migrated 110M weights from {args.init_from}: "
              f"{len(state)} tensors, {len(missing)} missing, "
              f"{len(unexpected)} unexpected", flush=True)
    optimizer = build_optimizer(model, train_cfg, train_cfg.optimizer)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    step = tokens_done = data_position = 0
    route_price = model_cfg.joint_route.price
    cost_ema = None
    cumulative_route_counts: dict[str, list[float]] = {}
    # Cumulative *executed* analytical cost, so the run can state what compute it
    # actually spent rather than what it was aiming at.  The predecessor
    # experiment was invalidated by a 1.3% gap between the two.
    cumulative_cost_sum = 0.0
    cumulative_cost_tokens = 0
    price_offset = 0.0
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
        saved_route = checkpoint.get("joint_route_state", {})
        route_price = float(saved_route.get("price", route_price))
        cost_ema = saved_route.get("cost_ema")
        cumulative_route_counts = saved_route.get("route_mode_counts", {})
        price_offset = float(saved_route.get("price_offset", 0.0))
        cumulative_cost_sum = float(saved_route.get("cumulative_cost_sum", 0.0))
        cumulative_cost_tokens = int(saved_route.get("cumulative_cost_tokens", 0))
        epochs_done = int(saved_route.get("epochs_done", 0))
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

    stream = SequentialTokenStream(
        train_path, train_cfg.sequence_length, train_cfg.micro_batch_size, data_position,
        max_epochs=args.max_epochs, epochs=epochs_done,
    )
    remaining = (len(stream.tokens) - data_position) + (
        (args.max_epochs - 1 - epochs_done) * len(stream.tokens)
        if args.max_epochs is not None else 0
    )
    wanted = train_cfg.target_tokens - tokens_done
    if args.max_epochs is not None and wanted > remaining:
        raise SystemExit(
            f"the target needs {wanted:,} more tokens but only {remaining:,} remain "
            f"within --max-epochs {args.max_epochs} of a {len(stream.tokens):,}-token "
            f"pool.  Raise --max-epochs to reuse the pool deliberately."
        )
    dtype = torch.bfloat16 if train_cfg.precision == "bf16" else torch.float32
    log_path = run_dir / "train.jsonl"

    model.train()
    while tokens_done < train_cfg.target_tokens:
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        cost_sum = 0.0
        ntp_sum = 0.0
        mtp_sum = 0.0
        step_route_counts: dict[str, list[float]] = {}
        # every micro-batch's decisions, so the price is solved from the whole
        # step rather than from whichever micro-batch happened to run last
        step_decisions: list[list] = []
        for _ in range(train_cfg.gradient_accumulation_steps):
            ids = stream.next().to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=dtype,
                enabled=(dtype != torch.float32 and device.type != "cpu"),
            ):
                # Train with the *deployed* policy.  Gumbel sampling trains one
                # policy and deploys another: at 82M the sampled policy sat on
                # budget while its argmax counterpart spent 5.3% more, because
                # argmax always takes the expensive side of a distribution that
                # sampling only visits sometimes.  Selecting by argmax with a
                # straight-through gradient makes the trained and deployed
                # policies the same object, so one price closes both.
                output = model(
                    ids, labels=ids,
                    route_state=RouteState(
                        price=route_price, hard=True,
                        force_fixed=model_cfg.joint_route.force_fixed,
                    ),
                )
                # Fixed-Full runs without a controller, so it has no cost to
                # constrain; its executed cost is the 1.0 normaliser by definition.
                if output.expected_cost is None:
                    loss = output.loss / train_cfg.gradient_accumulation_steps
                else:
                    gap = output.expected_cost - output.cost_target
                    budget_loss = (
                        route_price * gap
                        + 0.5 * train_cfg.budget_rho * torch.relu(gap).square()
                    )
                    loss = (output.loss + budget_loss) / train_cfg.gradient_accumulation_steps
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step + 1}")
            loss.backward()
            loss_sum += float(loss.detach())
            if output.ntp_loss is not None:
                ntp_sum += float(output.ntp_loss.detach()) / train_cfg.gradient_accumulation_steps
            if output.mtp_loss is not None:
                mtp_sum += float(output.mtp_loss.detach()) / train_cfg.gradient_accumulation_steps
            cost_sum += (
                1.0 if output.expected_cost is None
                else float(output.expected_cost.detach())
            ) / train_cfg.gradient_accumulation_steps
            collect_route_counts(output.joint_decisions, step_route_counts)
            step_decisions.append(output.joint_decisions)

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite gradient norm at step {step + 1}")
        ffn_gradients = ffn_gradient_band_norms(
            model, model_cfg.joint_route.ffn_width_tiers
        )
        # The rate for the tokens this step is about to consume, not the ones
        # already consumed: at `tokens_done` the very first step would run at
        # 1/warmup of the peak, which is 7.6e-12 -- a step thrown away.
        learning_rate = learning_rate_at(tokens_done + train_cfg.tokens_per_step,
                                         train_cfg)
        apply_learning_rate(optimizer, learning_rate, train_cfg)
        optimizer.step()
        for organ, values in step_route_counts.items():
            if organ not in cumulative_route_counts:
                cumulative_route_counts[organ] = [0.0] * len(values)
            cumulative_route_counts[organ] = [
                a + b for a, b in zip(cumulative_route_counts[organ], values)
            ]
        cost_ema = cost_sum if cost_ema is None else (
            train_cfg.cost_ema_decay * cost_ema
            + (1.0 - train_cfg.cost_ema_decay) * cost_sum
        )
        if (output.cost_target is not None and step_decisions[-1]
                and not model_cfg.joint_route.force_fixed):
            # Solve the common price for the batch just seen instead of letting a
            # dual variable crawl towards it: with reinvestment tiers available
            # the controller can overspend faster than slow ascent can answer,
            # and an unclosed budget voids the whole comparison.
            price_costs = OrganCosts(model_cfg, train_cfg.sequence_length)
            # The solver already targets the argmax policy, and training now
            # *is* the argmax policy, so no correction term is needed.  The
            # integral term that used to live here was driven by the sampled
            # cost and therefore pulled the price away from closing the budget
            # for the policy that actually gets deployed.
            route_price = solve_batch_price(
                merge_decisions(step_decisions), price_costs.fixed_share,
                float(output.cost_target), route_price,
            )
        step += 1
        tokens_done += train_cfg.tokens_per_step
        cumulative_cost_sum += cost_sum * train_cfg.tokens_per_step
        cumulative_cost_tokens += train_cfg.tokens_per_step
        cumulative_cost = cumulative_cost_sum / max(cumulative_cost_tokens, 1)
        target_cost = 1.0 if output.cost_target is None else float(output.cost_target)
        budget_error = cumulative_cost - target_cost

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
            "joint_analytical_cost": cost_sum,
            "joint_cost_target": target_cost,
            "joint_full_policy_cost": output.full_policy_cost,
            "joint_price": route_price,
            "joint_price_offset": price_offset,
            "joint_cost_ema": cost_ema,
            "joint_cumulative_cost": cumulative_cost,
            "joint_budget_error": budget_error,
            "joint_budget_valid": bool(abs(budget_error) <= BUDGET_TOLERANCE),
            "route_usage": normalised_route_usage(step_route_counts),
            "route_usage_cumulative": normalised_route_usage(cumulative_route_counts),
            "ffn_widths": list(model_cfg.joint_route.ffn_width_tiers),
            "ffn_gradient_l2_by_channel_band": ffn_gradients,
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
                    {
                        "price": route_price, "cost_ema": cost_ema,
                        "route_mode_counts": cumulative_route_counts,
                        "cumulative_cost_sum": cumulative_cost_sum,
                        "cumulative_cost_tokens": cumulative_cost_tokens,
                        "epochs_done": stream.epochs,
                    },
                )
                print(f"[milestone] {tag} at {tokens_done:,} tokens", flush=True)

        if step % train_cfg.checkpoint_every_steps == 0:
            path = run_dir / f"step_{step:08d}.pt"
            save_checkpoint(
                path, model, optimizer, step, tokens_done, stream.position,
                model_cfg, train_cfg, manifest_hash,
                {
                    "price": route_price,
                    "cost_ema": cost_ema,
                    "route_mode_counts": cumulative_route_counts,
                    "cumulative_cost_sum": cumulative_cost_sum,
                    "cumulative_cost_tokens": cumulative_cost_tokens,
                    "epochs_done": stream.epochs,
                    "price_offset": price_offset,
                },
            )
            retain_latest(run_dir, train_cfg.keep_latest_checkpoints)

    final = run_dir / f"step_{step:08d}.pt"
    save_checkpoint(
        final, model, optimizer, step, tokens_done, stream.position,
        model_cfg, train_cfg, manifest_hash,
        {
            "price": route_price,
            "cost_ema": cost_ema,
            "route_mode_counts": cumulative_route_counts,
            "cumulative_cost_sum": cumulative_cost_sum,
            "cumulative_cost_tokens": cumulative_cost_tokens,
            "epochs_done": stream.epochs,
            "price_offset": price_offset,
        },
    )
    retain_latest(run_dir, train_cfg.keep_latest_checkpoints)
    summary = {
        "tokens_done": tokens_done,
        "steps": step,
        "cumulative_executed_cost": cumulative_cost_sum / max(cumulative_cost_tokens, 1),
        "cost_target": target_cost,
        "budget_tolerance": BUDGET_TOLERANCE,
        "budget_valid": bool(abs(budget_error) <= BUDGET_TOLERANCE),
        "final_price": route_price,
        "cost_ema": cost_ema,
        "route_usage_cumulative": normalised_route_usage(cumulative_route_counts),
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    verdict = "VALID" if summary["budget_valid"] else "INVALID - compute not matched"
    print(f"[budget] cumulative executed cost = {summary['cumulative_executed_cost']:.5f} "
          f"(target {summary['cost_target']}) -> {verdict}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument(
        "--arm", choices=["dense", "base", "adaptive"], default="dense",
        help="`dense` is the deployed model: no controller, every layer at full "
             "capacity.  `base` and `adaptive` keep the routing supernet and "
             "differ only in force_fixed; they exist to reproduce the routed "
             "comparison, which measured slower than dense on this hardware.",
    )
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
        "--optimizer", choices=["adamw", "muon"], default="adamw",
        help="`muon` puts the 2-D hidden matrices on Muon and everything else "
             "on an auxiliary AdamW.  Still unvalidated on this model: run the "
             "short same-init comparison in HANDOFF.md section 6 first",
    )
    parser.add_argument("--muon-lr", type=float, default=None)
    parser.add_argument(
        "--warmup-fraction", type=float, default=None,
        help="fraction of --schedule-tokens spent warming up.  The 1%% default is "
             "sized for a full run; a short comparison needs a larger fraction or "
             "the arms spend most of their steps still ramping",
    )
    parser.add_argument("--micro-batch", type=int, default=None)
    parser.add_argument(
        "--booster-scale", type=float, default=0.1,
        help="initial scale of the reinvestment bands' output projection",
    )
    parser.add_argument(
        "--init-from", default=None,
        help="checkpoint (or state-dict .pt) of a Standalone Base whose weights "
             "are copied into the supernet's Base slice, so the two arms start "
             "from identical shared weights instead of a shared seed",
    )
    train(parser.parse_args())


if __name__ == "__main__":
    main()
