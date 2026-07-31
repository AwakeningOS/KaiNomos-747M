#!/usr/bin/env python
"""Snapshot-only observations for KaiNomos-750M.

This module is deliberately not part of the training step.  A caller runs it on
a frozen observation snapshot and a small, fixed diagnostic batch.  Validation
NLL is measured per source and combined with frozen source weights; architecture
internals and optimizer state are reduced to JSON-safe statistics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Iterable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from config import KaiNomosConfig as Config
from generation import generate
from kda import KDAttention, lower_bounded_log_decay
from mla import GatedMLA
from model import KaiNomosForCausalLM as Model
from segments import mask_targets_at_boundaries

SCHEMA_VERSION = "kainomos_750m_observation_v1"
DEFAULT_PROMPTS = (
    "日本の首都は",
    "むかしむかし、あるところに",
    "1 + 2 + 3 + 4 + 5 =",
    "def fibonacci(n):",
)


@dataclass(frozen=True)
class ValidationSource:
    paths: tuple[Path, ...]
    weight: float = 1.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_stats(value: torch.Tensor | None) -> dict | None:
    if value is None:
        return None
    data = value.detach().float().reshape(-1)
    finite = data[torch.isfinite(data)]
    if not finite.numel():
        return {"numel": data.numel(), "finite_fraction": 0.0}
    quantiles = torch.quantile(
        finite,
        torch.tensor([0.01, 0.1, 0.5, 0.9, 0.99], device=finite.device),
    )
    return {
        "numel": data.numel(),
        "finite_fraction": finite.numel() / max(data.numel(), 1),
        "mean": float(finite.mean()),
        "std": float(finite.std(unbiased=False)),
        "rms": float(finite.square().mean().sqrt()),
        "min": float(finite.min()),
        "p01": float(quantiles[0]),
        "p10": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p90": float(quantiles[3]),
        "p99": float(quantiles[4]),
        "max": float(finite.max()),
    }


def _gate_stats(value: torch.Tensor) -> dict:
    result = _tensor_stats(value)
    result["saturation_low"] = float((value < 0.01).float().mean())
    result["saturation_high"] = float((value > 0.99).float().mean())
    return result


@torch.no_grad()
def _kda_document_reset_stats(module: KDAttention, x, segments) -> dict:
    """Replay a small diagnostic batch and expose state reset effectiveness."""
    if segments is None or x.shape[1] < 2:
        return {"boundaries": 0}
    q, k, v, _ = module._project(x, None, segments)
    q = F.normalize(q.float(), dim=-1, eps=module.cfg.l2_eps)
    k = F.normalize(k.float(), dim=-1, eps=module.cfg.l2_eps)
    decay = lower_bounded_log_decay(
        module._raw_decay(x), module.A_log, module.dt_bias,
        module.cfg.gate_lower_bound,
    )
    beta = module.beta_proj(x).float().sigmoid()
    state = q.new_zeros(
        q.shape[0], q.shape[2], q.shape[3], v.shape[-1], dtype=torch.float32
    )
    before = []
    after_reset = []
    after_first = []
    boundary_count = 0
    for position in range(x.shape[1]):
        reset = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        if position:
            reset = segments[:, position] != segments[:, position - 1]
        if reset.any():
            boundary_count += int(reset.sum())
            selected = state[reset]
            before.append(selected.square().mean(dim=(-3, -2, -1)).sqrt())
            state[reset] = 0
            after_reset.append(
                state[reset].square().mean(dim=(-3, -2, -1)).sqrt()
            )
        state = state * decay[:, position].exp().unsqueeze(-1)
        key = k[:, position]
        predicted = (key.unsqueeze(-1) * state).sum(-2)
        error = v[:, position].float() - predicted
        update = (
            beta[:, position].unsqueeze(-1) * key
        ).unsqueeze(-1) * error.unsqueeze(-2)
        state = state + update
        if reset.any():
            after_first.append(
                state[reset].square().mean(dim=(-3, -2, -1)).sqrt()
            )
    return {
        "boundaries": boundary_count,
        "state_rms_before_reset": _tensor_stats(
            torch.cat(before) if before else None
        ),
        "state_rms_after_reset": _tensor_stats(
            torch.cat(after_reset) if after_reset else None
        ),
        "state_rms_after_first_token": _tensor_stats(
            torch.cat(after_first) if after_first else None
        ),
    }


def load(path: Path, device: str):
    """Load either a weights-only observation or a resumable checkpoint."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    cfg = Config.from_dict(blob.get("config") or blob["model_config"])
    model = Model(cfg)
    model.load_state_dict(blob["model"])
    return model.to(device).eval(), cfg, blob


@torch.no_grad()
def measure_source(model, cfg, paths: Path | Iterable[Path], max_tokens: int,
                   batch: int, device: str) -> dict:
    """Fixed-source NLL, accuracy, and correct-token margin."""
    if isinstance(paths, Path):
        paths = (paths,)
    seq = cfg.context_length_train
    total_nll = margin_sum = 0.0
    counted = correct = positions_read = 0
    tails: list[np.ndarray] = []
    for path in paths:
        tokens = np.memmap(path, dtype=np.uint16, mode="r")
        windows = min(max((max_tokens - positions_read) // seq, 0),
                      (tokens.size - 1) // seq)
        for start in range(0, windows, batch):
            end = min(start + batch, windows)
            rows = np.stack([
                np.asarray(tokens[i * seq:(i + 1) * seq + 1], dtype=np.int64)
                for i in range(start, end)
            ])
            positions_read += (end - start) * seq
            ids = torch.from_numpy(rows).to(device)
            inputs, targets = ids[:, :-1], ids[:, 1:]
            with torch.autocast(device_type=torch.device(device).type,
                                dtype=torch.bfloat16,
                                enabled=torch.device(device).type != "cpu"):
                logits = model(inputs).logits.float()
            targets = mask_targets_at_boundaries(targets, inputs, cfg.eod_token_id)
            valid = targets != -100
            losses = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                reduction="none", ignore_index=-100,
            ).reshape_as(targets)
            total_nll += float(losses.sum())
            counted += int(valid.sum())
            tails.append(losses[valid].cpu().numpy())
            correct += int(((logits.argmax(-1) == targets) & valid).sum())
            safe = targets.masked_fill(~valid, 0)
            gold = logits.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
            best_other = logits.scatter(-1, safe.unsqueeze(-1), float("-inf")).max(-1).values
            margin_sum += float(((gold - best_other) * valid).sum())
    if not counted:
        return {}
    token_losses = np.concatenate(tails)
    nll = total_nll / counted
    return {
        "nll": nll,
        "perplexity": float(np.exp(min(nll, 20.0))),
        "top1_accuracy": correct / counted,
        "correct_token_margin": margin_sum / counted,
        "nll_p10": float(np.percentile(token_losses, 10)),
        "nll_p90": float(np.percentile(token_losses, 90)),
        "tokens": counted,
    }


@torch.no_grad()
def measure_fixed_validation(model, cfg, sources: dict[str, ValidationSource],
                             max_tokens: int, batch: int, device: str) -> dict:
    """Measure every source separately, then apply frozen source weights."""
    rows: dict[str, dict] = {}
    for name, source in sources.items():
        metrics = measure_source(model, cfg, source.paths, max_tokens, batch, device)
        if not metrics:
            continue
        rows[name] = {
            "weight": float(source.weight),
            "files": [
                {"path": str(path), "sha256": _sha256(path)} for path in source.paths
            ],
            **metrics,
        }
    weight_sum = sum(row["weight"] for row in rows.values())
    if weight_sum <= 0:
        raise ValueError("fixed validation source weights must sum to a positive value")
    fields = ("nll", "top1_accuracy", "correct_token_margin", "nll_p10", "nll_p90")
    weighted = {
        field: sum(row["weight"] * row[field] for row in rows.values()) / weight_sum
        for field in fields
    }
    weighted["perplexity"] = math.exp(min(weighted["nll"], 20.0))
    weighted["weight_sum"] = weight_sum
    weighted["tokens"] = sum(row["tokens"] for row in rows.values())
    return {"sources": rows, "weighted": weighted, "max_tokens_per_source": max_tokens}


class ArchitectureDiagnostics(AbstractContextManager):
    """Temporary hooks used only for an explicit observation pass."""

    def __init__(self, model: Model):
        self.model = model
        self.kda: list[dict] = []
        self.mla: list[dict] = []
        self._handles = []
        names = {id(module): name for name, module in model.named_modules()}
        for module in model.modules():
            if isinstance(module, KDAttention):
                self._handles.append(module.register_forward_hook(
                    self._kda_hook(names[id(module)]), with_kwargs=True
                ))
            elif isinstance(module, GatedMLA):
                self._handles.append(module.register_forward_hook(
                    self._mla_hook(names[id(module)]), with_kwargs=True
                ))

    def __exit__(self, exc_type, exc, traceback):
        for handle in self._handles:
            handle.remove()
        return False

    def _kda_hook(self, name):
        def hook(module, args, kwargs, output):
            x = args[0] if args else kwargs["x"]
            raw_decay = module._raw_decay(x)
            log_decay = lower_bounded_log_decay(
                raw_decay, module.A_log, module.dt_bias, module.cfg.gate_lower_bound
            )
            cache = output[1]
            self.kda.append({
                "module": name,
                "retention": _tensor_stats(log_decay.exp()),
                "beta": _tensor_stats(module.beta_proj(x).float().sigmoid()),
                "state": _tensor_stats(None if cache is None else cache.recurrent_state),
                "output_gate": _gate_stats(module.output_gate(x).float().sigmoid()),
                "document_reset": _kda_document_reset_stats(
                    module, x, kwargs.get("segments")
                ),
            })
        return hook

    def _mla_hook(self, name):
        def hook(module, args, kwargs, output):
            x = args[0] if args else kwargs["x"]
            q = module.q_norm(module.project_q(x))
            current_latent = module.project_latent(x)
            k, _ = module.expand_kv(current_latent)
            k = module.k_norm(k)
            logits = torch.einsum("bhtd,bhsd->bhts", q.float(), k.float()) * module.scale
            causal = torch.ones(logits.shape[-2:], dtype=torch.bool,
                                device=logits.device).tril()
            cache = output[1]
            self.mla.append({
                "module": name,
                "q": _tensor_stats(q),
                "k": _tensor_stats(k),
                "attention_logits": _tensor_stats(logits[..., causal]),
                "output_gate": _gate_stats(module.output_gate(x).float().sigmoid()),
                "latent": _tensor_stats(
                    current_latent if cache is None else cache.latent
                ),
            })
        return hook


@torch.no_grad()
def collect_architecture_diagnostics(model: Model, input_ids: torch.Tensor) -> dict:
    """One opt-in diagnostic forward; never called by the training loop."""
    with ArchitectureDiagnostics(model) as diagnostics:
        output = model(input_ids, use_cache=True, return_route_stats=True)
    return {
        "delta_routes": output.route_stats,
        "kda": diagnostics.kda,
        "mla": diagnostics.mla,
    }


def summarize_optimizer_state(optimizer_state: dict | None) -> dict | None:
    """Reduce a serialized optimizer state to bounded, JSON-safe statistics."""
    if not optimizer_state:
        return None
    groups = []
    for group in optimizer_state.get("param_groups", []):
        groups.append({
            key: group[key] for key in (
                "lr", "adamw_lr", "base_lr", "weight_decay", "use_muon",
                "momentum", "nesterov", "ns_steps", "update_rms",
            ) if key in group
        } | {"parameter_count": len(group.get("params", ()))})
    accumulators: dict[str, dict] = {}
    scalar_steps: list[float] = []
    for state in optimizer_state.get("state", {}).values():
        for name, value in state.items():
            if torch.is_tensor(value):
                data = value.detach().float().reshape(-1)
                finite = data[torch.isfinite(data)]
                row = accumulators.setdefault(name, {
                    "tensor_count": 0, "numel": 0, "finite": 0,
                    "sum_squares": 0.0, "max_abs": 0.0,
                })
                row["tensor_count"] += 1
                row["numel"] += data.numel()
                row["finite"] += finite.numel()
                if finite.numel():
                    row["sum_squares"] += float(finite.square().sum())
                    row["max_abs"] = max(row["max_abs"], float(finite.abs().max()))
            elif name == "step":
                scalar_steps.append(float(value))
    tensors = {}
    for name, row in accumulators.items():
        finite = row.pop("finite")
        squares = row.pop("sum_squares")
        tensors[name] = {
            **row,
            "finite_fraction": finite / max(row["numel"], 1),
            "rms": math.sqrt(squares / max(finite, 1)),
        }
    return {
        "groups": groups,
        "state_entries": len(optimizer_state.get("state", {})),
        "step_min": min(scalar_steps) if scalar_steps else None,
        "step_max": max(scalar_steps) if scalar_steps else None,
        "tensors": tensors,
    }


@torch.no_grad()
def sample(model, sp, prompts, steps: int, temperature: float, seed: int,
           device: str) -> list[dict]:
    """Generate via the cache-stable public generation API."""
    generator = torch.Generator(device=torch.device(device).type).manual_seed(seed)
    rows = []
    for prompt in prompts:
        prompt_ids = sp.encode(prompt)
        ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        output = generate(model, ids, steps, temperature=temperature, generator=generator)
        rows.append({
            "prompt": prompt,
            "continuation": sp.decode(output[0, len(prompt_ids):].tolist()),
        })
    return rows


def _parse_sources(path: Path) -> dict[str, ValidationSource]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw = raw.get("sources", raw)
    result = {}
    for name, value in raw.items():
        if isinstance(value, str):
            result[name] = ValidationSource((Path(value),))
        else:
            paths = value.get("paths") or [value["path"]]
            result[name] = ValidationSource(
                tuple(Path(item) for item in paths), float(value.get("weight", 1.0))
            )
    return result


def _manifest_validation_sources(manifest: dict, root: Path) -> dict[str, ValidationSource]:
    """Read the KaiNomos manifest directly without depending on the training CLI."""
    declared = manifest.get("splits", {}).get("validation", {})
    grouped: dict[str, list[Path]] = {}
    for shard in declared.get("shards", ()):
        relative = str(shard["path"] if isinstance(shard, dict) else shard)
        path = (root / relative).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as error:
            raise ValueError(f"validation shard escapes data directory: {relative}") from error
        source = (
            str(shard.get("source_id") or shard.get("source") or "all")
            if isinstance(shard, dict) else "all"
        )
        grouped.setdefault(source, []).append(path)
    weights = manifest.get("validation_source_weights", {})
    return {
        name: ValidationSource(tuple(paths), float(weights.get(name, 1.0)))
        for name, paths in grouped.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--domains", default=None,
                        help="JSON source map; each source may define paths and weight")
    parser.add_argument("--data-dir", default="data/pool")
    parser.add_argument("--max-tokens", type=int, default=500_000)
    parser.add_argument("--diagnostic-tokens", type=int, default=128)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--gen-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--prompts", default=None)
    parser.add_argument("--log", default="runs/kainomos_750m_growth.jsonl")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    model, cfg, blob = load(Path(args.snapshot), args.device)
    if sp.get_piece_size() != cfg.vocab_size:
        raise SystemExit("tokenizer vocabulary does not match the snapshot")
    if args.domains:
        sources = _parse_sources(Path(args.domains))
    else:
        root = Path(args.data_dir)
        manifest = json.loads((root / "manifest.json").read_text())
        sources = _manifest_validation_sources(manifest, root)
        if not sources:
            raise SystemExit("manifest has no validation shards")
    started = time.time()
    validation = measure_fixed_validation(
        model, cfg, sources, args.max_tokens, args.batch, args.device
    )
    first_path = next(iter(sources.values())).paths[0]
    diagnostic = np.memmap(first_path, dtype=np.uint16, mode="r")
    diagnostic_ids = torch.from_numpy(np.asarray(
        diagnostic[:args.diagnostic_tokens], dtype=np.int64
    )[None]).to(args.device)
    architecture = collect_architecture_diagnostics(model, diagnostic_ids)
    prompts = (Path(args.prompts).read_text().splitlines()
               if args.prompts else DEFAULT_PROMPTS)
    samples = sample(model, sp, [p for p in prompts if p.strip()], args.gen_tokens,
                     args.temperature, args.seed, args.device)
    record = {
        "schema_version": SCHEMA_VERSION,
        "snapshot": {
            "path": str(args.snapshot),
            "step": int(blob.get("step", 0)),
            "tokens_done": int(blob.get("tokens_done", 0)),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "architecture_id": cfg.architecture_id,
        },
        "fixed_validation": validation,
        "architecture": architecture,
        "optimizer": summarize_optimizer_state(blob.get("optimizer")),
        "generation": {
            "seed": args.seed, "temperature": args.temperature, "samples": samples,
        },
        "measured_seconds": round(time.time() - started, 3),
    }
    log = Path(args.log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(record["fixed_validation"]["weighted"], indent=2))
    print(f"appended observation to {log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
