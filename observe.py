#!/usr/bin/env python
"""Read a model as it grows: per-domain NLL, continuous capability, samples.

Run this at every rung of the observation ladder (`runs/*/observations/`).  It
appends one row per snapshot to a single growth log, so the trajectory can be read
as a file rather than reassembled from scattered runs.

**Accuracy alone cannot tell you that something appeared.**  A discrete score can
jump the moment a smooth internal improvement crosses a threshold, which is why a
jump in accuracy is not evidence of emergence on its own.  So three continuous
quantities are recorded next to it, none of which need a benchmark:

* `nll` -- mean negative log-likelihood per token
* `margin` -- the correct token's logit minus the best *incorrect* token's logit,
  averaged.  This keeps moving while accuracy is pinned at zero or one, so a change
  of slope is visible before and after any threshold is crossed.
* `nll_p10` / `nll_p90` -- the easy and hard tails.  A model that is getting better
  at what it already knew moves differently from one that has started to handle
  what it could not.

Treat a change as real only if it shows in the continuous measures, persists across
adjacent rungs, and appears in more than one domain.

Generation passes `respect_documents=False`.  This is not cosmetic: with boundary
handling on, an `<|eod|>` inside a prompt masks the model off from its own context,
and the hidden state moves by 2.8 in a measured tiny model.  A continuation has no
document boundaries in it, so the handling must be off.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from config import KaiNomosConfig as Config
from model import KaiNomosForCausalLM as Model
from segments import mask_targets_at_boundaries
from train import manifest_split_paths

DEFAULT_PROMPTS = [
    "日本の首都は",
    "むかしむかし、あるところに",
    "水が沸騰する温度は",
    "次の文を英語に訳してください。「今日は雨が降っています。」",
    "1 + 2 + 3 + 4 + 5 =",
    "def fibonacci(n):",
    "人工知能とは、",
    "問: なぜ空は青いのですか。 答:",
]


def load(path: Path, device: str):
    """A weights-only observation snapshot, or a full resumable checkpoint."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    raw = blob.get("config") or blob["model_config"]
    cfg = Config.from_dict(raw)
    model = Model(cfg)
    model.load_state_dict(blob["model"])
    return model.to(device).eval(), cfg, blob.get("tokens_done", 0), blob.get("step", 0)


@torch.no_grad()
def measure(model, cfg, paths: Path | list[Path], max_tokens: int,
            batch: int, device: str) -> dict:
    """NLL, top-1 accuracy and margin over held-out token shards."""
    if isinstance(paths, Path):
        paths = [paths]
    seq = cfg.context_length_train
    total_nll = 0.0
    counted = 0
    correct = 0
    margin_sum = 0.0
    per_token = []
    for path in paths:
        tokens = np.memmap(path, dtype=np.uint16, mode="r")
        windows = min(
            max((max_tokens - counted) // seq, 0),
            (tokens.size - 1) // seq,
        )
        for start in range(0, windows, batch):
            rows = np.stack([
                np.asarray(tokens[i * seq:(i + 1) * seq + 1], dtype=np.int64)
                for i in range(start, min(start + batch, windows))
            ])
            ids = torch.from_numpy(rows).to(device)
            inputs, targets = ids[:, :-1], ids[:, 1:]
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device != "cpu"):
                out = model(inputs)
            logits = out.logits.float()
            targets = mask_targets_at_boundaries(
                targets, inputs, cfg.eod_token_id
            )
            valid = targets != -100
            nll = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                reduction="none", ignore_index=-100,
            ).reshape_as(targets)
            total_nll += float(nll.sum())
            counted += int(valid.sum())
            per_token.append(nll[valid].float().cpu().numpy())

            correct += int(((logits.argmax(-1) == targets) & valid).sum())
            # margin: correct logit minus the best non-correct logit.
            safe_targets = targets.masked_fill(~valid, 0)
            gold = logits.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
            masked = logits.scatter(
                -1, safe_targets.unsqueeze(-1), float("-inf")
            )
            margin_sum += float(
                ((gold - masked.max(-1).values) * valid).sum()
            )

    if counted < 1:
        return {}

    stacked = np.concatenate([p.ravel() for p in per_token])
    return {
        "nll": total_nll / counted,
        "perplexity": float(np.exp(min(total_nll / counted, 20.0))),
        "top1": correct / counted,
        "margin": margin_sum / counted,
        "nll_p10": float(np.percentile(stacked, 10)),
        "nll_p90": float(np.percentile(stacked, 90)),
        "tokens": counted,
    }


@torch.no_grad()
def sample(model, cfg, sp, prompts, steps: int, temperature: float, seed: int,
           device: str) -> list[dict]:
    """Fixed prompts, fixed seed, so two rungs differ only by the model."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    out = []
    for text in prompts:
        ids = torch.tensor([sp.encode(text)], device=device)
        for _ in range(steps):
            # `respect_documents=False`: see the module docstring.
            hidden, *_ = model.model(ids, respect_documents=False)
            logits = model.lm_head(hidden)[:, -1].float() / temperature
            probability = logits.softmax(-1).cpu()
            nxt = torch.multinomial(probability, 1, generator=generator)
            ids = torch.cat([ids, nxt.to(device)], dim=1)
        produced = ids[0].tolist()[len(sp.encode(text)):]
        out.append({"prompt": text, "continuation": sp.decode(produced)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", required=True, help="obs_*.pt or a step_*.pt")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument(
        "--domains", default=None,
        help="JSON mapping domain -> validation .bin. Without it, validation "
             "shards are read from --data-dir/manifest.json as domain 'all'",
    )
    ap.add_argument("--data-dir", default="data/pool")
    ap.add_argument("--max-tokens", type=int, default=500_000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--gen-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--prompts", default=None, help="one prompt per line")
    ap.add_argument("--log", default="runs/growth_log.jsonl")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    model, cfg, tokens_done, step = load(Path(args.snapshot), args.device)
    if sp.get_piece_size() != cfg.vocab_size:
        raise SystemExit(
            f"tokenizer has {sp.get_piece_size()} pieces but the snapshot was "
            f"trained with {cfg.vocab_size}; the samples would be nonsense"
        )

    if args.domains:
        domains = {k: Path(v) for k, v in json.loads(Path(args.domains).read_text()).items()}
    else:
        data_dir = Path(args.data_dir)
        manifest = json.loads((data_dir / "manifest.json").read_text())
        domains = {
            "all": manifest_split_paths(manifest, data_dir, "validation")
        }

    started = time.time()
    scores = {}
    for name, paths in domains.items():
        check_paths = [paths] if isinstance(paths, Path) else paths
        missing = [path for path in check_paths if not path.exists()]
        if missing:
            print(f"  [skip] {name}: {missing[0]} missing")
            continue
        scores[name] = measure(
            model, cfg, check_paths, args.max_tokens, args.batch, args.device
        )
        row = scores[name]
        print(f"  {name:26s} nll {row['nll']:.4f}  ppl {row['perplexity']:8.2f}  "
              f"top1 {row['top1']*100:5.2f}%  margin {row['margin']:+7.3f}")

    prompts = (Path(args.prompts).read_text().splitlines()
               if args.prompts else DEFAULT_PROMPTS)
    samples = sample(model, cfg, sp, [p for p in prompts if p.strip()],
                     args.gen_tokens, args.temperature, args.seed, args.device)

    record = {
        "snapshot": str(args.snapshot),
        "tokens_done": tokens_done,
        "step": step,
        "params": sum(p.numel() for p in model.parameters()),
        "vocab_size": cfg.vocab_size,
        "domains": scores,
        "samples": samples,
        "sample_seed": args.seed,
        "sample_temperature": args.temperature,
        "measured_seconds": round(time.time() - started, 1),
    }
    log = Path(args.log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"\nappended to {log}   ({tokens_done:,} tokens)")
    for entry in samples[:3]:
        print(f"  {entry['prompt']!r} -> {entry['continuation']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
