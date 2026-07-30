#!/usr/bin/env python
"""Validation / test NLL for KaiNomos-747M."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from config import KaiNomosConfig
from model import KaiNomosForCausalLM
from segments import mask_targets_at_boundaries
from train import manifest_split_paths


@torch.no_grad()
def evaluate(model, bin_paths: list[Path], seq_len: int, max_tokens: int,
             batch_size: int = 4, device: str = "cuda") -> dict:
    model.eval()
    total, counted = 0.0, 0
    for bin_path in bin_paths:
        tokens = np.memmap(bin_path, dtype=np.uint16, mode="r")
        remaining = max_tokens - counted
        n = min(remaining // seq_len, (tokens.size - 1) // seq_len)
        if n <= 0:
            break
        for first in range(0, n, batch_size):
            last = min(first + batch_size, n)
            windows = np.stack([
                np.asarray(
                    tokens[i * seq_len:(i + 1) * seq_len + 1],
                    dtype=np.int64,
                )
                for i in range(first, last)
            ])
            batch = torch.from_numpy(windows).to(device)
            ids, targets = batch[:, :-1], batch[:, 1:]
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device != "cpu"):
                out = model(ids)
            targets = mask_targets_at_boundaries(
                targets, ids, model.config.eod_token_id
            )
            loss = F.cross_entropy(
                out.logits.reshape(-1, out.logits.shape[-1]).float(),
                targets.reshape(-1), reduction="sum", ignore_index=-100,
            )
            total += float(loss)
            counted += int((targets != -100).sum())

    nll = total / max(counted, 1)
    return {
        "context": seq_len,
        "nll": nll,
        "perplexity": float(np.exp(min(nll, 20.0))),
        "eval_tokens": counted,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--splits", default="validation,test")
    ap.add_argument("--contexts", default="1024")
    ap.add_argument("--max-tokens", type=int, default=2_000_000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = KaiNomosConfig.from_dict(blob.get("config") or blob["model_config"])
    model = KaiNomosForCausalLM(cfg).to(args.device)
    model.load_state_dict(blob["model"])

    data_dir = Path(args.data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    results = [
        evaluate(model, manifest_split_paths(manifest, data_dir, split), int(ctx),
                 args.max_tokens, device=args.device)
        | {"split": split}
        for split in args.splits.split(",")
        for ctx in args.contexts.split(",")
    ]
    print(json.dumps(results, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
