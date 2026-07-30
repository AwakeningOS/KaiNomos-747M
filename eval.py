#!/usr/bin/env python
"""Validation / test NLL for KaiNomos-110M.

NLL is measured together with the executed compute in the same forward pass: a
routing decision moves both at once, so a cost measured separately belongs to a
different policy than the loss it is paired with.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from config import K3MiniPlusPlusPlusConfig
from joint_router import RouteState
from model import K3MiniPlusPlusPlusForCausalLM


@torch.no_grad()
def evaluate(model, bin_path: Path, seq_len: int, max_tokens: int,
             batch_size: int = 4, device: str = "cuda", force_fixed: bool = False,
             price: float = 0.0) -> dict:
    """`price` must be the price the checkpoint was trained under.

    The controller selects `argmax(logits - price * unit_cost)`, so the price is
    part of the policy, not a training-time detail.  Evaluating at the default 0.0
    measures a *different* policy from the one that was trained: every priced term
    vanishes and the controller reads off unpriced logits, which is why both the
    NLL and the compute ratio then belong to a model that was never trained.
    """
    model.eval()
    tokens = np.memmap(bin_path, dtype=np.uint16, mode="r")
    n = min(max_tokens // seq_len, (tokens.size - 1) // seq_len)
    windows = np.stack([np.asarray(tokens[i * seq_len:(i + 1) * seq_len + 1], dtype=np.int64)
                        for i in range(n)])

    total, counted, costs = 0.0, 0, []
    for i in range(0, len(windows), batch_size):
        batch = torch.from_numpy(windows[i:i + batch_size]).to(device)
        ids, targets = batch[:, :-1], batch[:, 1:]
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device != "cpu"):
            out = model(ids, route_state=RouteState(
                price=price, hard=True, force_fixed=force_fixed))
        loss = F.cross_entropy(out.logits.reshape(-1, out.logits.shape[-1]).float(),
                               targets.reshape(-1), reduction="sum")
        total += float(loss)
        counted += int(targets.numel())
        costs.append(float(out.expected_cost) / out.cost_target)

    nll = total / max(counted, 1)
    return {
        "context": seq_len,
        "arm": "base" if force_fixed else "adaptive",
        "nll": nll,
        "perplexity": float(np.exp(min(nll, 20.0))),
        "compute_ratio": float(np.mean(costs)),
        "route_price": price,
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
    ap.add_argument("--force-fixed", action="store_true")
    ap.add_argument(
        "--price", type=float, default=None,
        help="override the route price; by default the price stored in the "
             "checkpoint is used, which is the one the policy was trained under",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = K3MiniPlusPlusPlusConfig.from_dict(blob.get("config") or blob["model_config"])
    model = K3MiniPlusPlusPlusForCausalLM(cfg).to(args.device)
    model.load_state_dict(blob["model"])

    # One price for every split.  Re-solving per split would let validation and
    # test be scored under two different policies, and the test number would then
    # not describe the model that validation selected.
    if args.price is not None:
        price = args.price
    else:
        price = float((blob.get("joint_route_state") or {}).get("price", 0.0))
    print(f"[eval] route price = {price:.6f}"
          f"{' (from checkpoint)' if args.price is None else ' (overridden)'}")

    results = [
        evaluate(model, Path(args.data_dir) / f"{split}.bin", int(ctx),
                 args.max_tokens, device=args.device, force_fixed=args.force_fixed,
                 price=price)
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
