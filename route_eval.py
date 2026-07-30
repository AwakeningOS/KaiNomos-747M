"""Evaluation for the core proposition: NLL paired with executed cost.

    Does input-dependent joint execution pruning keep language performance at a
    lower inference compute than fixed maximal execution?

Every NLL is recorded together with the analytical cost that produced it, in the
same forward pass -- measuring them separately is what made the predecessor
experiment uninterpretable, because a routing decision changes both at once.

The permutation control answers the input-dependence half.  Routes are shuffled
across samples within the same (layer, organ, position), so the number of samples
choosing each mode at each decision site -- and therefore the analytical cost --
is unchanged; only which input receives the expensive computation is destroyed.

    NLL unchanged   -> a better average fixed policy was found
    NLL degrades    -> the per-input allocation itself carries the value
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from joint_router import RouteDecision, RouteState


def permute_routes(
    per_layer: list[dict[str, torch.Tensor]], generator: torch.Generator | None = None
) -> list[dict[str, torch.Tensor]]:
    """Shuffle the sample axis independently at every (layer, organ, position)."""
    out = []
    for layer in per_layer:
        new_layer = {}
        for organ, oh in layer.items():
            if oh.dim() < 3 or oh.shape[0] < 2:
                new_layer[organ] = oh
                continue
            b, p = oh.shape[0], oh.shape[1]
            noise = torch.rand(b, p, device=oh.device, generator=generator)
            idx = noise.argsort(dim=0).unsqueeze(-1).expand_as(oh)
            new_layer[organ] = oh.gather(0, idx)
        out.append(new_layer)
    return out


def _windows(bin_path: Path, seq_len: int, max_tokens: int) -> np.ndarray:
    tokens = np.memmap(bin_path, dtype=np.uint16, mode="r")
    n = min(max_tokens // seq_len, (tokens.size - 1) // seq_len)
    return np.stack([np.asarray(tokens[i * seq_len:(i + 1) * seq_len + 1], dtype=np.int64)
                     for i in range(n)])


@torch.no_grad()
def evaluate_nll_and_cost(
    model,
    bin_path: Path,
    seq_len: int,
    max_tokens: int,
    batch_size: int = 8,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    routed: bool = True,
    permute: bool = False,
    seed: int = 0,
    collect_routes: bool = False,
    price: float = 0.0,
) -> dict:
    """`price` must be the price the checkpoint was trained under; see eval.py."""
    model.eval()
    wins = _windows(bin_path, seq_len, max_tokens)
    gen = torch.Generator(device=device).manual_seed(seed) if permute else None

    total_nll, total_tokens, costs = 0.0, 0, []
    records: list[dict] = []

    for i in range(0, len(wins), batch_size):
        batch = torch.from_numpy(wins[i:i + batch_size]).to(device)
        ids, tgt = batch[:, :-1], batch[:, 1:]

        state = RouteState(price=price, hard=True) if routed else None
        if routed and (permute or collect_routes):
            with torch.autocast(device_type=device, dtype=dtype, enabled=device != "cpu"):
                probe = model(ids, route_state=RouteState(price=price, hard=True))
            per_layer = [d.hard_modes for d in probe.joint_decisions]
            if permute:
                per_layer = permute_routes(per_layer, gen)
                state = RouteState(price=price, hard=True, route_override=per_layer)
            if collect_routes:
                records.append({f"L{li:02d}_{o}": oh.argmax(-1).to(torch.int8).cpu().numpy()
                                for li, layer in enumerate(per_layer)
                                for o, oh in layer.items()})

        with torch.autocast(device_type=device, dtype=dtype, enabled=device != "cpu"):
            out = model(ids, labels=tgt, route_state=state) if routed else model(ids, labels=tgt)

        # labels are the shifted targets, so no further shift here
        ls = F.cross_entropy(out.logits.reshape(-1, out.logits.shape[-1]).float(),
                             tgt.reshape(-1), reduction="sum")
        total_nll += float(ls)
        total_tokens += int(tgt.numel())
        costs.append(1.0 if out.expected_cost is None else float(out.expected_cost))

    nll = total_nll / max(total_tokens, 1)
    res = {
        "context": seq_len,
        "condition": "route_permuted" if permute else ("dynamic" if routed else "fixed_full"),
        "nll": nll,
        "perplexity": float(np.exp(min(nll, 20.0))),
        "executed_cost_over_fixed": float(np.mean(costs)),
        "executed_cost_std": float(np.std(costs)),
        "route_price": price,
        "eval_tokens": total_tokens,
        "windows": int(len(wins)),
    }
    if collect_routes:
        res["_records"] = records
    return res


def save_routes_npz(records: list[dict], path: Path, meta: dict) -> None:
    """Mode indices per (layer, organ), shaped [sample, position]."""
    if not records:
        return
    keys = [k for k in records[0] if not k.startswith("_")]
    arrays = {k: np.concatenate([r[k] for r in records], axis=0) for k in keys}
    arrays["meta_json"] = np.array(json.dumps(meta))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def route_usage(npz_path: Path, mode_counts: dict[str, int]) -> dict:
    z = np.load(npz_path, allow_pickle=False)
    out: dict = {}
    for key in z.files:
        if key == "meta_json" or "_" not in key:
            continue
        layer, organ = key.split("_")
        counts = np.bincount(z[key].ravel(), minlength=mode_counts[organ]).astype(float)
        out.setdefault(layer, {})[organ] = (counts / counts.sum()).tolist()
    return out


__all__ = ["permute_routes", "evaluate_nll_and_cost", "save_routes_npz", "route_usage"]


# --------------------------------------------------------------------------
# batch price solving
# --------------------------------------------------------------------------

def merge_decisions(decision_batches: list[list[RouteDecision]]) -> list[RouteDecision]:
    """Join per-forward decisions along the sample axis, one entry per layer.

    A training step is several gradient-accumulation micro-batches, but the price
    used to be solved from the *last* one alone -- one tenth of the step at
    mb=6 -- so the price that governed the next step was fitted to a tenth of the
    evidence.  Concatenating is not the same as passing the lists end to end:
    `_cost_at_price` sums one contribution per decision, so 10 micro-batches of 16
    layers appended together would be charged as 160 layers.  The sample axis is
    what has to grow, which is what this does.

    Probabilities are detached: they are read as values by the solver, and keeping
    them attached would hold ten backward graphs alive at once.
    """
    if not decision_batches:
        return []
    depth = len(decision_batches[0])
    if any(len(batch) != depth for batch in decision_batches):
        raise ValueError("every accumulation micro-batch must decide at every layer")
    merged = []
    for index in range(depth):
        first = decision_batches[0][index]
        joined = RouteDecision()
        for organ, probs in first.probs.items():
            joined.probs[organ] = torch.cat(
                [batch[index].probs[organ].detach() for batch in decision_batches], dim=0
            )
            # Unit costs depend only on the config and the context length, so they
            # are identical across micro-batches.
            joined.unit_costs[organ] = first.unit_costs[organ]
        merged.append(joined)
    return merged


def _cost_at_price(decisions, fixed_share: float, price_used: float, price: float) -> float:
    """Executed cost the controller would have spent at `price`.

    `probs` is softmax(logits - price_used * unit), so `log(probs)` recovers the
    priced scores up to a per-site constant, which argmax ignores.  Re-pricing is
    therefore exact without another backbone pass.
    """
    total = fixed_share
    delta = price - price_used
    for decision in decisions:
        for organ, probs in decision.probs.items():
            unit = decision.unit_costs[organ]
            score = probs.clamp_min(1e-30).log() - delta * unit
            sel = score.argmax(-1, keepdim=True)
            total += float(unit.expand_as(probs).gather(-1, sel).squeeze(-1).mean())
    return total


def solve_batch_price(
    decisions, fixed_share: float, target: float, price_used: float,
    lo: float = -1024.0, hi: float = 1024.0, iterations: int = 40,
) -> float:
    """Bisect the common price so the executed cost equals `target`.

    The price may go **negative**.  The budget is an equality -- spend exactly
    what Fixed-Full spends -- so the multiplier is sign-free: a positive price
    prunes when the controller overspends, a negative one subsidises the tiers
    above the Base width when it underspends.  Clamping at zero, as a one-sided
    "<= budget" constraint would, leaves the pruned compute unspent and the
    reinvestment modes are never chosen -- which is the entire mechanism under
    test, so the sign matters.

    Cost is non-increasing in the price, so bisection is well posed.
    """
    if _cost_at_price(decisions, fixed_share, price_used, hi) > target:
        return hi                      # cannot get cheap enough; clamp
    if _cost_at_price(decisions, fixed_share, price_used, lo) < target:
        return lo                      # cannot get expensive enough; clamp
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if _cost_at_price(decisions, fixed_share, price_used, mid) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


__all__ += ["solve_batch_price", "merge_decisions"]
