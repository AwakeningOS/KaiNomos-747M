#!/usr/bin/env python
"""Read the Muon / AdamW comparison out of the two training logs.

The endpoint alone does not decide this.  Muon is reported to lead early and then
lose over a long run when it lacks decoupled weight decay and update-RMS matching;
`muon.py` has both, but whether they worked is a statement about the *trend*, not
about the loss at 300M tokens.  So this reports:

* NLL at equal tokens, at several points along the run
* the slope over the last third of each curve, in nats per billion tokens
* whether the gap is widening or narrowing
* the same numbers against wall clock, for reference only -- the decision is per
  token, because the throughput penalty is a cost of the comparison rather than a
  term in it
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(run_dir: Path) -> list[dict]:
    path = run_dir / "train.jsonl"
    if not path.exists():
        raise SystemExit(f"no log at {path}")
    rows = {}
    for line in path.open(encoding="utf-8"):
        record = json.loads(line)
        rows[record["step"]] = record          # later wins: resume replays a step
    return [rows[k] for k in sorted(rows)]


def smoothed(rows: list[dict], key: str = "ntp_loss", window: int = 20) -> list[float]:
    values = [r[key] for r in rows]
    out = []
    for index in range(len(values)):
        low = max(0, index - window + 1)
        piece = values[low:index + 1]
        out.append(sum(piece) / len(piece))
    return out


def slope_per_billion(rows: list[dict], curve: list[float], fraction: float = 1 / 3):
    """Least-squares slope of NLL against tokens, over the final `fraction`."""
    start = int(len(rows) * (1 - fraction))
    xs = [rows[i]["tokens_done"] / 1e9 for i in range(start, len(rows))]
    ys = curve[start:]
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    denominator = sum((x - mx) ** 2 for x in xs)
    if denominator == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denominator


def at_tokens(rows: list[dict], curve: list[float], target: int) -> float | None:
    for index, row in enumerate(rows):
        if row["tokens_done"] >= target:
            return curve[index]
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adamw", default="runs/cmp_adamw")
    ap.add_argument("--muon", default="runs/cmp_muon")
    args = ap.parse_args()

    arms = {}
    for name, directory in (("adamw", args.adamw), ("muon", args.muon)):
        rows = read(Path(directory))
        arms[name] = (rows, smoothed(rows))

    shortest = min(rows[-1]["tokens_done"] for rows, _ in arms.values())
    marks = [m for m in (25, 50, 100, 150, 200, 250, 300) if m * 1_000_000 <= shortest]

    print(f"tokens compared: up to {shortest:,}\n")
    print(f"{'tokens':>10s}  {'AdamW':>8s}  {'Muon':>8s}  {'gap':>8s}")
    for mark in marks:
        a = at_tokens(*arms["adamw"], mark * 1_000_000)
        m = at_tokens(*arms["muon"], mark * 1_000_000)
        if a is None or m is None:
            continue
        print(f"{mark:>9d}M  {a:8.4f}  {m:8.4f}  {a - m:+8.4f}")

    print("\nslope over the final third (nats per billion tokens, more negative "
          "= still improving faster):")
    slopes = {}
    for name, (rows, curve) in arms.items():
        slopes[name] = slope_per_billion(rows, curve)
        print(f"  {name:6s} {slopes[name]:+9.4f}")

    print("\nverdict:")
    a_end = arms["adamw"][1][-1]
    m_end = arms["muon"][1][-1]
    ahead = "muon" if m_end < a_end else "adamw"
    print(f"  ahead at the end : {ahead} (adamw {a_end:.4f}, muon {m_end:.4f})")
    if ahead == "muon":
        if slopes["muon"] <= slopes["adamw"]:
            print("  gap             : widening -> adopt Muon; the final size can "
                  "then be h1536 / 722,500,872 (20.24 GB; AdamW needs 22.77 GB there)")
        else:
            print("  gap             : NARROWING -> this is the transient Moonshot "
                  "report.  Do not adopt on the endpoint alone; extend the run.")
    else:
        print("  -> keep AdamW; the largest safe size is then "
              "h1408 / 610,961,902 (20.33 GB)")

    print("\nwall clock (reference only; the decision is per token):")
    for name, (rows, _) in arms.items():
        steps = len(rows)
        tokens = rows[-1]["tokens_done"]
        print(f"  {name:6s} {steps:5d} steps, {tokens:,} tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
