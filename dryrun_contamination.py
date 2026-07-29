#!/usr/bin/env python
"""Dry run: report what contamination filtering *would* remove, and why.

Nothing is deleted.  The point is to see the evidence before trusting the rule:
which sources are affected, how strong each match is, and whether the matched
spans read like quoted benchmark text or like ordinary prose that happens to
share a phrase.
"""

import argparse
import gzip
import json
import random
import time
from collections import Counter
from pathlib import Path

from contamination import ContaminationIndex, check_document


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clean-root", default="data/clean")
    ap.add_argument("--index", default="data/contamination.npz")
    ap.add_argument("--limit", type=int, default=0, help="documents to scan (0 = all)")
    ap.add_argument("--examples", type=int, default=10)
    ap.add_argument("--out", default="data/contamination_dryrun.json")
    args = ap.parse_args()

    index = ContaminationIndex.load(Path(args.index))
    print(f"index: {index.ngrams.size:,} n-grams, {index.exact.size:,} exact records",
          flush=True)

    by_source = Counter()
    flagged_by_source = Counter()
    by_reason = Counter()
    spans = []
    samples: list[dict] = []
    rng = random.Random(11)
    scanned = 0
    t0 = time.time()

    for shard in sorted(Path(args.clean_root).glob("clean_*.jsonl.gz")):
        with gzip.open(shard, "rt", encoding="utf-8") as fh:
            for line in fh:
                record = json.loads(line)
                source = record.get("source", "")
                by_source[source] += 1
                scanned += 1
                verdict = check_document(record["text"], index)
                if verdict:
                    flagged_by_source[source] += 1
                    by_reason[verdict["reason"]] += 1
                    spans.append(verdict["span"])
                    if len(samples) < args.examples * 4 or rng.random() < 0.01:
                        samples.append({
                            "source": source, **verdict,
                            "excerpt": record["text"][:220],
                        })
                if scanned % 100_000 == 0:
                    print(f"  {scanned:,} scanned, {sum(flagged_by_source.values()):,} "
                          f"flagged, {time.time()-t0:.0f}s", flush=True)
                if args.limit and scanned >= args.limit:
                    break
        if args.limit and scanned >= args.limit:
            break

    total_flagged = sum(flagged_by_source.values())
    report = {
        "scanned": scanned,
        "flagged": total_flagged,
        "flagged_percent": round(100.0 * total_flagged / max(scanned, 1), 4),
        "by_reason": dict(by_reason),
        "span_stats": {
            "min": min(spans) if spans else None,
            "median": sorted(spans)[len(spans) // 2] if spans else None,
            "max": max(spans) if spans else None,
        },
        "by_source": {
            s: {"documents": by_source[s], "flagged": flagged_by_source[s],
                "percent": round(100.0 * flagged_by_source[s] / max(by_source[s], 1), 4)}
            for s in sorted(by_source)
        },
        "examples": samples[: args.examples * 3],
    }
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"\nscanned {scanned:,}  flagged {total_flagged:,} "
          f"({report['flagged_percent']}%)")
    print("by reason:", dict(by_reason))
    print("span: ", report["span_stats"])
    print(f"\n{'source':24} {'documents':>10} {'flagged':>9} {'percent':>8}")
    for s, v in report["by_source"].items():
        print(f"{s:24} {v['documents']:>10,} {v['flagged']:>9,} {v['percent']:>7.3f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
