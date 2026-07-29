#!/usr/bin/env python
"""Distribution of contamination evidence, before anything is deleted.

Reports every candidate match with the quantities the rules actually use --
coverage of the evaluation record, longest contiguous run, record length, and
which part of the benchmark matched -- so the thresholds can be judged against
real data rather than assumed.
"""

import argparse
import gzip
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from contamination_match import RULES, RecordIndex, analyse


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clean-root", default="data/clean")
    ap.add_argument("--index", default="data/record_index.json")
    ap.add_argument("--shards", default="")
    ap.add_argument("--per-shard", type=int, default=40000)
    ap.add_argument("--examples", type=int, default=20)
    ap.add_argument("--out", default="data/contamination_distribution.json")
    args = ap.parse_args()

    index = RecordIndex.load(Path(args.index))
    shards = args.shards.split(",") if args.shards else \
        [p.stem for p in sorted(Path(args.clean_root).glob("clean_*.jsonl.gz"))]

    scanned = Counter()
    flagged = Counter()
    per_benchmark = Counter()
    coverage_bins = defaultdict(Counter)
    all_matches = []
    examples = defaultdict(list)
    t0 = time.time()

    for name in shards:
        path = Path(args.clean_root) / f"{name}.jsonl.gz"
        if not path.exists():
            continue
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if args.per_shard and i >= args.per_shard:
                    break
                record = json.loads(line)
                source = record.get("source", "")
                scanned[source] += 1
                findings = [f for f in analyse(record["text"], index)
                            if f["coverage"] > 0.2 or f["substring"]]
                if not findings:
                    continue
                best = max(findings, key=lambda f: (f["contaminated"], f["coverage"]))
                all_matches.append({k: best[k] for k in
                                    ("benchmark", "coverage", "contiguous_units",
                                     "record_chars", "record_ngrams", "substring",
                                     "contaminated")} | {"source": source})
                bucket = min(int(best["coverage"] * 10) / 10, 1.0)
                coverage_bins[best["benchmark"]][bucket] += 1
                if best["contaminated"]:
                    flagged[source] += 1
                    per_benchmark[best["benchmark"]] += 1
                if len(examples[best["benchmark"]]) < args.examples:
                    examples[best["benchmark"]].append({
                        **{k: best[k] for k in ("coverage", "contiguous_units",
                                                "record_chars", "substring",
                                                "contaminated")},
                        "source": source,
                        "record": index.record_text[best["record_id"]][:130],
                        "excerpt": record["text"][:170].replace("\n", " | "),
                    })
        print(f"  {name}: {sum(scanned.values()):,} scanned, "
              f"{sum(flagged.values()):,} flagged, {time.time()-t0:.0f}s", flush=True)

    report = {
        "scanned": dict(scanned),
        "flagged": dict(flagged),
        "flagged_by_benchmark": dict(per_benchmark),
        "coverage_histogram": {b: dict(sorted(c.items()))
                               for b, c in coverage_bins.items()},
        "rules": {k: vars(v) for k, v in RULES.items()},
        "candidates": all_matches[:5000],
        "examples": {k: v for k, v in examples.items()},
    }
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))

    total_scanned = sum(scanned.values())
    total_flagged = sum(flagged.values())
    print(f"\nscanned {total_scanned:,}  candidates {len(all_matches):,}  "
          f"flagged {total_flagged:,} ({100*total_flagged/max(total_scanned,1):.4f}%)")
    print(f"\n{'source':24} {'scanned':>9} {'flagged':>8} {'percent':>8}")
    for s in sorted(scanned):
        print(f"{s:24} {scanned[s]:>9,} {flagged[s]:>8,} "
              f"{100*flagged[s]/max(scanned[s],1):>7.4f}%")
    print(f"\n{'benchmark':20} {'flagged':>8}")
    for b, n in per_benchmark.most_common():
        print(f"{b:20} {n:>8,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
