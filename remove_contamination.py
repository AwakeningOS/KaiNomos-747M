#!/usr/bin/env python
"""Contamination removal by quarantine, not deletion.

Flagged documents are moved to `data/quarantine/contamination_removed.jsonl`
with the full evidence for each decision, so any judgement made here can be
reviewed, argued with, or reversed without rebuilding the corpus.

A guard rail stops the run if any one source's removal rate departs wildly from
what the dry run measured (max observed: 0.08%). A sudden spike would mean the
matcher is behaving differently on unseen shards, and finding that out after
deleting a source is too late.
"""

import argparse
import gzip
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

from contamination_match import RecordIndex, analyse

ABORT_RATE = 0.05      # 5%: ~60x the highest rate the dry run saw


def decision_path(finding: dict) -> str:
    if finding["substring"]:
        return "substring_containment"
    if finding["coverage"] >= 0.7 and finding["contiguous_fraction"] >= 0.7:
        return "record_coverage"
    return "contiguous_span"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clean-root", default="data/clean")
    ap.add_argument("--out-root", default="data/decontaminated")
    ap.add_argument("--quarantine", default="data/quarantine")
    ap.add_argument("--index", default="data/record_index.json")
    ap.add_argument("--shard-documents", type=int, default=50_000)
    args = ap.parse_args()

    index = RecordIndex.load(Path(args.index))
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    quarantine_dir = Path(args.quarantine)
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    quarantine_path = quarantine_dir / "contamination_removed.jsonl"
    quarantine = open(quarantine_path, "w", encoding="utf-8")

    scanned, removed = Counter(), Counter()
    by_benchmark, by_path = Counter(), Counter()
    buffer: list[str] = []
    shard_no = 0
    t0 = time.time()

    def flush() -> None:
        nonlocal shard_no
        if not buffer:
            return
        path = out_root / f"final_{shard_no:05d}.jsonl.gz"
        tmp = path.with_suffix(".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as fh:
            fh.write("".join(buffer))
        os.replace(tmp, path)
        shard_no += 1
        buffer.clear()

    for shard in sorted(Path(args.clean_root).glob("clean_*.jsonl.gz")):
        with gzip.open(shard, "rt", encoding="utf-8") as fh:
            for line in fh:
                record = json.loads(line)
                source = record.get("source", "")
                text = record["text"]
                scanned[source] += 1

                hits = [f for f in analyse(text, index) if f["contaminated"]]
                if hits:
                    best = max(hits, key=lambda f: (f["coverage"], f["contiguous_units"]))
                    path_name = decision_path(best)
                    removed[source] += 1
                    by_benchmark[best["benchmark"]] += 1
                    by_path[path_name] += 1
                    quarantine.write(json.dumps({
                        "source": source,
                        "document_id": record.get("key"),
                        "document_sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "benchmark": best["benchmark"],
                        "record_id": best["record_id"],
                        "decision_path": path_name,
                        "coverage": best["coverage"],
                        "contiguous_units": best["contiguous_units"],
                        "contiguous_fraction": best["contiguous_fraction"],
                        "matched_text": best.get("matched_text", ""),
                        "source_shard": shard.name,
                        "text": text,
                    }, ensure_ascii=False) + "\n")
                    continue

                buffer.append(line)
                if len(buffer) >= args.shard_documents:
                    flush()

        done = sum(scanned.values())
        gone = sum(removed.values())
        print(f"  {shard.name}: {done:,} scanned, {gone:,} quarantined, "
              f"{time.time()-t0:.0f}s", flush=True)

        for src, n in scanned.items():
            if n >= 20_000 and removed[src] / n > ABORT_RATE:
                quarantine.close()
                flush()
                raise SystemExit(
                    f"ABORT: {src} removal rate {100*removed[src]/n:.2f}% exceeds "
                    f"{100*ABORT_RATE:.0f}%; the dry run peaked at 0.08%"
                )

    flush()
    quarantine.close()

    digest = hashlib.sha256()
    with open(quarantine_path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            digest.update(block)

    report = {
        "scanned": dict(scanned),
        "removed": dict(removed),
        "removal_rate": {s: round(100 * removed[s] / max(scanned[s], 1), 5)
                         for s in sorted(scanned)},
        "by_benchmark": dict(by_benchmark),
        "by_decision_path": dict(by_path),
        "total_scanned": sum(scanned.values()),
        "total_removed": sum(removed.values()),
        "output_shards": shard_no,
        "quarantine_file": str(quarantine_path),
        "quarantine_sha256": digest.hexdigest(),
        "seconds": round(time.time() - t0, 1),
    }
    (out_root / "decontamination_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False))

    print(f"\n{'source':24} {'scanned':>10} {'removed':>8} {'rate':>9}")
    for s in sorted(scanned):
        print(f"{s:24} {scanned[s]:>10,} {removed[s]:>8,} "
              f"{report['removal_rate'][s]:>8.4f}%")
    print(f"\n{'benchmark':22} {'removed':>8}")
    for b, n in by_benchmark.most_common():
        print(f"{b:22} {n:>8,}")
    print(f"\n{'decision path':24} {'count':>8}")
    for p, n in by_path.most_common():
        print(f"{p:24} {n:>8,}")
    print(f"\ntotal {report['total_removed']:,} / {report['total_scanned']:,} "
          f"({100*report['total_removed']/max(report['total_scanned'],1):.4f}%)")
    print(f"quarantine SHA256: {report['quarantine_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
