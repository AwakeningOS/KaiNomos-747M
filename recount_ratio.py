#!/usr/bin/env python
"""Per-source token-count ratio between two tokenizers, from a spread sample.

A full recount of DataMix v2 means tokenizing ~78 GB. This does not do that. It
measures the *ratio* between the old 32,768-piece tokenizer and the frozen
49,152-piece one, which is enough to rescale counts already made with the old one:

    new_count ~= old_count * ratio

Better compression *eats* headroom rather than creating it: the text is unchanged,
the token count falls, and the domain targets are stated in tokens.

## Sampling must be spread, not just reproducible

Reproducibility is easy and not sufficient. Reading the first row groups of the
first few files is perfectly deterministic and can still be systematically wrong,
because sources are usually *ordered* -- by crawl date, by language subset, by
quality score, by title. This pool is known to be ordered: `train.bin` turned out
to be laid out in blocks by source, so ordering inside a source is likely too.

So the sample is spread deterministically along three axes:

* **across files** -- every shard, or an evenly strided subset when there are many
* **across row groups** -- evenly spaced indices, not 0 upward
* **within a file** -- for seekable text, byte offsets spread across the whole file

Gzipped shards cannot be seeked without decompressing from the start, so those get
per-shard spread instead: every shard contributes, and each skips a different
deterministic number of leading records so the sample is not all shard heads.

Per-file ratios are reported alongside the aggregate. If the spread across files is
wide, the aggregate is doing a lot of work and a full recount matters more.
"""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
from pathlib import Path

TEXT_KEYS = ("text", "content", "raw_content", "document", "body")


def pick_spread(count: int, want: int) -> list[int]:
    """`want` indices spread evenly over `range(count)`, endpoints included."""
    if count <= want:
        return list(range(count))
    if want == 1:
        return [count // 2]
    return sorted({round(i * (count - 1) / (want - 1)) for i in range(want)})


def text_key(names) -> str | None:
    for key in TEXT_KEYS:
        if key in names:
            return key
    return None


def from_parquet(files: list[Path], per_file: int, groups_per_file: int):
    """Evenly spaced row groups from every file, not the first ones."""
    import pyarrow.parquet as pq

    out = []
    for path in files:
        handle = pq.ParquetFile(path)
        key = text_key([f.name for f in handle.schema_arrow])
        if key is None:
            continue
        indices = pick_spread(handle.metadata.num_row_groups, groups_per_file)
        collected: list[str] = []
        for index in indices:
            rows = handle.read_row_group(index, columns=[key])[key].to_pylist()
            take = max(1, per_file // max(len(indices), 1))
            # spread within the row group too
            for position in pick_spread(len(rows), take):
                if rows[position]:
                    collected.append(rows[position])
            if len(collected) >= per_file:
                break
        if collected:
            out.append((path, collected[:per_file]))
    return out


def from_seekable_jsonl(files: list[Path], per_file: int):
    """Byte offsets spread across the whole file, then read the next record."""
    out = []
    for path in files:
        size = path.stat().st_size
        collected: list[str] = []
        with path.open("rb") as handle:
            for offset in pick_spread(size, per_file * 2):
                handle.seek(offset)
                if offset:
                    handle.readline()          # discard the partial line
                line = handle.readline()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                key = text_key(record)
                if key and isinstance(record[key], str) and record[key]:
                    collected.append(record[key])
                if len(collected) >= per_file:
                    break
        if collected:
            out.append((path, collected))
    return out


def from_gzip_jsonl(files: list[Path], per_file: int):
    """Every shard contributes, each skipping a different number of records.

    A gzip member cannot be seeked into cheaply, so spread comes from using all
    shards and from a per-shard offset rather than from within-file seeking.
    """
    out = []
    for order, path in enumerate(files):
        skip = (order * 137) % 500          # deterministic, differs per shard
        collected: list[str] = []
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            for number, line in enumerate(handle):
                if number < skip:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = text_key(record)
                if key and isinstance(record[key], str) and record[key]:
                    collected.append(record[key])
                if len(collected) >= per_file:
                    break
        if collected:
            out.append((path, collected))
    return out


def gather(root: Path, documents: int, max_files: int, groups_per_file: int):
    """Returns (method, [(file, texts)]) or (reason, []) when nothing is readable."""
    for pattern, reader, method in (
        ("*.parquet", None, "parquet row groups spread across every file"),
        ("*.jsonl", None, "byte offsets spread across every file"),
        ("*.jsonl.gz", None, "all shards, per-shard record offset"),
    ):
        found = sorted(root.rglob(pattern))
        if not found:
            continue
        chosen = [found[i] for i in pick_spread(len(found), max_files)]
        per_file = max(1, documents // len(chosen))
        if pattern == "*.parquet":
            batches = from_parquet(chosen, per_file, groups_per_file)
        elif pattern == "*.jsonl":
            batches = from_seekable_jsonl(chosen, per_file)
        else:
            batches = from_gzip_jsonl(chosen, per_file)
        if batches:
            return method, batches, len(found), len(chosen)
        return f"{pattern} present but no recognised text field", [], len(found), 0
    return "no parquet or jsonl files found", [], 0, 0


def ratio_for(root: Path, old, new, documents: int, max_files: int,
              groups_per_file: int) -> dict:
    method, batches, found, used = gather(root, documents, max_files, groups_per_file)
    if not batches:
        return {"ok": False, "reason": method, "files_found": found}

    per_file = []
    old_total = new_total = char_total = row_total = 0
    for path, texts in batches:
        old_tokens = sum(len(v) for v in old.encode(texts))
        new_tokens = sum(len(v) for v in new.encode(texts))
        characters = sum(len(t) for t in texts)
        old_total += old_tokens
        new_total += new_tokens
        char_total += characters
        row_total += len(texts)
        per_file.append({
            "file": path.name, "documents": len(texts), "characters": characters,
            "old_tokens": old_tokens, "new_tokens": new_tokens,
            "ratio": new_tokens / old_tokens,
        })

    spread = [entry["ratio"] for entry in per_file]
    return {
        "ok": True,
        "extraction": method,
        "files_found": found,
        "files_sampled": used,
        "documents": row_total,
        "characters": char_total,
        "old_tokens": old_total,
        "new_tokens": new_total,
        "old_chars_per_token": char_total / old_total,
        "new_chars_per_token": char_total / new_total,
        "ratio": new_total / old_total,
        "reduction_percent": (1 - new_total / old_total) * 100,
        "ratio_min": min(spread),
        "ratio_max": max(spread),
        "ratio_stdev": statistics.stdev(spread) if len(spread) > 1 else 0.0,
        "per_file": per_file,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--old-tokenizer", default="data/tokenizer/kainomos.model")
    ap.add_argument("--new-tokenizer", required=True)
    ap.add_argument("--raw-root", required=True)
    ap.add_argument("--sources", default=None)
    ap.add_argument("--documents", type=int, default=2000)
    ap.add_argument("--max-files", type=int, default=12,
                    help="files sampled per source, strided evenly over all of them")
    ap.add_argument("--groups-per-file", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import sentencepiece as spm

    old = spm.SentencePieceProcessor(model_file=args.old_tokenizer)
    new = spm.SentencePieceProcessor(model_file=args.new_tokenizer)
    root = Path(args.raw_root)
    names = (args.sources.split(",") if args.sources
             else sorted(p.name for p in root.iterdir() if p.is_dir()))

    print(f"old {old.get_piece_size():,} -> new {new.get_piece_size():,}\n")
    print(f"{'source':40s} {'files':>9s} {'docs':>6s} {'old c/t':>8s} "
          f"{'new c/t':>8s} {'reduce':>8s} {'spread':>15s}")
    results, failures = {}, []
    for name in names:
        row = ratio_for(root / name, old, new, args.documents,
                        args.max_files, args.groups_per_file)
        results[name] = row
        if not row["ok"]:
            failures.append({"source": name, "reason": row["reason"],
                             "files_found": row["files_found"]})
            print(f"{name:40s} FAILED  {row['reason']}")
            continue
        print(f"{name:40s} {row['files_sampled']:4d}/{row['files_found']:<4d} "
              f"{row['documents']:6d} {row['old_chars_per_token']:8.3f} "
              f"{row['new_chars_per_token']:8.3f} {row['reduction_percent']:7.2f}% "
              f"{row['ratio_min']:.3f}-{row['ratio_max']:.3f}")

    if failures:
        print("\n読取不能（平均には混ぜていない）:")
        for entry in failures:
            print(f"  {entry['source']}: {entry['reason']} "
                  f"({entry['files_found']} files)")

    payload = {
        "old_tokenizer": args.old_tokenizer,
        "new_tokenizer": args.new_tokenizer,
        "documents_requested_per_source": args.documents,
        "max_files_per_source": args.max_files,
        "row_groups_per_file": args.groups_per_file,
        "sampling": "deterministic and spread: files strided evenly over all of "
                    "them, row groups evenly spaced rather than from 0, and byte "
                    "offsets spread within seekable files. Gzip shards cannot be "
                    "seeked, so those spread across shards with a per-shard record "
                    "offset instead.",
        "caveat": "sample-based ratios, not a full recount. Rescale old counts by "
                  "`ratio`; check `ratio_min`/`ratio_max`/`ratio_stdev` before "
                  "trusting the aggregate.",
        "succeeded": sorted(k for k, v in results.items() if v["ok"]),
        "failed": failures,
        "sources": results,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
