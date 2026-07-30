#!/usr/bin/env python
"""Per-source token-count ratio between two tokenizers, from a sample.

A full recount of DataMix v2 means tokenizing ~78 GB.  This does not do that.  It
measures the *ratio* between the old 32,768-piece tokenizer and the frozen
49,152-piece one on a bounded sample of each source, which is enough to rescale
counts that were already made with the old tokenizer:

    new_count ~= old_count * ratio

That is what the domain-budget question needs.  A source whose candidates were
measured at 1.778B tokens under the old tokenizer does not hold 1.778B tokens any
more once the tokenizer compresses better -- the text is the same, the token count
is smaller, and the *target* is stated in tokens.  Better compression therefore
eats headroom rather than creating it, which is the opposite of the intuition.

Sampling is deterministic: the same row groups and the same shuffle seed every
time, so two runs of this script agree and a later full recount can be checked
against it.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

TEXT_COLUMNS = ("text", "content", "raw_content", "document", "body")


def find_text_column(schema) -> str | None:
    names = [field.name for field in schema]
    for candidate in TEXT_COLUMNS:
        if candidate in names:
            return candidate
    return None


def _from_parquet(files, want: int) -> list[str]:
    import pyarrow.parquet as pq

    texts: list[str] = []
    for path in files:
        handle = pq.ParquetFile(path)
        column = find_text_column(handle.schema_arrow)
        if column is None:
            continue
        for group in range(handle.metadata.num_row_groups):
            texts.extend(handle.read_row_group(group, columns=[column])[column].to_pylist())
            if len(texts) >= want * 2:
                return texts
    return texts


def _from_jsonl(files, want: int) -> list[str]:
    """Plain or gzipped JSON lines, reading only as far as the sample needs."""
    import gzip

    texts: list[str] = []
    for path in files:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for key in TEXT_COLUMNS:
                    if isinstance(record.get(key), str):
                        texts.append(record[key])
                        break
                if len(texts) >= want * 2:
                    return texts
    return texts


def sample_texts(root: Path, want: int, seed: int, max_files: int = 2) -> list[str]:
    """A bounded sample, whatever the on-disk format is.

    Sources arrive as parquet, .jsonl or .jsonl.gz depending on where they came
    from; a ratio measured on only the parquet ones would silently omit the largest
    domain, which is exactly what happened on the first attempt here.
    """
    parquet = sorted(root.rglob("*.parquet"))[:max_files]
    texts = _from_parquet(parquet, want) if parquet else []
    if not texts:
        lines = (sorted(root.rglob("*.jsonl.gz"))[:max_files]
                 or sorted(root.rglob("*.jsonl"))[:max_files])
        texts = _from_jsonl(lines, want)
    random.Random(seed).shuffle(texts)
    return [t for t in texts[:want] if t]


def ratio_for(root: Path, old, new, want: int, seed: int) -> dict:
    texts = sample_texts(root, want, seed)
    if not texts:
        return {"error": "no readable parquet text column"}
    characters = sum(len(t) for t in texts)
    old_tokens = sum(len(v) for v in old.encode(texts))
    new_tokens = sum(len(v) for v in new.encode(texts))
    return {
        "documents": len(texts),
        "characters": characters,
        "old_tokens": old_tokens,
        "new_tokens": new_tokens,
        "old_chars_per_token": characters / old_tokens,
        "new_chars_per_token": characters / new_tokens,
        "ratio": new_tokens / old_tokens,
        "reduction_percent": (1 - new_tokens / old_tokens) * 100,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--old-tokenizer", default="data/tokenizer/kainomos.model")
    ap.add_argument("--new-tokenizer", required=True)
    ap.add_argument("--raw-root", required=True, help="the DataMix v2 raw/ directory")
    ap.add_argument("--sources", default=None,
                    help="comma-separated subdirectory names; default is all of them")
    ap.add_argument("--documents", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import sentencepiece as spm

    old = spm.SentencePieceProcessor(model_file=args.old_tokenizer)
    new = spm.SentencePieceProcessor(model_file=args.new_tokenizer)
    root = Path(args.raw_root)
    names = (args.sources.split(",") if args.sources
             else sorted(p.name for p in root.iterdir() if p.is_dir()))

    print(f"old {old.get_piece_size():,} pieces -> new {new.get_piece_size():,}\n")
    print(f"{'source':44s} {'docs':>6s} {'old c/t':>8s} {'new c/t':>8s} {'reduction':>10s}")
    results = {}
    for name in names:
        row = ratio_for(root / name, old, new, args.documents, args.seed)
        results[name] = row
        if "error" in row:
            print(f"{name:44s} {row['error']}")
            continue
        print(f"{name:44s} {row['documents']:6d} "
              f"{row['old_chars_per_token']:8.3f} {row['new_chars_per_token']:8.3f} "
              f"{row['reduction_percent']:9.2f}%")

    payload = {
        "old_tokenizer": args.old_tokenizer,
        "new_tokenizer": args.new_tokenizer,
        "sample_documents_per_source": args.documents,
        "seed": args.seed,
        "note": "sample-based ratios, not a full recount; rescale old counts by `ratio`",
        "sources": results,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
