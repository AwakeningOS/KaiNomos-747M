"""Light cleaning, run before the tokenizer is trained.

A 32,768-piece vocabulary is a fixed budget.  Every piece spent on navigation
chrome, cookie banners, mojibake or a duplicated boilerplate footer is a piece
unavailable to Japanese, and unlike a bad training example a bad vocabulary
entry cannot be diluted by more data -- it is baked into every future encode.
So the cheap, deterministic cleaning happens *first*:

    Unicode NFC
    HTML / navigation / boilerplate stripping
    exact-duplicate removal
    repeated-line collapse

The expensive near-duplicate (MinHash) and contamination passes run afterwards,
because they change how often text appears rather than what it looks like, and
the tokenizer only cares about the latter.

Deliberately *not* removed here: the semantic overlap between `ja_paraphrase`
and `ja_web`.  Paraphrases restate the same content by design; dropping them as
near-duplicates would delete the source's entire reason for existing.  Only
byte-identical text is treated as a duplicate.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import xxhash

# navigation and legal chrome that survives most extractors
_BOILERPLATE = re.compile(
    r"^\s*(?:"
    r"cookie(?:s)?\s+(?:policy|settings|preferences)"
    r"|(?:accept|manage)\s+(?:all\s+)?cookies"
    r"|skip\s+to\s+(?:main\s+)?content"
    r"|(?:全ての)?クッキー(?:を)?(?:受け入れる|設定|同意)"
    r"|このサイトはCookieを使用"
    r"|メニュー(?:を)?(?:開く|閉じる)"
    r"|コンテンツへスキップ"
    r"|ページの先頭へ(?:戻る)?"
    r"|前のページに戻る"
    r"|(?:copyright|©)\s*\d{4}"
    r"|all\s+rights\s+reserved"
    r")\s*$",
    re.IGNORECASE,
)
_HTML_TAG = re.compile(r"<[^>]{1,200}>")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MANY_BLANKS = re.compile(r"\n{4,}")
# a replacement character means the extractor already lost the original bytes
_MOJIBAKE = re.compile(r"�")


@dataclass
class CleanStats:
    documents_in: int = 0
    documents_out: int = 0
    exact_duplicates: int = 0
    too_short: int = 0
    mojibake: int = 0
    lines_dropped: int = 0
    shards: int = 0
    bytes_out: int = 0
    per_source: dict = field(default_factory=dict)


def normalise(text: str) -> str:
    """NFC, control characters gone, HTML gone, whitespace preserved otherwise.

    Indentation and newlines are kept exactly: code depends on them, and the
    tokenizer must learn them rather than be handed pre-collapsed text.
    """
    text = unicodedata.normalize("NFC", text)
    text = _HTML_TAG.sub(" ", text)
    text = _CONTROL.sub("", text)
    return text


def clean_lines(text: str) -> tuple[str, int]:
    """Drop boilerplate lines and collapse immediate line repetition."""
    kept: list[str] = []
    dropped = 0
    previous = None
    repeats = 0
    for line in text.split("\n"):
        if _BOILERPLATE.match(line):
            dropped += 1
            continue
        stripped = line.strip()
        if stripped and stripped == previous:
            repeats += 1
            if repeats >= 2:          # allow one repeat, drop runs
                dropped += 1
                continue
        else:
            repeats = 0
            previous = stripped
        kept.append(line)
    out = _MANY_BLANKS.sub("\n\n\n", "\n".join(kept))
    return out.strip("\n"), dropped


def clean_corpus(
    raw_root: Path,
    out_root: Path,
    min_characters: int = 64,
    max_mojibake_ratio: float = 0.001,
    shard_documents: int = 50_000,
) -> CleanStats:
    out_root.mkdir(parents=True, exist_ok=True)
    stats = CleanStats()
    seen: set[int] = set()
    buffer: list[str] = []

    def flush() -> None:
        if not buffer:
            return
        path = out_root / f"clean_{stats.shards:05d}.jsonl.gz"
        tmp = path.with_suffix(".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as fh:
            fh.write("".join(buffer))
        os.replace(tmp, path)
        stats.shards += 1
        buffer.clear()

    t0 = time.time()
    for shard in sorted(raw_root.rglob("shard_*.jsonl.gz")):
        source = shard.parent.name
        with gzip.open(shard, "rt", encoding="utf-8") as fh:
            for line in fh:
                record = json.loads(line)
                stats.documents_in += 1
                text = normalise(record["text"])

                if _MOJIBAKE.findall(text) and \
                        len(_MOJIBAKE.findall(text)) / max(len(text), 1) > max_mojibake_ratio:
                    stats.mojibake += 1
                    continue

                text, dropped = clean_lines(text)
                stats.lines_dropped += dropped
                if len(text) < min_characters:
                    stats.too_short += 1
                    continue

                digest = xxhash.xxh64(text.encode("utf-8")).intdigest()
                if digest in seen:
                    stats.exact_duplicates += 1
                    continue
                seen.add(digest)

                stats.documents_out += 1
                stats.bytes_out += len(text.encode("utf-8"))
                per = stats.per_source.setdefault(source, {"documents": 0, "bytes": 0})
                per["documents"] += 1
                per["bytes"] += len(text.encode("utf-8"))

                buffer.append(json.dumps(
                    {"key": record.get("key"), "source": source, "text": text},
                    ensure_ascii=False,
                ) + "\n")
                if len(buffer) >= shard_documents:
                    flush()
                    print(f"[clean] {stats.documents_out:,} kept / "
                          f"{stats.documents_in:,} read  "
                          f"{stats.bytes_out/1e9:.2f} GB  {time.time()-t0:.0f}s", flush=True)
    flush()
    (out_root / "clean_stats.json").write_text(
        json.dumps(stats.__dict__, indent=2, ensure_ascii=False)
    )
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", default="data/mix")
    ap.add_argument("--out-root", default="data/clean")
    ap.add_argument("--min-characters", type=int, default=64)
    args = ap.parse_args()

    stats = clean_corpus(Path(args.raw_root), Path(args.out_root), args.min_characters)
    print(json.dumps(stats.__dict__, indent=2, ensure_ascii=False)[:1500])
    kept = 100.0 * stats.documents_out / max(stats.documents_in, 1)
    print(f"\nkept {stats.documents_out:,} / {stats.documents_in:,} ({kept:.1f}%), "
          f"{stats.bytes_out/1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
