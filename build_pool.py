"""Tokenise the decontaminated corpus into the final fixed-ratio token pool.

Ratios are enforced here, on real token counts from this project's tokenizer,
not on the byte estimates that sized the download.  Those estimates were only
ever a way to decide how much text to fetch; a corpus counted in someone else's
tokens says nothing about how much of it this model reads.

Two passes:

  measure   tokenise a sample of each source to learn its true bytes-per-token,
            then compute how many documents each source contributes
  build     tokenise for real, stopping each source at its token quota, and
            write `train/validation/test.bin` as uint16 with `.idx` document
            offsets and a manifest carrying every SHA256

Documents are separated by `<|eod|>`; splits are assigned per document by a
stable hash so no document can straddle train and validation.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import xxhash

TARGET_TOKENS = 2_500_000_000
SPLIT_SEED = 20260729
VALIDATION_TOKENS = 10_000_000
TEST_TOKENS = 10_000_000

KIND_RATIO = {"ja": 0.75, "en": 0.10, "code": 0.10, "math": 0.05}
SOURCE_KIND = {
    "ja_web": "ja", "ja_paraphrase": "ja", "ja_instruct": "ja",
    "ja_wikipedia_reference": "ja", "en_edu": "en", "math": "math",
}
# Within Japanese, the design's shares of the whole mix.
JA_SHARE = {"ja_web": 0.35, "ja_paraphrase": 0.20,
            "ja_instruct": 0.10, "ja_wikipedia_reference": 0.10}


def kind_of(source: str) -> str:
    return SOURCE_KIND.get(source, "code" if source.startswith("code_") else "en")


def source_quota(sources: list[str], total: int = TARGET_TOKENS) -> dict[str, int]:
    """Token budget per source, from the design ratios."""
    quota: dict[str, int] = {}
    code_sources = [s for s in sources if kind_of(s) == "code"]
    for source in sources:
        kind = kind_of(source)
        if kind == "ja":
            quota[source] = int(total * JA_SHARE.get(source, 0.0))
        elif kind == "code":
            quota[source] = int(total * KIND_RATIO["code"] / max(len(code_sources), 1))
        else:
            quota[source] = int(total * KIND_RATIO[kind])
    return quota


def assign_split(document_id: str) -> str:
    bucket = xxhash.xxh64(str(document_id).encode(), seed=SPLIT_SEED).intdigest() % 10_000
    if bucket < 40:
        return "test"
    if bucket < 80:
        return "validation"
    return "train"


def iter_documents(root: Path):
    for shard in sorted(root.glob("final_*.jsonl.gz")):
        with gzip.open(shard, "rt", encoding="utf-8") as fh:
            for line in fh:
                yield json.loads(line), shard.name


def build(root: Path, out_dir: Path, tokenizer_path: Path,
          total_tokens: int = TARGET_TOKENS) -> dict:
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    eod = sp.piece_to_id("<|eod|>")
    if eod < 0:
        raise RuntimeError("tokenizer has no <|eod|> piece")

    sources = sorted({d.get("source", "") for d, _ in iter_documents(root)})
    quota = source_quota(sources, total_tokens)
    print(json.dumps({"quota": quota}, indent=2))

    out_dir.mkdir(parents=True, exist_ok=True)
    handles = {s: open(out_dir / f"{s}.bin", "wb") for s in ("train", "validation", "test")}
    offsets = {s: [] for s in handles}
    counts = {s: 0 for s in handles}
    per_source = Counter()
    per_source_docs = Counter()
    skipped_full = Counter()

    batch, batch_meta = [], []
    t0 = time.time()

    def emit(ids_list):
        for ids, (source, split) in zip(ids_list, batch_meta):
            if per_source[source] >= quota.get(source, 0):
                skipped_full[source] += 1
                continue
            ids = ids + [eod]
            offsets[split].append(counts[split])
            np.asarray(ids, dtype=np.uint16).tofile(handles[split])
            counts[split] += len(ids)
            per_source[source] += len(ids)
            per_source_docs[source] += 1

    for document, _shard in iter_documents(root):
        source = document.get("source", "")
        if per_source[source] >= quota.get(source, 0):
            continue
        split = assign_split(document.get("key") or document["text"][:64])
        batch.append(document["text"])
        batch_meta.append((source, split))
        if len(batch) >= 2000:
            emit(sp.encode(batch))
            batch, batch_meta = [], []
            done = sum(per_source.values())
            print(f"  {done/1e6:8.1f}M tokens  {sum(per_source_docs.values()):,} docs  "
                  f"{time.time()-t0:.0f}s", flush=True)
            if all(per_source[s] >= quota.get(s, 0) for s in sources):
                break
    if batch:
        emit(sp.encode(batch))

    for handle in handles.values():
        handle.close()
    for split, index in offsets.items():
        with open(out_dir / f"{split}.idx", "wb") as fh:
            np.save(fh, np.asarray(index, dtype=np.int64))

    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 22), b""):
                digest.update(block)
        return digest.hexdigest()

    achieved = {k: 0 for k in KIND_RATIO}
    for source, tokens in per_source.items():
        achieved[kind_of(source)] += tokens
    total = max(sum(achieved.values()), 1)

    manifest = {
        "name": "KaiNomos-DataMix-2.5B-v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tokenizer": str(tokenizer_path),
        "tokenizer_sha256": sha256(tokenizer_path),
        "vocab_size": sp.get_piece_size(),
        "target_tokens": total_tokens,
        "tokens_by_source": dict(per_source),
        "documents_by_source": dict(per_source_docs),
        "quota_by_source": quota,
        "achieved_ratio": {k: round(v / total, 4) for k, v in achieved.items()},
        "target_ratio": KIND_RATIO,
        "splits": {
            s: {"tokens": counts[s], "documents": len(offsets[s]),
                "bin_sha256": sha256(out_dir / f"{s}.bin"),
                "idx_sha256": sha256(out_dir / f"{s}.idx")}
            for s in handles
        },
        "split_rule": {"hash": "xxh64", "seed": SPLIT_SEED,
                       "test_buckets": [0, 40], "validation_buckets": [40, 80],
                       "train": "remaining", "unit": "document"},
        "eod_token_id": eod,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/decontaminated")
    ap.add_argument("--out-dir", default="data/pool")
    ap.add_argument("--tokenizer", default="data/tokenizer/kainomos.model")
    ap.add_argument("--total-tokens", type=int, default=TARGET_TOKENS)
    args = ap.parse_args()

    manifest = build(Path(args.root), Path(args.out_dir),
                     Path(args.tokenizer), args.total_tokens)
    print(f"\n{'source':24} {'tokens':>14} {'documents':>11}")
    for source in sorted(manifest["tokens_by_source"]):
        print(f"{source:24} {manifest['tokens_by_source'][source]:>14,} "
              f"{manifest['documents_by_source'][source]:>11,}")
    print(f"\nachieved ratio: {manifest['achieved_ratio']}")
    print(f"target ratio  : {manifest['target_ratio']}")
    for split, info in manifest["splits"].items():
        print(f"{split:11} {info['tokens']:>14,} tokens  "
              f"{info['documents']:>9,} docs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
