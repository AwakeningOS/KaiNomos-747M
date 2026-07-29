"""Near-duplicate removal and evaluation-contamination filtering.

Runs *after* the tokenizer is trained: these passes change how often text
appears, not what it looks like, so the vocabulary does not depend on them.

Two rules govern what is removed, and one governs what is not:

* **near-duplicates** are dropped by MinHash/LSH over character 5-gram shingles.
  Repeating the same document teaches the model that document's wording rather
  than the language.
* **evaluation contamination** is dropped by exact normalised 13-gram overlap
  with the benchmark sets this model will be measured on. A model that has read
  the test set does not tell you anything about generalisation.
* **paraphrase is kept.** `ja_paraphrase` restates `ja_web` content by design;
  it is semantically near-duplicate on purpose. Dropping it would delete the
  source's entire reason for existing, so paraphrase documents are exempt from
  the near-duplicate pass and only exact byte-duplicates are removed from them.
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

import numpy as np
import xxhash

SHINGLE = 5          # characters per shingle; short enough for Japanese
NUM_PERM = 128       # MinHash permutations
BANDS = 16           # LSH bands -> threshold ~ (1/BANDS)^(1/rows) with rows=8
NGRAM_N = 13         # contamination n-gram, in normalised word tokens

# Sources whose whole purpose is to restate other sources.
PARAPHRASE_SOURCES = {"ja_paraphrase", "ja_instruct"}

_LATIN_WORD = re.compile(r"[0-9A-Za-z]+")
_CJK = re.compile(r"[぀-ヿ一-鿿]")
# Japanese has no whitespace, so a "word" n-gram over it collapses to a handful
# of punctuation-delimited runs: a 35-character question yields two tokens and
# therefore zero 13-grams.  Contamination detection over Japanese has to work on
# characters.  20 characters carries roughly the information of 13 English words.
JA_CHAR_NGRAM = 20

# Benchmarks this model will be evaluated on.  Only the *evaluation samples*
# belong here -- questions, answers, choices, contexts.  A retrieval corpus that
# a benchmark happens to search over is not contamination: learning general
# knowledge from Wikipedia is the point of pre-training, and indexing all of
# Japanese Wikipedia here would delete the reference slice from training while
# catching nothing that matters.
#
# JGLUE is read from Yahoo's official fixed-tag JSON and NIILC from its official
# XML, both fetched as plain files.  No third-party dataset script is executed.
JGLUE_BASE = "https://raw.githubusercontent.com/yahoojapan/JGLUE/v1.1.0/datasets"
NIILC_BASE = "https://raw.githubusercontent.com/mynlp/niilc-qa/master/data"

HF_EVAL_SOURCES = [
    ("jmmlu", dict(path="nlp-waseda/JMMLU", split="train",
                   revision="refs/convert/parquet"),
     ("question", "A", "B", "C", "D")),
    ("gsm8k", dict(path="openai/gsm8k", name="main", split="test"),
     ("question", "answer")),
    ("humaneval", dict(path="openai/openai_humaneval", split="test"),
     ("prompt", "canonical_solution")),
    ("mbpp", dict(path="google-research-datasets/mbpp", name="full", split="test"),
     ("text", "code")),
]


def _get(url: str) -> str:
    import urllib.request

    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read().decode("utf-8")


def load_jglue() -> dict[str, list[str]]:
    """JCommonsenseQA, JNLI and JSQuAD evaluation samples as flat texts."""
    out: dict[str, list[str]] = {}

    rows = [json.loads(line) for line in
            _get(f"{JGLUE_BASE}/jcommonsenseqa-v1.1/valid-v1.1.json").splitlines() if line]
    out["jcommonsenseqa"] = [
        t for r in rows
        for t in [r["question"], *(r[f"choice{i}"] for i in range(5))] if t
    ]

    rows = [json.loads(line) for line in
            _get(f"{JGLUE_BASE}/jnli-v1.1/valid-v1.1.json").splitlines() if line]
    out["jnli"] = [t for r in rows for t in (r["sentence1"], r["sentence2"]) if t]

    data = json.loads(_get(f"{JGLUE_BASE}/jsquad-v1.1/valid-v1.1.json"))
    texts = []
    for article in data["data"]:
        for para in article["paragraphs"]:
            texts.append(para["context"])
            for qa in para["qas"]:
                texts.append(qa["question"])
                texts.extend(a["text"] for a in qa.get("answers", []))
    out["jsquad"] = [t for t in texts if t]
    return out


def load_niilc() -> list[str]:
    """NIILC dev and test: the questions and their answers, nothing else."""
    import xml.etree.ElementTree as ET

    texts: list[str] = []
    for name in ("NIILC-ECQA2015_dev.xml", "NIILC-ECQA2015_test.xml"):
        root = ET.fromstring(_get(f"{NIILC_BASE}/{name}"))
        for question in root.iter("question"):
            for tag in ("text", "answers", "answer"):
                for node in question.iter(tag):
                    if node.text and node.text.strip():
                        texts.append(node.text.strip())
    return texts


def normalise_words(text: str) -> list[str]:
    """Latin word tokens.  Only meaningful for scripts that separate words."""
    return _LATIN_WORD.findall(unicodedata.normalize("NFKC", text).lower())


def is_cjk(text: str, threshold: float = 0.15) -> bool:
    if not text:
        return False
    return len(_CJK.findall(text)) / len(text) >= threshold


def normalise_chars(text: str) -> str:
    """NFKC, lowercased, whitespace removed -- the unit Japanese matching uses."""
    return "".join(unicodedata.normalize("NFKC", text).lower().split())


def text_ngrams(text: str) -> list[int]:
    """Hashed n-grams, at the granularity the script actually supports.

    Japanese goes through character n-grams and Latin text through word
    n-grams; using word n-grams for both is what silently reduced every Japanese
    benchmark to a handful of entries.
    """
    if is_cjk(text):
        chars = normalise_chars(text)
        if len(chars) < JA_CHAR_NGRAM:
            return []
        return [xxhash.xxh64(chars[i:i + JA_CHAR_NGRAM].encode()).intdigest()
                for i in range(len(chars) - JA_CHAR_NGRAM + 1)]
    words = normalise_words(text)
    if len(words) < NGRAM_N:
        return []
    return [xxhash.xxh64(" ".join(words[i:i + NGRAM_N]).encode()).intdigest()
            for i in range(len(words) - NGRAM_N + 1)]


def shingles(text: str, size: int = SHINGLE) -> set[int]:
    text = unicodedata.normalize("NFKC", text)
    text = "".join(text.split())          # whitespace is not a Japanese boundary
    if len(text) < size:
        return set()
    return {xxhash.xxh64(text[i:i + size].encode()).intdigest()
            for i in range(len(text) - size + 1)}


def minhash(sig_shingles: set[int], seeds: np.ndarray) -> np.ndarray:
    if not sig_shingles:
        return np.full(len(seeds), np.uint64(-1), dtype=np.uint64)
    values = np.fromiter(sig_shingles, dtype=np.uint64, count=len(sig_shingles))
    # xor-shift style permutation family; cheap and adequate for LSH bucketing
    return np.array([np.min(values ^ np.uint64(s)) for s in seeds], dtype=np.uint64)


def band_keys(signature: np.ndarray, bands: int = BANDS) -> list[int]:
    rows = len(signature) // bands
    return [xxhash.xxh64(signature[b * rows:(b + 1) * rows].tobytes()).intdigest()
            for b in range(bands)]


# --------------------------------------------------------------------------
# contamination index
# --------------------------------------------------------------------------

def _index_texts(name: str, texts: list[str], hashes: set[int], meta: dict,
                 detail: dict) -> None:
    """Add one benchmark's n-grams and record enough to see it worked.

    A load that returns `ok` proves nothing about whether the content was read:
    the first version of this index reported `ok` for every Japanese benchmark
    while extracting four n-grams in total.
    """
    before = len(hashes)
    empty = sum(1 for t in texts if not (t or "").strip())
    for text in texts:
        hashes.update(text_ngrams(text))
    produced = len(hashes) - before
    detail[name] = {
        "status": "ok",
        "texts": len(texts),
        "empty_texts": empty,
        "ngrams": produced,
        "ngrams_per_text": round(produced / max(len(texts), 1), 2),
        "script": "cjk" if any(is_cjk(t) for t in texts[:50]) else "latin",
        "samples": [normalise_chars(t)[:60] if is_cjk(t) else " ".join(normalise_words(t))[:60]
                    for t in texts[:3]],
    }
    print(f"[eval-ngrams] {name}: {len(texts):,} texts -> {produced:,} n-grams "
          f"({detail[name]['ngrams_per_text']}/text, {detail[name]['script']})", flush=True)
    if produced == 0:
        print(f"[eval-ngrams] {name}: EXTRACTED NOTHING -- check the field names",
              flush=True)


def build_eval_ngrams(cache: Path) -> tuple[np.ndarray, dict]:
    meta_path = cache.with_suffix(".meta.json")
    if cache.exists() and meta_path.exists():
        return np.load(cache), json.loads(meta_path.read_text())

    from datasets import load_dataset

    hashes: set[int] = set()
    detail: dict = {}
    meta: dict = {"word_ngram_n": NGRAM_N, "char_ngram_n": JA_CHAR_NGRAM,
                  "sources": detail}

    try:
        for name, texts in load_jglue().items():
            _index_texts(name, texts, hashes, meta, detail)
    except Exception as exc:
        detail["jglue"] = {"status": "failed", "error": repr(exc)[:300]}
        print(f"[eval-ngrams] JGLUE FAILED: {exc!r}"[:200], flush=True)

    try:
        _index_texts("niilc", load_niilc(), hashes, meta, detail)
    except Exception as exc:
        detail["niilc"] = {"status": "failed", "error": repr(exc)[:300]}
        print(f"[eval-ngrams] NIILC FAILED: {exc!r}"[:200], flush=True)

    for name, kwargs, fields in HF_EVAL_SOURCES:
        try:
            data = load_dataset(**kwargs)
            texts = []
            for record in data:
                for field_name in fields:
                    value = record.get(field_name)
                    for text in (value if isinstance(value, list) else [value]):
                        if isinstance(text, str):
                            texts.append(text)
            _index_texts(name, texts, hashes, meta, detail)
            detail[name]["revision"] = kwargs.get("revision", "main")
            detail[name]["split"] = kwargs.get("split")
        except Exception as exc:
            detail[name] = {"status": "failed", "error": repr(exc)[:300]}
            print(f"[eval-ngrams] {name}: FAILED {exc!r}"[:200], flush=True)

    array = np.array(sorted(hashes), dtype=np.uint64)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, array)
    meta["total_ngrams"] = int(array.size)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return array, meta


def is_contaminated(text: str, index: np.ndarray, samples: int = 64) -> bool:
    if index.size == 0:
        return False
    grams = text_ngrams(text)
    if not grams:
        return False
    step = max(1, len(grams) // samples)
    probe = np.array(grams[::step][:samples], dtype=np.uint64)
    pos = np.clip(np.searchsorted(index, probe), 0, index.size - 1)
    return bool((index[pos] == probe).any())


# --------------------------------------------------------------------------
# main pass
# --------------------------------------------------------------------------

@dataclass
class DedupStats:
    documents_in: int = 0
    documents_out: int = 0
    near_duplicates: int = 0
    contaminated: int = 0
    paraphrase_exempt: int = 0
    shards: int = 0
    bytes_out: int = 0
    per_source: dict = field(default_factory=dict)


def run(clean_root: Path, out_root: Path, cache: Path,
        shard_documents: int = 50_000, seed: int = 20260729) -> DedupStats:
    out_root.mkdir(parents=True, exist_ok=True)
    index, meta = build_eval_ngrams(cache)
    print(f"[contamination] {index.size:,} benchmark {NGRAM_N}-grams", flush=True)

    rng = np.random.default_rng(seed)
    seeds = rng.integers(1, 2**63, size=NUM_PERM, dtype=np.uint64)
    buckets: dict[int, int] = {}
    stats = DedupStats()
    buffer: list[str] = []
    t0 = time.time()

    def flush() -> None:
        if not buffer:
            return
        path = out_root / f"dedup_{stats.shards:05d}.jsonl.gz"
        tmp = path.with_suffix(".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as fh:
            fh.write("".join(buffer))
        os.replace(tmp, path)
        stats.shards += 1
        buffer.clear()

    for shard in sorted(clean_root.glob("clean_*.jsonl.gz")):
        with gzip.open(shard, "rt", encoding="utf-8") as fh:
            for line in fh:
                record = json.loads(line)
                stats.documents_in += 1
                text, source = record["text"], record.get("source", "")

                if is_contaminated(text, index):
                    stats.contaminated += 1
                    continue

                if source in PARAPHRASE_SOURCES:
                    stats.paraphrase_exempt += 1
                else:
                    signature = minhash(shingles(text), seeds)
                    keys = band_keys(signature)
                    if any(k in buckets for k in keys):
                        stats.near_duplicates += 1
                        continue
                    for k in keys:
                        buckets[k] = 1

                stats.documents_out += 1
                size = len(text.encode("utf-8"))
                stats.bytes_out += size
                per = stats.per_source.setdefault(source, {"documents": 0, "bytes": 0})
                per["documents"] += 1
                per["bytes"] += size
                buffer.append(line)
                if len(buffer) >= shard_documents:
                    flush()
                    print(f"[dedup] {stats.documents_out:,}/{stats.documents_in:,} kept  "
                          f"near-dup {stats.near_duplicates:,}  "
                          f"contaminated {stats.contaminated:,}  "
                          f"{time.time()-t0:.0f}s", flush=True)
    flush()
    (out_root / "dedup_stats.json").write_text(
        json.dumps({**stats.__dict__, "contamination_sources": meta.get("sources", {})},
                   indent=2, ensure_ascii=False))
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clean-root", default="data/clean")
    ap.add_argument("--out-root", default="data/dedup")
    ap.add_argument("--cache", default="data/eval_ngrams.npy")
    args = ap.parse_args()

    stats = run(Path(args.clean_root), Path(args.out_root), Path(args.cache))
    kept = 100.0 * stats.documents_out / max(stats.documents_in, 1)
    print(json.dumps(stats.__dict__, indent=2, ensure_ascii=False)[:1200])
    print(f"\nkept {stats.documents_out:,}/{stats.documents_in:,} ({kept:.1f}%)  "
          f"{stats.bytes_out/1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
