"""Benchmark contamination detection over canonical evaluation records.

Three properties this has to get right, each of which the previous version got
wrong:

**Records, not fields.**  An evaluation example is indexed as one canonical text
-- question plus its choices, premise plus hypothesis, question plus answer --
rather than as separate fields.  Indexing fields separately makes short ones
disappear: a five-character Japanese answer produces no 20-character n-gram at
all, so `NIILC: 4年` was silently unmatched.  Answers are never indexed alone,
because a bare `4年` would mark every ordinary document containing it.

**Matching at the granularity the script supports.**  Japanese has no spaces, so
word n-grams collapse: a 35-character question yields two "words" and zero
13-grams.  Japanese is matched on 20-character n-grams, Latin text on 13-word
n-grams, and records too short for either are matched by exact equality of the
whole normalised record against a document or one of its lines.

**More than one hit before deleting.**  A single matching 20-character phrase is
not evidence: fixed expressions recur across ordinary Japanese text, and JSQuAD
contexts are long Wikipedia prose that legitimately overlaps a web corpus.
Deletion requires either an exact record match or several distinct n-grams whose
matches form a long contiguous span.
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import xxhash

JA_CHAR_NGRAM = 20        # ~ the information of 13 English words
EN_WORD_NGRAM = 13
MIN_SHORT_CHARS = 8       # below this a record is too generic to match on
MIN_DISTINCT_MATCHES = 2  # a single hit is never enough
MIN_SPAN_CHARS = 30       # contiguous matched span required, Japanese
MIN_SPAN_WORDS = 20       # contiguous matched span required, Latin

import re

_LATIN_WORD = re.compile(r"[0-9A-Za-z]+")
_CJK = re.compile(r"[぀-ヿ一-鿿]")

JGLUE_BASE = "https://raw.githubusercontent.com/yahoojapan/JGLUE/v1.1.0/datasets"
NIILC_BASE = "https://raw.githubusercontent.com/mynlp/niilc-qa/master/data"


def is_cjk(text: str, threshold: float = 0.15) -> bool:
    return bool(text) and len(_CJK.findall(text)) / len(text) >= threshold


def norm_chars(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).lower().split())


def norm_words(text: str) -> list[str]:
    return _LATIN_WORD.findall(unicodedata.normalize("NFKC", text).lower())


def char_ngrams(text: str) -> list[int]:
    chars = norm_chars(text)
    if len(chars) < JA_CHAR_NGRAM:
        return []
    return [xxhash.xxh64(chars[i:i + JA_CHAR_NGRAM].encode()).intdigest()
            for i in range(len(chars) - JA_CHAR_NGRAM + 1)]


def word_ngrams(text: str) -> list[int]:
    words = norm_words(text)
    if len(words) < EN_WORD_NGRAM:
        return []
    return [xxhash.xxh64(" ".join(words[i:i + EN_WORD_NGRAM]).encode()).intdigest()
            for i in range(len(words) - EN_WORD_NGRAM + 1)]


def both_granularities(text: str) -> dict[str, list[int]]:
    """Character *and* word n-grams for the same text.

    Granularity cannot be chosen from the document's dominant script: a Japanese
    page containing an English code block is `is_cjk` overall, and matching it
    only with character n-grams misses every English benchmark it might quote.
    Indexing and matching both ways costs a second pass and removes the blind
    spot entirely.
    """
    return {"cjk": char_ngrams(text), "latin": word_ngrams(text)}


# --------------------------------------------------------------------------
# canonical evaluation records
# --------------------------------------------------------------------------

def _get(url: str) -> str:
    import urllib.request

    with urllib.request.urlopen(url, timeout=180) as response:
        return response.read().decode("utf-8")


def _jsonl(text: str) -> list[dict]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def canonical_records() -> dict[str, list[str]]:
    """One text per evaluation example, per benchmark."""
    records: dict[str, list[str]] = {}

    rows = _jsonl(_get(f"{JGLUE_BASE}/jcommonsenseqa-v1.1/valid-v1.1.json"))
    records["jcommonsenseqa"] = [
        " ".join([r["question"], *(r[f"choice{i}"] for i in range(5))]) for r in rows
    ]

    rows = _jsonl(_get(f"{JGLUE_BASE}/jnli-v1.1/valid-v1.1.json"))
    records["jnli"] = [f'{r["sentence1"]} {r["sentence2"]}' for r in rows]

    data = json.loads(_get(f"{JGLUE_BASE}/jsquad-v1.1/valid-v1.1.json"))
    qa_records, context_records = [], []
    for article in data["data"]:
        for para in article["paragraphs"]:
            context = para["context"]
            context_records.append(context)
            for qa in para["qas"]:
                answers = " ".join(a["text"] for a in qa.get("answers", []))
                qa_records.append(f'{qa["question"]} {answers}'.strip())
    records["jsquad_qa"] = qa_records
    # contexts are long Wikipedia prose and are matched on their own, with the
    # span rule doing the work of separating quotation from coincidence
    records["jsquad_context"] = context_records

    import xml.etree.ElementTree as ET

    niilc: list[str] = []
    for name in ("NIILC-ECQA2015_dev.xml", "NIILC-ECQA2015_test.xml"):
        root = ET.fromstring(_get(f"{NIILC_BASE}/{name}"))
        for question in root.iter("question"):
            texts = [n.text.strip() for n in question.iter("text")
                     if n.text and n.text.strip()]
            answers = [n.text.strip() for n in question.iter("answer")
                       if n.text and n.text.strip()]
            if texts:
                # question + answer; an answer on its own is never indexed
                niilc.append(" ".join(texts[:1] + answers))
    records["niilc"] = niilc

    from datasets import load_dataset

    jmmlu = load_dataset("nlp-waseda/JMMLU", split="train",
                         revision="refs/convert/parquet")
    records["jmmlu"] = [
        " ".join(str(r.get(k, "")) for k in ("question", "A", "B", "C", "D"))
        for r in jmmlu
    ]

    gsm = load_dataset("openai/gsm8k", name="main", split="test")
    records["gsm8k"] = [f'{r["question"]} {r["answer"]}' for r in gsm]

    he = load_dataset("openai/openai_humaneval", split="test")
    records["humaneval"] = [f'{r["prompt"]}\n{r["canonical_solution"]}' for r in he]

    mbpp = load_dataset("google-research-datasets/mbpp", name="full", split="test")
    records["mbpp"] = [f'{r["text"]}\n{r["code"]}' for r in mbpp]

    return records


# --------------------------------------------------------------------------
# index
# --------------------------------------------------------------------------

@dataclass
class ContaminationIndex:
    ngrams: np.ndarray                       # sorted uint64
    exact: np.ndarray                        # sorted uint64, whole short records
    meta: dict = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, ngrams=self.ngrams, exact=self.exact)
        path.with_suffix(".meta.json").write_text(
            json.dumps(self.meta, indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path) -> "ContaminationIndex":
        blob = np.load(path)
        meta = json.loads(path.with_suffix(".meta.json").read_text())
        return cls(blob["ngrams"], blob["exact"], meta)


def build_index(records: dict[str, list[str]]) -> ContaminationIndex:
    ngram_set: set[int] = set()
    exact_set: set[int] = set()
    detail: dict = {}

    for name, texts in records.items():
        with_ngrams = short_exact = unmatchable = 0
        produced = 0
        for text in texts:
            grams = both_granularities(text)
            all_grams = grams["cjk"] + grams["latin"]
            if all_grams:
                ngram_set.update(all_grams)
                produced += len(all_grams)
                with_ngrams += 1
                continue
            key = norm_chars(text) if is_cjk(text) else " ".join(norm_words(text))
            if len(key) >= MIN_SHORT_CHARS:
                exact_set.add(xxhash.xxh64(key.encode()).intdigest())
                short_exact += 1
            else:
                unmatchable += 1
        detail[name] = {
            "examples": len(texts),
            "with_ngrams": with_ngrams,
            "short_exact_match": short_exact,
            "unmatchable": unmatchable,
            "coverage": round((with_ngrams + short_exact) / max(len(texts), 1), 4),
            "ngrams": produced,
            "script": "cjk" if any(is_cjk(t) for t in texts[:50]) else "latin",
            "samples": [(norm_chars(t) if is_cjk(t) else " ".join(norm_words(t)))[:70]
                        for t in texts[:3]],
        }
        print(f"[index] {name:16} {len(texts):>7,} examples  "
              f"n-gram {with_ngrams:>7,}  exact {short_exact:>5,}  "
              f"unmatchable {unmatchable:>4,}  coverage "
              f"{detail[name]['coverage']*100:5.1f}%", flush=True)

    meta = {
        "ja_char_ngram": JA_CHAR_NGRAM, "en_word_ngram": EN_WORD_NGRAM,
        "min_short_chars": MIN_SHORT_CHARS,
        "min_distinct_matches": MIN_DISTINCT_MATCHES,
        "min_span_chars": MIN_SPAN_CHARS, "min_span_words": MIN_SPAN_WORDS,
        "benchmarks": detail,
        "total_ngrams": len(ngram_set), "total_exact": len(exact_set),
    }
    return ContaminationIndex(
        np.array(sorted(ngram_set), dtype=np.uint64),
        np.array(sorted(exact_set), dtype=np.uint64),
        meta,
    )


# --------------------------------------------------------------------------
# matching
# --------------------------------------------------------------------------

def _in(index: np.ndarray, values: np.ndarray) -> np.ndarray:
    if index.size == 0 or values.size == 0:
        return np.zeros(values.shape, dtype=bool)
    pos = np.clip(np.searchsorted(index, values), 0, index.size - 1)
    return index[pos] == values


def check_document(text: str, index: ContaminationIndex,
                   screen_samples: int = 48) -> dict | None:
    """Decide whether a training document is contaminated, and say why.

    Cheap screen first: if a handful of sampled n-grams miss entirely, the
    document is clean and no full scan happens.  Only survivors pay for the
    exact span analysis, which is what makes a per-document rule affordable
    over millions of documents.
    """
    # exact match: the document, or one of its lines, *is* a short eval record
    if index.exact.size:
        keys = [norm_chars(text) if is_cjk(text) else " ".join(norm_words(text))]
        for line in text.split("\n"):
            line = line.strip()
            if line:
                keys.append(norm_chars(line) if is_cjk(line) else " ".join(norm_words(line)))
        probe = np.array([xxhash.xxh64(k.encode()).intdigest()
                          for k in keys if len(k) >= MIN_SHORT_CHARS], dtype=np.uint64)
        if probe.size and _in(index.exact, probe).any():
            return {"reason": "exact_record", "distinct": 1, "span": len(text)}

    grams = both_granularities(text)

    # cheap screen across both granularities before any full scan
    screened = False
    for series in grams.values():
        if not series:
            continue
        step = max(1, len(series) // screen_samples)
        probe = np.array(series[::step][:screen_samples], dtype=np.uint64)
        if _in(index.ngrams, probe).any():
            screened = True
            break
    if not screened:
        return None

    best_verdict = None
    for script, series in grams.items():
        if not series:
            continue
        values = np.array(series, dtype=np.uint64)
        hits = _in(index.ngrams, values)
        distinct = int(np.unique(values[hits]).size)
        if distinct < MIN_DISTINCT_MATCHES:
            continue
        best = run = 0
        for hit in hits:
            run = run + 1 if hit else 0
            best = max(best, run)
        if best == 0:
            continue
        unit = JA_CHAR_NGRAM if script == "cjk" else EN_WORD_NGRAM
        span = best + unit - 1
        minimum = MIN_SPAN_CHARS if script == "cjk" else MIN_SPAN_WORDS
        if span < minimum:
            continue
        verdict = {"reason": "ngram_span", "distinct": distinct,
                   "span": int(span), "script": script}
        if best_verdict is None or span > best_verdict["span"]:
            best_verdict = verdict
    return best_verdict


__all__ = [
    "ContaminationIndex", "build_index", "canonical_records", "check_document",
    "is_cjk", "norm_chars", "norm_words", "char_ngrams", "word_ngrams",
    "both_granularities",
]
