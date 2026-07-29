"""Per-record contamination matching: how much of an evaluation example was copied.

The decision is *coverage of the evaluation record*, not how many n-grams a
document happened to hit.  A hit count scales with document length, so a
threshold on it systematically keeps long documents and misses short questions --
exactly backwards, since a short Japanese question copied verbatim is the clearer
contamination.

For each candidate evaluation record the matcher reports:

    coverage           fraction of the record's n-grams present in the document
    contiguous         longest run of the record's n-grams matched in order
    substring          whether the whole normalised record appears in the document

and the per-benchmark rules in `RULES` decide from those. A record's parts are
scored separately where the parts mean different things: GSM8K's question is
evidence of contamination, its answer's arithmetic boilerplate is not.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import xxhash

from contamination import (
    JA_CHAR_NGRAM,
    EN_WORD_NGRAM,
    char_ngrams,
    is_cjk,
    norm_chars,
    norm_words,
    word_ngrams,
)

MIN_SUBSTRING_CHARS = 12   # below this a record is too generic to match as a substring


@dataclass
class Rule:
    """When a match counts as contamination, per benchmark."""

    min_coverage: float = 0.80          # fraction of the record matched
    min_contiguous: float = 0.80        # fraction matched as one run
    min_contiguous_units: int = 0       # absolute run length, 0 = unused
    allow_substring: bool = True        # whole record found inside the document
    note: str = ""


RULES: dict[str, Rule] = {
    # Short Japanese QA and multiple choice: the record either appears or it does
    # not.  Substring containment is what catches a short question embedded in a
    # web page; requiring the *whole* record means a bare answer like "4年" can
    # never trigger on its own.
    "jcommonsenseqa": Rule(0.80, 0.80, note="question plus all choices"),
    "jnli": Rule(0.80, 0.80, note="premise plus hypothesis"),
    "niilc": Rule(0.80, 0.80, note="question plus answer, answer never alone"),
    "jmmlu": Rule(0.80, 0.80, note="question plus four choices"),
    "jsquad_qa": Rule(0.80, 0.80, note="question plus answer"),
    # Wikipedia prose: a shared opening definition is not a copy.  Either a long
    # verbatim run or most of the passage.
    "jsquad_context": Rule(0.70, 0.0, min_contiguous_units=100, allow_substring=False,
                           note="100 contiguous characters, or 70% of the passage"),
    # Word problems: the question carries the identity, the worked answer is full
    # of arithmetic boilerplate shared across unrelated problems.
    "gsm8k_question": Rule(0.80, 0.0, min_contiguous_units=20,
                           note="80% of the question, or 20 words verbatim"),
    "gsm8k_answer": Rule(0.90, 0.90, allow_substring=False,
                         note="answers alone are weak evidence; near-total only"),
    # Programming: same task title is not contamination, same code is.
    "humaneval_prompt": Rule(0.80, 0.0, min_contiguous_units=20, note="prompt and signature"),
    "humaneval_solution": Rule(0.80, 0.0, min_contiguous_units=20, note="reference solution"),
    "mbpp_prompt": Rule(0.80, 0.0, min_contiguous_units=15, note="task description"),
    "mbpp_solution": Rule(0.80, 0.0, min_contiguous_units=15, note="reference code"),
}


def part_records() -> dict[str, list[str]]:
    """Canonical records with the part-scored benchmarks split into their parts.

    GSM8K, HumanEval and MBPP are scored on their question/prompt and their
    answer/solution separately: a shared arithmetic phrase or a common algorithm
    name is not contamination, a copied problem statement or reference solution
    is.
    """
    from contamination import canonical_records
    from datasets import load_dataset

    out = {k: v for k, v in canonical_records().items()
           if k not in ("gsm8k", "humaneval", "mbpp")}

    gsm = load_dataset("openai/gsm8k", name="main", split="test")
    out["gsm8k_question"] = [r["question"] for r in gsm]
    out["gsm8k_answer"] = [r["answer"] for r in gsm]

    he = load_dataset("openai/openai_humaneval", split="test")
    out["humaneval_prompt"] = [r["prompt"] for r in he]
    out["humaneval_solution"] = [r["canonical_solution"] for r in he]

    mbpp = load_dataset("google-research-datasets/mbpp", name="full", split="test")
    out["mbpp_prompt"] = [r["text"] for r in mbpp]
    out["mbpp_solution"] = [r["code"] for r in mbpp]
    return out


@dataclass
class RecordIndex:
    """n-gram -> record ids, plus each record's own n-gram sequence length."""

    gram_to_records: dict[int, np.ndarray] = field(default_factory=dict)
    record_length: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    record_benchmark: list[str] = field(default_factory=list)
    record_text: list[str] = field(default_factory=list)
    record_grams: list[list[int]] = field(default_factory=list)
    short_records: list[tuple[int, str]] = field(default_factory=list)
    all_grams: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.uint64))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "record_benchmark": self.record_benchmark,
            "record_text": self.record_text,
            "record_grams": [list(map(str, g)) for g in self.record_grams],
            "short_records": [[i, t] for i, t in self.short_records],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path) -> "RecordIndex":
        payload = json.loads(path.read_text())
        index = cls(
            record_benchmark=payload["record_benchmark"],
            record_text=payload["record_text"],
            record_grams=[[int(x) for x in g] for g in payload["record_grams"]],
            short_records=[(int(i), t) for i, t in payload["short_records"]],
        )
        index.finalise()
        return index

    def finalise(self) -> None:
        mapping: dict[int, list[int]] = {}
        for rid, grams in enumerate(self.record_grams):
            for g in set(grams):
                mapping.setdefault(g, []).append(rid)
        self.gram_to_records = {g: np.array(v, dtype=np.int32)
                                for g, v in mapping.items()}
        self.all_grams = np.array(sorted(mapping), dtype=np.uint64)
        self.record_length = np.array([len(g) for g in self.record_grams],
                                      dtype=np.int32)


def build_record_index(records: dict[str, list[str]]) -> RecordIndex:
    index = RecordIndex()
    for benchmark, texts in records.items():
        for text in texts:
            grams = char_ngrams(text) if is_cjk(text) else word_ngrams(text)
            rid = len(index.record_grams)
            index.record_benchmark.append(benchmark)
            index.record_text.append(text)
            index.record_grams.append(grams)
            key = norm_chars(text) if is_cjk(text) else " ".join(norm_words(text))
            if len(key) >= MIN_SUBSTRING_CHARS:
                index.short_records.append((rid, key))
    index.finalise()
    return index


def _longest_run(flags: list[bool]) -> tuple[int, int]:
    """(length, start index) of the longest contiguous run of matches."""
    best = best_start = run = run_start = 0
    for i, f in enumerate(flags):
        if f:
            if run == 0:
                run_start = i
            run += 1
            if run > best:
                best, best_start = run, run_start
        else:
            run = 0
    return best, best_start


def analyse(text: str, index: RecordIndex, screen: int = 48) -> list[dict]:
    """Every evaluation record this document covers enough to count."""
    doc_char = set(char_ngrams(text))
    doc_word = set(word_ngrams(text))
    doc_grams = doc_char | doc_word
    if not doc_grams:
        return []

    candidates: dict[int, int] = {}
    for g in doc_grams:
        rids = index.gram_to_records.get(g)
        if rids is not None:
            for rid in rids:
                candidates[int(rid)] = candidates.get(int(rid), 0) + 1
    doc_key_chars = norm_chars(text)
    doc_key_words = " ".join(norm_words(text))

    findings = []
    for rid, hits in candidates.items():
        grams = index.record_grams[rid]
        total = len(grams)
        if total == 0:
            continue
        flags = [g in doc_grams for g in grams]
        matched = sum(flags)
        coverage = matched / total
        contiguous, run_start = _longest_run(flags)
        benchmark = index.record_benchmark[rid]
        rule = RULES.get(benchmark)
        if rule is None:
            continue
        record = index.record_text[rid]
        key = norm_chars(record) if is_cjk(record) else " ".join(norm_words(record))
        substring = bool(rule.allow_substring and len(key) >= MIN_SUBSTRING_CHARS
                         and (key in doc_key_chars or key in doc_key_words))
        unit = JA_CHAR_NGRAM if is_cjk(record) else EN_WORD_NGRAM
        contiguous_units = contiguous + unit - 1 if contiguous else 0

        decided = (
            substring
            or (coverage >= rule.min_coverage
                and (rule.min_contiguous == 0
                     or contiguous / total >= rule.min_contiguous))
            or (rule.min_contiguous_units
                and contiguous_units >= rule.min_contiguous_units)
        )
        # the text that actually matched, so a quarantine decision can be read
        # back and judged later without re-running the matcher
        if is_cjk(record):
            span_text = norm_chars(record)[run_start:run_start + contiguous_units]
        else:
            span_text = " ".join(
                norm_words(record)[run_start:run_start + contiguous_units])
        findings.append({
            "benchmark": benchmark,
            "record_id": rid,
            "matched_text": span_text[:300],
            "record_chars": len(key),
            "record_ngrams": total,
            "matched": matched,
            "coverage": round(coverage, 4),
            "contiguous_ngrams": contiguous,
            "contiguous_units": contiguous_units,
            "contiguous_fraction": round(contiguous / total, 4),
            "substring": substring,
            "contaminated": bool(decided),
            "rule": rule.note,
        })

    # records with no n-grams at all are matched purely by substring
    for rid, key in index.short_records:
        if index.record_length[rid] > 0:
            continue
        benchmark = index.record_benchmark[rid]
        rule = RULES.get(benchmark)
        if rule is None or not rule.allow_substring:
            continue
        if key in doc_key_chars or key in doc_key_words:
            findings.append({
                "benchmark": benchmark, "record_id": rid, "record_chars": len(key),
                "record_ngrams": 0, "matched": 0, "coverage": 1.0,
                "contiguous_ngrams": 0, "contiguous_units": 0,
                "contiguous_fraction": 1.0, "substring": True,
                "contaminated": True, "rule": rule.note,
            })
    return findings


def is_contaminated(text: str, index: RecordIndex) -> dict | None:
    findings = analyse(text, index)
    hits = [f for f in findings if f["contaminated"]]
    if not hits:
        return None
    return max(hits, key=lambda f: (f["coverage"], f["contiguous_units"]))


__all__ = ["RecordIndex", "build_record_index", "analyse", "is_contaminated",
           "RULES", "Rule", "part_records"]
