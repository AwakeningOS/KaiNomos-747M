"""KaiNomos-DataMix-v1: acquisition of the fixed-ratio pre-training mix.

Ratios are defined over tokens *in this project's tokenizer*, not over the token
counts the sources advertise -- a corpus counted with a different tokenizer says
nothing about how much of it this model will actually read.  Acquisition
therefore collects raw text against a per-source byte target derived from a
measured bytes-per-token estimate, and the final ratios are enforced again at
tokenisation time, where the true counts are known.

    Japanese Organic Web      35%
    Japanese Paraphrase       20%
    Japanese Document-Instruct 10%
    Japanese Reference        10%
    English Educational Web   10%
    Educational Code          10%
    Math / Reasoning           5%

Stage order matters and is not the obvious one:

    collect -> light clean -> tokenizer -> MinHash -> contamination -> tokenize

The light clean (Unicode normalisation, boilerplate stripping, exact-duplicate
and repeated-line removal) runs *before* the tokenizer is trained.  Training a
32,768-piece vocabulary on unwashed web text spends pieces on navigation chrome,
cookie banners and mojibake -- vocabulary slots are a fixed budget, and anything
spent there is unavailable to Japanese.  The expensive near-duplicate pass can
wait until after, because it changes how often text appears rather than what the
text looks like.

Every stage is resumable: an interrupted download continues from the shard it
reached, never from zero.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import xxhash

SPLIT_SEED = 20260729
TARGET_TOKENS = 2_500_000_000

# Japanese needs ~1.5 characters per token with a Japanese-aware vocabulary,
# and a Japanese character is 3 UTF-8 bytes, so ~4.5 bytes/token.  Latin text and
# code sit near 3.6.  These only size the download; the real ratios are applied
# after tokenisation.
BYTES_PER_TOKEN = {"ja": 4.5, "en": 3.6, "code": 3.4, "math": 3.6}


@dataclass
class Source:
    name: str
    ratio: float
    kind: str                      # ja | en | code | math
    loader: dict                   # kwargs for datasets.load_dataset
    text_field: str = "text"
    note: str = ""
    filters: dict = field(default_factory=dict)

    def target_bytes(self, total_tokens: int = TARGET_TOKENS) -> int:
        return int(total_tokens * self.ratio * BYTES_PER_TOKEN[self.kind])


SOURCES: list[Source] = [
    Source(
        "ja_web", 0.35, "ja",
        dict(path="llm-jp/scaling-data-constrained-llms", data_dir="data/ja_web_9b",
             split="train", streaming=True),
        note="FineWeb2 Japanese, deduplicated and filtered per language",
    ),
    Source(
        "ja_paraphrase", 0.20, "ja",
        dict(path="llm-jp/scaling-data-constrained-llms", data_dir="data/ja_paraphrase_63b",
             split="train", streaming=True),
        note="rewrites of Japanese web text; semantic overlap with ja_web is the "
             "point of the source and must survive deduplication",
    ),
    Source(
        "ja_instruct", 0.10, "ja",
        dict(path="llm-jp/scaling-data-constrained-llms", data_dir="data/ja_instruct_63b",
             split="train", streaming=True),
        note="document-grounded QA, not open-domain chat",
    ),
    # DEVIATION from the design: this slice was specified as a Reference *mix*
    # (Wikipedia 70 / Wikibooks 10 / Wikiversity 5 / government documents 15).
    # Japanese Wikibooks and Wikiversity have no published dataset, and the
    # e-Gov corpus did not load reliably, so the slice is Wikipedia alone.  It is
    # named for what it is rather than for what was intended; formal and
    # administrative Japanese is therefore *not* covered and should be added back
    # if that register matters.
    Source(
        "ja_wikipedia_reference", 0.10, "ja",
        dict(path="wikimedia/wikipedia", name="20231101.ja", split="train", streaming=True),
        note="Wikipedia only -- NOT the full Reference mix; see the deviation note",
    ),
    Source(
        "en_edu", 0.10, "en",
        dict(path="HuggingFaceTB/dclm-edu", split="train", streaming=True),
        filters={"int_score_min": 3},
        note="educational English only; English is here for transfer, not coverage",
    ),
    # Stack-Edu is split per language, and the mix inside the 10% code budget is
    # deliberate: at 110M the useful thing to learn is the correspondence between
    # prose and code, not memorised libraries, so Markdown and docstring-heavy
    # Python dominate and no single systems language gets much room.
    *[
        Source(
            f"code_{name.lower()}", 0.10 * share, "code",
            dict(path="HuggingFaceTB/stack-edu", name=name, split="train",
                 streaming=True),
            text_field="text",
            note="educational code; docstrings and README prose kept deliberately",
        )
        for name, share in (
            ("Python", 0.50),
            ("Markdown", 0.20),
            ("JavaScript", 0.06),
            ("TypeScript", 0.04),
            ("Shell", 0.08),
            ("SQL", 0.07),
            ("C", 0.03),
            ("Cpp", 0.02),
        )
    ],
    Source(
        "math", 0.05, "math",
        dict(path="HuggingFaceTB/finemath", name="finemath-4plus", split="train",
             streaming=True),
        note="worked explanations rather than bare formulae",
    ),
]


@dataclass
class SourceState:
    documents: int = 0
    bytes_written: int = 0
    shards: int = 0
    records_seen: int = 0
    done: bool = False


class MixCollector:
    def __init__(self, root: Path, total_tokens: int = TARGET_TOKENS,
                 shard_documents: int = 50_000):
        self.root = root
        self.total_tokens = total_tokens
        self.shard_documents = shard_documents
        self.state_path = root / "collect_state.json"
        self.state: dict[str, SourceState] = {}
        if self.state_path.exists():
            raw = json.loads(self.state_path.read_text())
            self.state = {k: SourceState(**v) for k, v in raw.items()}

    def save(self) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({k: v.__dict__ for k, v in self.state.items()}, indent=2))
        os.replace(tmp, self.state_path)

    def collect(self, source: Source) -> SourceState:
        from datasets import load_dataset

        out_dir = self.root / source.name
        out_dir.mkdir(parents=True, exist_ok=True)
        st = self.state.setdefault(source.name, SourceState())
        target = source.target_bytes(self.total_tokens)
        if st.done or st.bytes_written >= target:
            st.done = True
            return st

        print(f"[{source.name}] target {target/1e9:.2f} GB "
              f"(have {st.bytes_written/1e9:.2f} GB), skipping {st.records_seen:,} records",
              flush=True)

        stream = load_dataset(**source.loader)
        if st.records_seen:
            stream = stream.skip(st.records_seen)

        buffer: list[str] = []
        t0 = time.time()

        def flush() -> None:
            if not buffer:
                return
            path = out_dir / f"shard_{st.shards:05d}.jsonl.gz"
            tmp = path.with_suffix(".tmp")
            with gzip.open(tmp, "wt", encoding="utf-8") as fh:
                fh.write("".join(buffer))
            os.replace(tmp, path)
            st.shards += 1
            buffer.clear()
            self.save()

        min_score = source.filters.get("int_score_min")
        for record in stream:
            st.records_seen += 1
            text = record.get(source.text_field) or ""
            if not text:
                continue
            if min_score is not None:
                score = record.get("int_score", record.get("edu_int_score"))
                if score is not None and score < min_score:
                    continue
            key = xxhash.xxh64(text.encode("utf-8"), seed=SPLIT_SEED).hexdigest()
            buffer.append(json.dumps(
                {"key": key, "source": source.name, "text": text}, ensure_ascii=False
            ) + "\n")
            st.documents += 1
            st.bytes_written += len(text.encode("utf-8"))
            if len(buffer) >= self.shard_documents:
                flush()
                print(f"[{source.name}] {st.bytes_written/1e9:.2f}/{target/1e9:.2f} GB "
                      f"{st.documents:,} docs {time.time()-t0:.0f}s", flush=True)
            if st.bytes_written >= target:
                break

        flush()
        st.done = st.bytes_written >= target
        self.save()
        print(f"[{source.name}] {'done' if st.done else 'exhausted'}: "
              f"{st.bytes_written/1e9:.2f} GB, {st.documents:,} docs", flush=True)
        return st


def write_manifest(root: Path, collector: "MixCollector") -> dict:
    """Record enough to rebuild this exact pool later.

    A ratio table alone cannot be reproduced: the same dataset name can point at
    different content after a revision, and the stream position decides which
    rows were actually taken.
    """
    import time as _time

    from huggingface_hub import HfApi

    api = HfApi()
    entries = []
    for source in SOURCES:
        state = collector.state.get(source.name)
        if state is None and source.name.startswith("code_"):
            # code sources keep their own state file, written by data_code.py
            code_state = root / source.name / "code_state.json"
            if code_state.exists():
                raw = json.loads(code_state.read_text())
                state = SourceState(
                    documents=raw.get("documents", 0),
                    bytes_written=raw.get("bytes_written", 0),
                    shards=raw.get("shards", 0),
                    records_seen=raw.get("records_seen", 0),
                    done=raw.get("done", False),
                )
        info = {}
        try:
            repo = api.repo_info(source.loader["path"], repo_type="dataset")
            info = {"revision_sha": repo.sha,
                    "license": (repo.card_data or {}).get("license")
                    if hasattr(repo, "card_data") else None}
        except Exception as exc:
            info = {"revision_sha": None, "error": repr(exc)[:200]}
        entries.append({
            "source": source.name,
            "ratio": source.ratio,
            "kind": source.kind,
            "dataset": source.loader["path"],
            "config": source.loader.get("name"),
            "data_dir": source.loader.get("data_dir"),
            "split": source.loader.get("split"),
            "filters": source.filters,
            "note": source.note,
            "documents": state.documents if state else 0,
            "bytes": state.bytes_written if state else 0,
            "shards": state.shards if state else 0,
            # the stream range actually consumed, so the same rows can be retaken
            "records_consumed": state.records_seen if state else 0,
            **info,
        })
    manifest = {
        "name": "KaiNomos-DataMix-v1",
        "created_utc": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
        "requested_target_tokens": collector.total_tokens,
        "split_seed": SPLIT_SEED,
        "bytes_per_token_estimate": BYTES_PER_TOKEN,
        "stage_order": ["collect", "clean", "tokenizer", "minhash",
                        "contamination", "tokenize"],
        "sources": entries,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def plan(total_tokens: int = TARGET_TOKENS) -> list[dict]:
    return [
        {
            "source": s.name, "ratio": s.ratio, "kind": s.kind,
            "target_tokens": int(total_tokens * s.ratio),
            "target_gb": round(s.target_bytes(total_tokens) / 1e9, 2),
            "dataset": s.loader.get("path"),
            "subset": s.loader.get("data_dir") or s.loader.get("name", ""),
        }
        for s in SOURCES
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["plan", "collect", "manifest"])
    ap.add_argument("--root", default="data/mix")
    ap.add_argument("--total-tokens", type=int, default=TARGET_TOKENS)
    ap.add_argument("--only", default=None, help="comma-separated source names")
    args = ap.parse_args()

    if args.command == "plan":
        rows = plan(args.total_tokens)
        total = sum(r["target_gb"] for r in rows)
        print(f"{'source':16} {'ratio':>6} {'tokens':>14} {'raw GB':>8}  dataset")
        for r in rows:
            print(f"{r['source']:16} {r['ratio']*100:5.0f}% {r['target_tokens']:>14,} "
                  f"{r['target_gb']:>8.2f}  {r['dataset']} {r['subset']}")
        print(f"{'TOTAL':16} {sum(r['ratio'] for r in rows)*100:5.0f}% "
              f"{sum(r['target_tokens'] for r in rows):>14,} {total:>8.2f}")
        return 0

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    collector = MixCollector(root, args.total_tokens)

    if args.command == "manifest":
        manifest = write_manifest(root, collector)
        print(json.dumps({k: v for k, v in manifest.items() if k != "sources"}, indent=2))
        for entry in manifest["sources"]:
            print(f"  {entry['source']:24} {entry['documents']:>10,} docs  "
                  f"{entry['bytes']/1e9:>6.2f} GB  rev {str(entry['revision_sha'])[:12]}")
        return 0

    wanted = set(args.only.split(",")) if args.only else None
    for source in SOURCES:
        if wanted and source.name not in wanted:
            continue
        collector.collect(source)
    write_manifest(root, collector)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
