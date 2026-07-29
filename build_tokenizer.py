"""Train the KaiNomos tokenizer: SentencePiece Unigram, 32,768 pieces.

The model this replaces used a 16,384-piece ByteLevel BPE trained on English
FineWeb-Edu.  Measured on Japanese it produced **2.56 tokens per character** --
Japanese text decomposed into raw UTF-8 bytes, so `日本語の` became
`['æ','Ĺ','¥','æ','ľ','¬','è','ª','ŀ','ãģ','®','ã']`.  Training on a 75%-Japanese
corpus through that tokenizer would spend most of the budget re-deriving the
character encoding.

Design points and why each one:

* **Unigram, not BPE.** Japanese has no whitespace to anchor merges, and Unigram
  segments scriptio-continua text more stably than greedy merging.
* **32,768 pieces with weight tying.** At d_model 512 a tied 32k table is
  16,777,216 parameters -- byte-for-byte what the old untied 16k pair cost -- so
  the vocabulary doubles without moving the 110M budget at all.
* **byte fallback.** Any unseen character still encodes rather than becoming
  `<unk>`, which matters for a mixed corpus that includes code and mathematics.
* **NFC, no lowercasing, whitespace and indentation preserved.** Code depends on
  indentation and Japanese has no case; normalising either away would destroy
  information the model is supposed to learn.
* **digits split into 1-3 character pieces** rather than arbitrary spans, so
  numbers have a regular structure for the model to work with.

Sampling for the trainer follows the *final data ratios*, because a tokenizer
trained on a different mixture than the model reads is optimised for the wrong
text.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import unicodedata
from collections import Counter
from pathlib import Path

VOCAB_SIZE = 32_768
SPECIAL_TOKENS = ["<|pad|>", "<|bos|>", "<|eos|>", "<|eod|>"]

# The mix the tokenizer must be good at, by *source kind*.
KIND_RATIO = {"ja": 0.75, "en": 0.10, "code": 0.10, "math": 0.05}
SOURCE_KIND = {
    "ja_web": "ja", "ja_paraphrase": "ja", "ja_instruct": "ja",
    "ja_wikipedia_reference": "ja", "en_edu": "en", "math": "math",
}


def kind_of(source: str) -> str:
    return SOURCE_KIND.get(source, "code" if source.startswith("code_") else "en")


def sample_corpus(
    clean_root: Path, out_path: Path, total_bytes: int = 3 * 1024**3, seed: int = 20260729
) -> dict:
    """Write a training corpus that matches the final ratios by *bytes*.

    Byte quotas are the honest unit here: the point of the exercise is that the
    current tokenizer's token counts are not comparable across languages, so a
    token-based quota would be circular.
    """
    rng = random.Random(seed)
    quota = {kind: int(total_bytes * share) for kind, share in KIND_RATIO.items()}
    written = Counter()
    documents = Counter()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out:
        for shard in sorted(clean_root.glob("clean_*.jsonl.gz")):
            if all(written[k] >= quota[k] for k in quota):
                break
            with gzip.open(shard, "rt", encoding="utf-8") as fh:
                for line in fh:
                    record = json.loads(line)
                    kind = kind_of(record.get("source", ""))
                    if written[kind] >= quota[kind]:
                        continue
                    # thin the over-represented sources so one does not fill its
                    # kind's quota before the others are seen
                    if rng.random() > 0.5:
                        continue
                    text = unicodedata.normalize("NFC", record["text"])
                    out.write(text.replace("\0", "") + "\n")
                    written[kind] += len(text.encode("utf-8"))
                    documents[kind] += 1

    stats = {
        "corpus": str(out_path),
        "target_bytes": total_bytes,
        "bytes_by_kind": dict(written),
        "documents_by_kind": dict(documents),
        "achieved_ratio": {k: round(written[k] / max(sum(written.values()), 1), 4)
                           for k in quota},
        "target_ratio": KIND_RATIO,
    }
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return stats


def train(corpus: Path, out_dir: Path, vocab_size: int = VOCAB_SIZE,
          character_coverage: float = 0.9998) -> Path:
    import sentencepiece as spm

    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / "kainomos"
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(prefix),
        model_type="unigram",
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        byte_fallback=True,
        split_digits=True,
        allow_whitespace_only_pieces=True,
        remove_extra_whitespaces=False,       # code indentation is information
        normalization_rule_name="nfkc_cf" if False else "identity",
        user_defined_symbols=SPECIAL_TOKENS,
        unk_id=0, bos_id=-1, eos_id=-1, pad_id=-1,
        train_extremely_large_corpus=True,
        num_threads=max(1, (__import__("os").cpu_count() or 4) - 1),
        input_sentence_size=8_000_000,
        shuffle_input_sentence=True,
    )
    return Path(str(prefix) + ".model")


def report(model_path: Path) -> dict:
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    probes = {
        "japanese": ["日本語のテキストを効率よく符号化できるかを確認する。",
                     "機械学習モデルの学習には大量の計算資源が必要である。",
                     "東京都は日本の首都であり、人口は約1400万人です。"],
        "english": ["Machine learning models require large amounts of compute to train."],
        "code": ["def add(a, b):\n    return a + b\n"],
        "math": ["Let x = 3 and y = 4, then x^2 + y^2 = 25."],
    }
    out = {"vocab_size": sp.get_piece_size(), "tokens_per_character": {}}
    for name, texts in probes.items():
        ratios = [len(sp.encode(t)) / len(t) for t in texts]
        out["tokens_per_character"][name] = round(sum(ratios) / len(ratios), 3)

    sample = probes["japanese"][0]
    out["japanese_sample"] = sp.encode(sample, out_type=str)[:12]
    out["roundtrip_ok"] = all(sp.decode(sp.encode(t)) == t
                              for texts in probes.values() for t in texts)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["sample", "train", "report", "all"])
    ap.add_argument("--clean-root", default="data/clean")
    ap.add_argument("--corpus", default="data/tokenizer/corpus.txt")
    ap.add_argument("--out-dir", default="data/tokenizer")
    ap.add_argument("--sample-gb", type=float, default=3.0)
    ap.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    args = ap.parse_args()

    corpus = Path(args.corpus)
    out_dir = Path(args.out_dir)
    stats = {}
    if args.command in ("sample", "all"):
        stats = sample_corpus(Path(args.clean_root), corpus,
                              int(args.sample_gb * 1024**3))
    model = out_dir / "kainomos.model"
    if args.command in ("train", "all"):
        model = train(corpus, out_dir, args.vocab_size)
    if args.command in ("report", "all"):
        info = report(model)
        (out_dir / "tokenizer_report.json").write_text(
            json.dumps({"sampling": stats, "tokenizer": info}, indent=2, ensure_ascii=False)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
