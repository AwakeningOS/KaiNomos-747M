"""Pack text or JSONL documents into KaiNomos uint16 shards.

Each non-empty plain-text line is one document. JSONL input must contain a
string ``text`` field. Every document receives the tokenizer's EOD token.
Running the command again for another source or split updates ``manifest.json``.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import sentencepiece as spm

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOKENIZER = ROOT / "tokenizer" / "kainomos-49152.model"
VOCAB_SIZE = 49_152
EOD_ID = 4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def open_text(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def documents(paths: list[Path]):
    for path in paths:
        jsonl = path.name.endswith((".jsonl", ".jsonl.gz"))
        with open_text(path) as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                if jsonl:
                    value = json.loads(line)
                    text = value.get("text")
                    if not isinstance(text, str):
                        raise TypeError(f"{path}:{line_number} needs a text field")
                else:
                    text = line.rstrip("\r\n")
                if text:
                    yield text


def safe_source_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(
            "--source-id must use letters, digits, dot, underscore, or hyphen"
        )
    return value


def load_manifest(output: Path, tokenizer: Path) -> dict:
    path = output / "manifest.json"
    if path.is_file():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        expected = manifest.get("tokenizer", {}).get("sha256")
        if expected != sha256(tokenizer):
            raise RuntimeError("existing manifest uses a different tokenizer")
        return manifest
    return {
        "schema_version": 1,
        "format": "document-indexed-uint16-shards",
        "eod_token_id": EOD_ID,
        "tokenizer": {
            "path": str(tokenizer.resolve()),
            "sha256": sha256(tokenizer),
            "vocab_size": VOCAB_SIZE,
        },
        "splits": {},
        "interleave": {"sources": []},
    }


def update_split(manifest: dict, split: str, record: dict) -> None:
    values = manifest["splits"].setdefault(
        split,
        {"tokens": 0, "documents": 0, "shards": []},
    )
    old = [item for item in values["shards"] if item["path"] != record["path"]]
    old.append(record)
    old.sort(key=lambda item: item["path"])
    values["shards"] = old
    values["tokens"] = sum(int(item["tokens"]) for item in old)
    values["documents"] = sum(int(item["documents"]) for item in old)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--weight", type=float, default=1.0)
    parser.add_argument("--minor", action="store_true")
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    args = parser.parse_args()
    source_id = safe_source_id(args.source_id)
    if args.weight <= 0:
        raise ValueError("--weight must be positive")
    for path in args.input:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.tokenizer.is_file():
        raise FileNotFoundError(args.tokenizer)

    tokenizer = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    if tokenizer.get_piece_size() != VOCAB_SIZE:
        raise ValueError("tokenizer must contain exactly 49,152 pieces")
    if tokenizer.id_to_piece(EOD_ID) != "<|eod|>":
        raise ValueError("tokenizer ID 4 must be <|eod|>")

    args.output.mkdir(parents=True, exist_ok=True)
    stem = f"{source_id}-{args.split}-00000"
    binary = args.output / f"{stem}.bin"
    index = args.output / f"{stem}.idx"
    binary_tmp = binary.with_suffix(".bin.tmp")
    index_tmp = index.with_suffix(".idx.tmp")
    offsets = [0]
    token_count = 0
    document_count = 0
    with binary_tmp.open("wb") as output:
        for text in documents(args.input):
            ids = tokenizer.encode(text, out_type=int)
            ids.append(EOD_ID)
            array = np.asarray(ids, dtype=np.uint16)
            output.write(array.tobytes())
            token_count += len(ids)
            document_count += 1
            offsets.append(token_count)
        output.flush()
        os.fsync(output.fileno())
    if not document_count:
        binary_tmp.unlink(missing_ok=True)
        raise RuntimeError("no documents were found")
    np.asarray(offsets, dtype=np.uint64).tofile(index_tmp)
    os.replace(binary_tmp, binary)
    os.replace(index_tmp, index)

    record = {
        "source_id": source_id,
        "path": binary.name,
        "index": index.name,
        "tokens": token_count,
        "documents": document_count,
        "sha256": sha256(binary),
        "index_sha256": sha256(index),
    }
    manifest = load_manifest(args.output, args.tokenizer)
    update_split(manifest, args.split, record)
    if args.split == "train":
        sources = manifest["interleave"]["sources"]
        sources = [item for item in sources if item["source_id"] != source_id]
        sources.append(
            {
                "source_id": source_id,
                "weight": args.weight,
                "major": not args.minor,
                "shards": [record],
            }
        )
        sources.sort(key=lambda item: item["source_id"])
        manifest["interleave"]["sources"] = sources
    atomic_json(args.output / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "split": args.split,
                "source_id": source_id,
                "documents": document_count,
                "tokens": token_count,
                "manifest": str(args.output / "manifest.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
