"""Code acquisition: Stack-Edu metadata plus Software Heritage content.

Stack-Edu ships *no source text*.  Each row is metadata -- `blob_id`, `repo_name`,
`path`, `score`, `detected_licenses` -- and the file itself has to be fetched
from the Software Heritage archive by blob id.  That is deliberate on their part
and it is why the code slice needs its own collector rather than the streaming
path the text sources use.

Two consequences shape this module:

* every kept file is a separate S3 GET, so fetching is done from a thread pool;
  sequentially it would take hours for a slice worth 250M tokens
* licensing has to be decided per file, not per dataset.  Only files whose
  detected licence is permissive are kept, and the manifest records
  `repo / path / SWHID / licence` for each one, so the corpus can be audited and
  a file can be withdrawn later without rebuilding the pool.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

SWH_BUCKET = "softwareheritage"

# Permissive licences only.  Copyleft is excluded not because it is unusable but
# because redistributing a derived model trained on it raises questions this
# project is not set up to answer; "no_license" means no grant at all.
PERMISSIVE = {
    "mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "bsd-3-clause-clear",
    "isc", "unlicense", "cc0-1.0", "0bsd", "zlib", "postgresql", "python-2.0",
    "artistic-2.0", "bsl-1.0",
}


def _client():
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    return boto3.client(
        "s3",
        config=Config(signature_version=UNSIGNED, max_pool_connections=64,
                      retries={"max_attempts": 3, "mode": "standard"}),
    )


def fetch_content(client, blob_id: str) -> str | None:
    """Retrieve one file's text from Software Heritage, or None if unavailable."""
    try:
        obj = client.get_object(Bucket=SWH_BUCKET, Key=f"content/{blob_id}")
        with gzip.GzipFile(fileobj=io.BytesIO(obj["Body"].read())) as fh:
            return fh.read().decode("utf-8", "replace")
    except Exception:
        return None


def is_permissive(record: dict) -> bool:
    licences = {str(x).lower() for x in (record.get("detected_licenses") or [])}
    return bool(licences) and licences.issubset(PERMISSIVE)


@dataclass
class CodeState:
    documents: int = 0
    bytes_written: int = 0
    shards: int = 0
    records_seen: int = 0
    rejected_license: int = 0
    fetch_failed: int = 0
    done: bool = False
    licenses: dict = field(default_factory=dict)


def collect_language(
    language: str,
    target_bytes: int,
    out_dir: Path,
    state: CodeState,
    workers: int = 32,
    batch: int = 256,
    shard_documents: int = 20_000,
    min_score: int = 3,
) -> CodeState:
    from datasets import load_dataset

    out_dir.mkdir(parents=True, exist_ok=True)
    client = _client()
    stream = load_dataset("HuggingFaceTB/stack-edu", name=language,
                          split="train", streaming=True)
    if state.records_seen:
        stream = stream.skip(state.records_seen)

    buffer: list[str] = []
    provenance: list[dict] = []
    pending: list[dict] = []
    t0 = time.time()

    def flush() -> None:
        if not buffer:
            return
        path = out_dir / f"shard_{state.shards:05d}.jsonl.gz"
        tmp = path.with_suffix(".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as fh:
            fh.write("".join(buffer))
        os.replace(tmp, path)
        with open(out_dir / "provenance.jsonl", "a", encoding="utf-8") as fh:
            for entry in provenance:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        state.shards += 1
        buffer.clear()
        provenance.clear()

    def drain() -> None:
        nonlocal pending
        if not pending:
            return
        with ThreadPoolExecutor(max_workers=workers) as pool:
            texts = list(pool.map(lambda r: fetch_content(client, r["blob_id"]), pending))
        for record, text in zip(pending, texts):
            if not text:
                state.fetch_failed += 1
                continue
            state.documents += 1
            state.bytes_written += len(text.encode("utf-8"))
            for lic in record.get("detected_licenses") or []:
                state.licenses[lic] = state.licenses.get(lic, 0) + 1
            buffer.append(json.dumps(
                {"key": record["blob_id"], "source": f"code_{language.lower()}",
                 "text": text}, ensure_ascii=False) + "\n")
            provenance.append({
                "blob_id": record["blob_id"],
                "swhid": f"swh:1:cnt:{record['blob_id']}",
                "repo": record.get("repo_name"),
                "path": record.get("path"),
                "licenses": record.get("detected_licenses"),
                "int_score": record.get("int_score"),
            })
        pending = []

    for record in stream:
        state.records_seen += 1
        if record.get("int_score", 0) < min_score:
            continue
        if not is_permissive(record):
            state.rejected_license += 1
            continue
        pending.append(record)

        if len(pending) >= batch:
            drain()
            if len(buffer) >= shard_documents:
                flush()
                print(f"[code_{language}] {state.bytes_written/1e6:.0f}/"
                      f"{target_bytes/1e6:.0f} MB  {state.documents:,} files  "
                      f"licence-rejected {state.rejected_license:,}  "
                      f"{time.time()-t0:.0f}s", flush=True)
            if state.bytes_written >= target_bytes:
                break

    drain()
    flush()
    state.done = state.bytes_written >= target_bytes
    return state


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/mix")
    ap.add_argument("--language", required=True)
    ap.add_argument("--target-mb", type=float, required=True)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = root / f"code_{args.language.lower()}"
    state_path = out_dir / "code_state.json"
    state = CodeState(**json.loads(state_path.read_text())) if state_path.exists() \
        else CodeState()

    state = collect_language(args.language, int(args.target_mb * 1e6), out_dir,
                             state, workers=args.workers)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state.__dict__, indent=2, ensure_ascii=False))
    print(json.dumps(state.__dict__, indent=2, ensure_ascii=False)[:800])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
