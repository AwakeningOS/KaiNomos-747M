"""Deterministic, exactly resumable source-balanced document interleave.

The KaiNomos data-order contract selects a source with a seeded weighted PRNG,
then advances only that source's cursor.  Documents are read through the
``uint16`` token shard and ``uint64`` document-offset files produced by the
KaiNomos packers.  A document longer than ``max_chunk_tokens`` is returned in
bounded chunks; source selection happens again between chunks, so no large
shard can monopolise the beginning of training.

Canonical manifests should declare ``interleave.sources`` explicitly.  The
existing DataMix-v2 manifest predates that schema.  ``adapt_manifest`` handles
that one format deliberately and visibly by grouping its filenames into the
only recoverable source identities (currently ``local`` and ``jpnmix``).  Its
aggregate domain totals cannot recover a domain for each packed document.
"""

from __future__ import annotations

import copy
import hashlib
import pickle
import random
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

DATA_ORDER_CONTRACT = "source_balanced_interleave_v1"


@dataclass
class SourceCursor:
    source_id: str
    shard_index: int = 0
    document_index: int = 0
    token_offset: int = 0
    rng_state: bytes = b""


@dataclass(frozen=True)
class TokenChunk:
    source_id: str
    tokens: np.ndarray
    shard_index: int
    document_index: int
    token_offset: int

    @property
    def token_count(self) -> int:
        return int(self.tokens.size)


@dataclass(frozen=True)
class ShardSpec:
    token_path: Path
    index_path: Path
    tokens: int
    documents: int


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    weight: float
    major: bool
    shards: tuple[ShardSpec, ...]


@dataclass(frozen=True)
class AdaptedManifest:
    sources: tuple[SourceSpec, ...]
    adapter_id: str
    note: str


def _safe_path(root: Path, name: str) -> Path:
    root = root.resolve()
    path = (root / name).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"data path escapes manifest directory: {name}") from error
    return path


def _shard_spec(item: Mapping[str, object], root: Path) -> ShardSpec:
    token_name = item.get("path", item.get("tokens_path"))
    index_name = item.get("index", item.get("index_path"))
    if not isinstance(token_name, str) or not isinstance(index_name, str):
        raise TypeError("each interleave shard needs path and index")
    token_path = _safe_path(root, token_name)
    index_path = _safe_path(root, index_name)
    if not token_path.is_file() or not index_path.is_file():
        missing = token_path if not token_path.is_file() else index_path
        raise FileNotFoundError(f"missing interleave shard file: {missing}")
    file_tokens = token_path.stat().st_size // np.dtype(np.uint16).itemsize
    index_items = index_path.stat().st_size // np.dtype(np.uint64).itemsize
    declared_tokens = int(item.get("tokens", file_tokens))
    declared_documents = int(item.get("documents", index_items - 1))
    if token_path.stat().st_size % np.dtype(np.uint16).itemsize:
        raise ValueError(f"invalid uint16 shard length: {token_path}")
    if index_path.stat().st_size % np.dtype(np.uint64).itemsize:
        raise ValueError(f"invalid uint64 index length: {index_path}")
    if declared_tokens != file_tokens:
        raise ValueError(f"token count mismatch for {token_path}")
    if declared_documents + 1 != index_items:
        raise ValueError(f"document count mismatch for {index_path}")
    return ShardSpec(token_path, index_path, declared_tokens, declared_documents)


def _canonical_sources(
    declarations: Sequence[Mapping[str, object]], root: Path
) -> tuple[SourceSpec, ...]:
    sources = []
    for item in declarations:
        source_id = str(item.get("source_id", ""))
        if not source_id:
            raise ValueError("each source needs a non-empty source_id")
        if "major" not in item:
            raise ValueError(f"source {source_id!r} must explicitly declare major")
        shards = item.get("shards")
        if not isinstance(shards, list) or not shards:
            raise ValueError(f"source {source_id!r} has no shards")
        sources.append(
            SourceSpec(
                source_id=source_id,
                weight=float(item["weight"]),
                major=bool(item["major"]),
                shards=tuple(_shard_spec(shard, root) for shard in shards),
            )
        )
    return tuple(sources)


def _legacy_source_id(path: str, split: str) -> str:
    match = re.fullmatch(rf"(.+)-{re.escape(split)}-\d+\.bin", Path(path).name)
    return match.group(1) if match else Path(path).stem.split("-", 1)[0]


def adapt_manifest(
    manifest: Mapping[str, object],
    data_dir: Path | str,
    *,
    split: str = "train",
    major_weight_threshold: float = 0.01,
) -> AdaptedManifest:
    """Return the canonical source view, including a named legacy adapter.

    Preferred schema::

        {"interleave": {"sources": [
          {"source_id": "...", "weight": 0.5, "major": true,
           "shards": [{"path": "...bin", "index": "...idx", ...}]}
        ]}}

    DataMix-v2 has only packed-shard provenance.  It is adapted by filename
    prefix and token-proportional weights.  This is intentionally a separate,
    named path rather than pretending its aggregate ``domain_tokens`` are a
    per-document source index.
    """
    root = Path(data_dir)
    interleave = manifest.get("interleave")
    if isinstance(interleave, Mapping) and isinstance(interleave.get("sources"), list):
        return AdaptedManifest(
            _canonical_sources(interleave["sources"], root),
            "canonical_interleave_v1",
            "manifest supplies explicit source weights, major flags, and shards",
        )

    splits = manifest.get("splits")
    declared = splits.get(split) if isinstance(splits, Mapping) else None
    if not isinstance(declared, Mapping) or not isinstance(declared.get("shards"), list):
        raise TypeError("manifest has neither interleave.sources nor indexed split shards")
    if manifest.get("format") != "document-indexed-uint16-shards":
        raise ValueError("legacy adapter only supports document-indexed-uint16-shards")

    grouped: dict[str, list[Mapping[str, object]]] = {}
    for shard in declared["shards"]:
        if not isinstance(shard, Mapping) or not isinstance(shard.get("path"), str):
            raise TypeError("invalid legacy shard declaration")
        source_id = _legacy_source_id(shard["path"], split)
        grouped.setdefault(source_id, []).append(shard)
    source_tokens = {
        source_id: sum(int(shard.get("tokens", 0)) for shard in shards)
        for source_id, shards in grouped.items()
    }
    total = sum(source_tokens.values())
    if total <= 0:
        raise ValueError("legacy manifest has no source token counts")
    sources = tuple(
        SourceSpec(
            source_id=source_id,
            weight=source_tokens[source_id] / total,
            major=source_tokens[source_id] / total >= major_weight_threshold,
            shards=tuple(_shard_spec(shard, root) for shard in shards),
        )
        for source_id, shards in sorted(grouped.items())
    )
    return AdaptedManifest(
        sources,
        "datamix_v2_filename_prefix_v1",
        "legacy v2 exposes packed local/jpnmix identity only; aggregate domain "
        "totals cannot identify each document",
    )


def _derived_seed(seed: int, source_id: str, epoch: int, purpose: str) -> int:
    payload = f"{seed}\0{source_id}\0{epoch}\0{purpose}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _cursor_rng_state(seed: int, source_id: str, epoch: int) -> bytes:
    return pickle.dumps(
        {"seed": int(seed), "source_id": source_id, "epoch": int(epoch)},
        protocol=5,
    )


class DeterministicInterleaver:
    """Weighted source interleave with independent, checkpointable cursors."""

    def __init__(
        self,
        sources: Sequence[SourceSpec],
        *,
        seed: int,
        max_chunk_tokens: int,
    ) -> None:
        if not sources:
            raise ValueError("at least one source is required")
        if max_chunk_tokens < 1:
            raise ValueError("max_chunk_tokens must be positive")
        ids = [source.source_id for source in sources]
        if len(ids) != len(set(ids)):
            raise ValueError("source_id values must be unique")
        if any(not np.isfinite(source.weight) or source.weight <= 0 for source in sources):
            raise ValueError("source weights must be finite and positive")
        if any(not source.shards for source in sources):
            raise ValueError("every source needs at least one shard")
        if any(sum(shard.tokens for shard in source.shards) <= 0 for source in sources):
            raise ValueError("every source needs at least one token")
        self.sources = tuple(sources)
        self.seed = int(seed)
        self.max_chunk_tokens = int(max_chunk_tokens)
        self._by_id = {source.source_id: source for source in self.sources}
        self._weights = tuple(float(source.weight) for source in self.sources)
        self._weight_total = sum(self._weights)
        self._selection_rng = random.Random(self.seed)
        self.cursors = {
            source.source_id: SourceCursor(
                source.source_id,
                rng_state=_cursor_rng_state(self.seed, source.source_id, 0),
            )
            for source in self.sources
        }
        self._indices: dict[tuple[str, int], np.memmap] = {}
        self._tokens: dict[tuple[str, int], np.memmap] = {}
        self._shard_orders: dict[tuple[str, int], np.ndarray] = {}
        self._document_orders: dict[tuple[str, int, int], np.ndarray] = {}
        self._accounting = {
            source.source_id: {"tokens": 0, "chunks": 0, "documents_completed": 0}
            for source in self.sources
        }
        self._step = 0
        self._major_seen: set[str] = set()

    @classmethod
    def from_manifest(
        cls,
        manifest: Mapping[str, object] | Path | str,
        data_dir: Path | str | None = None,
        *,
        split: str = "train",
        seed: int,
        max_chunk_tokens: int,
    ) -> DeterministicInterleaver:
        if isinstance(manifest, (str, Path)):
            import json

            manifest_path = Path(manifest)
            import_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            root = manifest_path.parent if data_dir is None else Path(data_dir)
        else:
            import_manifest = manifest
            if data_dir is None:
                raise ValueError("data_dir is required when manifest is already parsed")
            root = Path(data_dir)
        adapted = adapt_manifest(import_manifest, root, split=split)
        result = cls(adapted.sources, seed=seed, max_chunk_tokens=max_chunk_tokens)
        result.manifest_adapter_id = adapted.adapter_id
        result.manifest_adapter_note = adapted.note
        return result

    @property
    def major_source_ids(self) -> frozenset[str]:
        return frozenset(source.source_id for source in self.sources if source.major)

    def _epoch(self, cursor: SourceCursor) -> int:
        value = pickle.loads(cursor.rng_state)
        if value.get("seed") != self.seed or value.get("source_id") != cursor.source_id:
            raise ValueError(f"invalid RNG state for source {cursor.source_id!r}")
        return int(value["epoch"])

    def _shard_order(self, source: SourceSpec, epoch: int) -> np.ndarray:
        key = (source.source_id, epoch)
        if key not in self._shard_orders:
            rng = np.random.Generator(np.random.PCG64(
                _derived_seed(self.seed, source.source_id, epoch, "shards")
            ))
            self._shard_orders[key] = rng.permutation(len(source.shards))
        return self._shard_orders[key]

    def _document_order(
        self, source: SourceSpec, actual_shard_index: int, epoch: int
    ) -> np.ndarray:
        key = (source.source_id, epoch, actual_shard_index)
        if key not in self._document_orders:
            count = source.shards[actual_shard_index].documents
            rng = np.random.Generator(np.random.PCG64(
                _derived_seed(
                    self.seed, source.source_id, epoch,
                    f"documents:{actual_shard_index}",
                )
            ))
            self._document_orders[key] = rng.permutation(count)
        return self._document_orders[key]

    def _arrays(
        self, source: SourceSpec, actual_shard_index: int
    ) -> tuple[np.memmap, np.memmap]:
        key = (source.source_id, actual_shard_index)
        shard = source.shards[actual_shard_index]
        if key not in self._indices:
            offsets = np.memmap(shard.index_path, dtype=np.uint64, mode="r")
            if int(offsets[0]) != 0 or int(offsets[-1]) != shard.tokens:
                raise ValueError(f"invalid document offsets: {shard.index_path}")
            if np.any(offsets[1:] < offsets[:-1]):
                raise ValueError(f"non-monotonic document offsets: {shard.index_path}")
            self._indices[key] = offsets
            self._tokens[key] = np.memmap(shard.token_path, dtype=np.uint16, mode="r")
        return self._tokens[key], self._indices[key]

    def _select_source(self) -> SourceSpec:
        point = self._selection_rng.random() * self._weight_total
        cumulative = 0.0
        for source, weight in zip(self.sources, self._weights, strict=True):
            cumulative += weight
            if point < cumulative:
                return source
        return self.sources[-1]

    def _advance_epoch(self, cursor: SourceCursor) -> None:
        epoch = self._epoch(cursor) + 1
        cursor.shard_index = 0
        cursor.document_index = 0
        cursor.token_offset = 0
        cursor.rng_state = _cursor_rng_state(self.seed, cursor.source_id, epoch)

    def _next_from_source(self, source: SourceSpec) -> TokenChunk:
        cursor = self.cursors[source.source_id]
        pieces: list[np.ndarray] = []
        remaining = self.max_chunk_tokens
        first_location: tuple[int, int, int] | None = None
        while remaining:
            epoch = self._epoch(cursor)
            shard_order = self._shard_order(source, epoch)
            if cursor.shard_index >= len(shard_order):
                self._advance_epoch(cursor)
                continue
            actual_shard = int(shard_order[cursor.shard_index])
            document_order = self._document_order(source, actual_shard, epoch)
            if cursor.document_index >= len(document_order):
                cursor.shard_index += 1
                cursor.document_index = 0
                cursor.token_offset = 0
                continue
            actual_document = int(document_order[cursor.document_index])
            tokens, offsets = self._arrays(source, actual_shard)
            document_start = int(offsets[actual_document])
            document_end = int(offsets[actual_document + 1])
            available = document_end - document_start - cursor.token_offset
            if available <= 0:
                cursor.document_index += 1
                cursor.token_offset = 0
                self._accounting[source.source_id]["documents_completed"] += 1
                continue
            count = min(available, remaining)
            token_offset = cursor.token_offset
            values = np.asarray(
                tokens[document_start + token_offset:document_start + token_offset + count],
                dtype=np.uint16,
            ).copy()
            if first_location is None:
                first_location = (actual_shard, actual_document, token_offset)
            pieces.append(values)
            remaining -= count
            cursor.token_offset += count
            if cursor.token_offset == document_end - document_start:
                cursor.document_index += 1
                cursor.token_offset = 0
                self._accounting[source.source_id]["documents_completed"] += 1
            self._accounting[source.source_id]["tokens"] += count
        self._accounting[source.source_id]["chunks"] += 1
        if first_location is None:  # guarded by the positive-source-size check
            raise RuntimeError(f"source {source.source_id!r} produced no tokens")
        shard_index, document_index, token_offset = first_location
        return TokenChunk(
            source.source_id,
            np.concatenate(pieces),
            shard_index,
            document_index,
            token_offset,
        )

    def next_document_or_chunk(self) -> TokenChunk:
        return self._next_from_source(self._select_source())

    def source_token_accounting(self) -> dict[str, dict[str, int]]:
        return copy.deepcopy(self._accounting)

    def note_optimizer_step(self, chunks_or_source_ids: Iterable[TokenChunk | str]) -> dict:
        """Record and return the source composition of one optimizer step.

        The caller should persist this returned mapping in its ordinary step log.
        At step 100 the warmup coverage gate fails immediately if any source
        explicitly marked ``major`` has not appeared.
        """
        counts: Counter[str] = Counter()
        tokens: Counter[str] = Counter()
        for value in chunks_or_source_ids:
            if isinstance(value, TokenChunk):
                source_id = value.source_id
                tokens[source_id] += value.token_count
            else:
                source_id = str(value)
            if source_id not in self._by_id:
                raise ValueError(f"unknown source in optimizer step: {source_id!r}")
            counts[source_id] += 1
            self._major_seen.add(source_id)
        self._step += 1
        record = {
            "optimizer_step": self._step,
            "source_chunks": dict(sorted(counts.items())),
            "source_tokens": dict(sorted(tokens.items())),
        }
        if self._step == 100:
            self.assert_major_source_coverage()
        return record

    def assert_major_source_coverage(self) -> None:
        missing = sorted(self.major_source_ids - self._major_seen)
        if missing:
            raise RuntimeError(
                "major sources missing from the first 100 optimizer steps: "
                + ", ".join(missing)
            )

    def _source_fingerprint(self) -> tuple:
        return tuple(
            (
                source.source_id,
                source.weight,
                source.major,
                tuple(
                    (str(shard.token_path.resolve()), str(shard.index_path.resolve()),
                     shard.tokens, shard.documents)
                    for shard in source.shards
                ),
            )
            for source in self.sources
        )

    def state_dict(self) -> dict:
        return {
            "version": 1,
            "data_order_contract": DATA_ORDER_CONTRACT,
            "seed": self.seed,
            "max_chunk_tokens": self.max_chunk_tokens,
            "source_fingerprint": self._source_fingerprint(),
            "selection_rng_state": pickle.dumps(
                self._selection_rng.getstate(), protocol=5
            ),
            "cursors": {
                source_id: asdict(cursor)
                for source_id, cursor in self.cursors.items()
            },
            "source_accounting": self.source_token_accounting(),
            "optimizer_steps": self._step,
            "major_seen": sorted(self._major_seen),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if state.get("data_order_contract") != DATA_ORDER_CONTRACT:
            raise ValueError("checkpoint has a different data-order contract")
        for key, expected in (
            ("seed", self.seed),
            ("max_chunk_tokens", self.max_chunk_tokens),
            ("source_fingerprint", self._source_fingerprint()),
        ):
            if state.get(key) != expected:
                raise ValueError(f"interleave checkpoint mismatch: {key}")
        raw_cursors = state.get("cursors")
        if not isinstance(raw_cursors, Mapping) or set(raw_cursors) != set(self.cursors):
            raise ValueError("interleave checkpoint has different source cursors")
        restored = {}
        for source_id, raw in raw_cursors.items():
            if not isinstance(raw, Mapping):
                raise TypeError(f"invalid cursor for {source_id!r}")
            restored[source_id] = SourceCursor(**raw)
            self._epoch(restored[source_id])
        selection_state = state.get("selection_rng_state")
        if not isinstance(selection_state, bytes):
            raise TypeError("interleave checkpoint lacks selection RNG state")
        self._selection_rng.setstate(pickle.loads(selection_state))
        self.cursors = restored
        raw_accounting = state.get("source_accounting", {})
        self._accounting = {
            source_id: {
                key: int(raw_accounting.get(source_id, {}).get(key, 0))
                for key in ("tokens", "chunks", "documents_completed")
            }
            for source_id in self.cursors
        }
        self._step = int(state.get("optimizer_steps", 0))
        self._major_seen = set(map(str, state.get("major_seen", [])))


class InterleavedSequenceStream:
    """Pack interleaved document chunks into fixed training sequences.

    The not-yet-consumed tail is part of the checkpoint.  Advancing the source
    cursor when a chunk is read can therefore never skip its remainder on resume.
    """

    def __init__(self, interleaver: DeterministicInterleaver,
                 sequence_length: int, batch_size: int = 1):
        if sequence_length < 2 or batch_size < 1:
            raise ValueError("invalid fixed-sequence shape")
        self.interleaver = interleaver
        self.sequence_length = int(sequence_length)
        self.batch_size = int(batch_size)
        self.pending: list[tuple[str, np.ndarray]] = []

    def next_batch(self) -> tuple[np.ndarray, dict[str, int]]:
        required = self.sequence_length * self.batch_size
        pieces: list[np.ndarray] = []
        accounting: Counter[str] = Counter()
        while required:
            if not self.pending:
                chunk = self.interleaver.next_document_or_chunk()
                self.pending.append((chunk.source_id, chunk.tokens))
            source_id, values = self.pending[0]
            take = min(required, int(values.size))
            pieces.append(values[:take])
            accounting[source_id] += take
            required -= take
            if take == values.size:
                self.pending.pop(0)
            else:
                self.pending[0] = (source_id, values[take:].copy())
        batch = np.concatenate(pieces).reshape(
            self.batch_size, self.sequence_length
        )
        return batch, dict(accounting)

    def state_dict(self) -> dict:
        return {
            "version": 1,
            "sequence_length": self.sequence_length,
            "batch_size": self.batch_size,
            "interleaver": self.interleaver.state_dict(),
            "pending": [
                {"source_id": source_id, "tokens": values.copy()}
                for source_id, values in self.pending
            ],
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if int(state.get("sequence_length", -1)) != self.sequence_length:
            raise ValueError("sequence length changed across resume")
        if int(state.get("batch_size", -1)) != self.batch_size:
            raise ValueError("micro batch changed across resume")
        self.interleaver.load_state_dict(state["interleaver"])
        self.pending = []
        for item in state.get("pending", []):
            source_id = str(item["source_id"])
            if source_id not in self.interleaver.cursors:
                raise ValueError(f"unknown pending source {source_id!r}")
            values = np.asarray(item["tokens"], dtype=np.uint16).copy()
            self.pending.append((source_id, values))
