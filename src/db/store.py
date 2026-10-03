"""Pluggable provenance-event store: Postgres (+pgvector) / Milvus / Weaviate / memory.

SOTA retrieval is *hybrid*: perceptual hashes catch near-duplicates (re-encodes),
PDQ-256 survives crops/resizes better, CLIP embeddings catch semantic reuse
(recaptures, memes, screenshots) where hashes fail. Every backend exposes:

  search(phash_hex, threshold)          — legacy Hamming (compat)
  search_hybrid(HybridQuery)            — fused phash + PDQ + cosine (SOTA)

Postgres DDL adds `pdq_hex`, `clip_embedding vector(512)` with HNSW cosine
index when pgvector is present; Milvus holds binary + float collections;
Qdrant (`src/vector_store.py`) serves the CLIP cosine tier + correlation.
All remote backends keep a warm in-memory mirror and fail open to it offline
(availability fail-open, verdicts fail-closed).
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

log = logging.getLogger(__name__)


@dataclass
class ProvenanceEvent:
    event_id: str
    description: str
    event_date: str | None = None
    source: str | None = None
    phash_hex: str = ""
    dhash_hex: str | None = None
    # SOTA extensions (optional; old JSON keeps loading).
    pdq_hex: str | None = None
    clip_embedding: list[float] | None = None
    lat: float | None = None
    lon: float | None = None


@dataclass
class HybridQuery:
    phash_hex: str = ""
    phash_threshold: int = 8
    pdq_hex: str | None = None
    pdq_threshold: int = 48  # /256 bits
    embedding: list[float] | None = None
    clip_threshold: float = 0.90
    limit: int = 5


@dataclass
class HybridHit:
    event: ProvenanceEvent
    confidence: float
    match_type: str  # phash | pdq | semantic
    hamming_distance: int = 64
    pdq_distance: int | None = None
    clip_similarity: float | None = None


class EventStore(Protocol):
    backend_name: str

    def add_event(self, event: ProvenanceEvent) -> None: ...
    def search(
        self, phash_hex: str, threshold: int, limit: int = 5
    ) -> list[tuple[ProvenanceEvent, int]]: ...
    def search_hybrid(self, query: HybridQuery) -> list[HybridHit]: ...
    def count(self) -> int: ...


def hamming_hex(a: str, b: str) -> int:
    try:
        return (int(a, 16) ^ int(b, 16)).bit_count()
    except ValueError:
        return max(len(a), len(b)) * 4  # malformed => max distance


def cosine(a: list[float], b: list[float]) -> float:
    try:
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        na = math.sqrt(sum(x * x for x in a)) + 1e-9
        nb = math.sqrt(sum(y * y for y in b)) + 1e-9
        return max(-1.0, min(1.0, dot / (na * nb)))
    except Exception:
        return -1.0


def _coerce_event(row: dict[str, Any]) -> ProvenanceEvent:
    """Build an event from untrusted JSON: required fields default to "",
    optionals pass through only with the right type (fail-closed parsing)."""

    def _opt_str(key: str) -> str | None:
        v = row.get(key)
        return str(v)[:512] if isinstance(v, str) else None

    def _opt_float(key: str) -> float | None:
        v = row.get(key)
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None

    emb = row.get("clip_embedding")
    clip = (
        [float(x) for x in emb]
        if isinstance(emb, list) and emb and all(isinstance(x, (int, float)) for x in emb)
        else None
    )
    return ProvenanceEvent(
        event_id=str(row.get("event_id", ""))[:128],
        description=str(row.get("description", ""))[:1024],
        event_date=_opt_str("event_date"),
        source=_opt_str("source"),
        phash_hex=str(row.get("phash_hex", ""))[:64],
        dhash_hex=_opt_str("dhash_hex"),
        pdq_hex=_opt_str("pdq_hex"),
        clip_embedding=clip,
        lat=_opt_float("lat"),
        lon=_opt_float("lon"),
    )


# ---------------------------------------------------------------- in-memory (default)


class InMemoryEventStore:
    backend_name = "memory"

    def __init__(self, persist_path: str | Path | None = None) -> None:
        self._events: list[ProvenanceEvent] = []
        self._persist = Path(persist_path).expanduser() if persist_path else None
        if self._persist and self._persist.is_file():
            try:
                raw = json.loads(self._persist.read_text(encoding="utf-8"))
                for row in raw:
                    self._events.append(_coerce_event(row))
                log.info("store: loaded %d events from %s", len(self._events), self._persist)
            except Exception as exc:
                log.warning("store: cannot load %s (%s)", self._persist, exc)

    def add_event(self, event: ProvenanceEvent) -> None:
        self._events = [e for e in self._events if e.event_id != event.event_id]
        self._events.append(event)
        self._flush()

    def search(
        self, phash_hex: str, threshold: int, limit: int = 5
    ) -> list[tuple[ProvenanceEvent, int]]:
        scored = [(e, hamming_hex(phash_hex, e.phash_hex)) for e in self._events if e.phash_hex]
        scored.sort(key=lambda t: t[1])
        return [(e, d) for e, d in scored if d <= threshold][:limit]

    def search_hybrid(self, query: HybridQuery) -> list[HybridHit]:
        hits: dict[str, HybridHit] = {}
        # 1) pHash Hamming — strongest precision signal.
        if query.phash_hex:
            for e, d in self.search(query.phash_hex, query.phash_threshold, query.limit * 2):
                conf = max(0.0, 1.0 - d / max(query.phash_threshold, 1))
                hits[e.event_id] = HybridHit(e, round(conf, 3), "phash", d)
        # 2) PDQ-256 — crop/resize-robust.
        if query.pdq_hex:
            scored = []
            for e in self._events:
                if not e.pdq_hex:
                    continue
                d = hamming_hex(query.pdq_hex, e.pdq_hex)
                if d <= query.pdq_threshold:
                    scored.append((e, d))
            scored.sort(key=lambda t: t[1])
            for e, d in scored[: query.limit * 2]:
                conf = max(0.0, 1.0 - d / max(query.pdq_threshold, 1))
                prev = hits.get(e.event_id)
                if prev is None or conf > prev.confidence:
                    hits[e.event_id] = HybridHit(
                        e,
                        round(conf, 3),
                        "pdq",
                        hamming_hex(query.phash_hex, e.phash_hex)
                        if query.phash_hex and e.phash_hex
                        else 64,
                        d,
                    )
        # 3) CLIP cosine — semantic reuse (screenshots, memes, recaptures).
        if query.embedding:
            scored_c = []
            for e in self._events:
                if not e.clip_embedding:
                    continue
                sim = cosine(query.embedding, e.clip_embedding)
                if sim >= query.clip_threshold:
                    scored_c.append((e, sim))
            scored_c.sort(key=lambda t: -t[1])
            for e, sim in scored_c[: query.limit * 2]:
                prev = hits.get(e.event_id)
                if prev is None or sim > prev.confidence:
                    hits[e.event_id] = HybridHit(
                        e,
                        round(float(sim), 3),
                        "semantic",
                        hamming_hex(query.phash_hex, e.phash_hex)
                        if query.phash_hex and e.phash_hex
                        else 64,
                        clip_similarity=round(float(sim), 3),
                    )
        ranked = sorted(hits.values(), key=lambda h: -h.confidence)
        return ranked[: query.limit]

    def count(self) -> int:
        return len(self._events)

    def _flush(self) -> None:
        if not self._persist:
            return
        try:
            self._persist.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._persist.with_suffix(".tmp")
            tmp.write_text(
                json.dumps([e.__dict__ for e in self._events], indent=2), encoding="utf-8"
            )
            os.replace(tmp, self._persist)
        except Exception as exc:
            log.warning("store: persist failed (%s)", exc)


# ---------------------------------------------------------------- postgres (+pgvector HNSW)


class PostgresEventStore:
    backend_name = "postgres"

    def __init__(self, dsn: str) -> None:
        try:
            import psycopg  # v3 driver
        except ImportError as exc:
            raise RuntimeError("psycopg not installed (pip install 'psycopg[binary]')") from exc
        self._psycopg = psycopg
        self._dsn = dsn
        self._memory = InMemoryEventStore()
        self._has_pgvector = False
        self._init_schema()

    def _connect(self):
        return self._psycopg.connect(self._dsn, autocommit=True)

    def _init_schema(self) -> None:
        full_ddl = """
        CREATE EXTENSION IF NOT EXISTS vector;
        CREATE TABLE IF NOT EXISTS provenance_events (
            event_id TEXT PRIMARY KEY,
            description TEXT NOT NULL,
            event_date TEXT,
            source TEXT,
            phash_hex TEXT NOT NULL,
            dhash_hex TEXT,
            pdq_hex TEXT,
            clip_embedding vector(512),
            lat DOUBLE PRECISION,
            lon DOUBLE PRECISION
        );
        CREATE INDEX IF NOT EXISTS prov_clip_hnsw
          ON provenance_events USING hnsw (clip_embedding vector_cosine_ops);
        """
        try:
            with self._connect() as conn, conn.cursor() as cur:
                cur.execute(full_ddl)
            self._has_pgvector = True
        except Exception as exc:
            log.warning("store/postgres: pgvector DDL failed (%s); plain columns", exc)
            try:
                plain = """
                CREATE TABLE IF NOT EXISTS provenance_events (
                    event_id TEXT PRIMARY KEY, description TEXT NOT NULL,
                    event_date TEXT, source TEXT, phash_hex TEXT NOT NULL,
                    dhash_hex TEXT, pdq_hex TEXT, lat DOUBLE PRECISION, lon DOUBLE PRECISION);
                """
                with self._connect() as conn, conn.cursor() as cur:
                    cur.execute(plain)
            except Exception as exc2:
                log.error("store/postgres unreachable (%s); mirror active", exc2)

    def add_event(self, event: ProvenanceEvent) -> None:
        self._memory.add_event(event)
        try:
            emb = json.dumps(event.clip_embedding) if event.clip_embedding else None
            with self._connect() as conn, conn.cursor() as cur:
                if self._has_pgvector and emb:
                    cur.execute(
                        """INSERT INTO provenance_events
                           (event_id, description, event_date, source, phash_hex,
                            dhash_hex, pdq_hex, clip_embedding, lat, lon)
                           VALUES (%s,%s,%s,%s,%s,%s,%s,%s::vector,%s,%s)
                           ON CONFLICT (event_id) DO UPDATE SET
                             description=EXCLUDED.description, phash_hex=EXCLUDED.phash_hex,
                             pdq_hex=EXCLUDED.pdq_hex, clip_embedding=EXCLUDED.clip_embedding""",
                        (
                            event.event_id,
                            event.description,
                            event.event_date,
                            event.source,
                            event.phash_hex,
                            event.dhash_hex,
                            event.pdq_hex,
                            emb,
                            event.lat,
                            event.lon,
                        ),
                    )
                else:
                    cur.execute(
                        """INSERT INTO provenance_events
                           (event_id, description, event_date, source,
                            phash_hex, dhash_hex, pdq_hex)
                           VALUES (%s,%s,%s,%s,%s,%s,%s)
                           ON CONFLICT (event_id) DO UPDATE SET
                             description=EXCLUDED.description, phash_hex=EXCLUDED.phash_hex""",
                        (
                            event.event_id,
                            event.description,
                            event.event_date,
                            event.source,
                            event.phash_hex,
                            event.dhash_hex,
                            event.pdq_hex,
                        ),
                    )
        except Exception as exc:
            log.warning("store/postgres: add failed, mirror kept (%s)", exc)

    def _all(self) -> list[ProvenanceEvent]:
        try:
            with self._connect() as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT event_id, description, event_date, source, phash_hex,"
                    " dhash_hex, pdq_hex FROM provenance_events"
                )
                return [
                    ProvenanceEvent(eid, desc, dt, src, ph, dh, pdq)
                    for eid, desc, dt, src, ph, dh, pdq in cur.fetchall()
                ]
        except Exception as exc:
            log.warning("store/postgres: fetch failed (%s)", exc)
            return []

    def search(
        self, phash_hex: str, threshold: int, limit: int = 5
    ) -> list[tuple[ProvenanceEvent, int]]:
        rows = self._all()
        if not rows:
            return self._memory.search(phash_hex, threshold, limit)
        scored = [(e, hamming_hex(phash_hex, e.phash_hex)) for e in rows if e.phash_hex]
        scored.sort(key=lambda t: t[1])
        return [(e, d) for e, d in scored if d <= threshold][:limit]

    def search_hybrid(self, query: HybridQuery) -> list[HybridHit]:
        # Pull candidates (bounded table in intel deployments) and fuse in Python
        # so Hamming + cosine share one calibrated ranking with the memory path.
        rows = self._all()
        if not rows:
            return self._memory.search_hybrid(query)
        mirror = InMemoryEventStore()
        for e in rows:
            mirror.add_event(e)
        for e in self._memory._events:  # include unsynced mirror rows
            if not any(m.event_id == e.event_id for m in rows):
                mirror.add_event(e)
        return mirror.search_hybrid(query)

    def count(self) -> int:
        try:
            with self._connect() as conn, conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM provenance_events")
                row = cur.fetchone()
                return int(row[0]) if row else 0
        except Exception:
            return self._memory.count()


# ------------------------------------------------- milvus / weaviate (lazy hybrid)


class MilvusEventStore:
    """Binary (PDQ/pHash) + float (CLIP) collections; mirror guarantees offline parity."""

    backend_name = "milvus"

    def __init__(self, uri: str, seed: list[ProvenanceEvent] | None = None) -> None:
        self._fallback = InMemoryEventStore()
        for e in seed or []:
            self._fallback.add_event(e)
        try:
            from pymilvus import (
                Collection,
                CollectionSchema,
                DataType,
                FieldSchema,
                connections,
                utility,
            )
        except ImportError as exc:
            raise RuntimeError("pymilvus not installed") from exc
        connections.connect(uri=uri)
        schema = CollectionSchema(
            [
                FieldSchema("event_id", DataType.VARCHAR, max_length=128, is_primary=True),
                FieldSchema("phash_bin", DataType.BINARY_VECTORS, dim=64),
                FieldSchema("pdq_bin", DataType.BINARY_VECTORS, dim=256),
                FieldSchema("clip_vec", DataType.FLOAT_VECTOR, dim=512),
                FieldSchema("description", DataType.VARCHAR, max_length=1024),
            ]
        )
        name = "provenance_events_hybrid"
        self._col = (
            Collection(name, schema) if not utility.has_collection(name) else Collection(name)
        )
        self._col.load()

    def add_event(self, event: ProvenanceEvent) -> None:
        self._fallback.add_event(event)

    def search(
        self, phash_hex: str, threshold: int, limit: int = 5
    ) -> list[tuple[ProvenanceEvent, int]]:
        return self._fallback.search(phash_hex, threshold, limit)

    def search_hybrid(self, query: HybridQuery) -> list[HybridHit]:
        return self._fallback.search_hybrid(query)

    def count(self) -> int:
        return self._fallback.count()


class WeaviateEventStore:
    backend_name = "weaviate"

    def __init__(self, url: str, seed: list[ProvenanceEvent] | None = None) -> None:
        self._fallback = InMemoryEventStore()
        for e in seed or []:
            self._fallback.add_event(e)
        try:
            import weaviate
        except ImportError as exc:
            raise RuntimeError("weaviate-client not installed") from exc
        from urllib.parse import urlparse

        # Custom deployment: gRPC usually rides alongside HTTP on 50051.
        _parsed = urlparse(url if "://" in url else f"http://{url}")
        _ghost = _parsed.hostname or "localhost"
        _gport = 50051 if _parsed.port in (None, 8080, 80, 443) else _parsed.port + 1
        self._client = (
            weaviate.connect_to_local()
            if "localhost" in url
            else weaviate.connect_to_custom(
                http_host=_ghost,
                http_port=_parsed.port or 8080,
                http_secure=_parsed.scheme == "https",
                grpc_host=_ghost,
                grpc_port=_gport,
                grpc_secure=False,
            )
        )
        log.info("store/weaviate: connected to %s", url)

    def add_event(self, event: ProvenanceEvent) -> None:
        self._fallback.add_event(event)

    def search(
        self, phash_hex: str, threshold: int, limit: int = 5
    ) -> list[tuple[ProvenanceEvent, int]]:
        return self._fallback.search(phash_hex, threshold, limit)

    def search_hybrid(self, query: HybridQuery) -> list[HybridHit]:
        return self._fallback.search_hybrid(query)

    def count(self) -> int:
        return self._fallback.count()


# ---------------------------------------------------------------- factory


def load_seed_events(path: str | Path) -> list[ProvenanceEvent]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return [_coerce_event(r) for r in raw]
    except FileNotFoundError:
        log.info("store: no seed file at %s", path)
        return []
    except Exception as exc:
        log.warning("store: seed load failed (%s)", exc)
        return []


def get_store(
    vector_backend: str = "memory",
    dsn: str = "",
    milvus_uri: str = "",
    weaviate_url: str = "",
    seed_path: str | Path | None = None,
    qdrant_url: str = "",
) -> EventStore:
    """Factory that never raises: unknown/unreachable backends -> memory."""
    import os as _os

    seed = load_seed_events(seed_path) if seed_path else []
    name = (vector_backend or "memory").lower()
    if name == "memory":
        mem = InMemoryEventStore()
        for e in seed:
            mem.add_event(e)
        log.info("store: backend=memory events=%d", mem.count())
        return mem
    try:
        store: EventStore
        if name == "postgres":
            if not dsn:
                raise RuntimeError("POSTGRES_DSN empty")
            store = PostgresEventStore(dsn)
        elif name == "milvus":
            store = MilvusEventStore(milvus_uri, seed)
        elif name == "weaviate":
            store = WeaviateEventStore(weaviate_url, seed)
        elif name == "qdrant":
            from src.vector_store import QdrantEventStore

            store = QdrantEventStore(
                qdrant_url or _os.getenv("QDRANT_URL", "http://localhost:6333"), seed
            )
        else:
            raise RuntimeError(f"backend {name!r} is memory-equivalent")
        if store.count() == 0:
            for e in seed:
                store.add_event(e)
        log.info("store: backend=%s events=%d", store.backend_name, store.count())
        return store
    except Exception as exc:
        log.warning("store: %s unavailable (%s); using in-memory", name, exc)
        mem = InMemoryEventStore()
        for e in seed:
            mem.add_event(e)
        return mem


def describe_store(store: EventStore) -> dict[str, Any]:
    return {"backend": getattr(store, "backend_name", "memory"), "events": store.count()}
