"""Qdrant vector backend: multimodal embeddings + cross-case correlation.

Collection layout (`provenance_events`):
  - payload: event_id, description, event_date, source, phash_hex, pdq_hex
  - binary-ish vectors are stored as float for HNSW: `clip` (512-d cosine).
  - Hamming tiers (pHash/PDQ) run against the in-memory mirror (exact integer
    math); the semantic tier queries Qdrant directly with cosine threshold.

Modes: server (`QDRANT_URL`), in-memory (`QDRANT_URL=":memory:"`, used by
tests), or file-persisted path. Absent server/unreachable -> fail over to
the mirror with a warning (availability fail-open, verdicts fail-closed).
Always ships the same `EventStore` surface (`search` + `search_hybrid`).
"""

from __future__ import annotations

import logging
from typing import Any

from src.db.store import (
    HybridHit,
    HybridQuery,
    InMemoryEventStore,
    ProvenanceEvent,
    cosine,
    hamming_hex,
)

log = logging.getLogger(__name__)

COLLECTION = "provenance_events"
IB_COLLECTION = "provenance_ib"  # ImageBind-1024 joint space (no migration of CLIP-512)
_CLIP_DIM = 512
_IB_DIM = 1024


class QdrantEventStore:
    backend_name = "qdrant"

    def __init__(
        self, url: str = "http://localhost:6333", seed: list[ProvenanceEvent] | None = None
    ) -> None:
        self._mirror = InMemoryEventStore()
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError as exc:
            raise RuntimeError("qdrant-client not installed") from exc
        self._models = None
        try:
            from qdrant_client import models as _m

            self._models = _m
        except Exception:
            pass
        self._client = (
            QdrantClient(url) if url not in (":memory:", "memory") else QdrantClient(":memory:")
        )
        try:
            cols = [c.name for c in self._client.get_collections().collections]
            if COLLECTION not in cols:
                self._client.create_collection(
                    COLLECTION,
                    vectors_config=VectorParams(size=_CLIP_DIM, distance=Distance.COSINE),
                )
            if IB_COLLECTION not in cols:
                self._client.create_collection(
                    IB_COLLECTION,
                    vectors_config=VectorParams(size=_IB_DIM, distance=Distance.COSINE),
                )
            log.info("store/qdrant: connected (%s)", url)
        except Exception as exc:
            raise RuntimeError(f"qdrant unreachable at {url}: {exc}") from exc
        for e in seed or []:  # seeds go through add_event: mirror + Qdrant
            self.add_event(e)

    # -- writes go to mirror + Qdrant -------------------------------------
    def add_event(self, event: ProvenanceEvent) -> None:
        self._mirror.add_event(event)
        if self._models is None:
            log.warning("store/qdrant: models unavailable; mirror only")
            return
        # Hash-only events (no semantic vector) live in the mirror alone.
        for collection, vec, dim in (
            (COLLECTION, event.clip_embedding, _CLIP_DIM),
            (IB_COLLECTION, event.ib_embedding, _IB_DIM),
        ):
            if not vec or len(vec) != dim:
                continue
            try:
                import uuid as _uuid

                self._client.upsert(
                    collection_name=collection,
                    points=[
                        self._models.PointStruct(
                            id=str(_uuid.uuid5(_uuid.NAMESPACE_DNS, event.event_id)),
                            vector=[float(v) for v in vec],
                            payload={
                                "event_id": event.event_id,
                                "description": event.description,
                                "event_date": event.event_date,
                                "source": event.source,
                                "phash_hex": event.phash_hex,
                                "pdq_hex": event.pdq_hex,
                            },
                        )
                    ],
                )
            except Exception as exc:
                log.warning("store/qdrant: upsert to %s failed (%s); mirror kept", collection, exc)

    # -- reads: hash tiers on mirror (exact), semantic tier on Qdrant ------
    def search(
        self, phash_hex: str, threshold: int, limit: int = 5
    ) -> list[tuple[ProvenanceEvent, int]]:
        return self._mirror.search(phash_hex, threshold, limit)

    def search_hybrid(self, query: HybridQuery) -> list[HybridHit]:
        hits = {h.event.event_id: h for h in self._mirror.search_hybrid(query)}
        by_id = {e.event_id: e for e in self._mirror._events}
        for collection, vec, dim, threshold in (
            (COLLECTION, query.embedding, _CLIP_DIM, query.clip_threshold),
            (IB_COLLECTION, query.ib_embedding, _IB_DIM, query.ib_threshold),
        ):
            if not vec or len(vec) != dim:
                continue
            try:
                res = self._client.query_points(
                    collection_name=collection,
                    query=[float(v) for v in vec],
                    limit=query.limit * 2,
                    score_threshold=float(threshold),
                ).points
                for p in res:
                    eid = str((p.payload or {}).get("event_id") or "")
                    ev = by_id.get(eid)
                    if ev is None:
                        continue
                    sim = round(float(p.score), 3)
                    prev = hits.get(eid)
                    if prev is None or sim > prev.confidence:
                        hits[eid] = HybridHit(
                            ev,
                            sim,
                            "semantic",
                            hamming_hex(query.phash_hex, ev.phash_hex)
                            if query.phash_hex and ev.phash_hex
                            else 64,
                            clip_similarity=sim,
                        )
            except Exception as exc:
                log.warning("store/qdrant: %s query failed (%s); mirror used", collection, exc)
        return sorted(hits.values(), key=lambda h: -h.confidence)[: query.limit]

    def search_correlated(
        self, embedding: list[float], limit: int = 10, score_threshold: float = 0.80
    ) -> list[dict[str, Any]]:
        """Cross-case correlation: nearest neighbours across ALL stored events,
        regardless of hash tiers. Powers `GET /search/similar` + dashboard."""
        if not embedding or len(embedding) != _CLIP_DIM:
            # Hash-space fallback: rank mirror by PDQ when no CLIP vector.
            return []
        try:
            res = self._client.query_points(
                collection_name=COLLECTION,
                query=[float(v) for v in embedding],
                limit=limit,
                score_threshold=score_threshold,
            ).points
            return [
                {
                    "event_id": (p.payload or {}).get("event_id"),
                    "description": (p.payload or {}).get("description"),
                    "score": round(float(p.score), 3),
                }
                for p in res
            ]
        except Exception as exc:
            log.warning("store/qdrant: correlated search failed (%s)", exc)
            # Mirror cosine fallback over seeded embeddings.
            scored: list[dict[str, Any]] = []
            for e in self._mirror._events:
                if e.clip_embedding:
                    s = cosine(embedding, e.clip_embedding)
                    if s >= score_threshold:
                        scored.append(
                            {
                                "event_id": e.event_id,
                                "description": e.description,
                                "score": round(s, 3),
                            }
                        )
            return sorted(scored, key=lambda d: -d["score"])[:limit]

    def count(self) -> int:
        return self._mirror.count()
