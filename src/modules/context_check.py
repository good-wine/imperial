"""Module 3 — Context & Decontextualization (hybrid retrieval, SOTA).

Genuine pixels, false story: decontextualization is the dominant real-world
abuse of crisis imagery. One signal is insufficient:

  pHash/dHash (64-bit)  precise near-duplicates (re-encodes, resizes)
  PDQ-lite (256-bit)    crop/resize-robust (DCT, no extra dependency)
  CLIP (512-d cosine)   semantic reuse — screenshots, memes, recaptures

Fusion: `EventStore.search_hybrid` ranks max-confidence across signals with
calibrated thresholds (pHash ≤8, PDQ ≤48, CLIP ≥0.90). Geo-temporal note
cross-checks EXIF capture date against matched event dates when available.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.config import SETTINGS
from src.db.store import EventStore, HybridQuery, get_store
from src.schemas import ContextMatch, ContextResult
from src.security import is_video, validate_media_file

log = logging.getLogger(__name__)

_STORE: EventStore | None = None


def _store() -> EventStore:
    global _STORE
    if _STORE is None:
        _STORE = get_store(
            vector_backend=SETTINGS.vector_backend,
            dsn=SETTINGS.postgres_dsn,
            milvus_uri=SETTINGS.milvus_uri,
            weaviate_url=SETTINGS.weaviate_url,
            seed_path=SETTINGS.reference_db_path,
            qdrant_url=SETTINGS.qdrant_url,
        )
    return _STORE


def reset_store_cache() -> None:
    global _STORE
    _STORE = None


def check_context(
    file_path: str | Path, threshold: int | None = None, store: EventStore | None = None
) -> ContextResult:
    """Hybrid decontextualization check. Never raises on bad input."""
    limit = SETTINGS.phash_threshold if threshold is None else threshold
    try:
        resolved = validate_media_file(file_path, max_mb=SETTINGS.max_file_mb)
    except Exception as exc:
        log.warning("context: input rejected: %s", exc)
        return ContextResult(
            file_path=str(file_path),
            query_phash=None,
            query_dhash=None,
            backend="memory",
            errors=[str(exc)],
        )
    active = store or _store()
    result = ContextResult(
        file_path=str(resolved),
        query_phash=None,
        query_dhash=None,
        backend=getattr(active, "backend_name", "memory"),
        threshold=limit,
    )
    try:
        phash_hex, dhash_hex, pdq_hex, embedding, ib_embedding, _pil_for_geo = _hash_media_hybrid(
            resolved
        )
    except RuntimeError as exc:
        result.errors.append(str(exc))
        result.warnings.append("perceptual hashing unavailable; decontextualization check skipped")
        return result
    except Exception as exc:
        log.exception("context: hashing failed")
        result.errors.append(f"hashing error: {exc}")
        return result

    result.query_phash, result.query_dhash = phash_hex, dhash_hex
    result.query_pdq = pdq_hex
    result.clip_available = embedding is not None
    if embedding is None:
        result.warnings.append(
            "CLIP embeddings unavailable; semantic tier skipped (hash-only mode)"
        )
    if ib_embedding is None:
        result.warnings.append("ImageBind embeddings unavailable; joint-space tier skipped")

    try:
        if hasattr(active, "search_hybrid"):
            hits = active.search_hybrid(
                HybridQuery(
                    phash_hex=phash_hex,
                    phash_threshold=limit,
                    pdq_hex=pdq_hex,
                    pdq_threshold=SETTINGS.pdq_threshold,
                    embedding=embedding,
                    ib_embedding=ib_embedding,
                    clip_threshold=SETTINGS.clip_threshold,
                )
            )
        else:  # legacy store compat
            hits_legacy = active.search(phash_hex, threshold=limit, limit=5)
            from src.db.store import HybridHit

            hits = [
                HybridHit(e, round(max(0.0, 1 - d / max(limit, 1)), 3), "phash", d)
                for e, d in hits_legacy
            ]
    except Exception as exc:
        log.exception("context: store search failed")
        result.errors.append(f"search error: {exc}")
        return result

    for h in hits:
        result.matches.append(
            ContextMatch(
                event_id=h.event.event_id,
                description=h.event.description,
                event_date=h.event.event_date,
                source=h.event.source,
                hamming_distance=h.hamming_distance,
                confidence=h.confidence,
                match_type=h.match_type,
                pdq_distance=h.pdq_distance,
                clip_similarity=h.clip_similarity,
            )
        )
    result.is_decontextualized = len(result.matches) > 0
    result.geo_temporal_note = _geo_temporal_note(resolved, result.matches)
    if result.geo_temporal_note:
        result.warnings.append(result.geo_temporal_note)
    if result.is_decontextualized:
        best = result.matches[0]
        result.warnings.append(
            f"near-duplicate of {best.event_id} via {best.match_type} "
            f"(conf={best.confidence}): verify caption/date before publishing"
        )
        log.info(
            "context: %s matches %s [%s] conf=%.2f",
            resolved.name,
            best.event_id,
            best.match_type,
            best.confidence,
        )
    else:
        log.info(
            "context: %s clean (phash=%s pdq=%s clip=%s)",
            resolved.name,
            phash_hex[:8] if phash_hex else None,
            "yes" if pdq_hex else "no",
            "yes" if embedding else "no",
        )
    return result


def _hash_media_hybrid(path: Path):
    """Return (phash, dhash, pdq_hex, clip_vec, ib_vec, pil_image).

    Video => middle frame. Both semantic tiers are signature-gated and may be
    None when their model is absent (hash-only mode).
    """
    try:
        import imagehash
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "perceptual hashing needs Pillow+ImageHash (pip install Pillow ImageHash)"
        ) from exc
    from src.embeddings import clip_image_embedding, imagebind_image_embedding, pdq_lite_hex

    def _embed(rgb):
        return (
            clip_image_embedding(rgb, SETTINGS.clip_model_path),
            imagebind_image_embedding(rgb, SETTINGS.imagebind_model_path),
        )

    if not is_video(path):
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            ph, dh = str(imagehash.phash(rgb)), str(imagehash.dhash(rgb))
            pdq = pdq_lite_hex(pil_image=rgb)
            clip, ib = _embed(rgb)
            return ph, dh, pdq, clip, ib, rgb.copy()
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("video hashing needs opencv-python-headless") from exc
    from PIL import Image as _PILImage

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {path.name}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(total // 2, 0))
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError("no decodable frames for hashing")
        rgb_arr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = _PILImage.fromarray(rgb_arr)
        import imagehash as _ih

        clip, ib = _embed(pil)
        return (
            str(_ih.phash(pil)),
            str(_ih.dhash(pil)),
            pdq_lite_hex(pil_image=pil),
            clip,
            ib,
            pil,
        )
    finally:
        cap.release()


def _geo_temporal_note(path: Path, matches: list[ContextMatch]) -> str | None:
    """EXIF capture date vs matched event date: flag anachronistic reuse."""
    if not matches:
        return None
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS

        with Image.open(path) as img:
            exif = img.getexif()
            taken = None
            for tag_id, v in (exif or {}).items():
                if TAGS.get(tag_id) in ("DateTimeOriginal", "DateTime"):
                    taken = str(v)[:10]
                    break
    except Exception:
        taken = None
    if not taken:
        return None
    for m in matches[:2]:
        if m.event_date and taken and not taken.startswith(m.event_date[:4]):
            return (
                f"temporal tension: file EXIF {taken} vs matched event "
                f"{m.event_id} ({m.event_date}) — supports recaptioning hypothesis"
            )
    return None
