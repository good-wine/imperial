"""Weight integrity verification with a memoization cache.

`scripts/verify_weights.py` proved the value of checking pins (it is how the
ImageBind weight blob was found unpinned in the first place), but hashing
2.4 GB of tensors costs ~18 s. Run per request that is unusable; run never and
it is decoration. So the hash is memoized on `(size, mtime_ns)` and the check
is scheduled on `beat`.

Honest limit of the cache, stated because it is a security control: an
attacker who rewrites a weight file AND restores its mtime defeats the cache,
because no content hash is recomputed. What the cache does buy is that ordinary
edits, partial restores and truncated copies are caught, and that an
air-gapped operator can run a full uncached pass on demand
(`--no-cache`, which is what the transfer checklist uses). The signature gate
(`security.signature.gate_model_load`) is the control that does not depend on
filesystem metadata; this is the cheaper tripwire around it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_REGISTRY = "models/registry.json"
DEFAULT_CACHE = "data/model_integrity.json"


def sha256_file(path: Path) -> str:
    """Streamed SHA-256: never loads a multi-GB blob into memory."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def collect_entries(doc: dict[str, Any]) -> list[tuple[str, str, str]]:
    """Flatten the registry into (path, sha256_pin, label) triples.

    Models over the 2 GB protobuf limit keep their tensors in a sibling `.data`
    blob; without an extra pin the graph hash alone would not cover the
    weights, which is where all the substance is.
    """
    entries = [
        (m.get("path", ""), m.get("sha256", ""), m.get("name", "?")) for m in doc.get("models", [])
    ]
    for m in doc.get("models", []):
        ext = m.get("external_data") or {}
        if ext.get("path"):
            entries.append(
                (ext.get("path", ""), ext.get("sha256", ""), f"{m.get('name', '?')}[weights]")
            )
    entries += [
        (f"models/whisper/{k}", v, f"whisper/{k}") for k, v in (doc.get("whisper") or {}).items()
    ]
    return entries


def _load_cache(path: Path) -> dict[str, Any]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else {}
    except Exception:  # a corrupt cache is simply a cold cache
        return {}


def _save_cache(path: Path, cache: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)
    except Exception as exc:  # caching must never break the gate
        log.warning("weight-integrity: could not write cache (%s)", exc)


def verify_registry(
    registry_path: str | Path = DEFAULT_REGISTRY,
    *,
    cache_path: str | Path | None = DEFAULT_CACHE,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Check every present weight against its pin; fail closed on mismatch.

    Returns a report dict rather than raising: the caller decides whether a
    mismatch is a deploy blocker (CLI, exit 1) or an alertable event (beat).
    """
    started = time.monotonic()
    reg = Path(registry_path)
    try:
        doc = json.loads(reg.read_text(encoding="utf-8"))
    except Exception as exc:  # unreadable registry means "cannot verify"
        return {
            "ok": False,
            "error": f"unreadable registry: {exc}",
            "checked": 0,
            "mismatches": [],
            "skipped": [],
            "cached": 0,
            "elapsed_s": round(time.monotonic() - started, 2),
        }

    cache_file = Path(cache_path) if cache_path else None
    cache = _load_cache(cache_file) if (cache_file and use_cache) else {}
    fresh: dict[str, Any] = dict(cache) if cache_file and use_cache else {}

    checked = 0
    cached = 0
    mismatches: list[dict[str, str]] = []
    skipped: list[str] = []
    for path, pin, name in collect_entries(doc):
        p = Path(path)
        if not p.is_file():
            skipped.append(f"{name}: absent")
            continue
        if not pin:
            skipped.append(f"{name}: unpinned (BYO slot)")
            continue
        checked += 1
        st = p.stat()
        key = str(p)
        hit = cache.get(key) if cache else None
        if (
            hit
            and hit.get("size") == st.st_size
            and hit.get("mtime_ns") == st.st_mtime_ns
            and hit.get("pin") == pin
        ):
            digest = str(hit.get("sha256", ""))
            cached += 1
        else:
            try:
                digest = sha256_file(p)
            except OSError as exc:
                mismatches.append({"name": name, "path": key, "reason": f"unreadable: {exc}"})
                continue
            if fresh is not None:
                fresh[key] = {
                    "sha256": digest,
                    "pin": pin,
                    "size": st.st_size,
                    "mtime_ns": st.st_mtime_ns,
                    "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
        if digest != pin:
            mismatches.append(
                {
                    "name": name,
                    "path": key,
                    "expected": pin,
                    "actual": digest,
                    "reason": "sha256 mismatch",
                }
            )

    if cache_file and use_cache and fresh:
        _save_cache(cache_file, fresh)

    return {
        "ok": not mismatches,
        "checked": checked,
        "cached": cached,
        "hashed": checked - cached,
        "mismatches": mismatches,
        "skipped": skipped,
        "elapsed_s": round(time.monotonic() - started, 2),
    }
