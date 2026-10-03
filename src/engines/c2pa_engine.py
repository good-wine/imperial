"""Engine 1 — C2PA verification (production wrapper).

Adds signer-identity and timestamp-integrity extraction on top of
`src/modules/c2pa_verify.py`: parses claim_generator/issuer fields from the
fallback manifest (or the official binding JSON) into a stable
`signer_identity` + `timestamp_integrity` verdict for API/STIX consumers.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.modules.c2pa_verify import verify_c2pa
from src.schemas import C2paResult

log = logging.getLogger(__name__)


def run_c2pa_engine(file_path: str | Path) -> tuple[C2paResult, dict[str, Any]]:
    """Returns (C2paResult, intel_summary) with signer/timestamp assessment."""
    result = verify_c2pa(file_path)
    manifest = (result.manifest_info or {}).get("manifest", {})
    signer = str(manifest.get("claim_generator", "") or "").strip()
    if not signer and isinstance(manifest.get("cbor"), dict):
        signer = str(_deep_get(manifest["cbor"], "claim_generator") or "")
    vd = (result.manifest_info or {}).get("validation_details", {})
    summary = {
        "manifest_validity": (
            "VALID"
            if result.signature_valid is True
            else "INVALID"
            if result.signature_valid is False
            else "UNVERIFIED"
            if result.has_manifest
            else "ABSENT"
        ),
        "signer_identity": signer or "unknown",
        "timestamp_integrity": (
            vd.get("timestamp", {}).get("status", "SKIPPED") if isinstance(vd, dict) else "SKIPPED"
        ),
        "validation_details": vd,
    }
    log.info(
        "c2pa-engine: %s validity=%s signer=%s",
        Path(str(file_path)).name,
        summary["manifest_validity"],
        summary["signer_identity"],
    )
    return result, summary


def _deep_get(obj: Any, key: str, depth: int = 0) -> Any | None:
    if depth > 4 or not isinstance(obj, dict):
        return None
    if key in obj:
        return obj[key]
    for v in obj.values():
        hit = _deep_get(v, key, depth + 1)
        if hit is not None:
            return hit
    return None
