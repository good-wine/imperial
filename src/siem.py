"""SIEM/SOC export: CEF + LEEF + ECS-JSON from a FinalReport dict.

STIX serves intel sharing; SOC pipelines ingest CEF (ArcSight), LEEF (QRadar)
or ECS JSON (Elastic). One endpoint (`GET /export/siem/{id}?format=…`) covers
all three from the same deterministic fields — no LLM text crosses into
SIEM (fixed schema only, injection-safe by construction).
"""

from __future__ import annotations

import json
from typing import Any


def _base(report: dict[str, Any]) -> dict[str, Any]:
    ctx = report.get("context") or {}
    return {
        "case_sha256": ((report.get("c2pa") or {}).get("sha256") or "")[:64],
        "risk": int(report.get("risk_score", 0)),
        "verdict": str(report.get("verdict", "")),
        "confidence": str(report.get("analytic_confidence", "")),
        "tlp": str(report.get("tlp", "TLP:CLEAR")),
        "file": str(report.get("file_path", ""))[-128:],
        "phash": str(ctx.get("query_phash") or ""),
        "top_match": (ctx.get("matches") or [{}])[0].get("event_id", "")
        if ctx.get("matches")
        else "",
    }


def _esc_cef(v: str) -> str:
    return str(v).replace("\\", "\\\\").replace("|", "\\|").replace("=", "\\=")


def to_cef(report: dict[str, Any], vendor: str = "ProvenanceEngine") -> str:
    b = _base(report)
    sev = min(10, max(0, b["risk"] // 10))
    ext = " ".join(f"{k}={_esc_cef(v)}" for k, v in b.items())
    return (
        f"CEF:0|{vendor}|MediaProvenance|3.0|{b['verdict']}|"
        f"Authenticity verdict {b['verdict']}|{sev}|{ext}"
    )


def to_leef(report: dict[str, Any], vendor: str = "ProvenanceEngine") -> str:
    b = _base(report)
    sev = min(10, max(0, b["risk"] // 10))
    ext = "\t".join(f"{k}={str(v).replace(chr(9), ' ')}" for k, v in b.items())
    return f"LEEF:2.0|{vendor}|MediaProvenance|3.0|{b['verdict']}|\tsev={sev}\t{ext}"


def to_ecs(report: dict[str, Any]) -> dict[str, Any]:
    b = _base(report)
    return {
        "@timestamp": __import__("datetime")
        .datetime.now(__import__("datetime").timezone.utc)
        .isoformat(),
        "event": {
            "kind": "alert",
            "category": ["threat"],
            "severity": min(10, max(0, b["risk"] // 10)),
            "risk_score": b["risk"],
            "action": "media-provenance-assessment",
        },
        "file": {"name": b["file"], "hash": {"sha256": b["case_sha256"], "phash": b["phash"]}},
        "threat": {
            "indicator": {"description": f"verdict={b['verdict']}", "confidence": b["confidence"]}
        },
        "labels": {"tlp": b["tlp"], "top_match": b["top_match"]},
    }


def render(report: dict[str, Any], fmt: str) -> tuple[str, str]:
    """Return (body, mime). Raises ValueError on unknown format."""
    f = (fmt or "json").lower()
    if f == "cef":
        return to_cef(report), "text/plain"
    if f == "leef":
        return to_leef(report), "text/plain"
    if f in ("ecs", "json"):
        return json.dumps(to_ecs(report), indent=2), "application/json"
    raise ValueError("format must be cef | leef | ecs")
