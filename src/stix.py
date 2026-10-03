"""STIX 2.1 bundle builder (stdlib-only, deterministic IDs).

Maps a `FinalReport` dict to: identity + observed-data (file + hashes) +
indicator(s) for manipulation IoB/IoC (pHash/PDQ as `file:hashes` patterns,
MISP-importable) + opinion (analyst-equivalent assessment with risk score) +
report. IDs are UUID5 (stable for the same SHA-256 + pipeline version), so
re-exports deduplicate in MISP/OpenCTI.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

STIX_VERSION = "2.1"
NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # DNS namespace base


def _uid(*parts: str) -> str:
    return str(uuid.uuid5(NAMESPACE, "|".join(parts)))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_bundle(
    report: dict[str, Any], *, identity_name: str = "Provenance Engine", tlp: str = "TLP:CLEAR"
) -> dict[str, Any]:
    sha = ((report.get("c2pa") or {}).get("sha256") or "unknown")[:64]
    risk = int(report.get("risk_score", 0))
    verdict = str(report.get("verdict", "UNVERIFIABLE"))
    now = _now()
    ident_id = f"identity--{_uid('identity', identity_name)}"
    obs_id = f"observed-data--{_uid('observed-data', sha)}"
    objects: list[dict[str, Any]] = [
        {
            "type": "identity",
            "spec_version": STIX_VERSION,
            "id": ident_id,
            "created": now,
            "modified": now,
            "name": identity_name,
            "identity_class": "system",
        },
        {
            "type": "observed-data",
            "spec_version": STIX_VERSION,
            "id": obs_id,
            "created": now,
            "modified": now,
            "first_observed": now,
            "last_observed": now,
            "number_observed": 1,
            "created_by_ref": ident_id,
            "objects": {
                "0": {
                    "type": "file",
                    "name": str(report.get("file_path", ""))[-256:],
                    "hashes": {
                        "SHA-256": sha,
                        "PHASH": str((report.get("context") or {}).get("query_phash") or ""),
                        "PDQ": str((report.get("context") or {}).get("query_pdq") or ""),
                    },
                }
            },
        },
    ]
    # IoB/IoC indicators: one per context hit + one synthetic-media flag when
    # forensics fired hard. Patterns use MISP-friendly file-hash syntax.
    indicators: list[dict[str, Any]] = []
    for m in (report.get("context") or {}).get("matches", [])[:5]:
        eid = str(m.get("event_id", "unknown"))
        iid = f"indicator--{_uid('indicator', eid, sha)}"
        indicators.append(
            {
                "type": "indicator",
                "spec_version": STIX_VERSION,
                "id": iid,
                "created": now,
                "modified": now,
                "created_by_ref": ident_id,
                "name": f"Decontextualized reuse of {eid}",
                "labels": ["anomalous-activity"],
                "pattern": f"[file:hashes.SHA-256 = '{sha}']",
                "pattern_type": "stix",
                "valid_from": now,
                "description": f"Near-duplicate ({m.get('match_type')}, "
                f"conf={m.get('confidence')}) of {m.get('description', '')}"[:512],
                "confidence": int(float(m.get("confidence", 0)) * 100),
            }
        )
    if float((report.get("forensic") or {}).get("aggregate_anomaly", 0)) > 0.65:
        iid = f"indicator--{_uid('indicator', 'synthetic-media', sha)}"
        indicators.append(
            {
                "type": "indicator",
                "spec_version": STIX_VERSION,
                "id": iid,
                "created": now,
                "modified": now,
                "created_by_ref": ident_id,
                "name": "Suspected synthetic/manipulated media",
                "labels": ["malicious-activity"],
                "pattern": f"[file:hashes.SHA-256 = '{sha}']",
                "pattern_type": "stix",
                "valid_from": now,
                "description": f"Forensic anomaly "
                f"{(report.get('forensic') or {}).get('aggregate_anomaly')}",
                "confidence": risk,
            }
        )
    objects.extend(indicators)
    opinion_id = f"opinion--{_uid('opinion', sha)}"
    objects.append(
        {
            "type": "opinion",
            "spec_version": STIX_VERSION,
            "id": opinion_id,
            "created": now,
            "modified": now,
            "created_by_ref": ident_id,
            "explanation": str(report.get("executive_summary", ""))[:1024],
            "object_refs": [obs_id] + [i["id"] for i in indicators],
            "opinion": "strongly-disagree"
            if risk >= 80
            else ("disagree" if risk >= 60 else ("neutral" if risk >= 40 else "agree")),
            "x_risk_score": risk,
            "x_verdict": verdict,
            "x_analytic_confidence": str(report.get("analytic_confidence", "")),
        }
    )
    report_id = f"report--{_uid('report', sha)}"
    objects.append(
        {
            "type": "report",
            "spec_version": STIX_VERSION,
            "id": report_id,
            "created": now,
            "modified": now,
            "created_by_ref": ident_id,
            "name": f"Media provenance assessment {sha[:12]}",
            "report_types": ["threat-report"],
            "published": now,
            "object_refs": [o["id"] for o in objects if o["id"] != report_id],
            "labels": ["provenance", verdict.lower()],
            "x_tlp": tlp,
            "x_schema_version": str(report.get("schema_version", "2.0")),
        }
    )
    return {
        "type": "bundle",
        "id": f"bundle--{_uid('bundle', sha, str(risk))}",
        "spec_version": STIX_VERSION,
        "objects": objects,
    }
