"""Traffic Light Protocol enforcement (TLP 2.0: CLEAR/GREEN/AMBER/RED).

Ordering CLEAR < GREEN < AMBER < RED. A requester cleared for level L may see
records classified at level <= L. Higher-TLP records are either hidden
(`allows()` False) or field-redacted (`redact()`), never leaked.
"""

from __future__ import annotations

from typing import Any

ORDER = ("TLP:CLEAR", "TLP:GREEN", "TLP:AMBER", "TLP:RED")

# Fields visible per requester clearance. The full record stays in the
# encrypted audit trail; only the API payload is filtered.
VISIBLE_FIELDS: dict[str, set[str]] = {
    "TLP:CLEAR": {
        "file_path",
        "risk_score",
        "verdict",
        "executive_summary",
        "findings",
        "recommendations",
        "narrative",
        "hypotheses",
        "analytic_confidence",
        "tlp",
        "schema_version",
        "citations",
        "llm_backend",
        "calibration_note",
        "ai_act50",
    },
    "TLP:GREEN": {
        "file_path",
        "risk_score",
        "verdict",
        "executive_summary",
        "findings",
        "recommendations",
        "narrative",
        "hypotheses",
        "analytic_confidence",
        "tlp",
        "schema_version",
        "citations",
        "llm_backend",
        "calibration_note",
        "ai_act50",
    },
    "TLP:AMBER": {
        "file_path",
        "risk_score",
        "verdict",
        "executive_summary",
        "recommendations",
        "analytic_confidence",
        "tlp",
        "schema_version",
        "llm_backend",
    },
    "TLP:RED": {"verdict", "tlp", "schema_version"},
}


def normalize(tlp: str) -> str:
    t = (tlp or "").strip().upper().replace("WHITE", "CLEAR")
    if not t.startswith("TLP:"):
        t = f"TLP:{t}"
    return t if t in ORDER else "TLP:CLEAR"


def allows(requester: str, record: str) -> bool:
    """True when requester clearance dominates the record classification."""
    try:
        return ORDER.index(normalize(requester)) >= ORDER.index(normalize(record))
    except ValueError:
        return False


def clamp_clearance(declared: str, allowed: str) -> str:
    """Return the lower of two clearances — never escalates.

    Used to cap a caller-supplied clearance by the clearance their identity
    actually holds, so a self-declared `X-TLP` header cannot grant access the
    token does not back.
    """
    d, a = normalize(declared), normalize(allowed)
    return d if ORDER.index(d) <= ORDER.index(a) else a


def redact(report: dict[str, Any], requester: str) -> dict[str, Any]:
    """Return a clearance-filtered copy of a report dict (non-mutating)."""
    visible = VISIBLE_FIELDS.get(normalize(requester), VISIBLE_FIELDS["TLP:RED"])
    out = {k: v for k, v in report.items() if k in visible}
    out["tlp"] = report.get("tlp", "TLP:CLEAR")
    out["_redacted_for"] = normalize(requester)
    return out
