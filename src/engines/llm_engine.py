"""Engine 5 — Explainable LLM + STIX threat reporting (backend-agnostic).

Aggregates every engine leg (C2PA, visual/multimodal forensics, GEOINT,
context) into one deterministic risk score, the ICD-203 narrative, and a
STIX 2.1 bundle with manipulation IoB/IoCs. Provider swaps (Ollama today,
HF/LlamaIndex tomorrow) never touch callers: only `llm_backends.get_backend`.
"""

from __future__ import annotations

import logging
from typing import Any

from src.config import SETTINGS
from src.engines.forensics_engine import multimodal_risk_bump
from src.modules.explain_report import (
    analytic_confidence,
    build_hypotheses,
    compute_risk,
    score_to_verdict,
)
from src.modules.explain_report import generate_report as _gen
from src.modules.llm_backends import BaseLLMBackend
from src.schemas import (
    AudioResult,
    C2paResult,
    ContextResult,
    FinalReport,
    ForensicResult,
    GeointResult,
    LipSyncResult,
    Mp4Result,
    RppgResult,
)
from src.stix import build_bundle

log = logging.getLogger(__name__)


def aggregate_report(
    c2pa: C2paResult,
    forensic: ForensicResult,
    context: ContextResult,
    multi: dict[str, Any] | None = None,
    geoint: GeointResult | None = None,
    llm: BaseLLMBackend | None = None,
) -> FinalReport:
    """Full fusion: base score + multimodal bumps + GEOINT + narrative + STIX."""
    multi = multi or {}
    base, factors = compute_risk(c2pa, forensic, context)
    bump, notes = multimodal_risk_bump(multi)
    score = base
    if bump:
        score = min(100, round(base + bump))
        factors += notes
    if geoint is not None and geoint.available and geoint.consistent is False:
        score = min(100, score + 10)
        factors.append(
            f"solar geometry contradiction (+10, err={geoint.angular_error_deg}°): "
            "claimed time/place vs shadows"
        )
    verdict = score_to_verdict(score)
    # Reuse the tradecraft generator for narrative/hypotheses, then upgrade
    # the score-dependent fields deterministically (LLM never re-scores).
    tmp = _gen(c2pa, forensic, context, llm=llm)
    hyps = build_hypotheses(c2pa, forensic, context, score)
    conf = analytic_confidence(c2pa, forensic, context)
    if geoint is not None and geoint.available and conf == "High" and geoint.confidence < 0.5:
        conf = "Moderate"  # coarse auto-measure caps confidence honestly

    findings = list(factors)
    if geoint is not None and geoint.available and geoint.consistent is False:
        findings.append("GEOINT: shadow geometry contradicts claimed capture time/place.")
    multi_notes = {
        "audio": (multi.get("audio") or {}),
        "lipsync": (multi.get("lipsync") or {}),
        "rppg": (multi.get("rppg") or {}),
        "mp4": (multi.get("mp4") or {}),
        "geoint": geoint.to_dict() if geoint else {},
        "watermark": (multi.get("watermark") or {}),
        "speech": (multi.get("speech") or {}),
        "redaction": (multi.get("redaction") or {}),
        "prnu": (multi.get("prnu") or {}),
    }
    rep = FinalReport(
        file_path=tmp.file_path,
        risk_score=score,
        verdict=verdict,
        executive_summary=(
            tmp.executive_summary
            if score == base
            else tmp.executive_summary.replace(f"Risk {tmp.risk_score}/100", f"Risk {score}/100")
        ),
        findings=findings,
        recommendations=list(tmp.recommendations),
        narrative=tmp.narrative,
        llm_backend=tmp.llm_backend,
        c2pa=tmp.c2pa,
        forensic=tmp.forensic,
        context=tmp.context,
        multimodal=multi_notes,
        hypotheses=[h.to_dict() for h in hyps],
        analytic_confidence=conf,
        tlp=SETTINGS.tlp,
        citations=list(tmp.citations),
    )
    # Keep the summary verdict token consistent when the bump crossed a bucket.
    if score != base:
        rep.executive_summary = (
            f"Risk {score}/100 ({verdict.value}; {conf} confidence). "
            f"Multimodal legs added +{score - base}: "
            + "; ".join(
                notes
                + (
                    ["solar contradiction +10"]
                    if geoint is not None and geoint.available and geoint.consistent is False
                    else []
                )
            )
            + ". "
            + rep.executive_summary
        )
    log.info("llm-engine: risk=%d verdict=%s (base=%d)", score, verdict.value, base)
    return rep


def stix_for(report: FinalReport, tlp: str = "") -> dict[str, Any]:
    return build_bundle(report.to_dict(), tlp=tlp or report.tlp)


__all__ = [
    "AudioResult",
    "GeointResult",
    "LipSyncResult",
    "Mp4Result",
    "RppgResult",
    "aggregate_report",
    "stix_for",
]
