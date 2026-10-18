"""AI Act Article 50 compliance assessment (applicable since 2 August 2026).

A forensic verdict answers "is this authentic?". A deployer in the EU now also
has to answer a *different* question, with legal consequences:

  - Art. 50(2): is this content marked in a **machine-readable** format and
    detectable as artificially generated/manipulated? (Provider duty.)
  - Art. 50(4)+(5): if it is a deep fake, must it carry a **human-facing
    disclosure** — clear, distinguishable, at first exposure, accessible?

This module turns the forensic evidence already gathered into that second
answer, deliberately operationalising the 2026 guidance rather than guessing:

  - **Intent-independent.** The July 2026 Commission Guidelines (C(2026) 5054)
    confirm that "would falsely appear authentic" does NOT depend on intent to
    deceive. A sarcastic edit is still a deep fake; we do not require intent.
  - **Audience-aware.** Art. 50(4) is not an "average reasonable person" test:
    foreseeable exposure to children or low-digital-literacy audiences counts
    when judging whether content would falsely appear authentic.
  - **"Could have existed."** A realistic depiction of a *fictitious* person
    still qualifies. Clearly unrealistic content (fantasy, laws of physics
    broken) falls outside Art. 3(60).
  - **Exemptions are explicit.** Standard editing and law-enforcement
    investigation are carved out (Art. 50(2) and (4) respectively) and are
    surfaced as `exemption`, never silently applied.

What this is NOT: a legal opinion. The Commission's own wording on its icons is
that using them "does not establish legal compliance by itself"; the deployer
remains responsible. This module produces evidence and a recommendation with a
confidence, and says so in its own output (`legal_advice: false`).

Machine-readable marking is detected from the standards themselves:
C2PA `digitalSourceType` (2.2 introduced `trainedAlgorithmicData` and `empty`),
the presence of an invisible watermark, and soft-binding assertions. Absence of
a marker is reported as ABSENT — never as "human-made".
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Reference constants (dates and identifiers an assessor must not invent).
# --------------------------------------------------------------------------

AI_ACT_APPLICABLE_FROM = "2026-08-02"  # Art. 50 in application
AI_ACT_MARKING_TRANSITION_END = "2026-12-02"  # Art. 50(2) transitional relief
CODE_OF_PRACTICE_DATE = "2026-07-31"
GUIDELINES_DATE = "2026-07-20"

# C2PA digitalSourceType values (2.2+ URIs; the older short forms are accepted).
C2PA_AI_GENERATED = (
    "http://c2pa.org/digitalsourcetype/trainedalgorithmicdata",
    "c2pa.trainedAlgorithmicData",
)
C2PA_AI_MODIFIED = (
    "http://c2pa.org/digitalsourcetype/digitalcreation",
    "c2pa.digitalCreation",
)
C2PA_UNMARKED_HUMAN = (
    "http://c2pa.org/digitalsourcetype/digitalcapture",
    "c2pa.digitalCapture",
)

# Marking states. PRESENT_UNMARKED is deliberately distinct from
# PRESENT_HUMAN: a claim that is present but not AI-flavoured proves the
# opposite of authenticity — absence of an AI marker is not evidence of a
# human original.
MARKING_STATES = (
    "AI_GENERATED",
    "AI_MODIFIED",
    "HUMAN_DECLARED",
    "UNMARKED",
    "ABSENT",
)

SUBJECT_KINDS = ("PERSON", "PLACE", "OBJECT", "EVENT", "ABSTRACT")


def _norm(value: Any) -> str:
    """Normalize a C2PA digitalSourceType to lowercase, comparable form."""
    return str(value or "").strip().rstrip("/").lower()


def _xml_escape(value: Any) -> str:
    """Escape the five XML predefined entities (no defusedxml dependency).

    The badge is exported and rendered in unknown contexts, so every
    interpolated value is escaped; `xml.sax.saxutils.escape` is avoided to
    keep the parser attack surface out of the codebase (bandit B406).
    """
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


@dataclass
class AiAct50Result:
    """Machine-readable Art. 50 assessment. Every field is evidence-bearing."""

    # --- Art. 50(2): machine-readable marking ---
    machine_readable_marking: str = "ABSENT"  # one of MARKING_STATES
    marking_standard: str = ""  # e.g. "C2PA 2.x digitalSourceType"
    marking_evidence: list[str] = field(default_factory=list)
    marking_present: bool = False

    # --- Art. 3(60) + 50(4): deep fake determination ---
    deepfake: bool | None = None  # True / False / None = indeterminate
    deepfake_confidence: str = "Indeterminate"  # High | Moderate | Low | Indeterminate
    deepfake_basis: list[str] = field(default_factory=list)
    #: Signals that raise suspicion but cannot establish Art. 3(60) on their
    #: own. Kept separate from `deepfake_basis` so the determination never
    #: launders a weak indicator into a legal conclusion.
    supporting_evidence: list[str] = field(default_factory=list)
    depicted_subject: list[str] = field(default_factory=list)  # PERSON / PLACE / ...
    subject_resembles_reality: bool | None = None
    false_appearance_risk: str = "unknown"  # high | moderate | low | unknown
    intent_assessed: bool = False  # always False: intent is irrelevant here

    # --- Art. 50(4)+(5): disclosure ---
    disclosure_required: bool | None = None
    disclosure_obligations: list[str] = field(default_factory=list)
    label_text: str = ""
    icon_variant: str = ""  # AI_GENERATED | AI_MODIFIED | "" (EU icon variants)
    disclosure_timing: str = "at_first_exposure"
    accessibility: list[str] = field(default_factory=list)

    # --- carve-outs and posture ---
    exemption: str = ""
    #: The declared operator posture actually applied, echoed for audit so a
    #: report never leaves an applied carve-out unexplained.
    operator_posture: dict[str, Any] = field(default_factory=dict)
    legal_advice: bool = False
    legal_note: str = (
        "Automated assessment, not a legal opinion. The EU AI Office states that using its "
        "icons does not establish compliance by itself; the deployer remains responsible."
    )
    references: list[str] = field(
        default_factory=lambda: [
            "Regulation (EU) 2024/1689 Art. 50(2),(4),(5) — applicable from 2026-08-02",
            f"Commission Guidelines C(2026) 5054 final ({GUIDELINES_DATE})",
            f"Code of Practice on Transparency of AI-generated Content ({CODE_OF_PRACTICE_DATE})",
            "C2PA Technical Specification 2.2+ — digitalSourceType",
        ]
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def summary(self) -> str:
        """One line for the executive summary and for the audit trail."""
        if self.deepfake is None:
            verdict = "indeterminate"
        else:
            verdict = "deep fake" if self.deepfake else "not a deep fake"
        mark = self.machine_readable_marking.lower()
        label = f"; label required: {self.label_text}" if self.disclosure_required else ""
        return (
            f"AI Act Art. 50 — {verdict}; machine-readable marking: {mark}"
            f" (confidence {self.deepfake_confidence.lower()}){label}"
        )


# --------------------------------------------------------------------------
# Evidence extraction
# --------------------------------------------------------------------------


def _walk(node: Any, key: str, depth: int = 0, acc: list[Any] | None = None) -> list[Any]:
    """Collect every value of `key` in a nested manifest structure."""
    if acc is None:
        acc = []
    if depth > 6:
        return acc
    if isinstance(node, dict):
        for k, v in node.items():
            if k == key:
                acc.append(v)
            _walk(v, key, depth + 1, acc)
    elif isinstance(node, list):
        for v in node:
            _walk(v, key, depth + 1, acc)
    return acc


def extract_digital_source_types(manifest_info: dict[str, Any] | None) -> list[str]:
    """Every `digitalSourceType` declared in the C2PA manifest, normalized.

    Ingredients are what matter: an asset assembled from AI-generated parts
    inherits the strongest source type found, which is exactly the Art. 50(2)
    question ("did AI produce any part of this?").
    """
    if not manifest_info:
        return []
    raw = _walk(manifest_info, "digitalSourceType")
    out: list[str] = []
    for v in raw:
        if isinstance(v, str) and v.strip():
            out.append(_norm(v))
        elif isinstance(v, list):
            out.extend(_norm(x) for x in v if isinstance(x, str) and x.strip())
    return out


def extract_created_by(manifest_info: dict[str, Any] | None) -> str:
    """`createdBy` (a 2.4-era assertion field) can name the AI tool directly."""
    if not manifest_info:
        return ""
    vals = [str(v) for v in _walk(manifest_info, "createdBy") if isinstance(v, str)]
    return "; ".join(sorted(set(vals))[:3])


def detect_marking(
    c2pa_result: Any, watermark: dict[str, Any] | None = None
) -> tuple[str, list[str], str]:
    """Classify machine-readable marking from the standards, not from opinion.

    Returns (state, evidence, standard). Absence of a marker is reported as
    ABSENT/UNMARKED and explicitly *not* as "human-made": unmarked content is
    the overwhelming majority of real-world media.
    """
    evidence: list[str] = []
    info = getattr(c2pa_result, "manifest_info", None) or {}
    srcs = {_norm(s) for s in extract_digital_source_types(info)}
    created_by = extract_created_by(info)

    if srcs & {_norm(s) for s in C2PA_AI_GENERATED} or re.search(
        r"trainedAlgorithmicData", created_by, re.IGNORECASE
    ):
        evidence.append(f"C2PA digitalSourceType declares AI generation: {sorted(srcs)}")
        if created_by:
            evidence.append(f"C2PA createdBy: {created_by}")
        return "AI_GENERATED", evidence, "C2PA digitalSourceType (2.2+)"

    if srcs & {_norm(s) for s in C2PA_AI_MODIFIED}:
        evidence.append(f"C2PA digitalSourceType declares digital creation: {sorted(srcs)}")
        return "AI_MODIFIED", evidence, "C2PA digitalSourceType (2.2+)"

    if getattr(c2pa_result, "has_manifest", False) and srcs:
        evidence.append(f"C2PA digitalSourceType present, no AI value: {sorted(srcs)}")
        return "HUMAN_DECLARED", evidence, "C2PA digitalSourceType (2.2+)"

    # A watermark is also a machine-readable marker: C2PA soft binding and the
    # SynthID-style invisible watermarks the Commission points at. These are the
    # real `WatermarkResult` fields: `ai_generated` is the boolean verdict and
    # `synthid` is a *status string* ("DETECTED"/"NOT_DETECTED"/"NEEDS_MODEL"/
    # "NOT_CHECKED"). Reading truthiness of that string, or of the keys
    # `detected`/`soft_binding` that no producer emits, marked every analysed
    # file as AI-generated.
    if watermark:
        markers: list[str] = []
        if watermark.get("ai_generated") is True:
            markers.append(str(watermark.get("generator") or "generative-tool metadata"))
        if str(watermark.get("synthid", "")).upper() == "DETECTED":
            markers.append("SynthID pixel watermark")
        if markers:
            evidence.append(f"machine-readable marker present: {', '.join(markers)}")
            return "AI_GENERATED", evidence, "C2PA soft binding / invisible watermark"

    if getattr(c2pa_result, "has_manifest", False):
        evidence.append("C2PA manifest present but declares no digitalSourceType")
        return "UNMARKED", evidence, "C2PA manifest (no digitalSourceType)"

    evidence.append("no C2PA manifest and no watermark detected: no machine-readable marking")
    return "ABSENT", evidence, "none detected"


# --------------------------------------------------------------------------
# Deep fake determination (Art. 3(60))
# --------------------------------------------------------------------------


def assess_subjects(
    forensic: Any, c2pa_result: Any, context_result: Any, multi: dict[str, Any] | None
) -> tuple[list[str], bool | None]:
    """Which Art. 3(60) subject classes are depicted, and could they exist?

    "Resembles existing persons, objects, places, entities or events" is read
    broadly per the 2026 Guidelines: a realistic *fictitious* person counts.
    What excludes content is implausibility, not fictionality.
    """
    subjects: set[str] = set()
    faces = _count_faces(forensic)
    if faces:
        subjects.add("PERSON")
    else:
        # A photo with no detected face is still a PLACE/OBJECT candidate;
        # absence of a face says nothing about the scene.
        subjects.add("PLACE")
        subjects.add("OBJECT")
    if multi:
        if _truthy((multi.get("geoint") or {}).get("available")):
            subjects.add("PLACE")
        if _truthy((multi.get("rppg") or {}).get("available")):
            subjects.add("PERSON")
    if getattr(c2pa_result, "has_manifest", False):
        subjects.add("OBJECT")
    ctx_matches = getattr(context_result, "matches", None) or []
    if ctx_matches:
        subjects.add("EVENT")

    plausible = bool(subjects)
    return sorted(subjects), (plausible or None)


def _frames(forensic: Any) -> list[Any]:
    return list(getattr(forensic, "frames", None) or [])


def _count_faces(forensic: Any) -> int:
    """Total faces across analysed frames.

    Reads `FrameMetrics.face_count` (the field the forensics engine writes).
    The previous `forensic.faces` lookup never matched the schema, so every
    image silently lost its PERSON subject class — and with it the elevated
    audience risk of Art. 50(4).
    """
    return sum(int(getattr(f, "face_count", 0) or 0) for f in _frames(forensic))


def _frame_max(attr: str, forensic: Any, default: float = 0.0) -> float:
    """Largest numeric value of `attr` across analysed frames."""
    vals = [getattr(f, attr, 0.0) for f in _frames(forensic)]
    nums = [float(v) for v in vals if isinstance(v, (int, float))]
    return max(nums, default=default)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value > 0
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "available", "detected"}
    return False


def _manipulation_signals(
    c2pa_result: Any,
    forensic: Any,
    context_result: Any,
    watermark: dict[str, Any] | None,
    multi: dict[str, Any] | None,
) -> tuple[list[str], list[str]]:
    """Split AI evidence into (decisive, supporting).

    Decisive signals establish Art. 3(60) on their own: the content declares
    itself AI-generated, or a dedicated media leg (lipsync, vocoder) reports
    manipulation. Supporting signals raise suspicion — an ELA/spectral
    anomaly is equally produced by re-compression, a bad scan, or aggressive
    denoising — and are reported separately so they cannot silently tip a
    legal determination.

    The prior implementation read `forensic.instability` and
    `forensic.splicing_detected`, neither of which exists on `ForensicResult`
    (it carries `aggregate_anomaly`, `temporal_consistency` and per-frame
    metrics). Every image therefore produced an empty signal list and was
    reported as "not a deep fake" regardless of anomaly score.
    """
    sig: list[str] = []
    supporting: list[str] = []

    marking, mark_ev, _ = detect_marking(c2pa_result, watermark)
    if marking in {"AI_GENERATED", "AI_MODIFIED"}:
        sig.append(f"machine-readable declaration: {marking}")
        sig.extend(mark_ev)

    # Dedicated media legs, read from their real result contracts. The previous
    # `manipulated`/`mismatch` keys exist on neither `LipSyncResult` nor
    # `AudioResult`, so the audio leg was unreachable. Thresholds mirror the
    # risk engine (`multimodal_risk_bump`) so the two paths agree on one file.
    lipsync = (multi or {}).get("lipsync") or {}
    if isinstance(lipsync, dict) and lipsync.get("verdict") == "DIVERGENT":
        sig.append("lipsync leg reports divergent mouth/audio correlation")
    audio = (multi or {}).get("audio") or {}
    if isinstance(audio, dict) and audio.get("available"):
        if float(audio.get("vocoder_anomaly", 0) or 0) > 0.6:
            sig.append(
                f"vocoder leg reports neural-vocoder artifacts ({audio.get('vocoder_anomaly')})"
            )
        elif float(audio.get("pitch_instability", 0) or 0) > 0.7:
            # Flat prosody is equally produced by dubbing, singing and heavy
            # compression, so it raises suspicion without establishing synthesis.
            supporting.append(
                "audio leg flags flat prosody (non-specific: dubbing, singing and "
                "compression flatten prosody too)"
            )

    env = getattr(context_result, "environment_mismatch", None)
    if env:
        supporting.append("context check flags environment/capture mismatch")

    # --- supporting: pointer to inspection, never proof of synthesis ---
    ano = getattr(forensic, "aggregate_anomaly", 0.0)
    if isinstance(ano, (int, float)) and ano > 0:
        detail = f"aggregate visual anomaly {ano:.3f}"
        if getattr(forensic, "is_video", False):
            detail += f", temporal consistency {getattr(forensic, 'temporal_consistency', 1.0):.2f}"
        supporting.append(f"{detail} (non-specific: recompression and rescaling also raise it)")

    face_peak = _frame_max("face_anomaly", forensic)
    if _count_faces(forensic) and face_peak > 0.45:
        supporting.append(f"face-region anomaly {face_peak:.2f} (identity-editing hotspot)")

    return sig, supporting


def _audience_risk(subjects: list[str]) -> str:
    """Art. 50(4) is audience-aware, not an average-reasonable-person test."""
    if "PERSON" in subjects:
        return "high"  # likenesses reach children and low-literacy audiences too
    if {"PLACE", "EVENT", "OBJECT"} & set(subjects):
        return "moderate"
    return "unknown"


# --------------------------------------------------------------------------
# Assessment
# --------------------------------------------------------------------------


VALID_ROLES = ("provider", "deployer", "both")


def operator_posture() -> dict[str, Any]:
    """Declared Art. 50 posture from configuration.

    Split from `assess()` so the carve-out decision is resolved in exactly one
    place and can be echoed into the audit trail. An out-of-range
    `AI_ACT_ROLE` fails closed to `deployer`: an unrecognised role must not
    silently disable a duty, and must not crash report generation either.
    """
    from src.config import SETTINGS

    role = (SETTINGS.ai_act_role or "").strip().lower()
    if role not in VALID_ROLES:
        if role:
            log.warning("ai_act50: unknown AI_ACT_ROLE %r; failing closed to deployer", role)
        role = "deployer"
    return {
        "role": role,
        "law_enforcement": bool(SETTINGS.ai_act_law_enforcement),
        "standard_editing_only": bool(SETTINGS.ai_act_standard_editing_only),
    }


def assess(
    c2pa_result: Any,
    forensic: Any,
    context_result: Any,
    *,
    watermark: dict[str, Any] | None = None,
    multi: dict[str, Any] | None = None,
    role: str | None = None,
    law_enforcement: bool | None = None,
    standard_editing_only: bool | None = None,
) -> AiAct50Result:
    """Full Art. 50 assessment. `role` is provider | deployer | both.

    Posture arguments default to `None`, meaning "use the declared operator
    posture" (`operator_posture()`). Callers therefore cannot forget to pass
    a carve-out: the module decides from configuration, and an explicit
    argument still wins for tests and one-off assessments.
    """
    declared = operator_posture()
    role = declared["role"] if role is None else role
    law_enforcement = declared["law_enforcement"] if law_enforcement is None else law_enforcement
    standard_editing_only = (
        declared["standard_editing_only"]
        if standard_editing_only is None
        else standard_editing_only
    )
    res = AiAct50Result()
    res.operator_posture = {
        "role": role,
        "law_enforcement": bool(law_enforcement),
        "standard_editing_only": bool(standard_editing_only),
    }

    marking, mark_ev, standard = detect_marking(c2pa_result, watermark)
    res.machine_readable_marking = marking
    res.marking_evidence = mark_ev
    res.marking_standard = standard
    res.marking_present = marking != "ABSENT"

    subjects, plausible = assess_subjects(forensic, c2pa_result, context_result, multi)
    res.depicted_subject = subjects
    res.subject_resembles_reality = plausible
    res.false_appearance_risk = _audience_risk(subjects)

    signals, supporting = _manipulation_signals(
        c2pa_result, forensic, context_result, watermark, multi
    )
    res.deepfake_basis = list(signals)
    res.supporting_evidence = list(supporting)

    if not subjects:
        # Nothing resembling a person, object, place or event: outside Art. 3(60).
        res.deepfake = False
        res.deepfake_confidence = "Moderate"
        res.deepfake_basis.append("no Art. 3(60) subject class detected in the media")
        return _finalize(
            res,
            role=role,
            law_enforcement=law_enforcement,
            standard_editing_only=standard_editing_only,
        )

    if not signals:
        if supporting:
            # Suspicious but non-specific: refuse to state a determination.
            # A deployer must not be handed a fabricated "not a deep fake"
            # simply because the only evidence is a generic anomaly score.
            res.deepfake = None
            res.deepfake_confidence = "Indeterminate"
            res.deepfake_basis.append(
                "no decisive AI signal, but non-specific forensic evidence is present; "
                "Art. 3(60) cannot be determined either way from this evidence"
            )
            return _finalize(
                res,
                role=role,
                law_enforcement=law_enforcement,
                standard_editing_only=standard_editing_only,
            )
        res.deepfake = False
        res.deepfake_confidence = "Low"
        res.deepfake_basis.append(
            "no AI-generation or manipulation signal detected; note that absence of a marker "
            "is not proof of a human original (Art. 50(2) marking is not universal)"
        )
        return _finalize(
            res,
            role=role,
            law_enforcement=law_enforcement,
            standard_editing_only=standard_editing_only,
        )

    res.deepfake = True
    res.deepfake_confidence = "High" if marking in {"AI_GENERATED", "AI_MODIFIED"} else "Moderate"
    if res.deepfake_confidence == "High":
        res.deepfake_basis.append(
            "subject class present AND content self-declares as AI-generated/modified"
        )

    return _finalize(
        res, role=role, law_enforcement=law_enforcement, standard_editing_only=standard_editing_only
    )


def _finalize(
    res: AiAct50Result,
    *,
    role: str,
    law_enforcement: bool,
    standard_editing_only: bool,
) -> AiAct50Result:
    """Apply role-dependent duties and the explicit carve-outs."""
    # Art. 50(4): deployer disclosure duty for deep fakes.
    if res.deepfake and not law_enforcement:
        res.disclosure_required = True
        res.label_text = (
            "AI-GENERATED" if res.machine_readable_marking == "AI_GENERATED" else "AI-MODIFIED"
        )
        res.icon_variant = (
            "AI_GENERATED" if res.machine_readable_marking == "AI_GENERATED" else "AI_MODIFIED"
        )
        res.disclosure_obligations = [
            "Art. 50(4): disclose that the content was artificially generated or manipulated.",
            (
                "Art. 50(5): the disclosure must be clear, distinguishable and provided "
                "no later than first exposure."
            ),
            (
                "Must be perceivable without technical tools or dedicated actions; "
                "end credits alone do not comply."
            ),
            (
                "Audio-only content: short audible disclaimer in plain language at the "
                "beginning, with reminders after interruptions for long-form or live audio."
            ),
            "Video: label at the beginning, after interruptions and at regular intervals.",
            (
                "Artistic/creative/satirical work: reduced disclosure, sufficient not to "
                "impair the work (CoP Section 2, Commitment 3)."
            ),
        ]
        res.accessibility = [
            "Conform to Directive 2019/882 / 2016/2102 accessibility requirements.",
            "Provide alt text / ARIA labels indicating the content is AI-generated or modified.",
            "Consider audio description, tactile cues and high-contrast presentation.",
        ]
    elif res.deepfake and law_enforcement:
        res.exemption = (
            "Art. 50(4) law-enforcement carve-out declared by the operator: authorised to "
            "detect, prevent, investigate or prosecute criminal offences. Verify the "
            "authorisation exists — this module cannot."
        )
        res.disclosure_required = None
    elif res.deepfake is False:
        res.disclosure_required = False
    elif res.deepfake is None:
        # No duty is established, and none is waived either: an indeterminate
        # determination still owes the deployer a documented decision.
        res.disclosure_required = None

    # Art. 50(2): provider marking duty, with the transitional relief deadline.
    if role in {"provider", "both"} and standard_editing_only:
        standard_editing = (
            "Art. 50(2) exemption: assistive standard editing that does not substantially alter "
            "the input data or its semantics."
        )
        # Combine, never drop: two declared carve-outs must both stay visible.
        res.exemption = (
            f"{res.exemption} {standard_editing}".strip() if res.exemption else standard_editing
        )

    return res


def render_disclosure_badge(res: AiAct50Result, width: int = 0) -> str | None:
    """An SVG label that satisfies the *form* of Art. 50(5) disclosure.

    Deliberately an equivalent label, not the EU icon assets: the icons are
    optional (their use "does not establish legal compliance by itself") and
    the Guidelines require plain-language labelling with assistive-technology
    support. Returns None when no disclosure is required.

    Accessibility is part of the obligation, not decoration: the SVG carries
    `role="img"` and a `<title>`/`<desc>`, and the caller is expected to emit
    the same text as alt text.
    """
    if not res.disclosure_required or not res.label_text:
        return None
    # Clamp the rendered width and escape every interpolated value: the badge is
    # exported and rendered in unknown contexts (a republication site), so a
    # stored `label_text` that ever became free-form must not become markup.
    w = max(120, min(int(width or 320), 2000))
    h = 64
    suffix = re.sub(r"[^a-z0-9]+", "-", res.label_text.lower()).strip("-") or "label"
    ids = f"aibadge-{suffix}"
    text = _xml_escape(res.label_text)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" role="img" '
        f'aria-labelledby="{ids}-title {ids}-desc">'
        f'<title id="{ids}-title">{text}</title>'
        f'<desc id="{ids}-desc">This content was artificially generated or manipulated '
        f"with an AI system.</desc>"
        f'<rect x="0" y="0" width="{w}" height="{h}" rx="8" fill="#111827" '
        f'stroke="#e5e7eb" stroke-width="1"/>'
        f'<text x="16" y="40" font-family="sans-serif" font-size="26" font-weight="700" '
        f'fill="#ffffff">{text}</text>'
        f"</svg>"
    )


__all__ = [
    "AI_ACT_APPLICABLE_FROM",
    "AI_ACT_MARKING_TRANSITION_END",
    "CODE_OF_PRACTICE_DATE",
    "GUIDELINES_DATE",
    "MARKING_STATES",
    "SUBJECT_KINDS",
    "AiAct50Result",
    "assess",
    "assess_subjects",
    "detect_marking",
    "extract_created_by",
    "extract_digital_source_types",
    "render_disclosure_badge",
]
