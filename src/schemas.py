"""Shared provenance schemas (v2).

Wire format is versioned (`REPORT_SCHEMA_VERSION`) so Python, Rust
microservices, and stored reports stay compatible. All new fields have
defaults — v1 readers ignore them, v2 readers get SOTA signals.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

REPORT_SCHEMA_VERSION = "2.0"


class Verdict(str, Enum):
    """Analyst-facing verdict buckets. Ordered by increasing risk."""

    AUTHENTIC = "AUTHENTIC"
    LOW_RISK = "LOW_RISK"
    UNVERIFIABLE = "UNVERIFIABLE"
    SUSPICIOUS = "SUSPICIOUS"
    HIGH_RISK = "HIGH_RISK"


# ICD-203 estimative-likelihood ladder used for hypothesis wording.
LIKELIHOOD_TERMS = (
    (0.05, "almost certainly not"),
    (0.20, "very unlikely"),
    (0.35, "unlikely"),
    (0.55, "roughly even chance"),
    (0.75, "likely"),
    (0.90, "very likely"),
    (1.01, "almost certainly"),
)


def likelihood_term(p: float) -> str:
    for bound, term in LIKELIHOOD_TERMS:
        if p < bound:
            return term
    return "almost certainly"


@dataclass
class C2paResult:
    file_path: str
    sha256: str
    file_size_bytes: int
    mime_type: str
    has_manifest: bool
    # None = not applicable (no manifest to validate).
    signature_valid: bool | None = None
    # SOTA: per-check validation detail (C2PA-Verify UX pattern). Keys:
    #   manifestFound/structureParse/claimSignature/certChain/dataHash/timestamp
    # each -> {"status": PASS|FAIL|SKIPPED, "reason": str}
    manifest_info: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    checker: str = "c2pa_verify"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FrameMetrics:
    frame_index: int
    timestamp_sec: float | None
    ela_score: float  # 0..1 recompression divergence
    blur_variance: float
    noise_score: float
    histogram_anomaly: float
    orb_clusters: int
    # --- SOTA spectral/compression signals (defaults keep v1 compat) ---
    fft_score: float = 0.0  # 0..1 spectral artifact energy (GAN/diffusion peaks)
    blockiness: float = 0.0  # 0..1 8x8 grid discontinuity (splice/recompress)
    dct_double_jpeg: float = 0.0  # 0..1 double-compression periodicity hint
    face_count: int = 0
    face_anomaly: float = 0.0  # 0..1 max face-region ELA anomaly
    anomaly_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ForensicResult:
    file_path: str
    device_used: str
    onnx_used: bool
    num_frames_analyzed: int
    frames: list[FrameMetrics] = field(default_factory=list)
    aggregate_anomaly: float = 0.0
    # --- SOTA aggregation ---
    temporal_consistency: float = 1.0  # video only; 1.0 = stable, low = flicker
    is_video: bool = False
    heatmap_path: str | None = None  # ELA heatmap when enabled
    model_versions: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ContextMatch:
    event_id: str
    description: str
    event_date: str | None
    source: str | None
    hamming_distance: int
    confidence: float
    # SOTA: which signal fired and fused evidence.
    match_type: str = "phash"  # phash | pdq | semantic
    pdq_distance: int | None = None
    clip_similarity: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ContextResult:
    file_path: str
    query_phash: str | None
    query_dhash: str | None
    backend: str
    matches: list[ContextMatch] = field(default_factory=list)
    is_decontextualized: bool = False
    threshold: int = 8
    # SOTA multi-signal query state.
    query_pdq: str | None = None
    clip_available: bool = False
    geo_temporal_note: str | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AudioResult:
    available: bool = False
    duration_sec: float | None = None
    sample_rate: int | None = None
    vocoder_anomaly: float = 0.0  # 0..1 neural-vocoder artifact score
    pitch_instability: float = 0.0  # 0..1 unnatural F0 stability (TTS flag)
    spectral_notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LipSyncResult:
    available: bool = False
    correlation: float | None = None  # mouth-motion vs audio-envelope Pearson
    verdict: str = "SKIPPED"  # SYNCED | DIVERGENT | SKIPPED
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RppgResult:
    available: bool = False
    heart_rate_bpm: float | None = None
    signal_quality: float = 0.0  # 0..1 spectral peak prominence
    waveform: list[float] = field(default_factory=list)  # downsampled pulse wave
    liveness: str = "UNKNOWN"  # LIVE | NO_PULSE | UNKNOWN
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Mp4Result:
    is_mp4: bool = False
    atoms: list[dict[str, Any]] = field(default_factory=list)
    faststart: bool | None = None  # moov-before-mdat (web-optimized ordering)
    anomalies: list[str] = field(default_factory=list)
    double_compression_hint: float = 0.0  # 0..1 container+DCT fusion
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GeointResult:
    available: bool = False
    latitude: float | None = None
    longitude: float | None = None
    timestamp_utc: str | None = None
    solar_elevation_deg: float | None = None
    solar_azimuth_deg: float | None = None
    expected_shadow_azimuth_deg: float | None = None
    measured_shadow_azimuth_deg: float | None = None
    angular_error_deg: float | None = None
    shadow_length_ratio_expected: float | None = None
    consistent: bool | None = None
    confidence: float = 0.0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Hypothesis:
    """One ACH row: H-authentic / H-manipulated / H-decontextualized."""

    name: str
    likelihood: float  # 0..1 posterior-ish weight from signals
    term: str = ""  # ICD-203 wording, auto-filled
    rationale: str = ""

    def __post_init__(self) -> None:
        if not self.term:
            self.term = likelihood_term(self.likelihood)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FinalReport:
    file_path: str
    risk_score: int
    verdict: Verdict
    executive_summary: str
    findings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    narrative: str = ""
    llm_backend: str = "rule-based"
    c2pa: dict[str, Any] = field(default_factory=dict)
    forensic: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    # Enterprise multimodal leg (audio/lipsync/rppg/mp4/geoint dicts).
    multimodal: dict[str, Any] = field(default_factory=dict)
    # --- SOTA tradecraft / compliance ---
    schema_version: str = REPORT_SCHEMA_VERSION
    hypotheses: list[dict[str, Any]] = field(default_factory=list)
    analytic_confidence: str = "Moderate"  # High | Moderate | Low (ICD-206)
    tlp: str = "TLP:CLEAR"
    citations: list[str] = field(default_factory=list)
    calibration_note: str = (
        "Risk is deterministic (auditable rules); LLM supplies narrative only. "
        "Scores are triage priors, not probabilities."
    )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["verdict"] = (
            self.verdict.value if isinstance(self.verdict, Verdict) else str(self.verdict)
        )
        return d

    def to_markdown(self) -> str:
        verdict = self.verdict.value if isinstance(self.verdict, Verdict) else self.verdict
        lines = [
            f"{self.tlp} — handle per distribution policy.",
            "",
            f"# Media Provenance Report (schema v{self.schema_version})",
            "",
            f"- **File:** `{self.file_path}`",
            f"- **Risk score:** `{self.risk_score}/100`",
            f"- **Verdict:** `{verdict}`",
            f"- **Analytic confidence:** `{self.analytic_confidence}`",
            f"- **LLM backend:** `{self.llm_backend}`",
            "",
            "## Executive summary",
            "",
            self.executive_summary,
            "",
            "## Narrative",
            "",
            self.narrative,
            "",
            "## Competing hypotheses (ACH)",
            "",
        ]
        if self.hypotheses:
            lines.extend(
                (
                    f"- **{h.get('name')}** — {h.get('term')} "
                    f"(p≈{h.get('likelihood'):.2f}): {h.get('rationale')}"
                )
                for h in self.hypotheses
            )
        else:
            lines.append("- No hypotheses recorded.")
        lines += ["", "## Findings", ""]
        lines += [f"- {x}" for x in self.findings] or ["- No findings."]
        lines += ["", "## Recommendations", ""]
        lines += [f"- {x}" for x in self.recommendations] or ["- No recommendations."]
        if self.citations:
            lines += ["", "## Citations", ""]
            lines += [f"- {c}" for c in self.citations]
        lines += [
            "",
            "## Raw signals (audit)",
            "",
            (
                f"- C2PA manifest: `{self.c2pa.get('has_manifest')}` "
                f"valid=`{self.c2pa.get('signature_valid')}`"
            ),
            (
                f"- Forensic anomaly: `{self.forensic.get('aggregate_anomaly')}` "
                f"on `{self.forensic.get('num_frames_analyzed')}` frame(s) "
                f"via `{self.forensic.get('device_used')}`"
            ),
            (
                f"- Context phash: `{self.context.get('query_phash')}` "
                f"pdq=`{self.context.get('query_pdq')}` "
                f"matches={len(self.context.get('matches', []))} "
                f"decontextualized=`{self.context.get('is_decontextualized')}`"
            ),
            "",
            f"*{self.calibration_note}*",
        ]
        return "\n".join(lines)
