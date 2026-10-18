"""EU AI Act Art. 50 transparency: module logic + end-to-end surfacing.

Art. 50 is a *legal duty* with two distinct questions (50(2) machine-readable
marking, 50(4) human-facing disclosure). These tests pin the properties that
make the assessment usable rather than decorative:

  - ABSENT marking is never reported as a human original.
  - Disclosure follows the deep-fake determination, not the risk score, and
    the Art. 50(4) law-enforcement carve-out is *declared*, never silent.
  - Every output path (JSON, markdown, STIX, API) carries the same marking
    state, so an exporter cannot drift from the analyst's screen.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.modules.ai_act50 import (
    AiAct50Result,
    assess,
    detect_marking,
    extract_digital_source_types,
    render_disclosure_badge,
)
from src.schemas import C2paResult, ContextResult, ForensicResult


def _c2pa(dst: str = "", manifest: bool = False, created_by: str = "") -> C2paResult:
    info: dict = {}
    if dst or created_by:
        info["cbor"] = {
            "assertions": [
                {
                    "digitalSourceType": dst,
                    "createdBy": created_by,
                }
            ]
        }
    return C2paResult(
        file_path="x.jpg",
        sha256="ab" * 32,
        file_size_bytes=1024,
        mime_type="image/jpeg",
        has_manifest=manifest or bool(dst),
        manifest_info=info,
    )


def _forensic(anomaly: float = 0.0, faces: int = 0) -> ForensicResult:
    from src.schemas import FrameMetrics

    return ForensicResult(
        file_path="x.jpg",
        device_used="cpu",
        onnx_used=False,
        num_frames_analyzed=1,
        frames=[FrameMetrics(0, 0.0, anomaly, 0.0, 0.0, 0.0, 0, face_count=faces)],
        aggregate_anomaly=anomaly,
    )


def _context(matches: int = 0) -> ContextResult:
    from src.schemas import ContextMatch

    return ContextResult(
        file_path="x.jpg",
        query_phash="00" * 8,
        query_dhash=None,
        backend="phash",
        matches=[ContextMatch("E1", "known event", None, None, 1, 0.9) for _ in range(matches)],
        is_decontextualized=bool(matches),
    )


# --------------------------------------------------------------- marking


def test_marking_states_are_distinguished():
    gen = detect_marking(_c2pa("c2pa.trainedAlgorithmicData"))
    assert gen[0] == "AI_GENERATED"

    human = detect_marking(_c2pa("c2pa.digitalCapture"))
    assert human[0] == "HUMAN_DECLARED"

    bare = detect_marking(_c2pa("", manifest=True))
    assert bare[0] == "UNMARKED"

    # The load-bearing distinction: no marker is NOT "human-made".
    absent = detect_marking(_c2pa())
    assert absent[0] == "ABSENT"
    assert absent[0] != "HUMAN_DECLARED"


def test_watermark_counts_as_machine_readable_marker():
    """Read the real `WatermarkResult` contract: `ai_generated` is the boolean
    verdict, `synthid` is a status string. Reading the status string as a flag
    marked *every* analysed file as AI-generated."""
    state, ev, std = detect_marking(_c2pa(), {"ai_generated": True, "generator": "Midjourney"})
    assert state == "AI_GENERATED" and ev and std

    synthid, _, _ = detect_marking(_c2pa(), {"ai_generated": False, "synthid": "DETECTED"})
    assert synthid == "AI_GENERATED"

    # The regression this pins: a not-detected status is a non-empty string.
    clean = detect_marking(_c2pa(), {"ai_generated": False, "synthid": "NOT_DETECTED"})
    assert clean[0] == "ABSENT"


def test_digital_source_type_walks_nested_ingredients():
    """AI-flavoured ingredient anywhere in the graph makes the asset AI."""
    info = {
        "cbor": {
            "manifest": {"assertions": [{"digitalSourceType": "c2pa.digitalCapture"}]},
            "ingredients": [{"digitalSourceType": "c2pa.trainedAlgorithmicData"}],
        }
    }
    found = extract_digital_source_types(info)
    assert "c2pa.trainedalgorithmicdata" in found and "c2pa.digitalcapture" in found
    c2pa = _c2pa()
    c2pa.manifest_info = info
    c2pa.has_manifest = True
    assert detect_marking(c2pa)[0] == "AI_GENERATED"  # strongest source type wins


# ----------------------------------------------------------- assessment


def test_declared_ai_with_person_subject_requires_disclosure():
    res = assess(
        _c2pa("c2pa.trainedAlgorithmicData"),
        _forensic(anomaly=0.1, faces=1),
        _context(),
    )
    assert res.deepfake is True
    assert res.deepfake_confidence == "High"
    assert res.disclosure_required is True
    assert res.label_text == "AI-GENERATED"
    assert res.false_appearance_risk == "high"  # audience-aware, not average-reasonable
    assert any("50(4)" in o for o in res.disclosure_obligations)
    assert res.intent_assessed is False
    assert res.legal_advice is False
    assert "art. 50" in res.summary.lower()


def test_unmarked_clean_media_is_not_called_a_deep_fake():
    res = assess(_c2pa(), _forensic(anomaly=0.0), _context())
    assert res.deepfake is False
    assert res.disclosure_required is False
    assert res.machine_readable_marking == "ABSENT"
    assert res.supporting_evidence == []
    # Honest about the limit of the negative finding.
    assert any("not proof of a human original" in b for b in res.deepfake_basis)


def test_law_enforcement_carveout_is_declared_not_silent():
    res = assess(
        _c2pa("c2pa.trainedAlgorithmicData"),
        _forensic(anomaly=0.1, faces=1),
        _context(),
        law_enforcement=True,
    )
    assert res.deepfake is True
    assert res.disclosure_required is None  # not False: the duty is carved out
    assert "law-enforcement" in res.exemption
    assert "cannot" in res.exemption  # module refuses to certify authorisation


def test_provider_standard_editing_exemption_is_declared():
    res = assess(
        _c2pa(),
        _forensic(anomaly=0.5, faces=1),
        _context(),
        role="provider",
        standard_editing_only=True,
    )
    assert "50(2)" in res.exemption


def test_badge_is_accessible_svg_or_none():
    deep = AiAct50Result(disclosure_required=True, label_text="AI-GENERATED", deepfake=True)
    svg = render_disclosure_badge(deep)
    assert svg and 'role="img"' in svg and "<title" in svg and "<desc" in svg
    assert "stroke=" in svg  # distinguishable on a dark page too, not only white
    assert render_disclosure_badge(AiAct50Result(disclosure_required=False)) is None


def test_badge_escapes_untrusted_label_text():
    """The badge is exported and rendered in unknown contexts (republication)."""
    res = AiAct50Result(disclosure_required=True, label_text="<script>alert(1)</script>")
    svg = render_disclosure_badge(res)
    assert svg and "<script" not in svg and "&lt;script&gt;" in svg


def test_media_legs_are_decisive_but_only_above_threshold():
    """A dedicated media leg establishes Art. 3(60) on its own. The previous
    implementation read `manipulated`/`mismatch` keys that exist on neither
    `LipSyncResult` nor `AudioResult`, so the audio leg was unreachable."""
    base = (_c2pa(), _forensic(anomaly=0.0, faces=1), _context())

    divergent = assess(*base, multi={"lipsync": {"verdict": "DIVERGENT"}})
    assert divergent.deepfake is True and divergent.disclosure_required is True

    vocoder = assess(*base, multi={"audio": {"available": True, "vocoder_anomaly": 0.8}})
    assert vocoder.deepfake is True

    # Flat prosody is non-specific (dubbing, singing, compression): it must
    # raise suspicion without establishing a determination.
    tts = assess(*base, multi={"audio": {"available": True, "pitch_instability": 0.9}})
    assert tts.deepfake is None and tts.supporting_evidence

    # Below threshold a leg score is not a determination.
    quiet = assess(*base, multi={"audio": {"available": True, "vocoder_anomaly": 0.2}})
    assert quiet.deepfake is False


def test_operator_posture_fails_closed_and_is_echoed(monkeypatch):
    """An unrecognised role must not silently drop a duty, and a declared
    carve-out must be echoed in the assessment."""
    import dataclasses

    from src import config as config_mod

    monkeypatch.setattr(
        config_mod,
        "SETTINGS",
        dataclasses.replace(config_mod.SETTINGS, ai_act_role="auditor"),
    )
    res = assess(_c2pa("c2pa.trainedAlgorithmicData"), _forensic(faces=1), _context())
    assert res.operator_posture["role"] == "deployer"

    monkeypatch.setattr(
        config_mod,
        "SETTINGS",
        dataclasses.replace(config_mod.SETTINGS, ai_act_law_enforcement=True),
    )
    carved = assess(_c2pa("c2pa.trainedAlgorithmicData"), _forensic(faces=1), _context())
    assert carved.disclosure_required is None  # carved out, not waived == not False
    assert "law-enforcement" in carved.exemption
    assert carved.operator_posture["law_enforcement"] is True


def test_combined_carveouts_are_both_reported():
    res = assess(
        _c2pa("c2pa.trainedAlgorithmicData"),
        _forensic(faces=1),
        _context(),
        role="provider",
        law_enforcement=True,
        standard_editing_only=True,
    )
    assert "law-enforcement" in res.exemption and "50(2)" in res.exemption


# ------------------------------------------------------ surfacing (output)


def test_aggregate_report_emits_one_disclosure_recommendation():
    """The two entry points share a single assessment, so the Art. 50(4)
    recommendation is emitted exactly once (a plain `any(...)` would hide a
    duplicate)."""
    from src.engines.llm_engine import aggregate_report

    rep = aggregate_report(_c2pa("c2pa.trainedAlgorithmicData"), _forensic(faces=1), _context())
    recs = [r for r in rep.recommendations if r.startswith("AI Act Art. 50(4)")]
    assert len(recs) == 1


def test_generate_report_reuses_a_supplied_assessment():
    """The worker hands its multimodal-aware assessment to the generator
    instead of letting it recompute a divergent one."""
    from src.modules.explain_report import generate_report

    given = AiAct50Result(
        machine_readable_marking="AI_GENERATED",
        deepfake=True,
        disclosure_required=True,
        label_text="AI-GENERATED",
    )
    rep = generate_report(_c2pa(), _forensic(anomaly=0.0), _context(), ai_act50=given)
    # `_c2pa()` declares nothing; the supplied assessment must win.
    assert rep.ai_act50["machine_readable_marking"] == "AI_GENERATED"
    assert rep.ai_act50["label_text"] == "AI-GENERATED"


def test_report_json_markdown_and_stix_agree():
    from src.engines.llm_engine import aggregate_report, stix_for
    from src.modules.explain_report import generate_report

    c2pa, fr, ctx = (
        _c2pa("c2pa.trainedAlgorithmicData"),
        _forensic(anomaly=0.2, faces=1),
        _context(),
    )
    for rep in (generate_report(c2pa, fr, ctx), aggregate_report(c2pa, fr, ctx)):
        assert rep.ai_act50["machine_readable_marking"] == "AI_GENERATED"
        assert rep.ai_act50["disclosure_required"] is True
        assert any("Art. 50(4)" in r for r in rep.recommendations)
        assert "EU AI Act Art. 50" in rep.to_markdown()
        opinion = next(o for o in stix_for(rep)["objects"] if o["type"] == "opinion")
        assert opinion["x_ai_act50"]["machine_readable_marking"] == "AI_GENERATED"


def test_marking_state_survives_clearance_redaction():
    """Art. 50 marking is a fact about the media, not case intelligence.

    `redact()` filters the `result` payload field-by-field. The marking block
    is served beside it (never inside it), so a CLEAR requester still learns
    whether a label must be rendered, while the narrative stays hidden.
    """
    from main import _ai_act50_payload
    from src.tlp import redact

    rep = {
        "ai_act50": AiAct50Result(machine_readable_marking="ABSENT").to_dict(),
        "verdict": "HIGH_RISK",
        "tlp": "TLP:RED",
        "narrative": "sensitive",
        "forensic": {"aggregate_anomaly": 0.9},
        "schema_version": "2.0",
    }
    red = redact(rep, "TLP:RED")
    assert set(red) >= {"verdict", "tlp", "schema_version"}
    assert "narrative" not in red and "forensic" not in red

    # A CLEAR requester still gets the marking, via the dedicated field.
    payload = {"result": redact(rep, "TLP:CLEAR")}
    payload["ai_act50"] = _ai_act50_payload(rep)
    assert payload["ai_act50"]["machine_readable_marking"] == "ABSENT"
    assert payload["ai_act50"]["available"] is True


def test_transparency_payload_withholds_case_intelligence_above_green():
    """The compliance fact is served at every clearance; the case-intelligence
    fields and operator posture are not served above GREEN, where `result` is
    already redacted to the verdict."""
    from main import _ai_act50_payload

    full = AiAct50Result(
        machine_readable_marking="AI_GENERATED",
        marking_present=True,
        deepfake=True,
        deepfake_confidence="High",
        disclosure_required=True,
        label_text="AI-GENERATED",
        depicted_subject=["PERSON"],
        false_appearance_risk="high",
        exemption="law-enforcement carve-out",
        operator_posture={"role": "deployer", "law_enforcement": True},
    ).to_dict()

    clear = _ai_act50_payload({"ai_act50": full}, "TLP:CLEAR")
    assert clear["deepfake"] is True and clear["operator_posture"]

    red = _ai_act50_payload({"ai_act50": full}, "TLP:RED")
    assert red["available"] is True
    assert red["machine_readable_marking"] == "AI_GENERATED"
    assert red["disclosure_required"] is True
    # Withheld must be explicit, not silently dropped (so a consumer cannot
    # read the absence as "assessed: none").
    assert red["restricted"] is True and "deepfake" in red["withheld_fields"]
    for hidden in (
        "deepfake",
        "deepfake_confidence",
        "depicted_subject",
        "false_appearance_risk",
        "exemption",
        "operator_posture",
    ):
        assert hidden not in red

    # A requester cleared *above* the record keeps full detail: the block
    # follows the served level, so it never contradicts `result`.
    assert _ai_act50_payload({"ai_act50": full}, "TLP:GREEN")["deepfake"] is True


def test_api_payload_shape_and_badge():
    import main

    _ai_act50_payload = main._ai_act50_payload
    deep = AiAct50Result(
        machine_readable_marking="AI_GENERATED",
        marking_present=True,
        deepfake=True,
        deepfake_confidence="High",
        disclosure_required=True,
        label_text="AI-GENERATED",
        depicted_subject=["PERSON"],
        false_appearance_risk="high",
    )
    payload = main._ai_act50_payload({"ai_act50": deep.to_dict()})
    assert payload["available"] is True
    assert payload["deepfake"] is True and payload["disclosure_timing"] == "at_first_exposure"
    assert payload["disclosure_badge_svg"].startswith("<svg")
    assert payload["legal_advice"] is False

    # Never conflate "no assessment" with "no marking".
    missing = main._ai_act50_payload({})
    assert missing["available"] is False and "machine_readable_marking" not in missing

    # A stored shape we do not recognise must not 500 a compliance read.
    assert main._disclosure_badge({"unexpected": 1}) is None


@pytest.mark.parametrize(
    "anomaly,expected,conf",
    [
        (0.0, False, "Low"),  # nothing detected -> negative, honestly caveated
        (0.6, None, "Indeterminate"),  # generic anomaly must NOT become "yes"
    ],
)
def test_forensic_anomaly_alone_never_establishes_deepfake(anomaly, expected, conf):
    """An ELA/spectral anomaly is non-specific: recompression raises it too.

    The prior implementation read a `forensic.instability` field that does not
    exist, so this evidence was silently dropped. Restored, it must inform the
    determination without asserting one.
    """
    res = assess(_c2pa(), _forensic(anomaly=anomaly, faces=1), _context())
    assert res.machine_readable_marking == "ABSENT"
    assert res.deepfake is expected and res.deepfake_confidence == conf
    if expected is None:
        assert res.supporting_evidence and res.deepfake_basis
        assert res.disclosure_required is None  # duty neither established nor waived
    else:
        assert any("not proof of a human original" in b for b in res.deepfake_basis)


def test_pipeline_end_to_end_carries_assessment(tmp_path):
    """The CLI path (generate_report) must expose the field analysts read."""
    from PIL import Image

    from main import analyze

    p = tmp_path / "aiact.png"
    arr = np.tile(np.linspace(0, 255, 96, dtype=np.uint8), (96, 1))
    Image.fromarray(np.dstack([arr, arr[::-1], arr])).save(p)
    rep = analyze(str(p))
    a = rep["ai_act50"]
    assert a["machine_readable_marking"] == "ABSENT"
    assert a["legal_advice"] is False
    assert rep["schema_version"] == "2.0"
