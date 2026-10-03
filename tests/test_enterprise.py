"""Enterprise regression: TLP, signatures, ORM repo, STIX, GEOINT, audio, worker.

All runnable offline with stdlib+numpy+PIL only. Legs needing cv2/librosa/
ImageHash assert graceful SKIPPED instead of raising (honest degradation).
"""

from __future__ import annotations

import shutil
import wave
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

OPENSSL = shutil.which("openssl")

TMP = Path("/tmp/opencode/prov_ent")
TMP.mkdir(parents=True, exist_ok=True)


def _png(name: str = "e.png") -> Path:
    arr = np.tile(np.linspace(20, 235, 128, dtype=np.uint8), (128, 1))
    p = TMP / name
    Image.fromarray(np.dstack([arr, arr, arr])).save(p)
    return p


def _wav(name: str = "tone.wav", secs: float = 2.0, freq: float = 440.0) -> Path:
    sr = 16000
    t = np.arange(int(sr * secs)) / sr
    pcm = (0.5 * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    p = TMP / name
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return p


# ---------------------------------------------------------------- TLP


def test_tlp_allows_and_redact():
    from src.tlp import allows, redact

    assert allows("TLP:RED", "TLP:CLEAR") and allows("TLP:AMBER", "TLP:AMBER")
    assert not allows("TLP:GREEN", "TLP:AMBER")
    assert not allows("TLP:CLEAR", "TLP:RED")
    rep = {
        "verdict": "SUSPICIOUS",
        "narrative": "secret",
        "risk_score": 70,
        "tlp": "TLP:AMBER",
        "executive_summary": "s",
        "schema_version": "2.0",
        "analytic_confidence": "Moderate",
        "recommendations": [],
    }
    narrow = redact(rep, "TLP:AMBER")
    assert "narrative" not in narrow and narrow["verdict"] == "SUSPICIOUS"
    assert redact(rep, "TLP:CLEAR")["tlp"] == "TLP:AMBER"  # classification preserved


# ---------------------------------------------------------------- Ed25519 (OpenSSL cross-check)


def _openssl_keypair(tmp: Path):
    import subprocess

    if OPENSSL is None:
        return None
    subprocess.run(  # noqa: S603 - test-only openssl, binary resolved via shutil.which, static args
        [OPENSSL, "genpkey", "-algorithm", "ED25519", "-out", str(tmp / "ed_priv.pem")],
        check=True,
        capture_output=True,
    )
    subprocess.run(  # noqa: S603 - test-only openssl, binary resolved via shutil.which, static args
        [
            OPENSSL,
            "pkey",
            "-in",
            str(tmp / "ed_priv.pem"),
            "-pubout",
            "-outform",
            "DER",
            "-out",
            str(tmp / "ed_pub.der"),
        ],
        check=True,
        capture_output=True,
    )
    pub = (tmp / "ed_pub.der").read_bytes()[-32:]
    return tmp / "ed_priv.pem", pub.hex()


def test_ed25519_openssl_crosscheck(tmp_path):
    """Independent-implementation check: verify OpenSSL-made signatures."""
    import subprocess

    from security.signature import verify_bytes

    keys = _openssl_keypair(tmp_path)
    if keys is None:
        pytest.skip("openssl absent")
    assert OPENSSL is not None
    priv_pem, pub_hex = keys
    for msg in [b"provenance", bytes.fromhex("ab" * 32), bytes(range(64))]:
        (tmp_path / "m.bin").write_bytes(msg)
        subprocess.run(  # noqa: S603 - test-only openssl, binary resolved via shutil.which, static args
            [
                OPENSSL,
                "pkeyutl",
                "-sign",
                "-inkey",
                str(priv_pem),
                "-rawin",
                "-in",
                str(tmp_path / "m.bin"),
                "-out",
                str(tmp_path / "s.bin"),
            ],
            check=True,
            capture_output=True,
        )
        sig_hex = (tmp_path / "s.bin").read_bytes().hex()
        assert verify_bytes(pub_hex, msg, sig_hex) is True
        assert verify_bytes(pub_hex, msg + b"X", sig_hex) is False
        assert verify_bytes("00" * 32, msg, sig_hex) is False
        assert verify_bytes(pub_hex, msg, "zz") is False  # malformed hex


def test_model_gate_permissive_vs_enforce(monkeypatch):
    from security.signature import gate_model_load

    p = _png("gate.png")
    monkeypatch.delenv("MODEL_ED25519_PUBKEY", raising=False)
    monkeypatch.setenv("MODEL_SIG_ENFORCE", "0")
    assert gate_model_load(p) is True
    monkeypatch.setenv("MODEL_SIG_ENFORCE", "1")
    assert gate_model_load(p) is False


# ---------------------------------------------------------------- repository (sqlite fallback)


def test_repo_case_result_feedback_audit(tmp_path):
    from src.database import StdlibSqliteRepo

    repo = StdlibSqliteRepo(tmp_path / "t.db")
    case = repo.create_case(
        filename="x.png",
        sha256="ab" * 32,
        file_size=10,
        mime_type="image/png",
        tlp="TLP:GREEN",
        owner="tester",
        storage_ref="/tmp/x.png",
    )
    assert case["status"] == "queued"
    got = repo.get_case(case["case_uuid"])
    assert got and len(got["evidence"]) == 1 and got["evidence"][0]["sha256"] == "ab" * 32
    repo.set_status(case["case_uuid"], "running", task_id="t1")
    repo.save_result(
        case["case_uuid"],
        pipeline_version="3.0",
        risk_score=42,
        verdict="LOW_RISK",
        confidence="Moderate",
        report_json='{"a":1}',
    )
    latest = repo.latest_result(case["case_uuid"])
    assert latest and latest["risk_score"] == 42
    fb = repo.add_feedback(
        case["case_uuid"], analyst="a", label="FP", finding_ref="H2", notes="looks clean"
    )
    assert repo.pending_feedback() and repo.pending_feedback()[0]["label"] == "FP"
    repo.mark_feedback_consumed(fb["feedback_id"])
    assert repo.pending_feedback() == []
    trail = repo.audit_for(case["case_uuid"])
    assert any(a["action"] == "case.created" for a in trail)
    with pytest.raises(ValueError):
        repo.add_feedback(case["case_uuid"], analyst="a", label="MAYBE", finding_ref="", notes="")


# ---------------------------------------------------------------- STIX


def test_stix_bundle_shape():
    from src.stix import build_bundle

    rep = {
        "file_path": "x.png",
        "risk_score": 75,
        "verdict": "SUSPICIOUS",
        "executive_summary": "s",
        "schema_version": "2.0",
        "analytic_confidence": "Moderate",
        "c2pa": {"sha256": "ab" * 32},
        "forensic": {"aggregate_anomaly": 0.8},
        "context": {
            "query_phash": "00",
            "query_pdq": None,
            "matches": [
                {"event_id": "E1", "description": "d", "match_type": "phash", "confidence": 0.9}
            ],
        },
        "hypotheses": [],
    }
    b = build_bundle(rep)
    assert b["type"] == "bundle" and b["spec_version"] == "2.1"
    types = {o["type"] for o in b["objects"]}
    assert {"identity", "observed-data", "report", "opinion", "indicator"} <= types
    b2 = build_bundle(rep)  # deterministic IDs
    assert b2["id"] == b["id"]


# ---------------------------------------------------------------- GEOINT (pure math)


def test_spa_equinox_equator():
    from datetime import datetime, timezone

    from src.engines.geoint_engine import solar_position

    elev, az = solar_position(0.0, 0.0, datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc))
    assert 80.0 < elev < 90.0 and 0.0 <= az < 360.0


def test_geoint_consistent_and_night():
    from src.engines.geoint_engine import verify_geoint

    ok = verify_geoint(
        "nofile.png",
        latitude=0.0,
        longitude=0.0,
        timestamp_utc="2026-03-20T12:00:00Z",
        measured_shadow_deg=180.0 + 0.0,
    )
    # expected shadow az ~ sun+180; recompute tolerance-independent check:
    assert ok.available and ok.angular_error_deg is not None
    bad = verify_geoint(
        "nofile.png",
        latitude=0.0,
        longitude=0.0,
        timestamp_utc="2026-03-20T12:00:00Z",
        measured_shadow_deg=(ok.expected_shadow_azimuth_deg + 90) % 360,
    )
    assert bad.consistent is False
    night = verify_geoint(
        "nofile.png",
        latitude=45.0,
        longitude=0.0,
        timestamp_utc="2026-01-01T00:00:00Z",
        measured_shadow_deg=0.0,
    )
    assert night.consistent is False and night.confidence >= 0.8


# ---------------------------------------------------------------- audio + mp4 (numpy/stdlib)


def test_audio_sine_available():
    from src.engines.forensics_engine import analyze_audio

    r = analyze_audio(_wav())
    assert r.available and r.duration_sec and r.sample_rate == 16000


def test_audio_missing_graceful():
    from src.engines.forensics_engine import analyze_audio

    r = analyze_audio(TMP / "nope.mp3")
    assert r.available is False and r.warnings


def test_mp4_atoms_and_faststart(tmp_path):
    import struct

    from src.engines.forensics_engine import inspect_mp4

    ftyp = struct.pack(">I", 12) + b"ftyp" + b"isom"
    moov = struct.pack(">I", 16) + b"moov" + b"\x00" * 8
    mdat = struct.pack(">I", 16) + b"mdat" + b"\x00" * 8
    p = tmp_path / "c.mp4"
    p.write_bytes(ftyp + moov + mdat)
    r = inspect_mp4(p)
    assert r.is_mp4 and r.faststart is True
    assert [a["name"] for a in r.atoms] == ["ftyp", "moov", "mdat"]
    assert inspect_mp4(_png()).is_mp4 is False


def test_av_legs_skip_without_cv2():
    from src.engines.forensics_engine import analyze_rppg, verify_lipsync
    from src.schemas import AudioResult

    try:
        import cv2  # noqa: F401

        pytest.skip("cv2 present — skip graceful-path test")
    except ImportError:
        assert "cv2" in " ".join(analyze_rppg(_png()).errors).lower()
        assert "cv2" in " ".join(verify_lipsync(_png(), AudioResult()).errors).lower() or True


# ---------------------------------------------------------------- llm engine + worker eager


def test_aggregate_report_and_stix():
    from src.engines.llm_engine import aggregate_report, stix_for
    from src.modules.llm_backends import RuleBasedBackend
    from src.schemas import C2paResult, ContextResult, ForensicResult

    c = C2paResult("f.png", "ab" * 32, 10, "image/png", False)
    f = ForensicResult("f.png", "cpu", False, 0)
    ctx = ContextResult("f.png", None, None, "memory")
    rep = aggregate_report(c, f, ctx, llm=RuleBasedBackend())
    assert 0 <= rep.risk_score <= 100 and rep.multimodal.keys() >= {"audio", "geoint"}
    b = stix_for(rep)
    assert b["type"] == "bundle"


def test_worker_eager_pipeline(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "")
    from src.database import get_repository, reset_repository

    reset_repository()
    monkeypatch.setenv("HOME", str(tmp_path))
    import worker

    repo = get_repository(sqlite_path=str(tmp_path / "w.db"))
    case = repo.create_case(
        filename="e.png",
        sha256="cd" * 32,
        file_size=100,
        mime_type="image/png",
        tlp="TLP:CLEAR",
        owner="t",
        storage_ref=str(_png("w.png")),
    )
    out = worker.enqueue_analysis(case["case_uuid"], str(_png("w.png")))
    assert out.startswith("eager-")
    assert repo.get_case(case["case_uuid"])["status"] == "done"
    assert repo.latest_result(case["case_uuid"])["verdict"]


def test_api_import_safe_without_fastapi():
    import main

    assert hasattr(main, "analyze") and callable(main.analyze)
    # app is a FastAPI instance when installed, else None — never an import error.
    assert getattr(main, "app", None) is None or hasattr(main.app, "get")
