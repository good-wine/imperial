"""SOTA-closing regression: PRNU, SIEM, OIDC, calibration gate, cert pinning."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

TMP = Path("/tmp/opencode/prov_sota2")
TMP.mkdir(parents=True, exist_ok=True)


def _cam_images(seed: int, n: int = 6, shape=(96, 96)) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, shape)  # simulated fixed sensor pattern
    return [
        (128 + base * 6 + rng.normal(0, 3, shape)).clip(0, 255).astype(np.uint8) for _ in range(n)
    ]


def test_prnu_self_match_and_exclusion():
    from src.engines.prnu_engine import fingerprint, identify, pce_score, residual

    refs = _cam_images(7)
    F, q = fingerprint(refs)
    assert -1.0 <= q <= 1.0
    same = _cam_images(7, n=1)[0]
    other = _cam_images(99, n=1)[0]
    assert pce_score(F, residual(same.astype(np.float32))) > pce_score(
        F, residual(other.astype(np.float32))
    )
    db = TMP / "prnu"
    db.mkdir(exist_ok=True)
    np.save(db / "CAM-A.npy", F)
    hit = identify(same, db)
    assert (
        hit.available and hit.best_camera == "CAM-A" and hit.verdict in ("MATCH", "INCONCLUSIVE")
    )  # screening thresholds stay conservative
    miss = identify((np.zeros((96, 96), np.uint8)), TMP / "empty_db")
    assert miss.available is False


def test_siem_formats_shape():
    from src.siem import render

    rep = {
        "file_path": "x.png",
        "risk_score": 75,
        "verdict": "SUSPICIOUS",
        "analytic_confidence": "Moderate",
        "tlp": "TLP:AMBER",
        "c2pa": {"sha256": "ab" * 32},
        "context": {"query_phash": "00", "matches": [{"event_id": "E1"}]},
    }
    cef, mime = render(rep, "cef")
    assert cef.startswith("CEF:0|") and mime == "text/plain" and "risk=" in cef
    leef, _ = render(rep, "leef")
    assert leef.startswith("LEEF:2.0|") and "sev=" in leef
    ecs_body, mime2 = render(rep, "ecs")
    assert json.loads(ecs_body)["event"]["kind"] == "alert" and mime2 == "application/json"
    with pytest.raises(ValueError):
        render(rep, "syslog-ng")


def test_oidc_disabled_passthrough_and_reject():
    import os

    from src.oidc import analyst_id, verify_bearer

    os.environ.pop("OIDC_ISSUER", None)
    assert analyst_id(None, "field-op") == "field-op"
    with pytest.raises(ValueError):
        verify_bearer("x", "")  # disabled
    with pytest.raises(ValueError):
        verify_bearer("not.a.jwt", "https://idp/x")  # malformed, fail-closed


def test_calibrate_refuses_anecdotes(tmp_path):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "calibrate_ensemble", "scripts/calibrate_ensemble.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.argv = [
        "calibrate",
        "--db",
        str(tmp_path / "empty.db"),
        "--out",
        str(tmp_path / "w.json"),
        "--min-samples",
        "20",
    ]
    assert spec.loader.exec_module(mod) is None
    assert mod.main() == 2


def test_cert_pinning_paths():
    from src.modules.c2pa_verify import _inspect_cert_chain, _load_trust_anchors

    # No anchors, no certs -> SKIPPED (not FAIL): absence is not evidence.
    vd = _inspect_cert_chain(None, b"\xff\xd8" + b"\x00" * 100)
    assert vd["status"] == "SKIPPED"
    assert isinstance(_load_trust_anchors(), list)


def test_rate_limit_and_metrics():
    pytest.importorskip("fastapi")
    import os

    from fastapi.testclient import TestClient

    import main

    os.environ["API_RATE_LIMIT_PER_MIN"] = "2"
    try:
        api2 = main.build_api()  # rebuilt: picks up the test limit
        with TestClient(api2, raise_server_exceptions=False) as c:
            codes = [
                c.post("/api/v1/feedback", json={"case_uuid": "nope", "label": "XX"}).status_code
                for _ in range(4)
            ]
            assert 429 in codes  # limiter engaged (400s then 429s)
            assert c.get("/metrics").status_code == 200
    finally:
        os.environ.pop("API_RATE_LIMIT_PER_MIN", None)
