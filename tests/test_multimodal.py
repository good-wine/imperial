"""Delta regression: watermark, PII, speech, Qdrant, OSINT, similar-search.

Offline-safe: Whisper asserts SKIPPED without provisioned weights; Qdrant
runs in `:memory:` (no server); RSS parses synthetic XML; Telegram asserts
idle without token. API tests use FastAPI TestClient (no network).
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

TMP = Path("/tmp/opencode/prov_multi")
TMP.mkdir(parents=True, exist_ok=True)


def _comfy_png() -> Path:
    p = TMP / "comfy.png"
    arr = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (64, 1))
    img = Image.fromarray(np.dstack([arr, arr, arr]))
    meta = PngInfo()
    meta.add_text("prompt", '{"3": {"class_type": "KSampler", "inputs": {}}}')
    meta.add_text("workflow", '{"nodes": []}')
    img.save(p, pnginfo=meta)
    return p


def _a1111_jpg() -> Path:
    p = TMP / "sdxl.jpg"
    arr = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (64, 1))
    Image.fromarray(np.dstack([arr, arr, arr])).save(p, quality=90)
    raw = bytearray(p.read_bytes())
    com = b"parameters: Steps: 30, Sampler: DPM++, CFG scale: 7, Model: stable diffusion XL"
    seg = struct.pack(">H", len(com) + 2) + com
    raw[2:2] = b"\xff\xfe" + seg  # inject COM right after SOI
    p.write_bytes(bytes(raw))
    return p


def test_watermark_comfy_and_a1111():
    from src.engines.watermark_engine import detect_watermark

    r = detect_watermark(_comfy_png())
    assert r.ai_generated is True and r.generator == "comfyui" and r.metadata_hits
    r2 = detect_watermark(_a1111_jpg())
    assert r2.ai_generated is True and "stable diffusion" in (r2.generator or "")
    assert r.synthid in ("NEEDS_MODEL", "NOT_CHECKED", "NOT_DETECTED", "DETECTED")


def test_watermark_clean_image():
    from src.engines.watermark_engine import detect_watermark

    p = TMP / "clean.png"
    Image.fromarray(np.zeros((32, 32, 3), np.uint8)).save(p)
    r = detect_watermark(p)
    assert r.ai_generated is False and not r.metadata_hits


def test_pii_redaction_runs():
    from src.pii import redacted_copy

    rng = np.random.default_rng(3)
    img = rng.integers(0, 255, (160, 160, 3), dtype=np.uint8)
    p = TMP / "crowd.png"
    Image.fromarray(img).save(p)
    log = redacted_copy(p, out_dir=TMP)
    assert log.redacted_path and Path(log.redacted_path).is_file()
    assert "examiner review" in " ".join(log.notes)


def test_speech_skipped_without_weights(monkeypatch):
    from src.engines.speech_engine import transcribe

    monkeypatch.setenv("WHISPER_MODEL", "definitely-not-a-model-xyz")
    import wave

    wav = TMP / "s.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000)
    r = transcribe(wav)
    assert r.available is False and r.errors


def test_qdrant_memory_hybrid_and_correlated():
    from src.db.store import HybridQuery, ProvenanceEvent
    from src.vector_store import QdrantEventStore

    vec = [1.0] + [0.0] * 511
    vec = (np.array(vec) / np.linalg.norm(vec)).tolist()
    store = QdrantEventStore(
        ":memory:",
        seed=[
            ProvenanceEvent(
                "Q1", "memory event", "2024-01-01", "t", "ff00ff00ff00ff00", clip_embedding=vec
            )
        ],
    )
    hits = store.search_hybrid(
        HybridQuery(
            phash_hex="ff00ff00ff00ff00", phash_threshold=8, embedding=vec, clip_threshold=0.5
        )
    )
    assert hits and hits[0].event.event_id == "Q1"
    corr = store.search_correlated(vec, score_threshold=0.5)
    assert corr and corr[0]["event_id"] == "Q1"
    assert store.search("ff00ff00ff00ff00", 8)[0][0].event_id == "Q1"


def test_qdrant_factory_wiring():
    from src.db.store import get_store

    s = get_store("qdrant", seed_path="data/reference_events.json", qdrant_url=":memory:")
    assert getattr(s, "backend_name", "") == "qdrant" and s.count() >= 5
    bad = get_store("qdrant", qdrant_url="http://127.0.0.1:9")
    assert getattr(bad, "backend_name", "") == "memory"  # fail-open


def test_rss_parse_and_poll(monkeypatch, tmp_path):
    import asyncio

    from ingestion import rss

    xml = b"""<rss version="2.0"><channel><title>T</title>
      <item><title>Blast footage</title><link>http://x/1</link>
      <guid>g1</guid><pubDate>Mon, 01 Jan 2024 00:00:00 GMT</pubDate>
      <enclosure url="http://x/a.jpg" type="image/jpeg"/>
      <description><![CDATA[<img src="http://x/b.png"/> text]]></description></item>
      <item><title>No media here</title><link>http://x/2</link></item>
      </channel></rss>"""
    items = rss.parse_feed(xml, source="test")
    assert len(items) == 2 and len(items[0].media_urls) == 2  # parser is complete…
    assert rss.parse_feed(b"<broken", source="t") == []

    async def fake_fetch(url, timeout_s=20):
        return xml

    monkeypatch.setattr(rss, "_fetch_async", fake_fetch)
    fresh = asyncio.run(rss.poll_feeds(["http://feed"], seen_path=str(tmp_path / "s.json")))
    assert len(fresh) == 1
    again = asyncio.run(rss.poll_feeds(["http://feed"], seen_path=str(tmp_path / "s.json")))
    assert again == []  # deduped


def test_telegram_idle_without_token():
    import asyncio

    from ingestion.telegram import poll_bot_api

    assert asyncio.run(poll_bot_api("", ["1"])) == []


def test_device_providers_order():
    from src.device import get_onnx_providers

    provs = get_onnx_providers()
    assert provs[-1] == "CPUExecutionProvider" and len(provs) >= 1


def test_api_health_and_similar():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import main

    assert main.app is not None
    with TestClient(main.app) as client:
        assert client.get("/health").json()["status"] == "ok"
        r = client.get(
            "/api/v1/search/similar",
            params={"phash_hex": "ff00ff00ff00ff00", "clearance": "TLP:GREEN"},
        )
        assert r.status_code == 200 and "correlated" in r.json()
        denied = client.get(
            "/api/v1/search/similar", params={"phash_hex": "ff00", "clearance": "TLP:CLEAR"}
        )
        assert denied.status_code == 403


def test_stix_carries_watermark_bump():
    from src.engines.forensics_engine import multimodal_risk_bump

    bump, notes = multimodal_risk_bump(
        {"watermark": {"ai_generated": True, "generator": "comfyui"}}
    )
    assert bump >= 15 and notes
