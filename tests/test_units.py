"""Offline unit tests for pure logic: SIEM, OIDC, RSS/Telegram parsing,
LLM backend selection, Qdrant :memory: retrieval, fetch choke point,
ORM models, RAG retrieval, engine wrappers. No network, no servers,
no model downloads — all dependencies vendored or generated in-test.
"""

from __future__ import annotations

import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

TMP = Path("/tmp/opencode/prov_units")
TMP.mkdir(parents=True, exist_ok=True)


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


# ---------------------------------------------------------------- SIEM


def _report() -> dict:
    return {
        "file_path": "/cases/x.png",
        "risk_score": 75,
        "verdict": "SUSPICIOUS",
        "analytic_confidence": "Moderate",
        "tlp": "TLP:AMBER",
        "c2pa": {"sha256": "ab" * 32},
        "context": {"query_phash": "00" * 8, "matches": [{"event_id": "E1"}]},
    }


def test_siem_cef_shape():
    from src.siem import render

    body, mime = render(_report(), "cef")
    assert body.startswith("CEF:0|ProvenanceEngine|MediaProvenance|3.0|SUSPICIOUS|")
    assert "risk=75" in body and mime == "text/plain"


def test_siem_leef_escapes_tabs():
    from src.siem import to_leef

    rep = _report()
    line = to_leef(rep)
    assert line.startswith("LEEF:2.0|") and "\t" in line
    assert "\tsev=7\t" in line or "sev=7" in line


def test_siem_ecs_and_bad_format():
    from src.siem import render

    body, mime = render(_report(), "ecs")
    doc = json.loads(body)
    assert doc["event"]["kind"] == "alert" and doc["event"]["risk_score"] == 75
    assert doc["file"]["hash"]["sha256"] == "ab" * 32
    assert mime == "application/json"
    with pytest.raises(ValueError):
        render(_report(), "syslog")


# ---------------------------------------------------------------- OIDC


def _rsa_keypair():
    from cryptography.hazmat.primitives.asymmetric import rsa

    priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pub = priv.public_key().public_numbers()
    n = _b64url(pub.n.to_bytes(256, "big"))
    e = _b64url(pub.e.to_bytes(3, "big"))
    return priv, {"keys": [{"kty": "RSA", "kid": "t1", "n": n, "e": e}]}


def _jwt(priv, kid="t1", alg="RS256", payload=None):
    import time

    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding

    body = {"iss": "https://idp/x", "aud": "api", "sub": "analyst-7", "exp": int(time.time()) + 300}
    body.update(payload or {})
    head = _b64url(json.dumps({"alg": alg, "kid": kid}).encode())
    pay = _b64url(json.dumps(body).encode())
    signing = f"{head}.{pay}".encode()
    if alg == "RS256":
        sig = priv.sign(signing, padding.PKCS1v15(), hashes.SHA256())
    else:  # HS256 stand-in: wrong algorithm must be rejected before verify
        import hmac

        sig = hmac.new(b"secret", signing, "sha256").digest()
    return f"{head}.{pay}.{_b64url(sig)}"


def test_oidc_disabled_passthrough(monkeypatch):
    from src import oidc

    monkeypatch.delenv("OIDC_ISSUER", raising=False)
    assert oidc.analyst_id(None, "field-op") == "field-op"


def test_oidc_roundtrip_and_rejections(monkeypatch):
    from src import oidc

    priv, jwks = _rsa_keypair()
    monkeypatch.setattr(oidc, "_jwks", lambda issuer: jwks)
    monkeypatch.setenv("OIDC_ISSUER", "https://idp/x")
    monkeypatch.setenv("OIDC_AUDIENCE", "api")
    good = _jwt(priv)
    assert oidc.verify_bearer(good, "https://idp/x", "api")["sub"] == "analyst-7"
    assert oidc.analyst_id(f"Bearer {good}", "fallback") == "analyst-7"
    # Tampered payload
    head, _pay, sig = good.split(".")
    bad = f"{head}.{_b64url(b'{}')}.{sig}"
    with pytest.raises(ValueError):
        oidc.verify_bearer(bad, "https://idp/x", "api")
    # Wrong alg / expired / wrong issuer / garbage
    with pytest.raises(ValueError):
        oidc.verify_bearer(_jwt(priv, alg="HS256"), "https://idp/x", "api")
    import time

    with pytest.raises(ValueError):
        oidc.verify_bearer(
            _jwt(priv, payload={"exp": int(time.time()) - 10}), "https://idp/x", "api"
        )
    with pytest.raises(ValueError):
        oidc.verify_bearer(_jwt(priv), "https://evil/y", "api")
    with pytest.raises(ValueError):
        oidc.verify_bearer("not.a.jwt", "https://idp/x", "api")
    with pytest.raises(PermissionError):
        oidc.analyst_id("Bearer garbage", "fallback")


# ---------------------------------------------------------------- RSS / Telegram parsing


_RSS = """<?xml version="1.0"?>
<rss version="2.0" xmlns:media="http://search.yahoo.com/mrss/">
<channel><title>Crisis</title>
<item><title>Blast footage</title><link>http://x/1</link><guid>g1</guid>
<pubDate>Mon, 01 Jan 2024 00:00:00 GMT</pubDate>
<enclosure url="http://x/a.jpg" type="image/jpeg"/>
<media:content url="http://x/b.mp4" type="video/mp4"/>
<description><![CDATA[<img src="http://x/c.png"/> text]]></description></item>
<item><title>No media</title><link>http://x/2</link><guid>g2</guid></item>
</channel></rss>"""

_ATOM = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom"><title>A</title>
<entry><title>V</title><link href="http://x/3"/><id>h3</id>
<updated>2024-01-01T00:00:00Z</updated>
<content type="html">&lt;img src="http://x/d.jpg"/&gt;</content></entry>
</feed>"""


def test_rss_parse_variants():
    from ingestion.rss import parse_feed

    items = parse_feed(_RSS.encode(), source="t")
    assert len(items) == 2
    assert items[0].external_id == "g1" and len(items[0].media_urls) == 3
    assert items[1].media_urls == []
    atom = parse_feed(_ATOM.encode(), source="t")
    assert len(atom) == 1 and atom[0].external_id == "h3"
    assert parse_feed(b"<broken", source="t") == []
    assert (
        parse_feed(b'<?xml version="1.0"?><!DOCTYPE x [<!ENTITY xxe "pwn">]><rss/>', source="t")
        == []
    )  # entity attack neutralized


def test_telegram_poll_gating(monkeypatch):
    import asyncio

    import ingestion.telegram as tg

    def fake_api(token, method, params=None, timeout_s=25):
        return [
            {
                "message": {
                    "message_id": 1,
                    "chat": {"id": 7, "title": "Front"},
                    "photo": [{"file_id": "small"}, {"file_id": "big"}],
                    "caption": "boom",
                }
            },
            {
                "message": {
                    "message_id": 2,
                    "chat": {"id": 7, "title": "Front"},
                    "document": {"file_id": "d1", "mime_type": "video/mp4"},
                }
            },
            {
                "message": {
                    "message_id": 3,
                    "chat": {"id": 9, "title": "Other"},
                    "photo": [{"file_id": "x"}],
                }
            },
            {
                "message": {
                    "message_id": 4,
                    "chat": {"id": 7, "title": "Front"},
                    "document": {"file_id": "d2", "mime_type": "application/pdf"},
                }
            },
        ]

    monkeypatch.setattr(tg, "_api", fake_api)
    out = asyncio.run(tg.poll_bot_api("tok", ["7"]))
    assert [(m.external_id, m.file_id) for m in out] == [
        ("msg-1", "big"),
        ("msg-2", "d1"),
    ]  # largest photo, video doc; chat-gated, pdf dropped
    assert asyncio.run(tg.poll_bot_api("", ["7"])) == []


# ---------------------------------------------------------------- LLM backends


def test_rule_based_and_jury():
    from src.modules.llm_backends import BaseLLMBackend, JuryBackend, RuleBasedBackend

    rb = RuleBasedBackend()
    prompt = "risk 10\nverdict: low\nc2pa ok, anomaly low"
    assert "risk 10" in rb.generate(prompt)
    assert rb.health() is True

    class _Stub(BaseLLMBackend):
        name = "stub"

        def __init__(self, text):
            self._t = text

        def generate(self, prompt, system=""):
            return self._t

    draft = "verdict: LOW c2pa ok anomaly low"
    # Verifier confirms SUPPORTED -> draft passes through untouched.
    assert JuryBackend(_Stub(draft), _Stub("SUPPORTED")).generate("e") == draft
    # Single-model jury without SUPPORTED verdict -> dissent appended (by design).
    flagged = JuryBackend(_Stub(draft)).generate("evidence c2pa ok")
    assert "dissent" in flagged.lower()

    class _Boom(BaseLLMBackend):
        name = "boom"

        def generate(self, prompt, system=""):
            raise RuntimeError("down")

    # Dead primary propagates (caller falls back to rule-based); health is False.
    with pytest.raises(RuntimeError):
        JuryBackend(_Boom(), _Stub("SUPPORTED")).generate("e")
    assert _Boom().health() is False


def test_ollama_unreachable_and_selection():
    from src.modules.llm_backends import OllamaBackend, get_backend

    bad = OllamaBackend("http://127.0.0.1:9", "llama3.1:8b", timeout_s=2)
    with pytest.raises(RuntimeError):
        bad.generate("hi")
    assert bad.health() is False
    assert get_backend("rule-based").name == "rule-based"
    assert get_backend("definitely-unknown").name == "rule-based"


# ---------------------------------------------------------------- Qdrant :memory:


def test_qdrant_memory_hybrid_and_correlated():
    from src.db.store import HybridQuery, ProvenanceEvent
    from src.vector_store import QdrantEventStore

    vec = (np.arange(512, dtype=float) / 512).tolist()
    store = QdrantEventStore(
        ":memory:",
        seed=[
            ProvenanceEvent(
                "Q1", "memory event", "2024-01-01", "t", "ff00ff00ff00ff00", clip_embedding=vec
            )
        ],
    )
    assert store.count() == 1
    # Non-matching phash forces the semantic tier (hash precedence on ties).
    hits = store.search_hybrid(
        HybridQuery(phash_hex="00" * 8, phash_threshold=0, embedding=vec, clip_threshold=0.5)
    )
    assert hits and hits[0].event.event_id == "Q1" and hits[0].match_type == "semantic"
    # Exact phash keeps precedence over an equally-confident embedding.
    tied = store.search_hybrid(
        HybridQuery(
            phash_hex="ff00ff00ff00ff00", phash_threshold=8, embedding=vec, clip_threshold=0.5
        )
    )
    assert tied and tied[0].match_type == "phash"
    corr = store.search_correlated(vec, score_threshold=0.5)
    assert corr and corr[0]["event_id"] == "Q1"
    assert store.search("ff00ff00ff00ff00", 8)[0][0].event_id == "Q1"
    assert store.search_correlated([0.0] * 10) == []  # wrong dim: honest empty


# ---------------------------------------------------------------- fetch choke point


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = b"x" * 100
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def test_fetch_url_gate_and_happy_path():
    from src.security import UnsafeInputError, download_to_file, fetch_url

    with pytest.raises(UnsafeInputError):
        fetch_url("file:///etc/passwd")
    with pytest.raises(UnsafeInputError):
        fetch_url("gopher://x/1")
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{srv.server_port}"
        assert fetch_url(base) == b"x" * 100
        with pytest.raises(UnsafeInputError):
            fetch_url(base, max_bytes=10)
        dest = TMP / "dl.bin"
        assert download_to_file(base, dest) == 100 and dest.stat().st_size == 100
    finally:
        srv.shutdown()


# ---------------------------------------------------------------- ORM models


def test_orm_crud_sqlite_memory():
    import datetime

    import src.models as M

    assert M.HAS_SQLALCHEMY
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    eng = create_engine("sqlite:///:memory:")
    M.Base.metadata.create_all(eng)
    now = datetime.datetime.now(datetime.timezone.utc)
    with Session(eng) as s:
        case = M.Case(
            case_uuid="u1",
            filename="a.png",
            sha256="00" * 32,
            file_size=1,
            mime_type="image/png",
            tlp="TLP:CLEAR",
            owner="t",
            status="queued",
            task_id="",
            created_at=now,
            updated_at=now,
        )
        s.add(case)
        s.flush()
        s.add(
            M.Evidence(
                case_id=case.id,
                sha256="00" * 32,
                acquired_by="t",
                acquired_at=now,
                handler="t",
                transfer_log="[]",
                storage_ref="/tmp/a.png",
            )
        )
        s.add(
            M.Feedback(
                case_id=case.id,
                analyst="a",
                label="FP",
                finding_ref="H2",
                notes="n",
                consumed=0,
                created_at=now,
            )
        )
        s.add(
            M.AuditEntry(
                ts=now,
                actor="t",
                action="a",
                case_uuid="u1",
                detail="{}",
                prev_hash="GENESIS",
                row_hash="x",
            )
        )
        s.add(
            M.AnalysisResult(
                case_id=case.id,
                pipeline_version="3.0",
                risk_score=10,
                verdict="LOW_RISK",
                confidence="High",
                report_json="{}",
                stix_json="",
                created_at=now,
            )
        )
        s.commit()
        assert s.query(M.Case).count() == 1
        assert s.query(M.Feedback).filter_by(consumed=0).count() == 1


# ---------------------------------------------------------------- RAG + engine wrappers


def test_rag_retrieval_tmpdb(tmp_path):
    from src.rag import retrieve_snippets

    db = tmp_path / "ref.json"
    db.write_text(
        json.dumps(
            [
                {
                    "event_id": "EVT-1",
                    "event_date": "2020-08-04",
                    "description": "Beirut port explosion recaptioned as strike",
                },
                {
                    "event_id": "EVT-2",
                    "event_date": "2019-04-15",
                    "description": "Notre-Dame fire reused for later blazes",
                },
            ]
        ),
        encoding="utf-8",
    )
    hits = retrieve_snippets(
        "beirut explosion recaptioned strike",
        top_k=1,
        reference_db=str(db),
        reports_dir=str(tmp_path / "none"),
    )
    assert hits and hits[0]["source"] == "reference:EVT-1"
    assert retrieve_snippets("", top_k=3, reference_db=str(db), reports_dir=str(tmp_path)) == []


def _gradient_png(name: str) -> Path:
    arr = np.tile(np.linspace(20, 235, 128, dtype=np.uint8), (128, 1))
    p = TMP / name
    Image.fromarray(np.dstack([arr, arr, arr])).save(p)
    return p


def test_rate_limiter_memory_and_redis_stub():
    from src.ratelimit import RateLimiter

    assert RateLimiter(0).allowed("k") is True  # 0 disables
    m = RateLimiter(3)
    assert [m.allowed("k") for _ in range(3)] == [True, True, True]
    assert m.allowed("k") is False
    assert m.allowed("other") is True  # per-key budgets

    class _Pipe:
        def __init__(self, parent):
            self.p = parent

        def zremrangebyscore(self, *a):
            return self

        def zadd(self, *a):
            return self

        def zcard(self, k):
            self.p.calls.append(k)
            return self

        def expire(self, *a):
            return self

        def execute(self):
            self.p.n += 1
            return [None, None, self.p.n, None]

    class _Stub:
        def __init__(self):
            self.n = 0
            self.calls = []

        def pipeline(self, transaction=True):
            return _Pipe(self)

    stub = _Stub()
    a, b = RateLimiter(2), RateLimiter(2)
    a._client, b._client = stub, stub  # shared budget across replicas
    assert a.allowed("k") and b.allowed("k") and not a.allowed("k")
    assert stub.calls  # went through Redis, not memory

    class _Down:
        def pipeline(self, *a, **k):
            raise ConnectionError("no broker")

    d = RateLimiter(1)
    d._client = _Down()
    assert d.allowed("k") is True and d.allowed("k") is False  # fail-open to memory


def test_engine_wrappers_unsigned():
    from src.db.store import InMemoryEventStore
    from src.engines.c2pa_engine import run_c2pa_engine
    from src.engines.context_engine import run_context_engine

    p = _gradient_png("wrap.png")
    _res, summary = run_c2pa_engine(p)
    assert summary["manifest_validity"] == "ABSENT"
    assert summary["signer_identity"] == "unknown"
    _ctx, summ = run_context_engine(p, store=InMemoryEventStore())
    assert summ["decontextualization"] == "NOT_OBSERVED"
    assert summ["semantic_tier"] in ("active", "offline-hash-only")


def test_cases_listing_tlp_omission_and_filters(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import main
    from src.database import StdlibSqliteRepo

    repo = StdlibSqliteRepo(tmp_path / "list.db")
    uuids = {}
    for tlp in ("TLP:GREEN", "TLP:RED", "TLP:CLEAR"):
        c = repo.create_case(
            filename="x.png",
            sha256="ab" * 32,
            file_size=1,
            mime_type="image/png",
            tlp=tlp,
            owner="t",
            storage_ref="/tmp/x.png",
        )
        uuids[tlp] = c["case_uuid"]
    repo.set_status(uuids["TLP:GREEN"], "done")
    # Endpoints resolve get_repository lazily per request: patch the source.
    import src.database

    monkeypatch.setattr(src.database, "get_repository", lambda *a, **k: repo)
    with TestClient(main.app, raise_server_exceptions=False) as client:
        clear = client.get("/api/v1/cases", params={"clearance": "TLP:CLEAR"}).json()
        assert {c["tlp"] for c in clear["cases"]} == {"TLP:CLEAR"}
        red = client.get("/api/v1/cases", params={"clearance": "TLP:RED"}).json()
        assert len(red["cases"]) == 3
        done = client.get("/api/v1/cases", params={"clearance": "TLP:RED", "status": "done"}).json()
        assert [c["case_uuid"] for c in done["cases"]] == [uuids["TLP:GREEN"]]
        page = client.get(
            "/api/v1/cases", params={"clearance": "TLP:RED", "limit": 1, "offset": 1}
        ).json()
        assert page["returned"] == 1
