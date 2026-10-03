"""End-to-end MVP workflow: C2PA -> forensics -> context -> explainable report.

Usage:
    python main.py samples/crisis.jpg
    python main.py video.mp4 --out reports --llm-backend auto --threshold 8
    python main.py --self-test   # offline smoke test with a synthetic image

Exit codes: 0 analysed (any verdict), 2 bad input, 1 unexpected failure.
High-risk verdicts do NOT fail the process — the report IS the product.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.audit import append_audit, verify_chain
from src.config import SETTINGS
from src.device import describe_compute
from src.logging_setup import setup_logging
from src.modules.c2pa_verify import verify_c2pa
from src.modules.context_check import check_context
from src.modules.explain_report import generate_report
from src.modules.forensic_vision import analyze_forensics
from src.security import safe_mkdir

# FastAPI at module scope (NOT inside build_api): with
# `from __future__ import annotations`, function-local imports stay
# ForwardRefs that pydantic cannot resolve -> 500 on every upload endpoint.
try:
    from fastapi import FastAPI, File, Form, Header, HTTPException, Response, UploadFile
    from fastapi.responses import JSONResponse, PlainTextResponse

    _FASTAPI = True
except ImportError:
    _FASTAPI = False

log = logging.getLogger("pipeline")


def analyze(path: str, threshold: int | None = None) -> dict:
    """Run all four modules in order, return the report dict."""
    c2pa = verify_c2pa(path)
    # Forensics always run: for unsigned files they are the primary signal;
    # for signed files they corroborate the cryptographic verdict.
    forensic = analyze_forensics(path)
    context = check_context(path, threshold=threshold)
    report = generate_report(c2pa, forensic, context)
    return report.to_dict()


def _write_outputs(report: dict, markdown: str, out_dir: Path, stem: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jp, mp = out_dir / f"{stem}.report.json", out_dir / f"{stem}.report.md"
    jp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    mp.write_text(markdown, encoding="utf-8")
    return jp, mp


def _self_test() -> int:
    """Offline smoke test: synthetic gradient image through the full pipeline."""
    import numpy as np

    try:
        from PIL import Image
    except ImportError:
        print("self-test needs Pillow (pip install -r requirements.txt)", file=sys.stderr)
        return 1
    tmp = Path("/tmp/opencode/provenance_selftest.png")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    arr = np.tile(np.linspace(0, 255, 256, dtype=np.uint8), (256, 1))
    Image.fromarray(np.dstack([arr, arr[::-1], arr])).save(tmp)
    report = analyze(str(tmp))
    print(json.dumps({k: report[k] for k in ("risk_score", "verdict", "llm_backend")}, indent=2))
    print(f"self-test OK: risk={report['risk_score']} verdict={report['verdict']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Geopolitical Media Provenance & Verification Engine (MVP)"
    )
    parser.add_argument("file", nargs="?", help="image/video to analyse")
    parser.add_argument("--out", default=SETTINGS.report_out_dir, help="report output dir")
    parser.add_argument(
        "--threshold",
        type=int,
        default=SETTINGS.phash_threshold,
        help="pHash Hamming threshold 0..64 (default from env)",
    )
    parser.add_argument("--self-test", action="store_true", help="offline synthetic smoke test")
    parser.add_argument("--log-level", default=SETTINGS.log_level)
    args = parser.parse_args(argv)
    setup_logging(args.log_level)
    log.info("compute: %s", describe_compute())

    if args.self_test:
        return _self_test()
    if not args.file:
        parser.error("FILE is required (or use --self-test)")
    try:
        from src.modules.explain_report import score_to_verdict  # noqa: F401
        from src.schemas import FinalReport, Verdict  # noqa: F401

        c2pa = verify_c2pa(args.file)
        forensic = analyze_forensics(args.file)
        if c2pa.errors and forensic.errors and not c2pa.file_size_bytes:
            print(f"ERROR: invalid input: {c2pa.errors[0]}", file=sys.stderr)
            return 2
        context = check_context(args.file, threshold=args.threshold)
        final = generate_report(c2pa, forensic, context)
        ok, chain_msg = verify_chain(SETTINGS.audit_log)
        if not ok:
            log.error("audit chain broken: %s", chain_msg)
        append_audit(
            {
                "file": final.file_path,
                "sha256": c2pa.sha256,
                "risk": final.risk_score,
                "verdict": final.verdict.value,
                "confidence": final.analytic_confidence,
            },
            SETTINGS.audit_log,
        )
        out_dir = safe_mkdir(args.out)
        stem = Path(final.file_path).stem if final.file_path else "report"
        jp, mp = _write_outputs(final.to_dict(), final.to_markdown(), out_dir, stem)
        print(final.to_markdown())
        print(f"\nSaved: {jp}\nSaved: {mp}", file=sys.stderr)
        return 0
    except Exception as exc:
        log.exception("pipeline failed")
        print(f"ERROR: pipeline failure: {exc}", file=sys.stderr)
        return 1


# ---------------------------------------------------------------- FastAPI (uvicorn main:app)
# Dual-mode: `python main.py file` = CLI triage; `uvicorn main:app` = full
# async API (Celery worker). FastAPI is optional: CLI never needs it.
#
# Layout: small `_register_*` functions (each under the complexity budget),
# shared `_ApiCtx` state, and pure helpers. Route behavior is covered by
# tests/test_multimodal.py::test_api_health_and_similar.

_ALLOWED_SUFFIXES = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".tif",
        ".tiff",
        ".webp",
        ".bmp",
        ".mp4",
        ".mov",
        ".mkv",
        ".avi",
        ".webm",
        ".wav",
    }
)

#: Flush upload stream to disk in chunks of this size (bounds RAM on 500 MB inputs).
_UPLOAD_FLUSH_BYTES = 8 << 20


@dataclass
class _ApiCtx:
    """Shared mutable API state: rate limit + operational counters."""

    rate_limit: int = 30
    metrics: dict[str, int] = field(
        default_factory=lambda: {
            "verify_total": 0,
            "feedback_total": 0,
            "stix_exports": 0,
            "osint_polls": 0,
        }
    )


def _clearance(x_tlp: str | None, param: str = "TLP:CLEAR") -> str:
    from src.tlp import normalize

    return normalize(x_tlp or param)


def _write_chunks(dest: Path, chunks: list[bytes]) -> None:
    """Blocking disk write — always called from a worker thread."""
    with open(dest, "wb") as fh:
        fh.writelines(chunks)


async def _save_upload_async(file: UploadFile, dest: Path, max_bytes: int) -> int:
    """Stream an upload to disk without blocking the event loop.

    Chunks are read with async I/O; the blocking write runs in a thread.
    Raises HTTPException(413) before anything is persisted on oversize.
    """
    import asyncio

    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = await file.read(1 << 20)
        if not chunk:
            break
        size += len(chunk)
        if size > max_bytes:
            raise HTTPException(413, "file exceeds MAX_FILE_MB")
    await asyncio.to_thread(_write_chunks, dest, chunks)
    return size


def _parse_geoint(
    latitude: str | None, longitude: str | None, timestamp_utc: str | None
) -> dict | None:
    """Parse optional GEOINT claim fields; 400 on malformed floats."""
    if not (latitude and longitude and timestamp_utc):
        return None
    try:
        return {
            "latitude": float(latitude),
            "longitude": float(longitude),
            "timestamp_utc": timestamp_utc,
        }
    except ValueError as exc:
        raise HTTPException(400, "bad geoint floats") from exc


def _get_case_or_404(repo, case_uuid: str) -> dict:
    """Fetch a case or raise the canonical 404 (single choke point)."""
    case = repo.get_case(case_uuid)
    if case is None:
        raise HTTPException(404, "unknown case")
    return case


def _require_clearance(clearance: str, record_tlp: str, action: str) -> None:
    from src.tlp import allows

    if not allows(clearance, record_tlp):
        raise HTTPException(403, f"clearance insufficient for this TLP ({action})")


def _register_middleware(api: FastAPI, ctx: _ApiCtx) -> None:
    """Shared-budget rate limiting on ingestion-grade endpoints (DoS shield)."""
    from src.ratelimit import RateLimiter

    limiter = RateLimiter(ctx.rate_limit, os.getenv("CELERY_BROKER_URL", ""))

    @api.middleware("http")
    async def _rate_limit(request, call_next):
        guarded = request.url.path.startswith(
            ("/api/v1/verify", "/api/v1/osint", "/api/v1/feedback")
        )
        if guarded:
            ip = request.client.host if request.client else "?"
            if not limiter.allowed(f"rl:{ip}"):
                return JSONResponse({"detail": "rate limit exceeded"}, status_code=429)
        return await call_next(request)


def _register_ops(api: FastAPI, ctx: _ApiCtx) -> None:
    @api.get("/metrics")
    def metrics() -> dict:
        return {"metrics": dict(ctx.metrics), "tlp": SETTINGS.tlp}

    @api.get("/health")
    def health() -> dict:
        from src.device import describe_compute as _dc

        return {"status": "ok", "pipeline": "3.0", "compute": _dc()}


def _register_verify(api: FastAPI, ctx: _ApiCtx) -> None:
    import shutil
    import uuid as _uuid

    from src.security import guess_mime, sha256_of
    from src.tlp import normalize

    def _suffix_or_400(filename: str) -> str:
        suffix = Path(filename or "upload.bin").suffix.lower()
        if suffix not in _ALLOWED_SUFFIXES:
            raise HTTPException(400, f"unsupported media type {suffix!r}")
        return suffix

    @api.post("/api/v1/verify")
    async def verify(
        file: UploadFile = File(...),
        tlp: str = Form("TLP:CLEAR"),
        owner: str = Form("analyst"),
        latitude: str | None = Form(None),
        longitude: str | None = Form(None),
        timestamp_utc: str | None = Form(None),
        authorization: str | None = Header(None, alias="Authorization"),
    ) -> JSONResponse:
        from src.database import get_repository
        from src.oidc import analyst_id

        try:
            owner = analyst_id(authorization, owner or "analyst")
        except PermissionError as exc:
            raise HTTPException(401, str(exc)) from exc
        tlp_n = normalize(tlp)
        if owner and len(owner) > 128:
            raise HTTPException(400, "owner too long")
        _suffix_or_400(file.filename or "upload.bin")
        case_dir = Path("data/cases") / str(_uuid.uuid4())
        case_dir.mkdir(parents=True, exist_ok=True)
        dest = case_dir / Path(file.filename or "upload.bin").name.replace("/", "_")
        try:
            size = await _save_upload_async(file, dest, SETTINGS.max_file_mb * 1024 * 1024)
        except HTTPException:
            shutil.rmtree(case_dir, ignore_errors=True)
            raise
        sha = sha256_of(dest)
        repo = get_repository()
        case = repo.create_case(
            filename=dest.name,
            sha256=sha,
            file_size=size,
            mime_type=guess_mime(dest),
            tlp=tlp_n,
            owner=owner or "analyst",
            storage_ref=str(dest),
        )
        geo = _parse_geoint(latitude, longitude, timestamp_utc)
        from worker import enqueue_analysis

        task_id = enqueue_analysis(case["case_uuid"], str(dest), geo)
        ctx.metrics["verify_total"] += 1
        if not task_id.startswith("eager-"):
            repo.set_status(case["case_uuid"], "queued", task_id=task_id)
        else:  # eager path already ran to completion; record the reference
            fresh = repo.get_case(case["case_uuid"]) or {}
            repo.set_status(case["case_uuid"], fresh.get("status", "done"), task_id=task_id)
        repo.audit(
            actor=owner or "analyst",
            action="api.verify",
            case_uuid=case["case_uuid"],
            detail={"tlp": tlp_n, "sha256": sha},
        )
        return JSONResponse(
            {
                "case_uuid": case["case_uuid"],
                "sha256": sha,
                "tlp": tlp_n,
                "task_id": task_id,
                "status": "queued",
            }
        )


def _register_cases(api: FastAPI, ctx: _ApiCtx) -> None:
    from src.tlp import allows, redact

    @api.get("/api/v1/cases")
    def list_cases(
        limit: int = 50,
        offset: int = 0,
        status: str = "",
        tlp: str = "",
        clearance: str = "TLP:CLEAR",
        x_tlp: str | None = Header(None, alias="X-TLP"),
    ) -> JSONResponse:
        """Paginated case listing with status/TLP filters.

        TLP rule for collections: records above the requester clearance are
        silently omitted (a 403 would itself leak their existence).
        """
        from src.database import get_repository

        clr = _clearance(x_tlp, clearance)
        repo = get_repository()
        rows = [
            r
            for r in repo.list_cases(limit=limit, offset=offset, status=status, tlp=tlp)
            if allows(clr, r.get("tlp", "TLP:CLEAR"))
        ]
        return JSONResponse({"cases": rows, "returned": len(rows)})

    @api.get("/api/v1/cases/{case_uuid}")
    def get_case(
        case_uuid: str,
        clearance: str = "TLP:CLEAR",
        x_tlp: str | None = Header(None, alias="X-TLP"),
    ) -> JSONResponse:
        from src.database import get_repository

        clr = _clearance(x_tlp, clearance)
        repo = get_repository()
        case = _get_case_or_404(repo, case_uuid)
        _require_clearance(clr, case.get("tlp", "TLP:CLEAR"), "case fetch")
        payload: dict = {
            "case_uuid": case_uuid,
            "status": case.get("status"),
            "tlp": case.get("tlp"),
            "sha256": case.get("sha256"),
            "filename": case.get("filename"),
        }
        res = repo.latest_result(case_uuid)
        if res:
            import json as _json

            report = _json.loads(res["report_json"])
            # TLP-aware detail: higher-TLP records served at requester level.
            level = case.get("tlp", "TLP:CLEAR") if allows("TLP:RED", clr) else clr
            payload["result"] = redact(report, clr if not allows(clr, "TLP:GREEN") else level)
            payload["stix_ready"] = bool(res.get("stix_json"))
        payload["audit"] = repo.audit_for(case_uuid)
        return JSONResponse(payload)


def _fingerprint_for_search(case_uuid: str, phash_hex: str) -> tuple[str, str | None, list | None]:
    """Resolve (phash, pdq, embedding) from a sealed case or raw hash input."""
    from src.database import get_repository

    if not case_uuid:
        return phash_hex, None, None
    repo = get_repository()
    case = _get_case_or_404(repo, case_uuid)
    ev = (case.get("evidence") or [{}])[0]
    try:
        from src.modules.context_check import _hash_media_hybrid

        ph, _, pdq, emb, _ = _hash_media_hybrid(Path(ev.get("storage_ref", "")))
    except Exception as exc:
        raise HTTPException(422, f"media fingerprint failed: {exc}") from exc
    return ph, pdq, emb


def _register_similar(api: FastAPI, ctx: _ApiCtx) -> None:
    from src.config import SETTINGS as _S
    from src.db.store import HybridQuery
    from src.db.store import get_store as _get_store

    @api.get("/api/v1/search/similar")
    def search_similar(
        case_uuid: str = "",
        phash_hex: str = "",
        limit: int = 10,
        clearance: str = "TLP:CLEAR",
        x_tlp: str | None = Header(None, alias="X-TLP"),
    ) -> JSONResponse:
        """Cross-case correlation across the Qdrant vector store.

        Provide a finished `case_uuid` (embedding recomputed from sealed
        storage) or a raw `phash_hex` for hash-tier search. Returns correlated
        case/event IDs with scores. TLP:GREEN clearance required (aggregate
        view), individual records still gate on their own TLP on fetch.
        """
        import os as _os

        clr = _clearance(x_tlp, clearance)
        _require_clearance(clr, "TLP:GREEN", "similarity search")
        limit = max(1, min(limit, 50))
        store = _get_store(
            vector_backend=_S.vector_backend,
            dsn=_S.postgres_dsn,
            milvus_uri=_S.milvus_uri,
            weaviate_url=_S.weaviate_url,
            seed_path=_S.reference_db_path,
            qdrant_url=_os.getenv("QDRANT_URL", "http://localhost:6333"),
        )
        phash_hex_q, pdq_hex_q, embedding = _fingerprint_for_search(case_uuid, phash_hex)
        if hasattr(store, "search_correlated") and embedding:
            return JSONResponse(
                {
                    "query": case_uuid or "embedding",
                    "correlated": store.search_correlated(embedding, limit=limit)[:limit],
                }
            )
        # Hash-tier fusion (phash + PDQ) when no CLIP vector is available.
        if hasattr(store, "search_hybrid") and phash_hex_q:
            hits = store.search_hybrid(
                HybridQuery(
                    phash_hex=phash_hex_q,
                    phash_threshold=_S.phash_threshold,
                    pdq_hex=pdq_hex_q,
                    pdq_threshold=_S.pdq_threshold,
                    limit=limit,
                )
            )
            return JSONResponse(
                {
                    "query": case_uuid or phash_hex_q[:16],
                    "tier": "hash",
                    "correlated": [
                        {
                            "event_id": h.event.event_id,
                            "description": h.event.description,
                            "score": h.confidence,
                            "tier": h.match_type,
                        }
                        for h in hits
                    ],
                }
            )
        return JSONResponse(
            {
                "query": phash_hex_q[:16] if phash_hex_q else case_uuid,
                "correlated": [],
                "fallback": "no-index",
            }
        )


def _register_osint(api: FastAPI, ctx: _ApiCtx) -> None:
    @api.post("/api/v1/osint/poll")
    async def osint_poll(owner: str = "osint") -> JSONResponse:
        """Trigger one OSINT ingestion cycle (RSS/Telegram -> cases)."""
        from ingestion.pipeline import run_once
        from src.database import get_repository

        created = await run_once(owner)
        get_repository().audit(actor=owner, action="osint.poll", detail={"created": len(created)})
        ctx.metrics["osint_polls"] += 1
        return JSONResponse({"created": len(created), "cases": created})


def _render_siem_report(report_json: str, format: str) -> tuple[str, str]:
    """Parse + render a stored report to a SOC format (400 on bad format)."""
    import json as _json

    from src.siem import render as _render

    try:
        return _render(_json.loads(report_json), format)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


def _register_exports(api: FastAPI, ctx: _ApiCtx) -> None:
    def _gated_result(case_uuid: str, clr: str) -> tuple[dict, dict]:
        """Fetch case + latest result with TLP gate and readiness check."""
        from src.database import get_repository

        repo = get_repository()
        case = _get_case_or_404(repo, case_uuid)
        _require_clearance(clr, case.get("tlp", "TLP:CLEAR"), "export")
        res = repo.latest_result(case_uuid)
        if not res:
            raise HTTPException(409, "analysis not complete yet")
        return case, res

    @api.get("/api/v1/export/siem/{case_uuid}")
    def export_siem(
        case_uuid: str,
        format: str = "cef",
        clearance: str = "TLP:CLEAR",
        x_tlp: str | None = Header(None, alias="X-TLP"),
    ) -> Response:
        """SOC formats: ?format=cef (ArcSight) | leef (QRadar) | ecs (Elastic).
        Fixed schema only — no LLM prose crosses into SIEM (injection-safe)."""
        import json as _json

        from src.database import get_repository

        clr = _clearance(x_tlp, clearance)
        repo = get_repository()
        _, res = _gated_result(case_uuid, clr)
        body, mime = _render_siem_report(res["report_json"], format)
        repo.audit(
            actor="api",
            action="siem.exported",
            case_uuid=case_uuid,
            detail={"format": format, "clearance": clr},
        )
        ctx.metrics["stix_exports"] += 1
        if mime == "application/json":
            return JSONResponse(_json.loads(body))
        return PlainTextResponse(body, media_type=mime)

    @api.get("/api/v1/export/stix/{case_uuid}")
    def export_stix(
        case_uuid: str,
        clearance: str = "TLP:CLEAR",
        x_tlp: str | None = Header(None, alias="X-TLP"),
    ) -> JSONResponse:
        import json as _json

        from src.database import get_repository

        clr = _clearance(x_tlp, clearance)
        repo = get_repository()
        _, res = _gated_result(case_uuid, clr)
        if not res.get("stix_json"):
            raise HTTPException(409, "analysis not complete yet")
        repo.audit(
            actor="api", action="stix.exported", case_uuid=case_uuid, detail={"clearance": clr}
        )
        ctx.metrics["stix_exports"] += 1
        return JSONResponse(_json.loads(res["stix_json"]))


def _register_feedback(api: FastAPI, ctx: _ApiCtx) -> None:

    @api.post("/api/v1/feedback")
    async def feedback(
        payload: dict, authorization: str | None = Header(None, alias="Authorization")
    ) -> JSONResponse:
        from src.database import get_repository
        from src.oidc import analyst_id

        try:
            analyst = analyst_id(authorization, str(payload.get("analyst", "analyst")))
        except PermissionError as exc:
            raise HTTPException(401, str(exc)) from exc
        try:
            case_uuid = str(payload["case_uuid"])
            label = str(payload["label"]).upper()
            finding_ref = str(payload.get("finding_ref", ""))[:256]
            notes = str(payload.get("notes", ""))
        except (KeyError, TypeError) as exc:
            raise HTTPException(400, "need {case_uuid, label}") from exc
        if label not in ("FP", "FN", "CONFIRM"):
            raise HTTPException(400, "label must be FP | FN | CONFIRM")
        repo = get_repository()
        try:
            fb = repo.add_feedback(
                case_uuid, analyst=analyst, label=label, finding_ref=finding_ref, notes=notes
            )
        except KeyError as exc:
            raise HTTPException(404, "unknown case") from exc
        try:  # fan out to the Active Learning worker; DB stays source of truth
            from worker import ingest_feedback as _ingest

            if hasattr(_ingest, "delay"):
                _ingest.delay(fb["feedback_id"])
            else:
                _ingest(fb["feedback_id"])
        except Exception as exc:
            log.warning("api: AL fan-out failed (%s)", exc)
        ctx.metrics["feedback_total"] += 1
        return JSONResponse({"feedback_id": fb["feedback_id"], "status": "queued-for-learning"})


def _register_federated(api: FastAPI, ctx: _ApiCtx) -> None:
    @api.get("/api/v1/federated/indicators")
    def federated(
        clearance: str = "TLP:CLEAR", x_tlp: str | None = Header(None, alias="X-TLP")
    ) -> JSONResponse:
        """Anonymized IoB/IoC sharing: only TLP:CLEAR/GREEN records, PII stripped."""
        clr = _clearance(x_tlp, clearance)
        _require_clearance(clr, "TLP:GREEN", "federated sharing")
        return JSONResponse(
            {
                "indicators": [],
                "note": "wire live DB scan in deployment; demo returns empty set",
                "clearance": clr,
            }
        )


def build_api():
    """Construct the v1 API. Raises RuntimeError when FastAPI is absent."""
    if not _FASTAPI:
        raise RuntimeError("fastapi not installed (pip install fastapi uvicorn)")

    ctx = _ApiCtx(rate_limit=int(os.getenv("API_RATE_LIMIT_PER_MIN", "30")))
    api = FastAPI(title="Media Provenance & Verification Engine", version="3.0")
    _register_middleware(api, ctx)
    _register_ops(api, ctx)
    _register_verify(api, ctx)
    _register_cases(api, ctx)
    _register_similar(api, ctx)
    _register_osint(api, ctx)
    _register_exports(api, ctx)
    _register_feedback(api, ctx)
    _register_federated(api, ctx)
    return api


try:
    app = build_api()
except Exception as _e:  # FastAPI optional; CLI unaffected
    log.debug("api unavailable: %s", _e)
    app = None


if __name__ == "__main__":
    raise SystemExit(main())
