"""Asynchronous multimodal pipeline (Celery + Redis, eager fallback).

Tasks:
  `analyze_case`  full pipeline: C2PA -> visual/multimodal forensics ->
                  (optional) GEOINT -> context -> LLM/STIX -> persist.
  `ingest_feedback`  HITL correction -> Active Learning queue (DB + Redis list).

Offline posture: when Celery/Redis are absent the same functions run
synchronously (`run_eager=True`, used by tests and single-node air-gap).
Otherwise tasks retry with exponential backoff (60s, 5 attempts) and never
lose the case: status flips to `failed` with the error persisted.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))  # /app in containers

log = logging.getLogger(__name__)

BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
PIPELINE_VERSION = "3.0"

try:
    from celery import Celery

    _CELERY = True
except ImportError:
    _CELERY = False

if _CELERY:
    app = Celery("provenance", broker=BROKER_URL, backend=BROKER_URL)
    app.conf.update(
        task_acks_late=True,
        worker_prefetch_multiplier=1,
        task_reject_on_worker_lost=True,
        timezone="UTC",
    )
else:
    app = None


def _run_pipeline(
    case_uuid: str, file_path: str, geoint: dict[str, Any] | None = None, actor: str = "worker"
) -> dict[str, Any]:
    from src import audit as audit_jsonl
    from src.config import SETTINGS
    from src.database import get_repository
    from src.engines.c2pa_engine import run_c2pa_engine
    from src.engines.forensics_engine import run_forensics_engines
    from src.engines.geoint_engine import verify_geoint
    from src.engines.llm_engine import aggregate_report, stix_for
    from src.engines.speech_engine import transcribe
    from src.engines.watermark_engine import detect_watermark
    from src.schemas import REPORT_SCHEMA_VERSION

    repo = get_repository()
    repo.set_status(case_uuid, "running")
    c2pa, _ = run_c2pa_engine(file_path)
    multi = run_forensics_engines(file_path)
    multi["watermark"] = detect_watermark(file_path).to_dict()
    # Speech leg: video/audio inputs; images skip cheaply inside transcribe? No —
    # whisper needs audio, so gate on the audio leg to avoid ffmpeg churn.
    if multi.get("audio", {}).get("available"):
        multi["speech"] = transcribe(file_path).to_dict()
    else:
        multi["speech"] = {"available": False, "warnings": ["no audio track — transcription N/A"]}
    # Civilian protection: GREEN/CLEAR releases must not expose identifiable
    # faces/plates. Redacted derivative sealed alongside the report; original
    # stays in case storage for the examiner chain-of-custody.
    from src.security import is_video as _is_video

    try:
        case_tlp = (repo.get_case(case_uuid) or {}).get("tlp", "TLP:AMBER")
    except Exception:
        case_tlp = "TLP:AMBER"
    if case_tlp in ("TLP:CLEAR", "TLP:GREEN") and not _is_video(Path(file_path)):
        from src.pii import redacted_copy

        multi["redaction"] = redacted_copy(file_path).to_dict()
    else:
        multi["redaction"] = {
            "redacted_path": None,
            "notes": ["no release-tier redaction (restricted TLP or video)"],
        }
    # PRNU sensor attribution (enrolled cameras only; abstains otherwise).
    try:
        from pathlib import Path as _P

        import numpy as _np

        from src.engines.prnu_engine import identify as _prnu_id

        prnu_db = _P("data/prnu")
        if any(prnu_db.glob("*.npy")):
            if _is_video(_P(file_path)):
                import cv2 as _cv2

                cap = _cv2.VideoCapture(str(file_path))
                ok, frame = cap.read()
                cap.release()
                qimg = frame if ok else None
            else:
                from PIL import Image as _PIL

                with _PIL.open(file_path) as _im:
                    qimg = _np.asarray(_im.convert("RGB"))
            multi["prnu"] = (
                _prnu_id(qimg).to_dict()
                if qimg is not None
                else {"available": False, "errors": ["no decodable frame for PRNU"]}
            )
        else:
            multi["prnu"] = {
                "available": False,
                "warnings": ["no enrolled cameras — attribution abstains"],
            }
    except Exception as exc:
        multi["prnu"] = {"available": False, "errors": [f"prnu failed: {exc}"]}
    geo = None
    if geoint:
        geo = verify_geoint(
            file_path,
            latitude=float(geoint["latitude"]),
            longitude=float(geoint["longitude"]),
            timestamp_utc=str(geoint["timestamp_utc"]),
            measured_shadow_deg=geoint.get("measured_shadow_deg"),
            object_height_m=geoint.get("object_height_m"),
            shadow_length_m=geoint.get("shadow_length_m"),
        )
    from src.modules.context_check import check_context
    from src.modules.forensic_vision import analyze_forensics

    visual = analyze_forensics(file_path)
    # Reuse visual already computed inside multi? multi has dict form; rebuild
    # typed result cheaply is wasteful — instead fuse from stored dicts below.
    context = check_context(file_path)
    report = aggregate_report(c2pa, visual, context, multi=multi, geoint=geo)
    stix = stix_for(report)
    repo.save_result(
        case_uuid,
        pipeline_version=PIPELINE_VERSION,
        risk_score=report.risk_score,
        verdict=report.verdict.value,
        confidence=report.analytic_confidence,
        report_json=json.dumps(report.to_dict()),
        stix_json=json.dumps(stix),
    )
    repo.audit(
        actor=actor,
        action="analysis.completed",
        case_uuid=case_uuid,
        detail={
            "risk": report.risk_score,
            "verdict": report.verdict.value,
            "schema": REPORT_SCHEMA_VERSION,
        },
    )
    try:
        audit_jsonl.append_audit(
            {"case": case_uuid, "risk": report.risk_score, "verdict": report.verdict.value},
            SETTINGS.audit_log,
        )
    except Exception as exc:
        log.warning("worker: jsonl audit failed (%s)", exc)
    log.info("worker: case %s done risk=%d", case_uuid, report.risk_score)
    return {
        "case_uuid": case_uuid,
        "risk_score": report.risk_score,
        "verdict": report.verdict.value,
    }


if _CELERY:

    @app.task(name="provenance.analyze_case", bind=True, max_retries=5)
    def analyze_case(
        self,
        case_uuid: str,
        file_path: str,
        geoint: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            return _run_pipeline(case_uuid, file_path, geoint)
        except Exception as exc:
            from src.database import get_repository

            try:
                get_repository().set_status(case_uuid, "failed")
                get_repository().audit(
                    actor="worker",
                    action="analysis.failed",
                    case_uuid=case_uuid,
                    detail={"error": str(exc)[:300]},
                )
            except Exception:
                pass
            raise self.retry(exc=exc, countdown=60 * (2**self.request.retries)) from exc

    @app.task(name="provenance.ingest_feedback", bind=True, max_retries=5)
    def ingest_feedback(self, feedback_id: int) -> dict[str, Any]:
        """Active Learning consumer: fan out to Redis AL queue + mark consumed."""
        from src.database import get_repository

        repo = get_repository()
        try:
            import redis

            r = redis.Redis.from_url(BROKER_URL, socket_connect_timeout=2)
            for fb in repo.pending_feedback():
                if fb["id"] == feedback_id or True:
                    r.lpush(
                        "al_queue",
                        json.dumps(
                            {"id": fb["id"], "label": fb["label"], "finding_ref": fb["finding_ref"]}
                        ),
                    )
                    break
        except Exception as exc:
            log.warning("worker: redis AL push failed (%s); DB remains source of truth", exc)
        try:
            repo.mark_feedback_consumed(feedback_id)
        except Exception as exc:
            raise self.retry(exc=exc, countdown=30 * (2**self.request.retries)) from exc
        _maybe_trigger_retraining(repo)
        return {"feedback_id": feedback_id, "queued": True}

    @app.task(name="provenance.osint_poll")
    def osint_poll(owner: str = "osint") -> list[dict[str, Any]]:
        """Beat-scheduled OSINT cycle (feeds/channels -> cases -> analysis)."""
        import asyncio

        from ingestion.pipeline import run_once

        return asyncio.run(run_once(owner))
else:  # eager stand-ins with identical signatures

    def analyze_case(
        case_uuid: str,
        file_path: str,
        geoint: dict[str, Any] | None = None,
        run_eager: bool = True,
    ) -> dict[str, Any]:
        log.info("worker: celery absent — eager synchronous run")
        return _run_pipeline(case_uuid, file_path, geoint)

    def ingest_feedback(feedback_id: int, run_eager: bool = True) -> dict[str, Any]:
        from src.database import get_repository

        get_repository().mark_feedback_consumed(feedback_id)
        return {"feedback_id": feedback_id, "queued": True}

    def osint_poll(owner: str = "osint") -> list[dict[str, Any]]:
        import asyncio

        from ingestion.pipeline import run_once

        return asyncio.run(run_once(owner))


def _maybe_trigger_retraining(repo) -> None:
    """Active Learning -> re-training gate. When unconsumed corrections reach
    `AL_RETRAIN_THRESHOLD`, materialize a versioned JSONL dataset snapshot for
    the (offline, human-approved) fine-tune run and mark the trigger in audit.
    Training itself stays a deliberate human decision — this only stages data.
    """
    import os as _os

    try:
        threshold = int(_os.getenv("AL_RETRAIN_THRESHOLD", "50"))
    except ValueError:
        threshold = 50
    try:
        pending = repo.pending_feedback()
    except Exception:
        return
    if len(pending) < threshold:
        return
    snap_dir = Path("data/al_snapshots")
    snap_dir.mkdir(parents=True, exist_ok=True)
    snap = snap_dir / f"al_{len(pending)}_pending.jsonl"
    try:
        with open(snap, "w", encoding="utf-8") as fh:
            fh.writelines(
                json.dumps(
                    {
                        k: fb.get(k)
                        for k in (
                            "id",
                            "case_uuid",
                            "analyst",
                            "label",
                            "finding_ref",
                            "created_at",
                        )
                    }
                )
                + "\n"
                for fb in pending
            )
        repo.audit(
            actor="worker",
            action="al.retrain_triggered",
            detail={"pending": len(pending), "snapshot": str(snap)},
        )
        log.info("worker: AL retrain dataset staged (%d, %s)", len(pending), snap)
    except Exception as exc:
        log.warning("worker: AL snapshot failed (%s)", exc)


def _broker_reachable(timeout_s: float = 2.0) -> bool:
    """Fast TCP probe so a missing Redis falls back to eager in ~2 s,
    instead of stalling in kombu's reconnect loop (air-gap single-node)."""
    import socket
    from urllib.parse import urlparse

    try:
        parts = urlparse(BROKER_URL)
        host, port = parts.hostname or "localhost", parts.port or 6379
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def enqueue_analysis(case_uuid: str, file_path: str, geoint: dict[str, Any] | None = None) -> str:
    """API entry: Celery task id, or `eager-<uuid>` after synchronous run."""
    if _CELERY and app is not None and _broker_reachable():
        task = analyze_case.delay(case_uuid, file_path, geoint)
        return str(task.id)
    if _CELERY:
        log.warning("worker: broker unreachable — eager synchronous run")
    out = _run_pipeline(case_uuid, file_path, geoint, actor="api-eager")
    return f"eager-{out['case_uuid']}"
