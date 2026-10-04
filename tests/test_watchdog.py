"""Stale-claim watchdog: a worker that dies mid-task must not strand a case.

Scenario that produced this code, reproduced live against the running stack:
a case was `running`, the worker was SIGKILLed, and after the restart the case
sat in `running` forever. Celery's `acks_late` did NOT redeliver it (the broker
entry was gone), so the queue silently lost the work with nothing to show for
it. Two lessons are encoded here:

  1. the watchdog must re-dispatch onto the broker — flipping the DB row to
     `queued` is cosmetic, since Celery takes work from Redis, not from the
     cases table;
  2. if dispatch fails, the case must stay visibly stuck rather than be
     labelled `queued` with nobody working on it.
"""

from __future__ import annotations

import datetime
import sqlite3
from pathlib import Path


def _mkcase(repo, tmp: Path, name: str, status: str, *, age_s: int, keep_blob: bool = True) -> str:
    blob = tmp / f"{name}.bin"
    blob.write_bytes(b"evidence-bytes")
    case = repo.create_case(
        filename=blob.name,
        sha256="cd" * 32,
        file_size=14,
        mime_type="application/octet-stream",
        tlp="TLP:GREEN",
        owner="t",
        storage_ref=str(blob) if keep_blob else str(tmp / "gone.bin"),
    )
    repo.set_status(case["case_uuid"], status)
    old = (
        datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=age_s)
    ).isoformat()
    with sqlite3.connect(repo.path) as conn:
        conn.execute(
            "UPDATE cases SET created_at=?, updated_at=? WHERE case_uuid=?",
            (old, old, case["case_uuid"]),
        )
        conn.commit()
    return case["case_uuid"]


def _repo(tmp: Path):
    from src.database import StdlibSqliteRepo

    return StdlibSqliteRepo(tmp / "watch.db")


def _recorder(seen: list[tuple[str, str]], task_id: str = "task-test"):
    def _send(case_uuid: str, file_path: str) -> str:
        seen.append((case_uuid, file_path))
        return task_id

    return _send


def test_stale_running_case_is_requeued_and_redispatched(tmp_path):
    import worker

    repo = _repo(tmp_path)
    stuck = _mkcase(repo, tmp_path, "stuck", "running", age_s=3600)
    sent: list[tuple[str, str]] = []
    out = worker.reap_stale_cases(repo, stale_after_s=900, dispatch=_recorder(sent))
    assert stuck in out["requeued"]
    assert stuck not in out["failed"]
    assert stuck in out["redispatched"]
    # The broker is the source of work: nothing is recovered without dispatch.
    assert sent == [(stuck, str(tmp_path / "stuck.bin"))]
    row = repo.get_case(stuck)
    assert row["status"] == "queued"
    assert row["task_id"] == "task-test"
    assert any(a["action"] == "case.requeued" for a in repo.audit_for(stuck))


def test_requeue_without_dispatch_would_be_a_lie(tmp_path):
    """A `queued` row with no broker entry is indistinguishable from work in
    progress, which is exactly how the outage hid in the first place."""
    import worker

    repo = _repo(tmp_path)
    stuck = _mkcase(repo, tmp_path, "stuck", "running", age_s=3600)

    def _refuse(case_uuid: str, file_path: str) -> str:
        raise RuntimeError("broker unreachable")

    out = worker.reap_stale_cases(repo, stale_after_s=900, dispatch=_refuse)
    assert out["requeued"] == []
    assert out["errors"] == [stuck]
    row = repo.get_case(stuck)
    assert row["status"] == "running"  # untouched, still visibly stuck


def test_stale_case_without_evidence_is_abandoned_not_retried_forever(tmp_path):
    import worker

    repo = _repo(tmp_path)
    lost = _mkcase(repo, tmp_path, "lost", "running", age_s=3600, keep_blob=False)
    sent: list[tuple[str, str]] = []
    out = worker.reap_stale_cases(repo, stale_after_s=900, dispatch=_recorder(sent))
    assert lost in out["failed"]
    assert sent == []  # nothing to analyse, so nothing is dispatched
    assert repo.get_case(lost)["status"] == "failed"
    trail = repo.audit_for(lost)
    assert any(a["action"] == "case.abandoned" for a in trail)


def test_fresh_cases_are_left_alone(tmp_path):
    import worker

    repo = _repo(tmp_path)
    fresh = _mkcase(repo, tmp_path, "fresh", "running", age_s=5)
    sent: list[tuple[str, str]] = []
    out = worker.reap_stale_cases(repo, stale_after_s=900, dispatch=_recorder(sent))
    assert out["requeued"] == [] and out["failed"] == []
    assert sent == []
    assert repo.get_case(fresh)["status"] == "running"


def test_closed_cases_are_never_reaped(tmp_path):
    import worker

    repo = _repo(tmp_path)
    done = _mkcase(repo, tmp_path, "done", "done", age_s=99999)
    sent: list[tuple[str, str]] = []
    out = worker.reap_stale_cases(repo, stale_after_s=900, dispatch=_recorder(sent))
    assert done not in out["requeued"] and done not in out["failed"]
    assert sent == []
    assert repo.get_case(done)["status"] == "done"


def test_stale_window_is_configurable(tmp_path):
    import worker

    repo = _repo(tmp_path)
    five_min = _mkcase(repo, tmp_path, "five", "queued", age_s=400)
    worker.reap_stale_cases(repo, stale_after_s=300, dispatch=_recorder([]))
    assert repo.get_case(five_min)["status"] == "queued"  # inside the window
    worker.reap_stale_cases(repo, stale_after_s=60, dispatch=_recorder([]))
    assert repo.get_case(five_min)["status"] == "queued"  # requeued (was queued)


def test_reaper_is_idempotent(tmp_path):
    """A second sweep must not re-dispatch: reaping refreshes the stale clock."""
    import worker

    repo = _repo(tmp_path)
    stuck = _mkcase(repo, tmp_path, "stuck", "running", age_s=3600)
    worker.reap_stale_cases(repo, stale_after_s=900, dispatch=_recorder([]))
    sent: list[tuple[str, str]] = []
    again = worker.reap_stale_cases(repo, stale_after_s=900, dispatch=_recorder(sent))
    assert again["requeued"] == [] and again["failed"] == []
    assert sent == []
    actions = [a["action"] for a in repo.audit_for(stuck)]
    assert actions.count("case.requeued") == 1
    assert repo.get_case(stuck)["status"] == "queued"


def test_beat_schedule_registers_the_watchdog():
    import worker

    if not worker._CELERY:
        return  # celery absent: eager stand-in path
    assert "reap-stale-cases" in worker.app.conf.beat_schedule
    assert worker.app.conf.beat_schedule["reap-stale-cases"]["task"] == "provenance.reap_stale"
    assert worker.app.conf.beat_schedule["reap-stale-cases"]["schedule"] == 300.0
