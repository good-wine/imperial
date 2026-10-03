"""Retention policy tests: per-TLP windows, queued immunity, RED immunity,
file wipe, and audit-trail survival (custody log outlives evidence)."""

from __future__ import annotations

import datetime
import sqlite3
from pathlib import Path


def _mkcase(repo, tmp: Path, name: str, tlp: str, status: str, age_days: int) -> str:
    blob = tmp / f"{name}.bin"
    blob.write_bytes(b"evidence-bytes")
    case = repo.create_case(
        filename=blob.name,
        sha256="ab" * 32,
        file_size=14,
        mime_type="application/octet-stream",
        tlp=tlp,
        owner="t",
        storage_ref=str(blob),
    )
    repo.set_status(case["case_uuid"], status)
    old = (
        datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=age_days)
    ).isoformat()
    with sqlite3.connect(repo.path) as conn:
        conn.execute(
            "UPDATE cases SET created_at=?, updated_at=? WHERE case_uuid=?",
            (old, old, case["case_uuid"]),
        )
    return case["case_uuid"]


def test_retention_purge_matrix(tmp_path):
    from src.database import StdlibSqliteRepo

    repo = StdlibSqliteRepo(tmp_path / "r.db")
    old_green = _mkcase(repo, tmp_path, "g", "TLP:GREEN", "done", 100)  # window 90 -> purge
    old_clear = _mkcase(repo, tmp_path, "c", "TLP:CLEAR", "done", 40)  # window 30 -> purge
    old_amber = _mkcase(repo, tmp_path, "a", "TLP:AMBER", "done", 400)  # window 365 -> purge
    red = _mkcase(repo, tmp_path, "r", "TLP:RED", "done", 1000)  # never auto-purge
    young = _mkcase(repo, tmp_path, "y", "TLP:GREEN", "done", 5)  # fresh -> keep
    queued = _mkcase(repo, tmp_path, "q", "TLP:GREEN", "queued", 200)  # open -> keep

    import worker

    out = worker.purge_expired_cases(repo)
    assert set(out["purged"]) == {old_green, old_clear, old_amber}
    assert out["errors"] == []
    for gone in (old_green, old_clear, old_amber):
        assert repo.get_case(gone) is None
    for kept_uuid in (red, young, queued):
        assert repo.get_case(kept_uuid) is not None
    # Evidence files wiped from case storage...
    assert not (tmp_path / "g.bin").exists()
    assert (tmp_path / "y.bin").exists()
    # ...while the custody trail survives with purge records.
    trail = repo.audit_for(old_green)
    assert any(a["action"] == "case.purged" for a in trail)
    assert any(a["action"] == "case.created" for a in trail)


def test_prune_orphans_dry_run_and_guards(tmp_path):
    from scripts.prune_orphans import find_orphans

    d = tmp_path / "cases"
    d.mkdir()
    (d / "11111111-1111-4111-8111-111111111111").mkdir()
    (d / "22222222-2222-4222-8222-222222222222").mkdir()
    (d / "not-a-uuid").mkdir()
    (d / "stray.txt").write_text("x")
    orphans, skipped = find_orphans(d, {"11111111-1111-4111-8111-111111111111"})
    assert [o.name for o in orphans] == ["22222222-2222-4222-8222-222222222222"]
    assert skipped == ["not-a-uuid"]  # non-uuid names are never touched
    assert (d / "stray.txt").is_file()  # files at top level are never touched
