"""Audit hash-chain verification: what tampering the DB chain actually catches.

The DB chain seals `prev_hash + detail` per row. These tests pin down both
halves of that statement — the tampering it catches, and the tampering it
does not — so nobody later mistakes it for a full-record seal.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


def _repo(tmp: Path):
    from src.database import StdlibSqliteRepo

    repo = StdlibSqliteRepo(tmp / "audit.db")
    for i in range(5):
        repo.audit(actor="api", action=f"step.{i}", case_uuid="c1", detail={"i": i})
    return repo


def _rewrite(repo, sql: str, params: tuple = ()) -> None:
    with sqlite3.connect(repo.path) as conn:
        conn.execute(sql, params)
        conn.commit()


def test_intact_chain_verifies(tmp_path):
    repo = _repo(tmp_path)
    out = repo.verify_audit_chain()
    assert out["ok"] is True
    assert out["rows"] == 5
    assert out["broken_at"] is None


def test_concurrent_writers_do_not_fork_the_chain(tmp_path):
    """The failure this layer exists to prevent.

    Two writers read the same tip and both append: the second insert violates
    the unique index on prev_hash, so the writer re-links against the winner
    instead of forking the chain. Without the constraint the chain is
    unverifiable — which is exactly what the live database turned out to be.
    """
    import threading

    from src.database import StdlibSqliteRepo

    repo = StdlibSqliteRepo(tmp_path / "race.db")
    barrier = threading.Barrier(6)

    def _writer(i: int) -> None:
        barrier.wait()
        repo.audit(actor=f"w{i}", action="concurrent", detail={"i": i})

    threads = [threading.Thread(target=_writer, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    out = repo.verify_audit_chain()
    assert out["ok"] is True, out
    assert out["rows"] == 6


def test_verify_can_start_after_a_documented_break(tmp_path):
    """Operators must be able to verify the post-fix chain without the historic
    fork being silently repaired (that would be evidence tampering)."""
    repo = _repo(tmp_path)
    _rewrite(repo, "UPDATE audit_trail SET detail=? WHERE id=3", ('{"i": 99}',))
    assert repo.verify_audit_chain()["ok"] is False

    repo.audit(actor="api", action="after.fix", detail={"i": 5})
    assert repo.verify_audit_chain(since_id=5)["ok"] is True
    assert repo.verify_audit_chain()["ok"] is False  # history still shows the break


def test_since_id_seeds_prev_from_the_anchor_row(tmp_path):
    repo = _repo(tmp_path)
    repo.audit(actor="api", action="extra", detail={"i": 5})
    out = repo.verify_audit_chain(since_id=5)
    assert out["ok"] is True
    assert out["rows"] == 1


def test_edited_detail_payload_is_detected(tmp_path):
    """The substantive part of an audit row — its payload — is sealed."""
    repo = _repo(tmp_path)
    _rewrite(repo, "UPDATE audit_trail SET detail=? WHERE id=3", ('{"i": 99}',))
    out = repo.verify_audit_chain()
    assert out["ok"] is False
    assert out["broken_at"] == 3
    assert "row_hash" in out["reason"]


def test_deleted_middle_row_is_detected(tmp_path):
    """A hole in the chain: the next row's prev_hash can no longer match."""
    repo = _repo(tmp_path)
    _rewrite(repo, "DELETE FROM audit_trail WHERE id=3")
    out = repo.verify_audit_chain()
    assert out["ok"] is False
    assert out["broken_at"] == 4
    assert "chain" in out["reason"]


def test_reordered_rows_are_detected(tmp_path):
    repo = _repo(tmp_path)
    _rewrite(repo, "UPDATE audit_trail SET detail=? WHERE id=2", ('{"i": 3}',))
    out = repo.verify_audit_chain()
    assert out["ok"] is False


def test_unsealed_columns_are_documented_as_a_limit(tmp_path):
    """Editing `action` alone does NOT break the DB chain — by design of the
    current formula, and the reason the JSONL mirror exists. Asserted so the
    limitation cannot be quietly forgotten."""
    repo = _repo(tmp_path)
    _rewrite(repo, "UPDATE audit_trail SET action=? WHERE id=2", ("forged.action",))
    assert repo.verify_audit_chain()["ok"] is True

    from src import audit as audit_jsonl

    assert "prev_hash" in Path(audit_jsonl.__file__).read_text(encoding="utf-8")


def test_jsonl_mirror_seals_the_whole_event(tmp_path):
    """The compensating control: every field, not just detail."""
    from src import audit as audit_jsonl

    log = tmp_path / "audit.jsonl"
    for i in range(3):
        audit_jsonl.append_audit({"actor": "api", "action": f"a{i}", "i": i}, log)
    ok, _ = audit_jsonl.verify_chain(log)
    assert ok is True

    lines = log.read_text(encoding="utf-8").splitlines()
    entry = json.loads(lines[1])
    entry["event"]["actor"] = "attacker"
    lines[1] = json.dumps(entry, ensure_ascii=False)
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ok, reason = audit_jsonl.verify_chain(log)
    assert ok is False
    assert "tamper" in reason or "break" in reason
