"""Persistence layer: PostgreSQL (SQLAlchemy) with stdlib-sqlite fallback.

`get_repository()` is the only entry point. Connection choice:
  1. `DATABASE_URL` set + SQLAlchemy installed -> PostgreSQL/SQLAlchemy repo
     (encryption at-rest via pgcrypto volume / filesystem encryption is a
     deployment concern; sensitive free-text fields are stored encrypted
     through `src/database.py:seal()` Fernet-when-available, cleartext-safe).
  2. Otherwise -> stdlib `sqlite3` repo with an identical method surface and
     the same table shapes, so the API/worker/tests run anywhere offline.

Both repos are append-only for audit (no update/delete methods exist).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

log = logging.getLogger(__name__)

# How many times an audit append retries when another writer advanced the
# chain first. Contention is rare (one row per case transition) so a handful
# of attempts is plenty; exhaustion raises rather than writing unsealed data.
_AUDIT_WRITE_ATTEMPTS = 8

# Both engines raise their own IntegrityError; the sqlite repo must not miss
# the SQLAlchemy one (and vice versa) or a chain-collision retry turns into an
# unhandled crash in a worker thread.
_INTEGRITY_ERRORS: tuple[type[BaseException], ...] = (sqlite3.IntegrityError,)
try:  # SQLAlchemy is optional (stdlib-sqlite fallback).
    from sqlalchemy.exc import IntegrityError as _SAIntegrityError

    _INTEGRITY_ERRORS = (*_INTEGRITY_ERRORS, _SAIntegrityError)
except ImportError:  # pragma: no cover - exercised on the sqlite-only path
    pass


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_chain_unique(run: Callable[[str], Any]) -> None:
    """Serialize audit writers with a uniqueness rule on `prev_hash`.

    Two writers that read the same chain tip and both append produce two rows
    with the same `prev_hash` — the chain forks and no longer verifies. A
    unique index turns that race into a constraint violation the writer can
    retry, so the database enforces single-writer ordering instead of trusting
    application code to be fast enough.

    The index is PARTIAL (`WHERE id > N`, N = highest id at install time) on
    purpose: a chain that was already forked must keep its broken rows intact
    as forensic evidence. Silently rewriting or deduplicating them would be
    evidence tampering. Rows written from here on are protected; the historic
    break stays visible to `verify_audit_chain` until an operator resolves it
    on the record.
    """
    rows = run("SELECT COALESCE(MAX(id), 0) FROM audit_trail")
    high = int(rows[0][0]) if rows else 0
    try:
        run(
            "CREATE UNIQUE INDEX IF NOT EXISTS ux_audit_prev_hash ON audit_trail(prev_hash) "
            f"WHERE id > {high}"
        )
    except Exception as exc:  # advisory only: audit must keep working
        log.warning("database: could not install audit chain index (%s)", exc)
        return
    log.info("database: audit chain serialized for rows > %d", high)


def _jsonable(obj: Any) -> Any:
    """PostgreSQL returns native datetime/bytes; SQLite returns strings.
    Normalize at the repository boundary so every consumer gets JSON-safe
    structures regardless of backend."""
    from datetime import date

    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, (bytes, memoryview)):
        return bytes(obj).hex()
    return obj


def seal(text: str) -> str:
    """Field-level encryption when `CASES_FERNET_KEY` is provisioned, else passthrough.

    At-rest encryption for sensitive case data without a hard dependency:
    provision a Fernet key via env in classified deployments; air-gapped
    laptops without the key still function (defense in depth, documented).
    """
    key = os.getenv("CASES_FERNET_KEY", "")
    if not key:
        return text
    try:
        from cryptography.fernet import Fernet

        return "fernet:" + Fernet(key.encode()).encrypt(text.encode()).decode()
    except Exception as exc:
        log.warning("database: field encryption unavailable (%s)", exc)
        return text


def unseal(blob: str) -> str:
    if not blob.startswith("fernet:"):
        return blob
    try:
        from cryptography.fernet import Fernet

        return (
            Fernet(os.getenv("CASES_FERNET_KEY", "").encode())
            .decrypt(blob[len("fernet:") :].encode())
            .decode()
        )
    except Exception as exc:
        log.error("database: field decryption failed (%s)", exc)
        return ""


class Repository(Protocol):
    def create_case(
        self,
        *,
        filename: str,
        sha256: str,
        file_size: int,
        mime_type: str,
        tlp: str,
        owner: str,
        storage_ref: str,
    ) -> dict[str, Any]: ...
    def get_case(self, case_uuid: str) -> dict[str, Any] | None: ...
    def list_cases(
        self, limit: int = 50, offset: int = 0, status: str = "", tlp: str = ""
    ) -> list[dict[str, Any]]: ...
    def set_status(self, case_uuid: str, status: str, task_id: str = "") -> None: ...
    def save_result(
        self,
        case_uuid: str,
        *,
        pipeline_version: str,
        risk_score: int,
        verdict: str,
        confidence: str,
        report_json: str,
        stix_json: str = "",
    ) -> None: ...
    def latest_result(self, case_uuid: str) -> dict[str, Any] | None: ...
    def add_feedback(
        self, case_uuid: str, *, analyst: str, label: str, finding_ref: str, notes: str
    ) -> dict[str, Any]: ...
    def pending_feedback(self) -> list[dict[str, Any]]: ...
    def mark_feedback_consumed(self, feedback_id: int) -> None: ...
    def closed_cases(self) -> list[dict[str, Any]]: ...
    def stale_open_cases(self, before_iso: str) -> list[dict[str, Any]]:
        """Open cases not touched since `before_iso` (stale-claim watchdog)."""
        ...

    def all_case_uuids(self) -> set[str]:
        """Every known case UUID (for orphan-file reconciliation)."""
        ...

    def shareable_indicators(self, limit: int = 200) -> list[dict[str, Any]]:
        """Anonymized IoB/IoC rows: finished GREEN/CLEAR cases only, with all
        identifying fields (uuid, filename, owner, notes) stripped at the
        query layer — never selected, not merely redacted."""
        ...

    def delete_case(self, case_uuid: str) -> list[str]:
        """Hard-delete a case and all dependents. Returns storage refs so the
        caller can wipe sealed files. Audit trail rows are RETAINED (custody
        log must survive the evidence it describes)."""
        ...

    def audit(
        self, *, actor: str, action: str, case_uuid: str = "", detail: dict[str, Any] | None = None
    ) -> None: ...
    def audit_for(self, case_uuid: str) -> list[dict[str, Any]]: ...
    def verify_audit_chain(self, since_id: int = 0) -> dict[str, Any]:
        """Recompute the audit hash chain; report the first break, if any."""
        ...


_SCHEMA = """
CREATE TABLE IF NOT EXISTS cases(
  id INTEGER PRIMARY KEY AUTOINCREMENT, case_uuid TEXT UNIQUE, filename TEXT,
  sha256 TEXT, file_size INTEGER, mime_type TEXT, tlp TEXT, owner TEXT,
  status TEXT, task_id TEXT, created_at TEXT, updated_at TEXT);
CREATE TABLE IF NOT EXISTS evidence(
  id INTEGER PRIMARY KEY AUTOINCREMENT, case_id INTEGER, sha256 TEXT,
  acquired_by TEXT, acquired_at TEXT, handler TEXT, transfer_log TEXT, storage_ref TEXT);
CREATE TABLE IF NOT EXISTS analysis_results(
  id INTEGER PRIMARY KEY AUTOINCREMENT, case_id INTEGER, pipeline_version TEXT,
  risk_score INTEGER, verdict TEXT, confidence TEXT, report_json TEXT, stix_json TEXT,
  created_at TEXT);
CREATE TABLE IF NOT EXISTS feedback(
  id INTEGER PRIMARY KEY AUTOINCREMENT, case_id INTEGER, analyst TEXT, label TEXT,
  finding_ref TEXT, notes TEXT, consumed INTEGER DEFAULT 0, created_at TEXT);
CREATE TABLE IF NOT EXISTS audit_trail(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, actor TEXT, action TEXT,
  case_uuid TEXT, detail TEXT, prev_hash TEXT, row_hash TEXT);
"""


class StdlibSqliteRepo:
    """Zero-dependency repo (stdlib sqlite3). Same surface as the SQLAlchemy one."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.executescript(_SCHEMA)
            _ensure_chain_unique(lambda sql: conn.execute(sql).fetchall())

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def create_case(
        self,
        *,
        filename: str,
        sha256: str,
        file_size: int,
        mime_type: str,
        tlp: str,
        owner: str,
        storage_ref: str,
    ) -> dict[str, Any]:
        cuuid, now = str(uuid.uuid4()), utcnow()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO cases(case_uuid,filename,sha256,file_size,mime_type,tlp,"
                "owner,status,task_id,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (cuuid, filename, sha256, file_size, mime_type, tlp, owner, "queued", "", now, now),
            )
            # Portable PK lookup (cursor.lastrowid is not reliable on
            # PostgreSQL without RETURNING; case_uuid is UNIQUE).
            row = conn.execute("SELECT id FROM cases WHERE case_uuid=?", (cuuid,)).fetchone()
            case_id = row["id"] if row else None
            if case_id is None:
                raise RuntimeError("case insert not visible")
            conn.execute(
                "INSERT INTO evidence(case_id,sha256,acquired_by,acquired_at,handler,"
                "transfer_log,storage_ref) VALUES(?,?,?,?,?,?,?)",
                (
                    case_id,
                    sha256,
                    owner,
                    now,
                    owner,
                    json.dumps([{"ts": now, "by": owner, "event": "acquired"}]),
                    storage_ref,
                ),
            )
        self.audit(
            actor=owner,
            action="case.created",
            case_uuid=cuuid,
            detail={"sha256": sha256, "tlp": tlp},
        )
        return {"case_uuid": cuuid, "status": "queued", "sha256": sha256, "tlp": tlp}

    def get_case(self, case_uuid: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM cases WHERE case_uuid=?", (case_uuid,)).fetchone()
            if row is None:
                return None
            d = dict(row)
            ev = conn.execute("SELECT * FROM evidence WHERE case_id=?", (d["id"],)).fetchall()
            d["evidence"] = [dict(r) for r in ev]
            return _jsonable(d)

    def list_cases(
        self, limit: int = 50, offset: int = 0, status: str = "", tlp: str = ""
    ) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 200))
        offset = max(0, offset)
        # Static query shapes only (no interpolated SQL: column names and
        # operators are fixed literals, values always bound parameters).
        base = "SELECT case_uuid, filename, sha256, tlp, owner, status, created_at FROM cases "
        tail = " ORDER BY id DESC LIMIT ? OFFSET ?"
        q: str
        params: list[Any]
        if status and tlp:
            q, params = base + "WHERE status=? AND tlp=?" + tail, [status, tlp]
        elif status:
            q, params = base + "WHERE status=?" + tail, [status]
        elif tlp:
            q, params = base + "WHERE tlp=?" + tail, [tlp]
        else:
            q, params = base + tail.lstrip(), []
        with self._conn() as conn:
            rows = conn.execute(q, (*params, limit, offset)).fetchall()
            return _jsonable([dict(r) for r in rows])

    def set_status(self, case_uuid: str, status: str, task_id: str = "") -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE cases SET status=?, task_id=?, updated_at=? WHERE case_uuid=?",
                (status, task_id, utcnow(), case_uuid),
            )

    def save_result(
        self,
        case_uuid: str,
        *,
        pipeline_version: str,
        risk_score: int,
        verdict: str,
        confidence: str,
        report_json: str,
        stix_json: str = "",
    ) -> None:
        with self._conn() as conn:
            row = conn.execute("SELECT id FROM cases WHERE case_uuid=?", (case_uuid,)).fetchone()
            if row is None:
                raise KeyError(f"unknown case {case_uuid}")
            conn.execute(
                "INSERT INTO analysis_results(case_id,pipeline_version,risk_score,verdict,"
                "confidence,report_json,stix_json,created_at) VALUES(?,?,?,?,?,?,?,?)",
                (
                    row["id"],
                    pipeline_version,
                    risk_score,
                    verdict,
                    confidence,
                    seal(report_json),
                    stix_json,
                    utcnow(),
                ),
            )
            conn.execute(
                "UPDATE cases SET status='done', updated_at=? WHERE case_uuid=?",
                (utcnow(), case_uuid),
            )

    def latest_result(self, case_uuid: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT ar.* FROM analysis_results ar JOIN cases c ON c.id=ar.case_id "
                "WHERE c.case_uuid=? ORDER BY ar.id DESC LIMIT 1",
                (case_uuid,),
            ).fetchone()
            if row is None:
                return None
            d = dict(row)
            d["report_json"] = unseal(d["report_json"])
            return _jsonable(d)

    def add_feedback(
        self, case_uuid: str, *, analyst: str, label: str, finding_ref: str, notes: str
    ) -> dict[str, Any]:
        if label not in ("FP", "FN", "CONFIRM"):
            raise ValueError("label must be FP | FN | CONFIRM")
        with self._conn() as conn:
            row = conn.execute("SELECT id FROM cases WHERE case_uuid=?", (case_uuid,)).fetchone()
            if row is None:
                raise KeyError(f"unknown case {case_uuid}")
            conn.execute(
                "INSERT INTO feedback(case_id,analyst,label,finding_ref,notes,consumed,"
                "created_at) VALUES(?,?,?,?,?,?,?)",
                (row["id"], analyst, label, finding_ref[:256], seal(notes), 0, utcnow()),
            )
            fid_row = conn.execute(
                "SELECT id FROM feedback WHERE case_id=? ORDER BY id DESC LIMIT 1", (row["id"],)
            ).fetchone()
            fid = fid_row["id"] if fid_row else -1
        self.audit(
            actor=analyst,
            action="feedback.submitted",
            case_uuid=case_uuid,
            detail={"label": label, "finding_ref": finding_ref},
        )
        return {"feedback_id": fid, "label": label}

    def pending_feedback(self) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT f.*, c.case_uuid FROM feedback f JOIN cases c ON c.id=f.case_id "
                "WHERE f.consumed=0 ORDER BY f.id"
            ).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                d["notes"] = unseal(d["notes"])
                out.append(_jsonable(d))
            return out

    def mark_feedback_consumed(self, feedback_id: int) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE feedback SET consumed=1 WHERE id=?", (feedback_id,))

    def closed_cases(self) -> list[dict[str, Any]]:
        """Finished cases (done/failed) with raw timestamps for retention sweeps."""
        with self._conn() as conn:
            return _jsonable(
                [
                    dict(r)
                    for r in conn.execute(
                        "SELECT case_uuid, tlp, status, created_at FROM cases "
                        "WHERE status IN ('done','failed')"
                    ).fetchall()
                ]
            )

    def all_case_uuids(self) -> set[str]:
        with self._conn() as conn:
            return {
                str(r["case_uuid"]) for r in conn.execute("SELECT case_uuid FROM cases").fetchall()
            }

    def stale_open_cases(self, before_iso: str) -> list[dict[str, Any]]:
        """Open cases with no progress since `before_iso` (stale-claim watchdog).

        A worker killed mid-task (deploy, OOM, host restart) leaves the case in
        `running` forever: acks_late redelivers, but if the task was already
        acknowledged nothing picks it up again. That is how a queue silently
        loses work, so the watchdog needs to find exactly those rows.
        """
        # Literal status set on purpose: no SQL text is ever built from a
        # parameter, so there is nothing for an injection to reach. Only
        # `before_iso` is bound. Closed states are excluded here rather than
        # filtered in Python, so a settled case can never be reaped.
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT case_uuid, status, task_id, created_at, updated_at FROM cases "
                "WHERE status IN ('queued','running') "
                "AND COALESCE(updated_at, created_at) <= ?",
                (before_iso,),
            ).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                ev = conn.execute(
                    "SELECT storage_ref FROM evidence WHERE case_id="
                    "(SELECT id FROM cases WHERE case_uuid=?)",
                    (d["case_uuid"],),
                ).fetchall()
                d["storage_ref"] = str(ev[0]["storage_ref"]) if ev else ""
                out.append(d)
            return _jsonable(out)

    def shareable_indicators(self, limit: int = 200) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 500))
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT c.sha256, c.tlp, c.created_at, ar.verdict, ar.risk_score, "
                "ar.confidence FROM cases c JOIN analysis_results ar ON ar.case_id = c.id "
                "WHERE c.status='done' AND c.tlp IN ('TLP:CLEAR','TLP:GREEN') "
                "ORDER BY ar.id DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return _jsonable(
                [
                    {
                        "type": "file-sha256",
                        "value": r["sha256"],
                        "verdict": r["verdict"],
                        "risk_score": r["risk_score"],
                        "confidence": r["confidence"],
                        "tlp": r["tlp"],
                        "assessed_at": r["created_at"],
                    }
                    for r in rows
                ]
            )

    def delete_case(self, case_uuid: str) -> list[str]:
        with self._conn() as conn:
            row = conn.execute("SELECT id FROM cases WHERE case_uuid=?", (case_uuid,)).fetchone()
            if row is None:
                return []
            case_id = row["id"]
            refs = [
                r["storage_ref"]
                for r in conn.execute(
                    "SELECT storage_ref FROM evidence WHERE case_id=?", (case_id,)
                ).fetchall()
            ]
            conn.execute("DELETE FROM feedback WHERE case_id=?", (case_id,))
            conn.execute("DELETE FROM analysis_results WHERE case_id=?", (case_id,))
            conn.execute("DELETE FROM evidence WHERE case_id=?", (case_id,))
            conn.execute("DELETE FROM cases WHERE id=?", (case_id,))
            return [str(r) for r in refs if r]

    def audit(
        self, *, actor: str, action: str, case_uuid: str = "", detail: dict[str, Any] | None = None
    ) -> None:
        payload = json.dumps(detail or {}, sort_keys=True, ensure_ascii=False)
        # Read-tip-then-insert is NOT atomic: the SQLAlchemy adapter commits
        # after every statement, so two writers (api + worker + watchdog) can
        # read the same tip and both append — forking the chain. Observed live:
        # rows 83/84 were exact duplicates and `verify_audit_chain` reported
        # the break. The unique index on prev_hash (see _ensure_chain_unique)
        # turns the race into a constraint violation, and the retry below
        # re-links against the winner. The chain is serialized by the database
        # rather than by hope.
        for attempt in range(_AUDIT_WRITE_ATTEMPTS):
            with self._conn() as conn:
                last = conn.execute(
                    "SELECT row_hash FROM audit_trail ORDER BY id DESC LIMIT 1"
                ).fetchone()
                prev = last["row_hash"] if last else "GENESIS"
                row_hash = hashlib.sha256((prev + payload).encode()).hexdigest()
                try:
                    conn.execute(
                        "INSERT INTO audit_trail"
                        "(ts,actor,action,case_uuid,detail,prev_hash,row_hash)"
                        " VALUES(?,?,?,?,?,?,?)",
                        (utcnow(), actor[:128], action[:128], case_uuid, payload, prev, row_hash),
                    )
                except _INTEGRITY_ERRORS:
                    log.warning(
                        "audit: chain advanced concurrently, re-linking (attempt %d)", attempt
                    )
                    time.sleep(0.01 * (attempt + 1))
                    continue
                return
        # Exhausted: fail loudly rather than silently writing an unsealed row.
        log.error(
            "audit: could not append after %d attempts: chain contention", _AUDIT_WRITE_ATTEMPTS
        )
        raise RuntimeError("audit: chain contention, append refused")

    def verify_audit_chain(self, since_id: int = 0) -> dict[str, Any]:
        """Recompute the audit hash chain and report the first break.

        `since_id` starts verification after a given row id (seeding the
        expected `prev` from that row). It exists because the live chain has a
        documented pre-fix fork: operators can verify everything written after
        the constraint was installed without pretending the old break away.

        What this proves: no row's `detail` payload was altered, reordered or
        removed, because each row_hash seals `prev_hash + detail`.

        What it does NOT prove, stated plainly rather than glossed over:
          - `actor`, `action`, `case_uuid` and `ts` are stored but not sealed
            by row_hash, so editing those columns alone is undetectable here.
            The JSONL mirror (`src/audit.py`) hashes the whole event, so it is
            the authoritative custody log; this check is the DB-side backstop.
          - truncating the TAIL is undetectable by construction: a shortened
            chain still verifies. Pair it with off-host backups.
        """
        with self._conn() as conn:
            rows = [
                dict(r)
                for r in conn.execute(
                    "SELECT id, ts, actor, action, case_uuid, detail, prev_hash, row_hash "
                    "FROM audit_trail WHERE id > ? ORDER BY id",
                    (since_id,),
                ).fetchall()
            ]
            prev = "GENESIS"
            if since_id:
                anchor = conn.execute(
                    "SELECT row_hash FROM audit_trail WHERE id = ?", (since_id,)
                ).fetchone()
                prev = anchor["row_hash"] if anchor else "GENESIS"
        for r in rows:
            expected = hashlib.sha256((prev + str(r["detail"])).encode()).hexdigest()
            if str(r["prev_hash"]) != prev:
                return {
                    "ok": False,
                    "rows": len(rows),
                    "broken_at": int(r["id"]),
                    "reason": "prev_hash does not chain to the previous row",
                }
            if str(r["row_hash"]) != expected:
                return {
                    "ok": False,
                    "rows": len(rows),
                    "broken_at": int(r["id"]),
                    "reason": "detail payload no longer matches its row_hash",
                }
            prev = expected
        return {"ok": True, "rows": len(rows), "broken_at": None, "reason": None}

    def audit_for(self, case_uuid: str) -> list[dict[str, Any]]:
        with self._conn() as conn:
            return _jsonable(
                [
                    dict(r)
                    for r in conn.execute(
                        "SELECT * FROM audit_trail WHERE case_uuid=? ORDER BY id", (case_uuid,)
                    )
                ]
            )


class SQLAlchemyRepo(StdlibSqliteRepo):
    """Production repo. Reuses the identical SQL surface through SQLAlchemy's
    engine connection so behavior (incl. hash-chained audit) matches the
    fallback bit-for-bit; ORM classes in `src/models.py` serve migrations,
    admin tooling, and typed queries elsewhere."""

    def __init__(self, database_url: str) -> None:
        try:
            from sqlalchemy import create_engine, text

            import src.models as M
        except ImportError as exc:
            raise RuntimeError("SQLAlchemy not installed") from exc
        self._sa = __import__("sqlalchemy")
        self.engine = create_engine(database_url, pool_pre_ping=True, future=True)
        M.Base.metadata.create_all(self.engine)
        self._text = text
        self._ensure_compat()
        log.info("database: SQLAlchemy repo on %s", database_url.split("@")[-1])

    def _ensure_compat(self) -> None:
        # Serialize audit writers here too (PostgreSQL): same partial-unique
        # trick, installed against the current head.
        def _run(conn, sql: str) -> list[tuple[Any, ...]]:
            res = conn.execute(self._text(sql))
            try:
                return [tuple(r) for r in res.fetchall()]
            except Exception:  # DDL (CREATE INDEX) yields no rows
                return []

        try:
            with self.engine.begin() as conn:
                _ensure_chain_unique(lambda sql: _run(conn, sql))
        except Exception as exc:  # advisory only: audit must keep working
            log.warning("database: audit chain index not installed (%s)", exc)
        """Lightweight migrations for tables created by older model versions."""
        if self.engine.dialect.name != "postgresql":
            return
        try:
            with self.engine.begin() as conn:
                conn.execute(
                    self._text(
                        "ALTER TABLE feedback ALTER COLUMN consumed "
                        "TYPE INTEGER USING (consumed::int)"
                    )
                )
                log.info("database: migrated feedback.consumed to INTEGER")
        except Exception as exc:
            log.debug("database: consumed migration skipped (%s)", exc)

    def _conn(self):
        conn = self.engine.connect()
        # Adapt the DBAPI connection to the sqlite3-like usage in base methods.
        # For PostgreSQL we translate `?` placeholders to named params.
        return _SAConnection(conn, self._text)


class _SAConnection:
    """Minimal adapter: exposes sqlite3-style execute/fetch for our fixed queries."""

    def __init__(self, conn, text):
        self._conn = conn
        self._text = text

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self._conn.close()

    def execute(self, sql: str, params: tuple = ()):
        named: dict[str, Any] = {f"p{i}": v for i, v in enumerate(params)}
        q = sql
        for i in range(len(params)):
            q = q.replace("?", f":p{i}", 1)
        res = self._conn.execute(self._text(q), named)
        self._conn.commit()
        return _SARows(res)


class _SARows:
    def __init__(self, res):
        self._res = res
        try:
            self._rows = [dict(r._mapping) for r in res.fetchall()]
        except Exception:
            self._rows = []
        self.lastrowid = (self._rows[0].get("id") if self._rows else None) or 0

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return self._rows

    def __iter__(self):  # sqlite3 cursors iterate rows; mirror that
        return iter(self._rows)


_REPO: Repository | None = None


def get_repository(database_url: str = "", sqlite_path: str = "data/provenance.db") -> Repository:
    """Factory (cached). Prefers SQLAlchemy/PostgreSQL, else stdlib sqlite."""
    global _REPO
    if _REPO is not None:
        return _REPO
    url = database_url or os.getenv("DATABASE_URL", "")
    if url:
        try:
            _REPO = SQLAlchemyRepo(url)
            return _REPO
        except Exception as exc:
            log.warning("database: SQLAlchemy repo failed (%s); sqlite fallback", exc)
    _REPO = StdlibSqliteRepo(sqlite_path)
    return _REPO


def reset_repository() -> None:
    global _REPO
    _REPO = None
