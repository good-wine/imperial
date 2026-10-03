"""SQLAlchemy ORM tables — Cases, Evidence, Results, Feedback, Audit.

ISO/IEC 27037 chain-of-custody: every Evidence row carries SHA-256, acquisition
timestamps, handler identity, and transfer history; the Audit table is
append-only (no UPDATE/DELETE in repository code) and each row chains the
previous row hash (tamper-evident, mirroring `src/audit.py` JSONL mode).

Import-safe: when SQLAlchemy is absent (minimal air-gapped bootstrap), light
placeholder classes with identical attribute names are defined so module
imports never break; `src/database.py` then serves the stdlib-sqlite repo.
"""

from __future__ import annotations

try:
    from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
    from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

    _SA = True
except ImportError:  # pragma: no cover - offline bootstrap path
    _SA = False

HAS_SQLALCHEMY = _SA

if _SA:

    class Base(DeclarativeBase):
        pass

    class Case(Base):
        """One verification case = one ingested file + workflow state."""

        __tablename__ = "cases"
        id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
        case_uuid: Mapped[str] = mapped_column(String(36), unique=True, index=True)
        filename: Mapped[str] = mapped_column(String(512))
        sha256: Mapped[str] = mapped_column(String(64), index=True)
        file_size: Mapped[int] = mapped_column(Integer, default=0)
        mime_type: Mapped[str] = mapped_column(String(128), default="")
        tlp: Mapped[str] = mapped_column(String(16), default="TLP:CLEAR", index=True)
        owner: Mapped[str] = mapped_column(String(128), default="analyst")
        status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
        task_id: Mapped[str] = mapped_column(String(128), default="")
        created_at: Mapped[str] = mapped_column(DateTime, nullable=True)
        updated_at: Mapped[str] = mapped_column(DateTime, nullable=True)
        evidence: Mapped[list[Evidence]] = relationship(
            back_populates="case", cascade="all, delete-orphan"
        )

    class Evidence(Base):
        """Chain-of-custody record per ISO/IEC 27037 § acquisition/handling."""

        __tablename__ = "evidence"
        id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
        case_id: Mapped[int] = mapped_column(ForeignKey("cases.id"), index=True)
        sha256: Mapped[str] = mapped_column(String(64), index=True)
        acquired_by: Mapped[str] = mapped_column(String(128), default="ingest")
        acquired_at: Mapped[str] = mapped_column(DateTime, nullable=True)
        handler: Mapped[str] = mapped_column(String(128), default="")
        transfer_log: Mapped[str] = mapped_column(Text, default="[]")  # JSON list
        storage_ref: Mapped[str] = mapped_column(String(512), default="")
        case: Mapped[Case] = relationship(back_populates="evidence")

    class AnalysisResult(Base):
        """One row per pipeline run (re-runs versioned, never overwritten)."""

        __tablename__ = "analysis_results"
        id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
        case_id: Mapped[int] = mapped_column(ForeignKey("cases.id"), index=True)
        pipeline_version: Mapped[str] = mapped_column(String(32), default="2.0")
        risk_score: Mapped[int] = mapped_column(Integer, default=0)
        verdict: Mapped[str] = mapped_column(String(16), default="UNVERIFIABLE")
        confidence: Mapped[str] = mapped_column(String(16), default="Low")
        report_json: Mapped[str] = mapped_column(Text, default="{}")
        stix_json: Mapped[str] = mapped_column(Text, default="")
        created_at: Mapped[str] = mapped_column(DateTime, nullable=True)

    class Feedback(Base):
        """HITL analyst corrections feeding the Active Learning queue."""

        __tablename__ = "feedback"
        id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
        case_id: Mapped[int] = mapped_column(ForeignKey("cases.id"), index=True)
        analyst: Mapped[str] = mapped_column(String(128), default="")
        label: Mapped[str] = mapped_column(String(16), default="")  # FP | FN | CONFIRM
        finding_ref: Mapped[str] = mapped_column(String(256), default="")
        notes: Mapped[str] = mapped_column(Text, default="")
        # Integer (not Boolean): the shared raw-SQL surface binds 0/1 on both
        # SQLite and PostgreSQL without datatype mismatches.
        consumed: Mapped[int] = mapped_column(Integer, default=0)
        created_at: Mapped[str] = mapped_column(DateTime, nullable=True)

    class AuditEntry(Base):
        """Immutable append-only audit trail. Repository exposes insert+select only."""

        __tablename__ = "audit_trail"
        id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
        ts: Mapped[str] = mapped_column(DateTime, nullable=True)
        actor: Mapped[str] = mapped_column(String(128), default="system")
        action: Mapped[str] = mapped_column(String(128), default="")
        case_uuid: Mapped[str] = mapped_column(String(36), default="", index=True)
        detail: Mapped[str] = mapped_column(Text, default="{}")
        prev_hash: Mapped[str] = mapped_column(String(64), default="GENESIS")
        row_hash: Mapped[str] = mapped_column(String(64), default="")

    TABLES = {
        "cases": Case,
        "evidence": Evidence,
        "analysis_results": AnalysisResult,
        "feedback": Feedback,
        "audit_trail": AuditEntry,
    }
else:  # placeholder mirrors (attribute-compatible, no persistence)
    from typing import Any

    Base: Any = type("Base", (), {"metadata": None})  # type: ignore[no-redef]  # optional-SQLAlchemy fallback branch

    class _Placeholder:
        __tablename__ = ""

        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

    Case: Any = type("Case", (_Placeholder,), {"__tablename__": "cases"})  # type: ignore[no-redef]  # optional-SQLAlchemy fallback branch
    Evidence: Any = type("Evidence", (_Placeholder,), {"__tablename__": "evidence"})  # type: ignore[no-redef]  # optional-SQLAlchemy fallback branch
    AnalysisResult: Any = type(  # type: ignore[no-redef]  # optional-SQLAlchemy fallback branch
        "AnalysisResult", (_Placeholder,), {"__tablename__": "analysis_results"}
    )
    Feedback: Any = type("Feedback", (_Placeholder,), {"__tablename__": "feedback"})  # type: ignore[no-redef]  # optional-SQLAlchemy fallback branch
    AuditEntry: Any = type("AuditEntry", (_Placeholder,), {"__tablename__": "audit_trail"})  # type: ignore[no-redef]  # optional-SQLAlchemy fallback branch

    TABLES: dict = {}  # type: ignore[no-redef]  # optional-SQLAlchemy fallback branch
