"""Initial schema: mirrors src/models.py (cases/evidence/results/feedback/audit).

Revision ID: 0001_initial
Revises: None
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "cases",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("case_uuid", sa.String(length=36), nullable=False),
        sa.Column("filename", sa.String(length=512), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("file_size", sa.Integer(), nullable=False),
        sa.Column("mime_type", sa.String(length=128), nullable=False),
        sa.Column("tlp", sa.String(length=16), nullable=False),
        sa.Column("owner", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("task_id", sa.String(length=128), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_cases_case_uuid", "cases", ["case_uuid"], unique=True)
    op.create_index("ix_cases_sha256", "cases", ["sha256"], unique=False)
    op.create_index("ix_cases_tlp", "cases", ["tlp"], unique=False)
    op.create_index("ix_cases_status", "cases", ["status"], unique=False)

    op.create_table(
        "evidence",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("case_id", sa.Integer(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("acquired_by", sa.String(length=128), nullable=False),
        sa.Column("acquired_at", sa.DateTime(), nullable=True),
        sa.Column("handler", sa.String(length=128), nullable=False),
        sa.Column("transfer_log", sa.Text(), nullable=False),
        sa.Column("storage_ref", sa.String(length=512), nullable=False),
        sa.ForeignKeyConstraint(["case_id"], ["cases.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_evidence_case_id", "evidence", ["case_id"], unique=False)
    op.create_index("ix_evidence_sha256", "evidence", ["sha256"], unique=False)

    op.create_table(
        "analysis_results",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("case_id", sa.Integer(), nullable=False),
        sa.Column("pipeline_version", sa.String(length=32), nullable=False),
        sa.Column("verdict", sa.String(length=16), nullable=False),
        sa.Column("risk_score", sa.Integer(), nullable=False),
        sa.Column("confidence", sa.String(length=16), nullable=False),
        sa.Column("report_json", sa.Text(), nullable=False),
        sa.Column("stix_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["case_id"], ["cases.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_analysis_results_case_id", "analysis_results", ["case_id"], unique=False
    )

    op.create_table(
        "feedback",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("case_id", sa.Integer(), nullable=False),
        sa.Column("analyst", sa.String(length=128), nullable=False),
        sa.Column("label", sa.String(length=16), nullable=False),
        sa.Column("finding_ref", sa.String(length=256), nullable=False),
        sa.Column("notes", sa.Text(), nullable=False),
        # Integer by contract (shared raw-SQL surface binds 0/1 portably).
        sa.Column("consumed", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["case_id"], ["cases.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_feedback_case_id", "feedback", ["case_id"], unique=False)

    op.create_table(
        "audit_trail",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("ts", sa.DateTime(), nullable=True),
        sa.Column("actor", sa.String(length=128), nullable=False),
        sa.Column("action", sa.String(length=128), nullable=False),
        sa.Column("case_uuid", sa.String(length=36), nullable=False),
        sa.Column("detail", sa.Text(), nullable=False),
        sa.Column("prev_hash", sa.String(length=64), nullable=False),
        sa.Column("row_hash", sa.String(length=64), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_audit_trail_case_uuid", "audit_trail", ["case_uuid"], unique=False)


def downgrade() -> None:
    for table in ("audit_trail", "feedback", "analysis_results", "evidence", "cases"):
        op.drop_table(table)
