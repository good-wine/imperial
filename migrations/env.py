"""Alembic environment: versioned schema evolution for PostgreSQL.

Fresh bootstraps and tests still use `Base.metadata.create_all()` (zero-config);
production deployments evolve via `make migrate` (alembic upgrade head).
The initial revision mirrors `src/models.py` exactly; future changes arrive
as new revisions, never as silent DDL.
"""
from __future__ import annotations

import os
import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import engine_from_config, pool

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alembic import context

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

config.set_main_option(
    "sqlalchemy.url", os.getenv("DATABASE_URL", "sqlite:///data/provenance.db")
)

import src.models as M  # noqa: E402

target_metadata = M.Base.metadata if M.HAS_SQLALCHEMY else None


def run_migrations_offline() -> None:
    context.configure(url=config.get_main_option("sqlalchemy.url"),
                      target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
