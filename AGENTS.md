# AGENTS.md

Offline-first crisis-media provenance engine: C2PA → multimodal forensics →
GEOINT → context search → LLM/STIX report. FastAPI API (`main.py`), Celery
worker (`worker.py`), Streamlit console (`dashboard/`). Read
`docs/ARCHITECTURE.md` (ops runbook, measured decisions, honest gaps) and
`docs/SECURITY.md` (trust boundaries, audit incident) before deep changes.

## Commands

- Setup: `make install`, then `cp .env.example .env`. Local runs need no
  services: SQLite + eager Celery + rule-based LLM are the fallbacks.
- Always call tools as `.venv/bin/python -m <tool>` (as the Makefile does).
  Bare `.venv/bin/pytest`/`ruff` fail: entry-point shebangs point to a stale
  path after the repo was moved.
- Quality gate: `make gate` = format-check → lint → typecheck → security →
  test. CI (`.github/workflows/ci.yml`) adds `make audit`, `make coverage`
  (55% ratchet — never regress), `make eval` (adversarial harness).
- Single test: `.venv/bin/python -m pytest tests/test_units.py -q -k <name>`
  from the repo root. There is no conftest and no `pythonpath` setting: tests
  find `src.*` only because `python -m pytest` puts CWD on `sys.path`.
- Full suite is offline (no net/DB/GPU). Markers exist (`slow`, `fuzz`,
  `needs_gpu`, `needs_net`); only `slow` is currently applied.
- `make migrate` (Alembic) is the only sanctioned DDL path for Postgres.
  Fresh boots use `create_all()`; `migrations/` mirrors `src/models.py`.
- Rust crate `rust/provenance-jumbf`: run `cargo test` there (must stay
  `cargo fmt`-clean). Excluded from ruff/mypy/bandit; gated by CI's separate
  `rust` job, not by `make gate`.
- Model weights are gitignored and provisioned, never committed. Every model
  needs a `.sig` (Ed25519, `scripts/sign_models.py`) and a pin in
  `models/registry.json`; check with `make verify-weights` (`--no-cache`
  before air-gap transfer).

## Architecture

- `src/modules/*` holds the real engines (`c2pa_verify`, `forensic_vision`,
  `context_check`, `explain_report`, `ai_act50`); `src/engines/*` are async
  worker wrappers that delegate to them. Put logic in `modules`, keep
  `engines` thin.
- `src/schemas.py`: `FinalReport.to_dict()` is the frozen wire format
  (`REPORT_SCHEMA_VERSION = "2.0"`), also consumed by the Rust bridge over
  JSON stdin/stdout. Don't rename/remove fields without bumping the version.
- Fail-open analysis, fail-closed verdicts: every leg degrades to SKIPPED
  with a logged reason. Broad excepts (ruff BLE001/S110/S112, bandit
  B110/B112) are the documented contract, not a defect to clean up.
- TLP 2.0 (`src/tlp.py`) is enforced on every response via the `X-TLP`
  header/clearance; redaction strips fields per level (RED serves verdict
  only). A new API field must respect `redact()`.
- Audit is hash-chained. The known fork at DB rows 83/84 is intentionally
  preserved as forensic evidence — never "repair" it; verify with
  `python scripts/verify_audit_chain.py --since-id 85`. The JSONL mirror
  (`reports/audit.jsonl`) is the authoritative custody log.

## Gotchas

- Never touch `data/cases/` or case rows with ad-hoc shell loops — one
  empty-glob loop once deleted live evidence blobs. Orphan cleanup only via
  `scripts/prune_orphans.py` (dry-run default, explicit flag to execute).
- Don't `pkill -f` a server cmdline on the same line as the invocation (it
  matches the invoking shell); use `fuser -k 8001/tcp`.
- Docker rebuilds cost 30+ min (torch). For Python-only changes, tar-copy
  sources into the running container — exact command in
  `docs/ARCHITECTURE.md`.
- Config comes from `.env`; most settings are centralized and frozen in
  `src/config.py` (loaded once), but a few API knobs (e.g.
  `API_RATE_LIMIT_PER_MIN`) are read via `os.getenv` in `main.py`.
- Dashboard i18n (`dashboard/i18n.py`, EN/IT/FR/ES with English fallback):
  a new string must be added to all languages; TLP/verdict codes are never
  translated (tests enforce both).
- Conventional Commits; bump `pyproject.toml` version + `CHANGELOG.md`
  (Keep a Changelog); `v*` tags trigger the cosign-signed release workflow.
- Scratch space: `/tmp/opencode` (pre-approved), app work dir
  `/tmp/opencode/provenance`; tests write to `/tmp/opencode/provenance_tests`.
