# Contributing

Thanks for taking the time to contribute. This is a **crisis-media provenance
engine**: a wrong verdict is as serious as a crash, so "it runs" is not the bar
— evidence is.

Before a deep change, read [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) (ops
runbook, measured decisions, honest gaps) and [`docs/SECURITY.md`](docs/SECURITY.md)
(trust boundaries). Coding agents should read [`AGENTS.md`](AGENTS.md) instead.

## Prerequisites

- Python **3.10+** (CI runs 3.11)
- `make`, `git`
- Optional: Docker + Docker Compose, Redis, PostgreSQL, `cargo` (Rust crate)

No secrets, no paid account and no network are required for the test suite: it
runs fully offline (SQLite + eager Celery + rule-based LLM fallbacks).

## Getting started

```bash
make install                       # venv + requirements + editable dev extras
cp .env.example .env               # local runs need no running services

.venv/bin/python main.py --self-test          # sanity check
.venv/bin/python main.py path/to/media.jpg    # single-file triage
```

> **Always call tools as `.venv/bin/python -m <tool>`** (as the Makefile does).
> Bare `.venv/bin/pytest` / `.venv/bin/ruff` fail because the entry-point
> shebangs point to a stale path.

## Quality gate

Every change must leave the gate green — the same order CI runs:

```bash
make gate          # format-check → lint → typecheck → security → test
make coverage      # tests + 55% coverage ratchet (never regress)
make eval          # adversarial harness: pristine vs spliced vs recaptioned
make sbom          # supply-chain SBOM (Syft CycloneDX or pip freeze)
```

`pre-commit` is configured (ruff + ruff-format + mypy) if you want the fast
inner loop: `pre-commit install`.

Run a single test from the repo root (tests find `src.*` only because
`python -m pytest` puts the CWD on `sys.path`):

```bash
.venv/bin/python -m pytest tests/test_units.py -q -k <name>
```

Markers: `slow`, `fuzz`, `needs_gpu`, `needs_net` — only `slow` is applied
today. The default suite must stay offline.

## Where code goes

- `src/modules/*` holds the **real engines** (`c2pa_verify`, `forensic_vision`,
  `context_check`, `explain_report`, `ai_act50`). Put logic here.
- `src/engines/*` are **thin async worker wrappers** that delegate to modules.
- `src/schemas.py`: `FinalReport.to_dict()` is the **frozen wire format**
  (`REPORT_SCHEMA_VERSION`), also consumed by the Rust bridge. Do not
  rename/remove fields without bumping the version.
- **Fail-open analysis, fail-closed verdicts**: every leg degrades to `SKIPPED`
  with a logged reason. The broad `except` blocks (ruff BLE001/S110/S112,
  bandit B110/B112) are the documented contract, not a defect to clean up.
- **TLP 2.0** (`src/tlp.py`) is enforced on every response: a new API field must
  respect `redact()`, and must be field-scoped by clearance.
- **Audit is hash-chained.** The known fork at DB rows 83/84 is intentionally
  preserved as forensic evidence — never "repair" it.
- Dashboard strings are i18n'd (`dashboard/i18n.py`, EN/IT/FR/ES with English
  fallback): a new string goes in **all** languages. TLP/verdict codes are never
  translated.
- A few API knobs (`API_RATE_LIMIT_PER_MIN`, …) are read via `os.getenv` in
  `main.py`; everything else lives in `src/config.py`. Document new knobs in
  `.env.example`.

## Commits and pull requests

- [Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`,
  `docs:`, `refactor:`, `test:`, `chore:`, `ci:`.
- Bump `pyproject.toml` version + `CHANGELOG.md` (Keep a Changelog) for
  user-visible changes. `v*` tags trigger the signed release workflow.
- One logical change per PR. Fill in the PR template, including the **command
  and its output** as evidence.
- Never commit secrets, `.env`, case evidence or media you are not cleared to
  share. Never `git push --force` on `main`.

See [`.github/PULL_REQUEST_TEMPLATE.md`](.github/PULL_REQUEST_TEMPLATE.md) for
the checklist.

## Model weights and supply chain

Weights are gitignored and provisioned, never committed. Every model needs a
`.sig` (Ed25519, `scripts/sign_models.py`) and a pin in `models/registry.json`;
check with `make verify-weights` (`--no-cache` before an air-gap transfer).

## Rust crate

`rust/provenance-jumbf` is gated by its own CI job (`cargo fmt --check` +
`cargo test`), not by `make gate`:

```bash
cargo test --manifest-path rust/provenance-jumbf/Cargo.toml
```

## Common gotchas

- Never touch `data/cases/` or case rows with ad-hoc shell loops — an
  empty-glob loop once deleted live evidence blobs. Orphan cleanup only via
  `scripts/prune_orphans.py` (dry-run by default).
- Don't `pkill -f` a server cmdline on the same line as the invocation (it
  matches the invoking shell); use `fuser -k 8001/tcp`.
- Docker rebuilds cost 30+ min (torch). For Python-only changes, tar-copy
  sources into the running container — the exact command is in
  `docs/ARCHITECTURE.md`.
- `make migrate` (Alembic) is the only sanctioned DDL path for Postgres.

## Reporting a security issue

Do **not** open a public issue. Use the private advisory channel described in
[`docs/SECURITY.md`](docs/SECURITY.md).
