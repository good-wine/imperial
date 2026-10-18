# Security model

## Trust boundaries
Untrusted: uploaded media, filenames, EXIF/XMP, feed content, Telegram
captions — all sanitized before LLM use (`_sanitize`, instruction-pattern
redaction) and never executed. Trusted: analyst input, signed models,
pinned C2PA trust anchors.

## TLP 2.0 enforcement (`src/tlp.py`)
`CLEAR < GREEN < AMBER < RED`. Records serve only at/below requester
clearance (`X-TLP` header or `clearance` param); higher-TLP fields are
stripped by `redact()` (narrative/hashes leave AMBER+, everything but
verdict leaves RED). The classification travels in the payload; the full
record stays server-side + audit.

`GET /api/v1/cases/{uuid}/transparency` is clearance-gated on the case TLP
(`_require_clearance`) and audit-logged (`ai_act50.transparency_read`). The
Art. 50 block is assembled by `_ai_act50_payload()` and field-scoped at the
*served* record level (the same level `redact()` uses for `result` in the same
response): the *compliance duty* (marking state, disclosure requirement,
label/badge) is served at every level, while `deepfake`, `depicted_subject` and
the operator posture/`exemption` wording are withheld above GREEN and named in
a `withheld_fields` list. Note the duty is not neutral information — a present
`disclosure_required` implies the deep-fake determination; the field scoping
removes the *subject* and *operator* intelligence, not the obligation. The same
block is attached to `GET /api/v1/cases/{uuid}` (top-level `ai_act50`, served
level) and, inside `result`, only survives at CLEAR/GREEN (`src/tlp.py`). The
STIX bundle and the SIEM export DO carry the determination: they are
case-TLP-gated export paths, not the per-requester API payload — stated here so
that no one reads the API scoping as a property of every output surface.

Caveat and mitigation: by default TLP clearance is **requester-declared**
(`X-TLP` header or `clearance` param) — TLP is a handling caveat, not an access
control, and the API and dashboard ports are therefore bound to loopback by
default in `docker-compose.yml` (reached through Caddy, or the oauth2-proxy in
front of the dashboard). For classified networks, `OIDC_BIND_CLEARANCE=1` binds
the effective clearance to the bearer token's claim (`OIDC_TLP_CLAIM`, default
`tlp_clearance`): the declared header is **capped** by the token — never
escalated — and fails closed to `TLP:CLEAR` on a missing/invalid/unverifiable
token. Binding also **requires** `OIDC_AUDIENCE`: without an audience check a
token minted for another client at the same issuer would be accepted, so an
empty audience fails closed. Enabling OIDC now also makes a valid bearer
*mandatory*: the self-declared `owner`/`X-Analyst`/form field is no longer
accepted as identity (it would otherwise be written into the case owner and the
audit trail unverified), and `OIDC_ALLOW_ANONYMOUS=1` is the explicit opt-out
for a trusted lab. Note what the shipped compose does *not* do: Caddy is a plain
TLS reverse proxy (no bearer validation) and the `oauth2-proxy` service is behind
the `sso` profile with no Caddy route to it, so under the default `full` profile
the console is reachable without authentication either. Identity-binding the
ingress in front of the API and console (mTLS gateway or an authenticating
proxy) is therefore the deployment's responsibility; the endpoints enforce the
cap, they do not authenticate on their own.

## Chain of custody (ISO/IEC 27037)
SHA-256 at ingestion, UTC timestamps, handler/owner, transfer log on
Evidence rows; hash-chained append-only audit in PostgreSQL
(`audit_trail`, `prev_hash`/`row_hash`) mirrored to JSONL (`src/audit.py`
with content-seal `row_hash` + legacy-compat verify). Tamper test in suite.

Verify either side: `python scripts/verify_audit_chain.py` (DB table, exit 1
on break) and `python -c "from src import audit; print(audit.verify_chain(
'reports/audit.jsonl'))"` (JSONL mirror).

### Incident 2026-10-04: forked audit chain, live (open, by design)
`scripts/verify_audit_chain.py` reported **BROKEN at row 84 of 85** on the
live Postgres. Diagnosis, not speculation: rows 83 and 84 were exact
duplicates — same `prev_hash`, same `row_hash`, same actor/action and the
same second. Root cause: `audit()` read the chain tip and inserted as two
separate statements, and the SQLAlchemy adapter commits after *every*
statement, so there was no transaction spanning the read-modify-write. Two
processes (worker + API/watchdog) read the same tip and both appended.

Fix (shipped): a partial unique index `ux_audit_prev_hash ON audit_trail
(prev_hash) WHERE id > 85` turns the race into a constraint violation, and
`audit()` retries against the new tip, so the **database** enforces
single-writer ordering instead of trusting application timing. Verified with
four concurrent processes against live Postgres: 20 rows, zero forks,
`--since-id 85` verifies clean.

Deliberately **not** repaired: the duplicate rows are untouched forensic
evidence. Deleting or rewriting them would be evidence tampering, so the
historic break stays visible to `verify_audit_chain()` and operators verify
the post-fix chain with `--since-id`. Resolving rows 83/84 needs a human
decision on the record, not a patch.

Known limits of the DB chain, stated rather than glossed: `row_hash` seals
`prev_hash + detail`, so edits confined to `actor`, `action`, `case_uuid` or
`ts` are **not** detected by it (a test pins this so the limitation cannot be
quietly forgotten) — the JSONL mirror seals the whole event and is the
authoritative custody log. A truncated **tail** is undetectable by any chained
structure; that is what `scripts/backup_volumes.sh` and off-host copies are
for.

## At-rest protection
`CASES_FERNET_KEY` enables field-level encryption (report JSON, analyst
notes); without it the stack still runs (documented, defense-in-depth).
Postgres volume + case artifacts persist on encrypted host storage in
classified deployments (deployment concern, noted in compose).

## Supply chain
Every model in `models/registry.json` ships `<model>.sig` (Ed25519 over
SHA-256, `scripts/sign_models.py`); `gate_model_load()` refuses unsigned
loads when `MODEL_SIG_ENFORCE=1` (pure-Python RFC 8032 fallback, OpenSSL
cross-validated). C2PA trust anchors pinned in `config/trust_anchors.json`.
Python dependencies are pinned in `requirements.lock` (`make lock` to
refresh); `make audit` runs `pip-audit` with one documented waiver —
PYSEC-2026-3740 (nltk model-persistence sandbox bypass, unpatched upstream,
unreachable: no caller-controlled model paths exist in our code). SBOM via
`make sbom` (Syft CycloneDX or pip freeze).

## API hardening
Multipart type allowlist + `MAX_FILE_MB` cap, per-IP token-bucket rate
limit on ingest/feedback/OSINT (`API_RATE_LIMIT_PER_MIN`), TLP gating,
structured JSON logs, no secrets in code (env only).
