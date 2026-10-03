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

## Chain of custody (ISO/IEC 27037)
SHA-256 at ingestion, UTC timestamps, handler/owner, transfer log on
Evidence rows; hash-chained append-only audit in PostgreSQL
(`audit_trail`, `prev_hash`/`row_hash`) mirrored to JSONL (`src/audit.py`
with content-seal `row_hash` + legacy-compat verify). Tamper test in suite.

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
