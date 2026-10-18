# Changelog

All notable changes to the Provenance Engine follow
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/) versioning
(`MAJOR.MINOR.PATCH` on `pyproject.toml`).

## [4.5.0] - 2026-10-04

### Added
- EU AI Act Art. 50 transparency assessment (`src/modules/ai_act50.py`):
  machine-readable marking detection from C2PA `digitalSourceType`, invisible
  watermarks and soft binding; Art. 3(60) deep-fake determination that is
  audience-aware and intent-independent; Art. 50(4) disclosure duty with an
  accessible SVG badge; declared (never silently applied) carve-outs; and
  `legal_advice: false` on every payload.
- Assessment now runs on every report (`generate_report()` and
  `aggregate_report()`) and reaches all four output paths: `report.ai_act50`
  in the schema v2 JSON, an `## EU AI Act Art. 50 transparency` block in the
  markdown (carrying the required label text), `x_ai_act50` on the STIX
  `opinion` object, and `GET /api/v1/cases/{uuid}/transparency` for
  publication pipelines.
- `tests/test_ai_act50.py`: marking-state separation, decisive-vs-supporting
  evidence split, carve-out declaration, badge accessibility, output parity,
  and TLP interaction.

### Fixed
- Art. 50 evidence extraction read three fields that do not exist on
  `ForensicResult` (`faces`, `instability`, `splicing_detected`). Subject
  detection therefore never found a PERSON, audience risk never rose to
  `high`, and the signal list was always empty — every image was assessed
  "not a deep fake" regardless of anomaly score. Now reads per-frame
  `face_count`, `aggregate_anomaly`, `temporal_consistency` and
  `face_anomaly`.
- A generic forensic anomaly no longer tips the Art. 3(60) determination to
  "deep fake". Recompression and rescaling raise the same score, so such
  evidence moves to the new `supporting_evidence` field and yields
  `Indeterminate`, keeping decisive and weak evidence separately auditable.
- Watermark marking read keys no producer emits (`detected`, `soft_binding`)
  and treated the `synthid` *status string* as a boolean, so every analysed
  file without a C2PA AI declaration was reported `AI_GENERATED` with
  `deepfake=True`. Now reads the real `WatermarkResult` contract
  (`ai_generated is True`, `synthid == "DETECTED"`).
- The audio media leg read `manipulated`/`mismatch`/`verdict` keys absent from
  `AudioResult`, so it was unreachable; it now reads `vocoder_anomaly` at the
  risk-engine threshold, while flat prosody (`pitch_instability`) is
  supporting-only (dubbing, singing and compression flatten prosody too).
- The CLI and worker paths assessed the same media twice with different inputs
  (the worker's assessment carries the multimodal evidence); the assessment is
  now computed once and shared, removing a duplicated Art. 50(4)
  recommendation.
- The transparency payload is field-scoped by clearance: the compliance duty
  (marking state, disclosure requirement, label/badge) is served at every
  level, while `deepfake`, `depicted_subject` and the operator
  posture/`exemption` wording are withheld above GREEN and named in a
  `withheld_fields` list instead of being served beside a RED-redacted
  `result`. The STIX bundle and the SIEM export still carry the
  determination (case-TLP-gated export paths).
- The disclosure badge escapes interpolated text, uses per-label element ids
  and carries a border (visible on a dark page).

## [4.4.0] - 2026-10-03

### Added
- Multilingual dashboard (EN/IT/FR/ES, zero new dependencies):
  `dashboard/i18n.py` with English-fallback contract, sidebar language
  selector, `DASHBOARD_LANG` deployment default (wired through compose).
- `tests/test_i18n.py`: key parity, placeholder consistency, fallback and
  invariant (TLP/verdict codes never translated) enforcement.

## [4.3.0] - 2026-10-03

### Added
- Retention purge per TLP with Celery beat schedule, Alembic migrations,
  oauth2-proxy SSO gate for the dashboard, cosign keyless release workflow,
  red-team suite for the LLM guardrail, Redis sliding-window rate limiting.

## [4.2.0] - 2026-10-03

### Added
- Real spectral synthesis head: shared `spectral_profile()` feature,
  `scripts/train_spectral_head.py` (bootstrap corpus, 0.70 AUC refusal
  gate, sklearn↔ONNX parity check), `input: spectral32` dispatch in the
  forensic ensemble, Ed25519-signed and registered.
- Ground-truth AL bootstrap (`scripts/seed_feedback.py`, 24 construction-
  labeled rows) closing the loop into `calibrate_ensemble.py` (CV 0.78).
- Redis sliding-window rate limiting (`src/ratelimit.py`) with memory
  fallback; shared budget across replicas.
- Hypothesis property tests (`tests/test_fuzz.py`) for all hostile-input
  parsers (JUMBF/MP4/PNG/seed-JSON/hamming/C2PA entry).
- DR + SSO: `scripts/backup_volumes.sh` (pg_dump, Qdrant snapshots, RDB),
  `config/keycloak_realm.json` template.
- Tests: 56 → 66 passing; coverage 61% → 62%.

### Fixed
- Clean gradient scored above spliced image by v1 head (out-of-domain
  pristine class) → scene-invariant high-pass texture feature.
- `pyproject.toml` version now tracks CHANGELOG (4.2.0).

## [4.1.0] - 2026-10-03
### Added
- `tests/test_units.py`: 14 offline unit tests (SIEM shapes, OIDC RS256
  roundtrip + rejections, RSS/Atom parsing incl. entity-attack, Telegram
  gating, Jury semantics, Qdrant `:memory:` hybrid/correlation, fetch
  choke point incl. size cap, ORM CRUD, RAG retrieval, engine wrappers).
- Repo hygiene: `LICENSE` (Apache-2.0), `.gitignore`, `.editorconfig`,
  `requirements.lock` (`make lock`), `CHANGELOG.md`, `make audit`
  (pip-audit with documented nltk waiver), CI dependency-audit step.
- `GET /metrics`, per-IP rate limiting, `docs/ARCHITECTURE.md`,
  `docs/SECURITY.md` supply-chain section.

### Fixed
- `list.extend()` on async generator (Telegram scraper TypeError).
- `FeedItem`/`TgMedia` variable confusion in OSINT pipeline.
- `cert.fingerprint()` called with `hashlib` object instead of
  `hashes.SHA256()` (c2pa_verify, pin_trust_anchor).
- Centralized outbound fetch with scheme allowlist (`S310` class closed).

### Security
- `pip-audit`: 1 known vuln (nltk PYSEC-2026-3740, unpatched upstream,
  unreachable from our code paths) — waived with justification in
  `Makefile`/`docs/SECURITY.md`, re-audited in CI.

## [4.0.0] - 2026-10-03

### Added
- `GET /api/v1/export/siem` (CEF/LEEF/ECS) and `GET /metrics`, per-IP
  token-bucket rate limiting on ingestion endpoints.
- PRNU sensor fingerprinting (`src/engines/prnu_engine.py`).
- C2PA trust-anchor pinning (`scripts/pin_trust_anchor.py`, cert
  carve-and-compare in `certChain` validation detail).
- Optional OIDC analyst auth (`src/oidc.py`, Keycloak `sso` profile).
- Ensemble calibration from HITL feedback (`scripts/calibrate_ensemble.py`).
- Centralized outbound fetch (`src/security.fetch_url`, scheme allowlist).
- Engineering: `ruff`/`mypy`/`bandit` clean, `Makefile` + CI + pre-commit,
  coverage ratchet, SBOM script, lockfile.

### Fixed
- `list.extend()` on async generator in Telegram scraper (runtime TypeError).
- `FeedItem`/`TgMedia` variable confusion in OSINT pipeline.
- `cert.fingerprint(hashlib.sha256())` → `hashes.SHA256()` (correct API).
- PostgreSQL triad: `lastrowid`, non-iterable adapter rows, naive JSON dump
  of `datetime`, `consumed` boolean-vs-integer.
- Bytes leak in JUMBF tree breaking `json.dumps` on JPEG/MP4 reports.

## [3.0.0] - 2026-10-02

### Added
- Enterprise stack: FastAPI v1, Celery worker (eager fallback), SQLAlchemy
  ORM + stdlib-sqlite fallback, Streamlit HITL console, OSINT ingestion
  (Telegram/RSS), Qdrant correlation, Whisper STT, watermark screening,
  PII redaction, GEOINT solar analysis, STIX 2.1, TLP enforcement,
  Ed25519 model gating, Rust JUMBF crate.
