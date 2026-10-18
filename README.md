# Geopolitical Media Provenance & Verification Engine

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Types: mypy](https://img.shields.io/badge/types-mypy-green.svg)](https://mypy-lang.org/)
[![Commits: Conventional](https://img.shields.io/badge/commits-conventional-fe5196.svg)](https://www.conventionalcommits.org/)

**Alpha research build** — interfaces and the report wire format
(`REPORT_SCHEMA_VERSION`) can still change; review
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for measured decisions and the
gaps that are known and not yet closed.

Offline-first intelligence platform for crisis-zone media: **C2PA crypto →
multimodal forensics (video+audio+rPPG+lip-sync+MP4) → GEOINT solar/shadow →
hybrid decontextualization → tradecraft report + STIX 2.1**, served over an
async API (FastAPI + Celery/Redis), persisted in PostgreSQL/SQLAlchemy with
ISO 27037 chain-of-custody, reviewed in a Streamlit HITL console.

Standards: **C2PA 2.x**, **ICD-203/206 tradecraft**, **TLP 2.0**, **STIX 2.1**
(MISP-ready IoB/IoC), **Ed25519** model signing, NREL solar geometry.

v4 delta: **Qdrant** multimodal correlation + `GET /search/similar`, **Whisper**
local STT/translation, **AI-watermark** screening (SynthID + ComfyUI/A1111/
Midjourney metadata), **PII redaction** (face/plate blur on GREEN/CLEAR release),
**OSINT ingestion** (Telegram + RSS -> cases), ONNX providers
TensorRT/OpenVINO/CUDA/CoreML/CPU, AL retraining dataset trigger.

v5 delta (this): **c2pa-python** real chain binding, **PRNU** sensor
fingerprinting, **trust-anchor pinning** (`pin_trust_anchor.py` + cert
carve-and-compare), **OIDC** analyst auth (Keycloak profile, optional),
**SIEM export** (CEF/LEEF/ECS), **ensemble calibration** from feedback
(sklearn LR → `models/ensemble_weights.json`), API **rate-limit** +
`/metrics`, **Whisper small** + **CLIP ONNX** provisioning scripts.

## Quickstart

```bash
make install            # venv + requirements + editable dev extras
cp .env.example .env

# Single-file triage (no services needed)
.venv/bin/python main.py path/to/media.jpg
.venv/bin/python main.py --self-test

# Quality gate (same order as CI): format → lint → types → security → tests
make gate
make coverage           # tests + 55% coverage ratchet (never regress)
make eval               # pristine vs spliced vs recaptioned harness
make sbom               # supply-chain SBOM (Syft CycloneDX or pip freeze)

# Full async stack
docker compose up --build            # api + worker + redis + postgres
docker compose --profile full up     # + ollama + dashboard + milvus + weaviate
docker compose --profile gpu up      # NVIDIA worker overlay
make run-api run-worker               # API + worker without docker
API_BASE=http://127.0.0.1:8001 make run-dashboard   # analyst console
.venv/bin/python scripts/sign_models.py --sign-all   # Ed25519 provision models
```

Quality bars (enforced): `ruff check` + `ruff format --check` clean,
`mypy` zero errors on 42 files, `bandit -c .bandit` zero issues,
`pytest` green with coverage ratchet, adversarial `eval_metrics` PASS.

## Folder structure

```
.
├── main.py                  # CLI triage + FastAPI v1 (uvicorn main:app)
├── worker.py                # Celery tasks (eager fallback offline)
├── src/
│   ├── models.py            # SQLAlchemy ORM: cases/evidence/results/feedback/audit
│   ├── database.py          # Postgres repo + stdlib-sqlite fallback, sealed fields
│   ├── tlp.py               # TLP 2.0 enforcement + response redaction
│   ├── stix.py              # STIX 2.1 bundle (deterministic IDs, MISP-ready)
│   ├── audit.py             # hash-chained JSONL custody log
│   ├── engines/
│   │   ├── c2pa_engine.py   # manifest validity + signer + timestamp
│   │   ├── forensics_engine.py # denoise/norm + MP4 atoms + audio vocoder + lip-sync + rPPG
│   │   ├── geoint_engine.py # NREL solar position + shadow consistency
│   │   ├── context_engine.py# hybrid phash/PDQ/CLIP wrapper + intel summary
│   │   └── llm_engine.py    # fusion scoring + narrative + STIX
│   ├── ...                  # v2 modules unchanged (c2pa/forensic/context/llm)
├── security/signature.py    # Ed25519 gate before every model load
├── dashboard/app.py         # Streamlit HITL console (EN/IT/FR/ES via i18n.py)
├── scripts/sign_models.py   # offline model provisioning
├── scripts/eval_metrics.py  # adversarial eval harness
├── Dockerfile               # targets: api | worker | dashboard | cli (multi-arch)
└── docker-compose.yml       # api/worker(-gpu)/redis/postgres/ollama/dashboard
```

## Design decisions

- **Tri-state C2PA + per-check detail**: `manifestFound/structureParse/
  claimSignature/certChain/dataHash/timestamp` each PASS/FAIL/SKIPPED.
  A `dataHash` FAIL floors risk at 80 — content changed post-signing.
- **Deterministic score, LLM narrative**: auditable rules decide; the model
  explains under a faithfulness self-check (verdict + factors echoed, else
  an addendum is appended). Prompt-injection strings are sanitized pre-LLM.
- **Hybrid retrieval**: pHash (≤8) for near-dups, PDQ-256 (≤48) for crops,
  CLIP cosine (≥0.90) for semantic reuse. Fail-open availability,
  fail-closed verdicts.
- **Rust-ready**: `schemas.py` v2 dicts are the frozen wire format;
  `rust/provenance-jumbf` already parses JUMBF with property tests.

## EU AI Act Art. 50 transparency

`src/modules/ai_act50.py` answers a **different question from authenticity**:
is the media machine-readably marked, and must a human see a label? `assess()`
runs on every report and reaches four surfaces: `report.ai_act50` (schema v2
JSON), the markdown `## EU AI Act Art. 50 transparency` block (carrying the
required label text), `x_ai_act50` on the STIX `opinion` object, and
`GET /api/v1/cases/{uuid}/transparency` (marking state, disclosure duty,
accessible SVG badge) for publication pipelines. Carve-outs are declared via
`AI_ACT_ROLE`/`AI_ACT_LAW_ENFORCEMENT`/`AI_ACT_STANDARD_EDITING`, never applied
silently; every payload carries `legal_advice: false`.

## Offline / Docker

```bash
docker build -t provenance-mvp .
docker run --rm -v $PWD/reports:/app/reports provenance-mvp --self-test
docker compose --profile full up     # + postgres/milvus/weaviate/ollama
```

`VECTOR_BACKEND=postgres` + `POSTGRES_DSN` for PG-HNSW, `LLM_BACKEND=ollama|jury`
+ `OLLAMA_MODEL=llama3.1:8b` for narratives, `TLP_MARKING` for handling caveat.
Vendored ONNX files go in `models/` and are listed in `models/registry.json`.

## Local domain with HTTPS (`imperial.local`)

The `proxy` service (Caddy, offline `tls internal` CA) exposes
`https://api.imperial.local` and `https://app.imperial.local` on host
:443/:80. Two one-time client steps (see `scripts/trust-proxy-ca.sh`):

```bash
# One-shot setup (idempotent): DNS check (+pkexec /etc/hosts if missing),
# CA export + browser trust (no root), end-to-end HTTPS verify.
bash scripts/setup-local-domain.sh
# Manual equivalents if you prefer:
# 1. trust the machine-local CA
bash scripts/trust-proxy-ca.sh   # exports config/imperial-root.crt + prints next step
# 2. name resolution (needs sudo, once)
sudo sh -c 'echo "127.0.0.1 imperial.local api.imperial.local app.imperial.local" >> /etc/hosts'
# without sudo, verify immediately via SNI override:
curl --resolve api.imperial.local:443:127.0.0.1 --cacert config/imperial-root.crt \
  https://api.imperial.local/health
```

## Contributing, security and licence

- **[CONTRIBUTING.md](CONTRIBUTING.md)** — setup, the quality gate, where code
  goes (`src/modules` vs `src/engines`), the frozen wire format, and the
  repo's gotchas.
- **[docs/SECURITY.md](docs/SECURITY.md)** — trust boundaries, the TLP model,
  and how to report a vulnerability privately.
- **[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)** — Contributor Covenant 2.1.
- **[CHANGELOG.md](CHANGELOG.md)** — Keep a Changelog.
- Licensed under the **Apache License 2.0** ([LICENSE](LICENSE)).

