# Architecture (v4) — data flow & service map

```
                    ┌─────────────┐     multipart/TLP      ┌──────────────┐
                    │  Dashboard  │◄──────────────────────►│  FastAPI     │──┐
                    │  :8501/:8502 │      analyst HITL      │  :8000/:8001 │  │
                    └──────┬──────┘                        └──────┬───────┘  │
                           │ verify/poll                   enqueue│          │
                           ▼                                      ▼          ▼
┌────────┐  poll   ┌────────────────┐  tasks   ┌─────────┐  ┌────────────┐  ┌──────┐
│Telegram│────────►│ Celery beat:   │─────────►│  Redis  │◄─│   Worker   │─►│ ollama│
│RSS     │  media  │ osint_poll     │  results │ :6379   │  │ (GPU flag) │  │:11434 │
└────────┘         └────────────────┘          └─────────┘  └─────┬──────┘  └──────┘
                                                                  │ read/write
                                              ┌───────────────────┼───────────────┐
                                              ▼                   ▼               ▼
                                        ┌──────────┐        ┌───────────┐  ┌──────────┐
                                        │ Postgres │        │  Qdrant   │  │ Milvus / │
                                        │ pgvector │        │  :6333    │  │ Weaviate │
                                        │  :5432   │        │  CLIP-512 │  │ (opt-in) │
                                        └──────────┘        └───────────┘  └──────────┘
```

Pipeline per case: `C2PA (c2pa-python → JUMBF tree) → visual forensics
(ELA/FFT/DCT/faces/flicker + ONNX registry, Ed25519-gated) → audio (vocoder)
→ lip-sync → rPPG → MP4 atoms → watermark (metadata + SynthID gate) →
speech (Whisper, audio-gated) → GEOINT (opt-in coords) → context hybrid
(pHash/PDQ-256/CLIP → Qdrant + mirror) → PII redaction (GREEN/CLEAR) →
deterministic risk + ACH + LLM narrative → STIX 2.1`.

## Service map

| Service | Local | Compose | State |
|---|---|---|---|
| API | :8001 (eager, sqlite) | :8000 (celery, postgres) | both healthy |
| Dashboard | :8502 → API 8001 | :8501 → api:8000 | both up |
| Ollama | :11434 (llama3.1:8b ✓) | internal (model pulled ✓) | LLM verified E2E |
| Redis/Postgres/Qdrant | — | :6379/:5432/:6333 | healthy |
| Milvus/Weaviate | — | up (opt-in backends) | running |

## Ops runbook

- Logs: `tail /tmp/opencode/api.log` (local) · `docker compose logs -f api worker`
- Restart local API: `fuser -k 8001/tcp; bash /tmp/opencode/start_api.sh`
  (NEVER `pkill -f` with the server cmdline on the same line — it matches
  the invoking shell itself.)
- Hot-sync Python-only changes into running containers (avoids a 30+ min
  torch rebuild when PyPI is slow; image files are root-owned so extract
  as root, then restart the service):
  `tar -c --exclude=__pycache__ src ingestion security main.py worker.py \
   | docker compose exec -T -u root api tar -x -C /app`
  (repeat per service, then `docker compose restart api worker`).
- Rate limit: `API_RATE_LIMIT_PER_MIN` (default 30, 0 disables).
- Metrics: `GET /metrics` (verify/feedback/stix/osint counters).
- Air-gap: bake `models/` (+`.sig` via `scripts/sign_models.py`), set
  `MODEL_SIG_ENFORCE=1`, run compose with `--network none` after first boot.
- GPU: `docker compose --profile gpu` (worker-gpu, NVIDIA runtime); torch
  auto-selects cuda, ONNX prefers TensorRT.
- High availability: this stack is single-instance by design (field/air-gap
  default). For fixed-site production: Postgres → Patroni/Stolon cluster
  (`DATABASE_URL` already a single knob), Redis → Sentinel or managed
  ElastiCache (only `CELERY_BROKER_URL` changes; the rate limiter shares it),
  Qdrant → distributed mode with the same collection name, API/worker →
  plain replicas behind the Caddy edge (rate budget is already shared via
  Redis). Backups: `scripts/backup_volumes.sh` (pg_dump + Qdrant snapshots
  + Redis RDB); retention purge runs daily via the `beat` service.
- Known honest gaps: ImageBind (CLIP-512 implemented; adapter = same
  `search_correlated` surface), SynthID pixel check (needs DeepMind weights;
  metadata tier is decisive), Milvus/Weaviate (connected, not primary).
- v5 additions: PRNU (`src/engines/prnu_engine.py`, enroll under
  `data/prnu/`), trust pinning (`scripts/pin_trust_anchor.py`), OIDC
  (Keycloak `--profile sso`, `OIDC_ISSUER`), SIEM (`GET /export/siem`),
  calibration (`scripts/calibrate_ensemble.py`), rate-limit + `/metrics`.
