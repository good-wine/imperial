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
                                        │  :5432   │        │ CLIP-512 +│  │ (opt-in) │
                                         │          │        │  IB-1024  │  │          │
                                        └──────────┘        └───────────┘  └──────────┘
```

Pipeline per case: `C2PA (c2pa-python → JUMBF tree) → visual forensics
(ELA/FFT/DCT/faces/flicker + ONNX registry, Ed25519-gated) → audio (vocoder)
→ lip-sync → rPPG → MP4 atoms → watermark (metadata + SynthID gate) →
speech (Whisper, audio-gated) → GEOINT (opt-in coords) → context hybrid
(pHash/PDQ-256/CLIP-512/ImageBind-1024 → Qdrant + mirror) → PII redaction (GREEN/CLEAR) →
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
- **Stuck-case watchdog** (`provenance.reap_stale`, beat every 5 min,
  `STALE_CASE_SECONDS` default 900). Live-measured failure it exists for:
  `docker compose kill -s SIGKILL worker` mid-task left the case in `running`
  **forever** — Celery's `acks_late` did NOT redeliver it, so the queue lost
  the work silently and nothing alerted. The watchdog re-dispatches the task
  onto the broker (flipping the DB row to `queued` is cosmetic and would be a
  lie: Celery takes work from Redis, not from the cases table), marks cases
  whose evidence blob is gone as `failed` instead of looping forever, and
  audits every action (`case.requeued` / `case.abandoned`). If dispatch fails
  the case is left visibly stuck, never labelled `queued`. Verified live:
  SIGKILL → stranded → reap → `done` in ~15 s.
- **Rejected: two-phase verdict** (deterministic verdict signed first, LLM
  narrative as a second write). Proposed to fix a 5-minute turnaround seen
  during an earlier session; measured before building it and the premise was
  wrong. Component timings on this host: C2PA 0.09 s, visual forensics
  3.3 s, context (CLIP + hashes) 10 s, watermark 0.04 s, redaction 0.01 s,
  report generation 0.03 s — `LLM_BACKEND=auto` resolves to `rule-based` on
  this host (Ollama at `localhost:11434` was not reachable from the process
  that measured it), so the narrative was never the cost. A live end-to-end
  submit took **22 s**
  (queued → running → done). The 5 minutes came from worker restarts during
  that session, and chasing it uncovered the real defect: a killed worker
  strands its case (see the watchdog entry above). Nothing was built for a
  latency problem that did not exist; the dashboard streams the verdict as
  soon as it lands regardless.
- ImageBind export provenance: the checkpoint is pinned by SHA-256 (that is
  the authoritative anchor), but the upstream clone's commit was **not
  recorded** — it lived in `/tmp` and was wiped before the field existed.
  `models/registry.json` says `commit: UNRECORDED` rather than carrying an
  invented sha. `scripts/export_imagebind_onnx.py` now refuses to export
  without `--upstream-commit`, verifies it against the clone's HEAD, and
  writes the resolved sha back into the registry, so the next rebuild is
  attributable. Until then the shipped graph is verified-but-unattributed.
- **Restore drill**: `make restore-drill BACKUP=backups/<set>` restores a
  backup set into a throwaway database inside the postgres container and drops
  it afterwards (never touches live, never stops a service). It checks that
  the dump restores, that cases present in both copies carry the same sha256,
  that every case which left live after the backup has a
  `case.purged`/`case.abandoned` audit entry, that stored reports are intact
  JSON, that the Qdrant snapshot names its collection and that the Redis RDB
  has the REDIS magic. Custody-log criterion: the restored chain may
  reproduce the **documented** 2026-10-04 fork (row 84), but must introduce
  no break the live database does not also have. Result on
  `backups/drill-fresh`: 15 passed, 0 failed.
- Case storage is NEVER touched by ad-hoc shell loops (an empty-expansion
  class of error once deleted live rows' blobs). Orphan cleanup goes only
  through `python scripts/prune_orphans.py` (dry-run default, explicit
  `--execute --i-understand-that-this-deletes-evidence`, audited).
- Rate limit: `API_RATE_LIMIT_PER_MIN` (default 30, 0 disables).
- Metrics: `GET /metrics` (verify/feedback/stix/osint counters).
- Weight integrity is **scheduled**, not decorative: `provenance.verify_models`
  runs hourly on `beat` (~0.1 s) and re-checks every pin in
  `models/registry.json`; a mismatch writes a `models.pin_mismatch` audit
  entry. Digests are memoized on `(size, mtime_ns)` in
  `data/model_integrity.json` because a full re-hash of the 2.4 GB blob costs
  ~18 s. Honest limit: that cache trusts filesystem metadata, so an attacker
  who rewrites a weight *and* restores its mtime defeats it — it is a
  tripwire around the signature gate, not a replacement. Use
  `make verify-weights` / `python scripts/verify_weights.py --no-cache` before
  an air-gap transfer, where the full hash is required.
- Air-gap: bake `models/` (+`.sig` via `scripts/sign_models.py`), set
  `MODEL_SIG_ENFORCE=1`, run compose with `--network none` after first boot.
  Transfer `models/imagebind_vision.onnx` AND `models/imagebind_vision.onnx.data`
  together (the graph is 840 KB, the weights 2.4 GB — the `.onnx` alone is inert);
  `python scripts/verify_weights.py` checks both pins before you cut the link.
  `.gitignore` deliberately excludes `*.onnx.data`: weights are provisioned,
  not versioned.
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
- Known honest gaps: ImageBind is shipped **disabled by default and measured,
  not assumed**: `scripts/eval_decontextualization.py` runs the reuse benchmark
  (7 real images x 5 transformations + 15 distractors) through the real
  fusion path and prints a verdict. Current result: end-to-end recall 0.714 →
  0.743 (+1 query of 35), no change in false positives — i.e. **inside noise at
  this corpus size**, not a win. It stays `IMAGEBIND_ENABLED=0` until the
  harness runs on ≥20 images. The vision tower itself is real and signed
  (`models/imagebind_vision.onnx` + `.data`, `provenance_ib` collection, its
  own `IB_THRESHOLD` because its similarity scale is lower than CLIP's). The
  **audio tower** remains unbuilt: it needs a torchaudio whose native
  extension loads — the stock wheel links libcudart and fails on CPU hosts.
  Also open: SynthID pixel check (needs DeepMind weights; metadata tier is
  decisive), Milvus/Weaviate (connected, not primary).
- v5 additions: PRNU (`src/engines/prnu_engine.py`, enroll under
  `data/prnu/`), trust pinning (`scripts/pin_trust_anchor.py`), OIDC
  (Keycloak `--profile sso`, `OIDC_ISSUER`), SIEM (`GET /export/siem`),
  calibration (`scripts/calibrate_ensemble.py`), rate-limit + `/metrics`.
