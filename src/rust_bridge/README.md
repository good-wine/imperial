# Rust microservices (future) — Standard Stable boundary

Python owns orchestration + AI glue in this MVP. Compute-heavy stages
(frame extraction, pHash batch ingest, JUMBF parsing) are designed to move to
Rust services without changing module interfaces:

```
src/modules/forensic_vision.py  ->  rust/provenance-frames  (ffmpeg + rav1d bindings)
src/modules/c2pa_verify.py      ->  rust/provenance-jumbf   (nom-based box parser)
src/db/store.py                 ->  rust/provenance-hash    (Hamming/ANN index)
```

Contract: each Rust service speaks JSON over stdin/stdout (or gRPC later)
using `src/schemas.py` field names. `FinalReport.to_dict()` is the frozen
wire format — do not rename fields without bumping the pipeline version.

Suggested first slice (all Standard Stable, no unsafe needed):
  `cargo new --lib provenance-jumbf` with `nom` for MP4/JPEG box walking,
  mirroring `_scan_jumbf()` semantics, property-tested against this Python code.
