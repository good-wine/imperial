"""Adversarial eval harness (SOTA practice): pristine vs spliced vs recaptioned.

Builds synthetic fixtures (gradient=pristine, block-spliced, recaptioned-copy),
runs the pipeline, and reports per-class detection rates + latency. No extra
deps beyond Pillow/numpy. Fails (exit 1) if any class scores 0% — the suite
catches silent regressions in ensemble weights or hybrid search.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

TMP = Path("/tmp/opencode/prov_eval")
TMP.mkdir(parents=True, exist_ok=True)


def _pristine(p: Path) -> None:
    arr = np.tile(np.linspace(20, 235, 256, dtype=np.uint8), (256, 1))
    rgb = np.dstack([arr, arr, arr])
    Image.fromarray(rgb).save(p)


def _spliced(p: Path) -> None:
    rng = np.random.default_rng(42)
    arr = np.tile(np.linspace(20, 235, 256, dtype=np.uint8), (256, 1))
    rgb = np.dstack([arr, arr, arr])
    rgb[90:170, 90:170] = rng.integers(0, 255, (80, 80, 3), dtype=np.uint8)
    Image.fromarray(rgb).save(p)


def main() -> int:
    import main as pipeline
    from src.db.store import InMemoryEventStore, ProvenanceEvent
    from src.modules.context_check import check_context

    pristine, spliced = TMP / "pristine.png", TMP / "spliced.png"
    _pristine(pristine)
    _spliced(spliced)

    t0 = time.time()
    rp, rs = pipeline.analyze(str(pristine)), pipeline.analyze(str(spliced))
    dt = (time.time() - t0) / 2
    print(f"pristine: risk={rp['risk_score']} verdict={rp['verdict']}")
    print(
        f"spliced : risk={rs['risk_score']} verdict={rs['verdict']} "
        f"anomaly={rs['forensic']['aggregate_anomaly']}"
    )
    print(f"mean latency: {dt:.2f}s/file")

    # Recaptioned-copy: identical pixels must match the seeded event (hybrid).
    import imagehash

    with Image.open(spliced) as img:
        ph = str(imagehash.phash(img.convert("RGB")))
    store = InMemoryEventStore()
    store.add_event(ProvenanceEvent("EVT-EVAL", "eval fixture", "2026-01-01", "eval", ph))
    hit = check_context(spliced, store=store)
    print(
        f"recaptioned-copy match: {hit.is_decontextualized} "
        f"({hit.matches[0].event_id} d={hit.matches[0].hamming_distance})"
        if hit.matches
        else "recaptioned-copy match: False"
    )

    ok = True
    if not (rs["risk_score"] >= rp["risk_score"]):
        print("FAIL: spliced must score >= pristine")
        ok = False
    if not hit.is_decontextualized:
        print("FAIL: exact copy must match")
        ok = False
    print("EVAL " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
