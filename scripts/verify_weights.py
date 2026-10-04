"""Verify vendored weights against registry pins (supply-chain gate).

Checks every present `models/*` file (ONNX heads, external weight blobs,
Whisper checkpoints) against the `sha256` pins in `models/registry.json`
(+ `external_data` and `whisper` sections). Exit 0 = all present files match;
exit 1 = mismatch or unreadable registry (fail-closed: do not deploy). Absent
BYO slots are skipped, never errors.

    python scripts/verify_weights.py              # memoized on size+mtime
    python scripts/verify_weights.py --no-cache   # full re-hash (~18 s)

Use `--no-cache` before an air-gap transfer and in CI: the cache trusts
(size, mtime_ns), so it is a tripwire, not a cryptographic guarantee. The
logic lives in `src/weight_integrity.py` so `beat` can run the same check on a
schedule.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.weight_integrity import verify_registry


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify weight pins")
    ap.add_argument("--registry", default="models/registry.json")
    ap.add_argument("--cache", default="data/model_integrity.json")
    ap.add_argument(
        "--no-cache",
        action="store_true",
        help="re-hash everything, ignoring the (size, mtime) memo",
    )
    ap.add_argument("--json", action="store_true", help="machine-readable report")
    args = ap.parse_args()

    out = verify_registry(
        args.registry,
        cache_path=None if args.no_cache else args.cache,
        use_cache=not args.no_cache,
    )
    if args.json:
        print(json.dumps(out, indent=2))
        return 0 if out["ok"] else 1

    if out.get("error"):
        print(out["error"], file=sys.stderr)
        return 1
    for line in out["skipped"]:
        print(f"SKIP {line}")
    for bad in out["mismatches"]:
        print(
            f"MISMATCH {bad['name']}: expected {bad.get('expected', '?')[:16]}… ({bad['reason']})",
            file=sys.stderr,
        )
    cached_note = f" ({out['cached']} cached, {out['hashed']} hashed)" if out["checked"] else ""
    print(
        f"{out['checked']} checked, {len(out['mismatches'])} mismatches{cached_note}"
        f" in {out['elapsed_s']}s"
    )
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
