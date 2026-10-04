"""Verify vendored weights against registry pins (supply-chain gate).

Checks every present `models/*` file (ONNX heads, external weight blobs,
Whisper checkpoints) against the `sha256` pins in `models/registry.json`
(+ `external_data` and `whisper` sections).
Exit 0 = all present files match; exit 1 = mismatch or unreadable registry
(fail-closed: do not deploy). Absent BYO slots are skipped, never errors.
Run in CI (`make verify-weights`) and before air-gap transfer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def collect_entries(doc: dict) -> list[tuple[str, str, str]]:
    """Flatten the registry into (path, sha256_pin, label) triples.

    Models over the 2 GB protobuf limit keep their tensors in a sibling .data
    blob; without an extra pin the graph hash alone would not cover the
    weights, which is where all the substance is.
    """
    entries = [
        (m.get("path", ""), m.get("sha256", ""), m.get("name", "?")) for m in doc.get("models", [])
    ]
    for m in doc.get("models", []):
        ext = m.get("external_data") or {}
        if ext.get("path"):
            entries.append((ext["path"], ext.get("sha256", ""), f"{m.get('name', '?')}[weights]"))
    entries += [
        (f"models/whisper/{k}", v, f"whisper/{k}") for k, v in (doc.get("whisper") or {}).items()
    ]
    return entries


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify weight pins")
    ap.add_argument("--registry", default="models/registry.json")
    args = ap.parse_args()
    try:
        doc = json.loads(Path(args.registry).read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"unreadable registry: {exc}", file=sys.stderr)
        return 1
    failures = 0
    checked = 0
    entries = collect_entries(doc)
    for path, pin, name in entries:
        p = Path(path)
        if not p.is_file():
            print(f"SKIP absent: {name}")
            continue
        if not pin:
            print(f"SKIP unpinned: {name} (BYO slot)")
            continue
        checked += 1
        if _sha(p) == pin:
            print(f"OK {name}")
        else:
            print(f"MISMATCH {name}: expected {pin[:16]}…", file=sys.stderr)
            failures += 1
    print(f"{checked} checked, {failures} mismatches")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
