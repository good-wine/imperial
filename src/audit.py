"""Hash-chained audit log for chain-of-custody (SOTA intel requirement).

Each entry embeds the SHA-256 of the previous line: silent tampering with
`reports/audit.jsonl` breaks the chain and is detectable with `verify_chain()`.
No third-party dependency; safe for air-gapped append-only volumes.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def _prev_hash(log_path: Path) -> str:
    if not log_path.is_file():
        return "GENESIS"
    try:
        last = log_path.read_text(encoding="utf-8").strip().splitlines()[-1]
        return json.loads(last).get("row_hash", "UNKNOWN")
    except Exception:
        return "UNKNOWN"


def _row_hash(prev: str, event: dict[str, Any]) -> str:
    payload = json.dumps(event, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256((prev + payload).encode()).hexdigest()


def append_audit(event: dict[str, Any], log_path: str | Path) -> dict[str, str]:
    p = Path(log_path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        prev = _prev_hash(p)
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "prev": prev,
            "event": event,
            "row_hash": _row_hash(prev, event),
        }
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return {"prev": prev, "status": "appended"}
    except Exception as exc:
        log.warning("audit: append failed (%s)", exc)
        return {"prev": "", "status": f"failed: {exc}"}


def verify_chain(log_path: str | Path) -> tuple[bool, str]:
    p = Path(log_path)
    if not log_path or not p.is_file():
        return True, "no log yet"
    try:
        prev = "GENESIS"
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("prev") != prev:
                return False, f"chain break at line {i + 1}"
            if "row_hash" in entry:
                # Content seal: any edit (even of the last line) breaks row_hash.
                if entry["row_hash"] != _row_hash(entry["prev"], entry.get("event", {})):
                    return False, f"content tamper at line {i + 1}"
                prev = entry["row_hash"]
            else:  # legacy pre-seal entries: linkage only, head recomputed
                payload = json.dumps(entry.get("event", {}), sort_keys=True, ensure_ascii=False)
                prev = hashlib.sha256((entry["prev"] + payload).encode()).hexdigest()
        return True, f"chain intact, head={prev[:16]}"
    except Exception as exc:
        return False, f"verify error: {exc}"
