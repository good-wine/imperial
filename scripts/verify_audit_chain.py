"""Verify the audit hash chain held in the database.

The JSONL mirror (`src/audit.py verify_chain`) seals whole events; this
checks the `audit_trail` table, which seals each row's `detail` payload into
`prev_hash + detail`. Run it after any restore, incident or privilege change,
and wire it into monitoring: a non-zero exit means the custody log can no
longer be trusted.

    python scripts/verify_audit_chain.py            # against DATABASE_URL
    python scripts/verify_audit_chain.py --json     # machine-readable

Honest limits (also printed on success): `actor`, `action`, `case_uuid` and
`ts` are not sealed by the DB row hash, and a truncated TAIL cannot be
detected by any chained structure. Use the JSONL mirror plus off-host backups
for those.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify the audit_trail hash chain")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument(
        "--since-id",
        type=int,
        default=0,
        help="verify only rows after this id (use to check the post-fix chain)",
    )
    ap.add_argument(
        "--all-breaks",
        action="store_true",
        help="list every bad row instead of stopping at the first",
    )
    args = ap.parse_args()

    from src.database import get_repository

    repo = get_repository()
    try:
        out = repo.verify_audit_chain(since_id=args.since_id, all_breaks=args.all_breaks)
    except Exception as exc:  # CLI must report the failure, not traceback
        print(f"audit chain: UNVERIFIABLE ({type(exc).__name__}: {exc})")
        return 2

    if args.json:
        print(json.dumps(out, indent=2))
    elif out["ok"]:
        print(f"audit chain OK: {out['rows']} rows, every detail payload sealed")
    else:
        print(f"audit chain BROKEN at row {out['broken_at']} of {out['rows']}: {out['reason']}")
        for b in out.get("breaks", [])[1:]:
            print(f"  also broken at row {b['id']}: {b['reason']}")

    if out["ok"] and not args.json:
        print(
            "note: this seals the detail payload only. actor/action/ts are not in the row hash, "
            "and a truncated tail is undetectable — keep the JSONL mirror and off-host backups."
        )
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
