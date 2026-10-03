"""Prune orphan case directories (disk state without DB rows).

Root cause it prevents: ad-hoc `rm -rf` loops on case storage once deleted
more than the orphan set (empty-expansion / stale-snapshot class of error).
This tool is the ONLY sanctioned way to remove case dirs outside retention:

  python scripts/prune_orphans.py --dry-run            # report only (default)
  python scripts/prune_orphans.py --execute --i-understand-that-this-deletes-evidence

Each deletion is audit-logged (actor ops, action orphans.pruned). Directory
names must parse as UUIDs or they are left alone and reported.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _is_uuid(name: str) -> bool:
    try:
        return str(uuid.UUID(name)) == name.lower()
    except ValueError:
        return False


def find_orphans(cases_dir: Path, known: set[str]) -> tuple[list[Path], list[str]]:
    """Return (orphan_dirs, skipped_non_uuid). Never deletes."""
    orphans, skipped = [], []
    if not cases_dir.is_dir():
        return orphans, skipped
    for entry in sorted(cases_dir.iterdir()):
        if not entry.is_dir():
            continue
        if not _is_uuid(entry.name):
            skipped.append(entry.name)
            continue
        if entry.name not in known:
            orphans.append(entry)
    return orphans, skipped


def main() -> int:
    ap = argparse.ArgumentParser(description="Prune orphan case directories")
    ap.add_argument("--cases-dir", default="data/cases")
    ap.add_argument("--dry-run", action="store_true", default=True)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument(
        "--i-understand-that-this-deletes-evidence", action="store_true", dest="confirm"
    )
    args = ap.parse_args()

    from src.database import get_repository

    repo = get_repository()
    orphans, skipped = find_orphans(Path(args.cases_dir), repo.all_case_uuids())
    print(f"orphans: {len(orphans)}  skipped-non-uuid: {len(skipped)}")
    for o in orphans[:20]:
        print(f"  orphan: {o.name}")
    if len(orphans) > 20:
        print(f"  ... and {len(orphans) - 20} more")
    if not args.execute:
        print("dry-run: nothing deleted (pass --execute --i-understand-... to delete)")
        return 0
    if not args.confirm:
        print(
            "refusing: --execute requires --i-understand-that-this-deletes-evidence",
            file=sys.stderr,
        )
        return 2
    removed = 0
    for o in orphans:
        if _rmtree_quiet(o):
            removed += 1
    repo.audit(
        actor="ops", action="orphans.pruned", detail={"removed": removed, "of": len(orphans)}
    )
    print(f"removed {removed}/{len(orphans)} (audited as orphans.pruned)")
    return 0 if removed == len(orphans) else 1


def _rmtree_quiet(path: Path) -> bool:
    """Best-effort recursive delete (isolated try so the sweep loop stays clean)."""
    try:
        shutil.rmtree(path)
        return True
    except Exception as exc:
        print(f"FAILED {path.name}: {exc}", file=sys.stderr)
        return False


if __name__ == "__main__":
    raise SystemExit(main())
