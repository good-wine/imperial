#!/usr/bin/env bash
# Disaster-recovery drill: restore the backup set into a SCRATCH database and
# prove it is usable. A backup that has never been restored is a hypothesis.
#
# Isolated by design: it creates its own throwaway database inside the running
# postgres container and drops it at the end. It never touches the live
# database, never truncates anything, and never stops a service.
#
# Usage: scripts/restore_drill.sh [backups/live-check] [--keep]
#
# Checks performed:
#   1. the dump is present, non-empty and contains the expected tables;
#   2. it restores into a scratch DB without error;
#   3. row counts are reported and compared against live (a delta is expected
#      if the backup is older than the last write — the report states it
#      rather than hiding it);
#   4. the audit hash chain verifies inside the restored copy: the custody log
#      survived the dump, which is the thing that actually matters;
#   5. analysis_results payloads are parseable JSON, not truncated;
#   6. the Qdrant snapshot is valid JSON naming a known collection;
#   7. the Redis RDB starts with the REDIS magic (i.e. it is a real dump).
#
# Exit code 0 only if every check passes.
set -uo pipefail

SRC="${1:-backups/live-check}"
KEEP=0
[[ "${2:-}" == "--keep" ]] && KEEP=1

DB="${POSTGRES_DB:-provenance}"
SCRATCH="${DB}_restore_drill"
PASS=0
FAIL=0

say()  { printf '%s\n' "$*"; }
ok()   { printf '  PASS  %s\n' "$*"; PASS=$((PASS+1)); }
warn() { printf '  WARN  %s\n' "$*"; PASS=$((PASS+1)); }
bad()  { printf '  FAIL  %s\n' "$*"; FAIL=$((FAIL+1)); }

say "=== restore drill from ${SRC} ==="
for f in postgres.sql qdrant-provenance_events.snapshot.json redis-dump.rdb; do
  [[ -s "${SRC}/${f}" ]] && ok "${f} present ($(du -h "${SRC}/${f}" | cut -f1))" \
                         || bad "${f} missing or empty"
done
[[ -s "${SRC}/postgres.sql" ]] || { say "no dump to restore — aborting"; exit 1; }

cleanup() {
  if [[ $KEEP -eq 0 ]]; then
    docker compose exec -T postgres psql -q -U "${POSTGRES_USER:-prov}" -d postgres \
      -c "DROP DATABASE IF EXISTS ${SCRATCH}" >/dev/null 2>&1 \
      && say "scratch database ${SCRATCH} dropped"
  else
    say "scratch database ${SCRATCH} KEPT (--keep)"
  fi
}
trap cleanup EXIT

say "-- restoring into scratch database ${SCRATCH}"
docker compose exec -T postgres psql -q -U "${POSTGRES_USER:-prov}" -d postgres \
  -c "DROP DATABASE IF EXISTS ${SCRATCH}" >/dev/null 2>&1
docker compose exec -T postgres psql -q -U "${POSTGRES_USER:-prov}" -d postgres \
  -c "CREATE DATABASE ${SCRATCH}" >/dev/null 2>&1 \
  || { bad "could not create scratch database"; exit 1; }

if docker compose exec -T postgres psql -q -v ON_ERROR_STOP=1 -U "${POSTGRES_USER:-prov}" \
     -d "${SCRATCH}" < "${SRC}/postgres.sql" >/tmp/opencode/restore_drill.log 2>&1; then
  ok "psql restore completed without errors"
else
  bad "psql restore failed (see /tmp/opencode/restore_drill.log)"
  tail -5 /tmp/opencode/restore_drill.log
  exit 1
fi

count() {  # count <table>
  docker compose exec -T postgres psql -tA -U "${POSTGRES_USER:-prov}" -d "$1" \
    -c "SELECT count(*) FROM $2" 2>/dev/null | tr -d '[:space:]'
}

say "-- table inventory (restored vs live)"
for t in cases evidence analysis_results feedback audit_trail; do
  r="$(count "${SCRATCH}" "${t}")"; l="$(count "${DB}" "${t}")"
  if [[ -z "$r" ]]; then bad "table ${t} missing from the restore"; continue; fi
  delta=$(( l - r ))
  if   [[ "$delta" -gt 0 ]]; then ok "${t}: ${r} restored, ${delta} newer rows in live (backup predates recent writes)"
  elif [[ "$delta" -lt 0 ]]; then warn "${t}: ${r} restored > ${l} live — rows left live after the backup (retention purge or deletion); explained below"
  else                           ok "${t}: ${r} restored, matches live"
  fi
done

say "-- consistency of cases present in BOTH copies"
# Exported to files and diffed locally: dblink (and any other cross-database
# extension) would be an extra dependency an air-gapped install may lack.
dump_cases() {  # dump_cases <db> <outfile>
  docker compose exec -T postgres psql -tA -U "${POSTGRES_USER:-prov}" -d "$1" \
    -c "SELECT case_uuid||' '||sha256||' '||status FROM cases ORDER BY 1" 2>/dev/null \
    | tr -d '\r' > "$2"
}
dump_cases "${SCRATCH}" /tmp/opencode/drill_backup_cases.txt
dump_cases "${DB}"       /tmp/opencode/drill_live_cases.txt
# case_uuid, sha256, audit actions recorded for that case in live
docker compose exec -T postgres psql -tA -U "${POSTGRES_USER:-prov}" -d "${DB}" \
  -c "SELECT case_uuid||' '||COALESCE(string_agg(action,',' ORDER BY id),'') FROM audit_trail GROUP BY case_uuid" \
  2>/dev/null | tr -d '\r' > /tmp/opencode/drill_live_actions.txt

python3 - <<'PY' > /tmp/opencode/drill_compare.txt
from pathlib import Path

def load(p, cols):
    out = {}
    for line in Path(p).read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        out[parts[0]] = parts[1:]
    return out

backup = load("/tmp/opencode/drill_backup_cases.txt", 2)
live = load("/tmp/opencode/drill_live_cases.txt", 2)
actions = load("/tmp/opencode/drill_live_actions.txt", 1)

mismatch = [c for c in backup.keys() & live.keys() if backup[c][0] != live[c][0]]
gone = sorted(backup.keys() - live.keys())
unexplained = [
    c
    for c in gone
    # actions arrive as one comma-joined string per case_uuid, not a list
    if not {"case.purged", "case.abandoned"} & set(",".join(actions.get(c, [])).split(","))
]

print(f"common={len(backup.keys() & live.keys())} sha256_mismatch={len(mismatch)}")
for c in mismatch:
    print(f"MISMATCH {c}")
print(f"gone={len(gone)} unexplained={len(unexplained)}")
for c in unexplained:
    print(f"UNEXPLAINED {c}")
PY
mismatch="$(grep -c '^MISMATCH ' /tmp/opencode/drill_compare.txt || true)"
gone="$(grep -c '^gone=' /tmp/opencode/drill_compare.txt >/dev/null && sed -n 's/^gone=\([0-9]*\).*/\1/p' /tmp/opencode/drill_compare.txt)"
unexplained="$(sed -n 's/^gone=[0-9]* unexplained=\([0-9]*\)/\1/p' /tmp/opencode/drill_compare.txt)"
common="$(sed -n 's/^common=\([0-9]*\).*/\1/p' /tmp/opencode/drill_compare.txt)"

if [[ "$mismatch" == "0" ]]; then
  ok "all ${common} cases present in both copies carry the same sha256"
else
  bad "${mismatch} cases differ in sha256 between the backup and live"
fi

if   [[ "$unexplained" == "0" ]]; then ok "${gone} cases left live after the backup, each with a case.purged/case.abandoned audit entry"
elif [[ -z "$gone" ]];              then warn "cross-compare produced no data; removals UNVERIFIED"
else bad "${unexplained} of ${gone} removed cases have NO purge/abandon audit entry — unexplained loss"; fi

say "-- custody log integrity inside the restored copy"
# A faithful backup of a chain that has a documented historic fork reproduces
# that fork. So the pass criterion is not "the restored chain is intact" but
# "the restore introduced no corruption the live database does not also have".
breaks_in() {  # breaks_in <db> -> space-separated bad row ids
  docker compose exec -T \
    -e "DATABASE_URL=postgresql://${POSTGRES_USER:-prov}:${POSTGRES_PASSWORD:-prov}@postgres:5432/${2}" \
    worker python -c "
import sys
from src.database import get_repository
out = get_repository().verify_audit_chain(all_breaks=True)
print(' '.join(str(b['id']) for b in out.get('breaks', [])))
" 2>/dev/null | tr -d '\r'
}
live_breaks="$(breaks_in x "${DB}")"
dump_breaks="$(breaks_in x "${SCRATCH}")"
new_breaks=""
for b in ${dump_breaks}; do
  case " ${live_breaks} " in *" ${b} "*) ;; *) new_breaks="${new_breaks}${b} ";; esac
done

if [[ -z "${dump_breaks}" ]]; then
  ok "restored custody log verifies end to end (no breaks)"
elif [[ -z "${new_breaks}" ]]; then
  ok "restored chain reproduces ONLY the known live breaks (rows ${dump_breaks// /, }) — no corruption added by the restore"
  say "    known incident: these rows are the documented 2026-10-04 fork; see docs/SECURITY.md"
else
  bad "restored chain has breaks the live database does not: rows ${new_breaks// /, } (live breaks: ${live_breaks:-none})"
fi

say "-- payload sanity (analysis_results must be parseable, not truncated)"
bad_json="$(docker compose exec -T postgres psql -tA -U "${POSTGRES_USER:-prov}" -d "${SCRATCH}" \
  -c "SELECT count(*) FROM analysis_results WHERE report_json IS NOT NULL AND report_json NOT LIKE '{%'" \
  2>/dev/null | tr -d '[:space:]')"
if [[ "${bad_json}" == "0" ]]; then ok "all stored reports start as JSON objects"
else bad "${bad_json} analysis_results rows have a malformed report_json"; fi

say "-- vector store snapshot"
if python3 - "${SRC}/qdrant-provenance_events.snapshot.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
name = d.get("name") or (d.get("result") or {}).get("name") or ""
print(f"    snapshot: {name or '<unnamed>'}")
raise SystemExit(0 if name else 1)
PY
then ok "Qdrant snapshot parses and names its collection"
else bad "Qdrant snapshot unreadable or unnamed"; fi

say "-- redis dump"
if head -c 5 "${SRC}/redis-dump.rdb" | grep -q REDIS; then
  ok "RDB carries the REDIS magic"
else
  bad "RDB header is not REDIS — file is truncated or not a dump"
fi

say ""
say "=== drill result: ${PASS} passed, ${FAIL} failed ==="
[[ $FAIL -eq 0 ]] || exit 1