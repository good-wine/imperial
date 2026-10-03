#!/usr/bin/env bash
# Offline disaster recovery: Postgres dump + Qdrant snapshots + Redis RDB.
# No extra images: uses the running compose services only.
# Usage: scripts/backup_volumes.sh [backups/2026-10-03]
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-backups/$(date +%F)}"
mkdir -p "$OUT"
: "${POSTGRES_USER:=prov}" "${POSTGRES_DB:=provenance}"

echo "-> pg_dump $POSTGRES_DB"
docker compose exec -T postgres pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
  > "$OUT/postgres.sql"

echo "-> qdrant snapshots (per collection)"
for col in $(curl -s http://localhost:6333/collections \
  | python3 -c "import json,sys; print(' '.join(c['name'] for c in json.load(sys.stdin)['result']['collections']))" 2>/dev/null || true); do
  curl -s -X POST "http://localhost:6333/collections/$col/snapshots" \
    -o "$OUT/qdrant-$col.snapshot.json"
  echo "   snapshot: $col"
done

echo "-> redis RDB"
docker compose exec -T redis redis-cli --rdb /tmp/dump.rdb >/dev/null
docker compose cp redis:/tmp/dump.rdb "$OUT/redis-dump.rdb"

ls -la "$OUT"
echo "backup complete: $OUT"
echo "RESTORE: psql < postgres.sql | qdrant PUT /collections/<c>/snapshots/recover | redis restart with RDB"
