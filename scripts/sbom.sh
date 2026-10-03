#!/usr/bin/env bash
# Supply-chain SBOM for releases/air-gap review.
# Prefers Syft (CycloneDX) when installed; falls back to a pinned pip freeze.
# Usage: scripts/sbom.sh [.venv] [reports/sbom.json]
set -euo pipefail
VENV="${1:-.venv}"
OUT="${2:-reports/sbom.json}"
mkdir -p "$(dirname "$OUT")"
if command -v syft >/dev/null 2>&1; then
  syft "$VENV" -o cyclonedx-json="$OUT"
  echo "SBOM (CycloneDX) -> $OUT"
else
  "$VENV/bin/python" -m pip freeze --exclude-editable > "${OUT%.json}.freeze.txt"
  echo "syft absent; pip freeze -> ${OUT%.json}.freeze.txt"
fi
