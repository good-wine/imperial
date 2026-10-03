#!/usr/bin/env bash
# Trust Caddy's local CA + wire imperial.local resolution.
# Part 1 (automatic): export the CA cert where curl/Python/browsers can use it.
# Part 2 (one manual sudo step): /etc/hosts is root-owned; print the exact line.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1/2: exporting Caddy local CA =="
docker compose cp proxy:/data/caddy/pki/authorities/local/root.crt ./config/imperial-root.crt
echo "CA -> ./config/imperial-root.crt (git-ignored, machine-local)"
echo "   curl:   curl --cacert config/imperial-root.crt https://api.imperial.local/health"
echo "   python: requests.get(url, verify='config/imperial-root.crt')"
echo "   system: sudo cp config/imperial-root.crt /usr/local/share/ca-certificates/ && sudo update-ca-certificates"
echo
echo "== 2/2: name resolution (requires sudo, once) =="
if grep -q "imperial.local" /etc/hosts 2>/dev/null; then
  echo "/etc/hosts already covers imperial.local"
else
  echo "Run ONCE:"
  echo "  sudo sh -c 'echo \"127.0.0.1 imperial.local api.imperial.local app.imperial.local\" >> /etc/hosts'"
  echo "No-sudo alternative (Chrome/Chromium only):"
  echo "  chromium --host-resolver-rules='MAP imperial.local 127.0.0.1, MAP *.imperial.local 127.0.0.1'"
fi
