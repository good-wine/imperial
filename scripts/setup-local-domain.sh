#!/usr/bin/env bash
# One-shot local setup for https://*.imperial.local (idempotent, re-runnable).
# Automates everything possible WITHOUT root; the single privileged step
# (/etc/hosts) goes through pkexec (graphical password prompt — the script
# never sees your password) only if mDNS already covers the names.
set -euo pipefail
cd "$(dirname "$0")/.."

HOSTS_LINE="127.0.0.1 imperial.local api.imperial.local app.imperial.local"
NAMES="imperial.local api.imperial.local app.imperial.local"

echo "== 1/3: name resolution =="
MISSING=0
for n in $NAMES; do
  getent hosts "$n" >/dev/null 2>&1 || MISSING=1
done
if [ "$MISSING" -eq 0 ]; then
  echo "/etc/hosts already covers imperial.local (or mDNS does)"
else
  echo "Requesting one-time privileged /etc/hosts entry (pkexec prompt)…"
  pkexec sh -c "grep -q imperial.local /etc/hosts || echo '$HOSTS_LINE' >> /etc/hosts"
  for n in $NAMES; do getent hosts "$n" >/dev/null || {
    echo "still unresolvable: $n"; exit 1;
  }; done
  echo "/etc/hosts updated"
fi

echo "== 2/3: export proxy CA =="
docker compose cp proxy:/data/caddy/pki/authorities/local/root.crt ./config/imperial-root.crt 2>/dev/null \
  || { echo "proxy not running? start it: docker compose --profile full up -d proxy"; exit 1; }
echo "CA -> ./config/imperial-root.crt"

echo "== 3/3: trust CA in browser NSS stores (no root) =="
CA=./config/imperial-root.crt
if [ -d "$HOME/.pki/nssdb" ]; then
  certutil -D -d "sql:$HOME/.pki/nssdb" -n "imperial.local CA" >/dev/null 2>&1 || true
  certutil -A -d "sql:$HOME/.pki/nssdb" -t "C,," -n "imperial.local CA" -i "$CA"
  echo "Chrome/Chromium: trusted"
else
  echo "Chrome/Chromium: no NSS db found, skipped"
fi
for prof in "$HOME"/.mozilla/firefox/*.default* "$HOME"/snap/firefox/common/.mozilla/firefox/*.default*; do
  [ -d "$prof" ] || continue
  certutil -D -d "sql:$prof" -n "imperial.local CA" >/dev/null 2>&1 || true
  certutil -A -d "sql:$prof" -t "C,," -n "imperial.local CA" -i "$CA"
  echo "Firefox: trusted ($prof)"
done

echo "== verify =="
curl -s -o /dev/null -w "api: %{http_code}\n" --cacert "$CA" https://api.imperial.local/health
curl -s -o /dev/null -w "app: %{http_code}\n" --cacert "$CA" https://app.imperial.local/
echo "Open https://app.imperial.local/ — no warnings expected."
