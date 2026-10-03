"""Concurrency drill for the API: mixed verify/poll/export load with p50/p95.

Stdlib only (urllib + threads) so it runs anywhere, including air-gapped
hosts. Default target is the local single-node API; point at compose
(:8000) to measure the full Celery path. Verdicts are not asserted here —
this measures responsiveness under concurrency, not correctness (see
tests/ + eval_metrics for that). Exit 1 on any 5xx or p95 over budget.
"""

from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _req(
    method: str,
    base: str,
    path: str,
    *,
    files=None,
    params=None,
    body=None,
    headers=None,
    timeout=120,
):
    url = base + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    data, ctype = None, None
    if files:
        boundary = "DRILLBOUNDARY42"
        buf = io.BytesIO()
        for name, (fname, content, mt) in files.items():
            buf.write(
                f"--{boundary}\r\nContent-Disposition: form-data; "
                f'name="{name}"; filename="{fname}"\r\n'
                f"Content-Type: {mt}\r\n\r\n".encode()
            )
            buf.write(content + b"\r\n")
        for k, v in (body or {}).items():
            buf.write(
                f"--{boundary}\r\nContent-Disposition: form-data; "
                f'name="{k}"\r\n\r\n{v}\r\n'.encode()
            )
        buf.write(f"--{boundary}--\r\n".encode())
        data, ctype = buf.getvalue(), f"multipart/form-data; boundary={boundary}"
    elif body is not None:
        data, ctype = json.dumps(body).encode(), "application/json"
    t0 = time.time()
    try:
        from src.security import fetch_url

        hdrs = dict(headers or {})
        if ctype:
            hdrs["Content-Type"] = ctype
        body = fetch_url(url, method=method, data=data, headers=hdrs, timeout_s=timeout)
        return 200, time.time() - t0, body[:256]
    except urllib.error.HTTPError as e:
        return e.code, time.time() - t0, b""
    except Exception as e:
        return -1, time.time() - t0, str(e).encode()[:120]


def _pct(xs: list[float], p: float) -> float:
    return statistics.quantiles(sorted(xs), n=100)[min(int(p), 99)] if xs else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description="API concurrency drill")
    ap.add_argument("--base", default="http://127.0.0.1:8001")
    ap.add_argument("--file", default="")
    ap.add_argument("--n", type=int, default=20, help="verify uploads")
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--clearance", default="TLP:GREEN")
    ap.add_argument("--p95-budget", type=float, default=60.0)
    args = ap.parse_args()

    blob = Path(args.file).read_bytes() if args.file else bytes(20000)
    fname = Path(args.file).name if args.file else "drill.bin"
    results: list[tuple[int, float]] = []

    def _one(i: int):
        code, dt, _ = _req(
            "POST",
            args.base,
            "/api/v1/verify",
            files={"file": (fname, blob, "application/octet-stream")},
            body={"tlp": "TLP:GREEN", "owner": "load-drill"},
            headers={"X-TLP": args.clearance},
        )
        return code, dt

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for code, dt in ex.map(_one, range(args.n)):
            results.append((code, dt))
    codes = [c for c, _ in results]
    lat = [d for _, d in results]
    ok = sum(1 for c in codes if c == 200)
    bad = [c for c in codes if c >= 500 or c == -1]
    print(
        f"uploads={args.n} conc={args.concurrency} ok={ok} "
        f"5xx/conn-fail={len(bad)} "
        f"p50={_pct(lat, 50):.2f}s p95={_pct(lat, 95):.2f}s max={max(lat):.2f}s"
    )
    code, dt, _ = _req("GET", args.base, "/metrics")
    print(f"metrics: http={code} in {dt:.2f}s")
    if bad or _pct(lat, 95) > args.p95_budget:
        print("DRILL FAIL", file=sys.stderr)
        return 1
    print("DRILL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
