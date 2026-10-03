"""Pin a C2PA signer trust anchor from a known-good signed file.

Without pinned anchors the crypto leg can never report VALID (correct
fail-closed default). On the CONNECTED provisioning station:

    python scripts/pin_trust_anchor.py signed-by-agency.jpg --alias agency-cam

extracts the leaf-signer SHA-256 fingerprint (via c2pa-python manifest or
raw x5c/carved cert, verified with `cryptography`) and appends it to
`config/trust_anchors.json`. Copy that file to the air-gapped host; from
then on, `_inspect_cert_chain()` can report PINNED vs UNPINNED signers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _leaf_fingerprint(asset: Path) -> tuple[str, str]:
    """Return (fingerprint_hex, method). Tries binding manifest, then carve."""
    detail = ""
    try:
        import c2pa  # type: ignore[import-not-found]

        if hasattr(c2pa, "Reader"):
            with open(asset, "rb") as fh:
                manifest = json.loads(c2pa.Reader(fh).json())  # type: ignore[attr-defined]
            detail = json.dumps(manifest)[:2000]
    except Exception as exc:
        detail = f"binding unreadable: {exc}"
    # Carve DER certs (30 82 .. 30 0d 06 09 2a 86 48 86 f7 0d 0101 05) for x5c.
    raw = asset.read_bytes()
    import re

    for m in re.finditer(rb"\x30\x82....\x30\x0d\x06\x09\x2a\x86\x48\x86\xf7\x0d\x01\x01\x05", raw):
        der = raw[m.start() : m.start() + 2048]
        try:
            from cryptography.hazmat.primitives import hashes
            from cryptography.x509 import load_der_x509_certificate

            cert = load_der_x509_certificate(der)
            fp = cert.fingerprint(hashes.SHA256()).hex()
            return fp, f"carved-x5c subject={cert.subject.rfc4514_string()[:80]}"
        except Exception:
            continue
    raise RuntimeError(f"no extractable signer cert ({detail[:120]})")


def main() -> int:  # type: ignore[no-untyped-def]
    ap = argparse.ArgumentParser(description="Pin C2PA signer trust anchor")
    ap.add_argument("file", help="known-good C2PA-signed asset")
    ap.add_argument("--alias", default="", help="human label for the anchor")
    ap.add_argument("--anchors", default="config/trust_anchors.json")
    args = ap.parse_args()
    try:
        fp, method = _leaf_fingerprint(Path(args.file))
    except Exception as exc:
        print(f"pin failed: {exc}", file=sys.stderr)
        return 1
    apath = Path(args.anchors)
    try:
        data = json.loads(apath.read_text(encoding="utf-8"))
    except Exception:
        data = {"fingerprints": []}
    fps = data.get("fingerprints", [])
    if fp not in fps:
        fps.append(fp)
    data["fingerprints"] = fps
    if args.alias:
        data.setdefault("aliases", {})[args.alias] = fp[:16]
    apath.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"pinned {fp[:32]}… ({method}) -> {apath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
