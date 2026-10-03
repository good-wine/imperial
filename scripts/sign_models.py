"""Offline model provisioning: Ed25519 sign every registry model.

Usage (on the CONNECTED provisioning station, private key never leaves it):
    python scripts/sign_models.py --generate-keypair config/model_signing/
    python scripts/sign_models.py --sign-all
    # then copy models/*.sig + MODEL_ED25519_PUBKEY (env) to the air-gapped host.

Verification on the air-gapped host happens in security/signature.py before
any model is mmap'd. Set MODEL_SIG_ENFORCE=1 to refuse unsigned loads.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _keypair(args) -> int:  # type: ignore[no-untyped-def]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        priv = Ed25519PrivateKey.generate()
        pub = priv.public_key()
        (out / "ed25519.sk").write_bytes(
            priv.private_bytes(
                serialization.Encoding.Raw,
                serialization.PrivateFormat.Raw,
                serialization.NoEncryption(),
            )
        )
        pub_hex = pub.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex()
        (out / "ed25519.pub.hex").write_text(pub_hex + "\n")
        print(f"pubkey (set MODEL_ED25519_PUBKEY to this):\n{pub_hex}")
        return 0
    except ImportError:
        print("cryptography lib required for keygen (pip install cryptography)", file=sys.stderr)
        return 1


def _sign_all(args) -> int:  # type: ignore[no-untyped-def]
    import hashlib

    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    except ImportError:
        print("cryptography lib required for signing", file=sys.stderr)
        return 1
    sk_path = Path(args.key)
    registry = json.loads(Path(args.registry).read_text(encoding="utf-8"))
    priv = Ed25519PrivateKey.from_private_bytes(sk_path.read_bytes())
    n = 0
    for m in registry.get("models", []):
        mp = Path(m["path"])
        if not mp.is_file():
            print(f"skip (missing): {mp}")
            continue
        h = hashlib.sha256()
        with open(mp, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        sig = priv.sign(bytes.fromhex(h.hexdigest())).hex()
        (mp.parent / (mp.name + ".sig")).write_text(sig + "\n")
        print(f"signed: {mp}")
        n += 1
    print(f"{n} model(s) signed")
    return 0


def main() -> int:  # type: ignore[no-untyped-def]
    ap = argparse.ArgumentParser(description="Ed25519 model provisioning")
    ap.add_argument("--generate-keypair", action="store_true")
    ap.add_argument("--out", default="config/model_signing")
    ap.add_argument("--sign-all", action="store_true")
    ap.add_argument("--key", default="config/model_signing/ed25519.sk")
    ap.add_argument("--registry", default="models/registry.json")
    args = ap.parse_args()
    if args.generate_keypair:
        return _keypair(args)
    if args.sign_all:
        return _sign_all(args)
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
