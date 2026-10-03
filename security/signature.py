"""Ed25519 model-signature verification for air-gapped loading (RFC 8032).

Policy: every ONNX/GGUF file listed in `models/registry.json` ships with a
detached `<model>.sig` (64-byte hex) made by `scripts/sign_models.py` with the
offline private key. `verify_file()` runs before ANY model is mmap'd into
memory; failure refuses the load (fail-closed) and the pipeline continues in
hash-only/heuristic mode with an explicit warning.

Backend order: `cryptography`/`PyNaCl` when installed, else a compact
pure-Python (stdlib `hashlib.sha512`) verifier — no dependency can be assumed
in an air-gapped bootstrap. Key provisioning stays offline by design.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

log = logging.getLogger(__name__)

# Ed25519 base point & field constants (RFC 8032 §5.1).
_P = 2**255 - 19
_D = (-121665 * pow(121666, _P - 2, _P)) % _P


def _xrecover(y: int) -> int:
    xx = (y * y - 1) * pow(_D * y * y + 1, _P - 2, _P)
    x = pow(xx, (_P + 3) // 8, _P)
    if (x * x - xx) % _P != 0:
        x = (x * pow(2, (_P - 1) // 4, _P)) % _P
    return x if x % 2 == 0 else _P - x


_Gx = _xrecover((4 * pow(5, _P - 2, _P)) % _P)
_Gy = (4 * pow(5, _P - 2, _P)) % _P
_G = (_Gx, _Gy, 1, (_Gx * _Gy) % _P)


def _edwards_add(p, q):
    x1, y1, z1, t1 = p
    x2, y2, z2, t2 = q
    a = ((y1 - x1) * (y2 - x2)) % _P
    b = ((y1 + x1) * (y2 + x2)) % _P
    c = (t1 * 2 * _D * t2) % _P
    d = (z1 * 2 * z2) % _P
    e = (b - a) % _P
    f = (d - c) % _P
    g = (d + c) % _P
    h = (b + a) % _P
    return ((e * f) % _P, (g * h) % _P, (f * g) % _P, (e * h) % _P)


def _scalarmult(p, e: int):
    q = (0, 1, 1, 0)
    while e > 0:
        if e & 1:
            q = _edwards_add(q, p)
        p = _edwards_add(p, p)
        e >>= 1
    return q


def _encodepoint(p) -> bytes:
    x, y, z, _ = p
    zi = pow(z, _P - 2, _P)
    x, y = (x * zi) % _P, (y * zi) % _P
    bits = [(y >> i) & 1 for i in range(256)]
    bits[255] = x & 1
    return bytes(sum(bits[i * 8 + j] << j for j in range(8)) for i in range(32))


def _decodepoint(s: bytes):
    y = sum(2**i * ((s[i // 8] >> (i % 8)) & 1) for i in range(255))
    x = _xrecover(y)
    if x & 1 != ((s[31] >> 7) & 1):
        x = _P - x
    return (x, y, 1, (x * y) % _P)


_L = 2**252 + 27742317777372353535851937790883648493


def _hint(m: bytes) -> int:
    h = hashlib.sha512(m).digest()
    return sum(2**i * ((h[i // 8] >> (i % 8)) & 1) for i in range(512)) % _L


def _pure_verify(pubkey: bytes, msg: bytes, sig: bytes) -> bool:
    if len(pubkey) != 32 or len(sig) != 64:
        return False
    try:
        a = _decodepoint(pubkey)
        r, s = sig[:32], int.from_bytes(sig[32:], "little")
        if s >= _L:
            return False
        r_pt = _decodepoint(r)
        h = _hint(r + pubkey + msg)
        lhs = _scalarmult(_G, s)
        rhs = _edwards_add(r_pt, _scalarmult(a, h))
        return _encodepoint(lhs) == _encodepoint(rhs)
    except Exception:
        return False


def verify_bytes(pubkey_hex: str, msg: bytes, sig_hex: str) -> bool:
    """Verify a detached Ed25519 signature. Tries libs first, then pure-python."""
    try:
        pub, sig = bytes.fromhex(pubkey_hex), bytes.fromhex(sig_hex)
    except ValueError:
        return False
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey,
        )

        Ed25519PublicKey.from_public_bytes(pub).verify(sig, msg)
        return True
    except ImportError:
        pass
    except Exception:
        return False
    try:
        import nacl.bindings

        nacl.bindings.crypto_sign_open(sig + msg, pub)
        return True
    except ImportError:
        pass
    except Exception:
        return False
    return _pure_verify(pub, msg, sig)


def verify_file(
    pubkey_hex: str, model_path: str | Path, sig_path: str | Path | None = None
) -> tuple[bool, str]:
    """Verify `<model>.sig` (hex, alongside the model by default)."""
    mp = Path(model_path)
    sp = Path(sig_path) if sig_path else mp.with_suffix(mp.suffix + ".sig")
    if not mp.is_file():
        return False, f"model missing: {mp}"
    if not sp.is_file():
        return False, f"signature missing: {sp} (refusing unsigned load)"
    try:
        sig_hex = sp.read_text(encoding="utf-8").strip().split()[0]
    except Exception as exc:
        return False, f"cannot read signature: {exc}"
    h = hashlib.sha256()
    with open(mp, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    # Signatures cover the SHA-256 digest (stable across transports).
    ok = verify_bytes(pubkey_hex, bytes.fromhex(h.hexdigest()), sig_hex)
    return ok, "signature valid" if ok else "SIGNATURE INVALID — refusing load"


def gate_model_load(model_path: str | Path, pubkey_hex: str = "") -> bool:
    """Enforcement point called before onnxruntime sessions are created.

    Returns True when loading is allowed. Empty pubkey + `MODEL_SIG_ENFORCE=1`
    refuses; empty pubkey otherwise warns (permissive bootstrap for first
    provisioning — sign models with scripts/sign_models.py, then enforce).
    """
    import os

    pubkey = pubkey_hex or os.getenv("MODEL_ED25519_PUBKEY", "")
    enforce = os.getenv("MODEL_SIG_ENFORCE", "0") == "1"
    if not pubkey:
        log.warning(
            "signature: no pubkey configured (%s)",
            "REFUSING load" if enforce else "permissive bootstrap",
        )
        return not enforce
    ok, reason = verify_file(pubkey, model_path)
    (log.info if ok else log.error)("signature: %s: %s", model_path, reason)
    return ok
