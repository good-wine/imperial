"""Optional OIDC identity (Keycloak/Entra) for classified networks.

Disabled by default (`OIDC_ISSUER` empty): identity falls back to the
`X-Analyst` header or the `analyst`/`owner` form field (field-laptop mode).
Enabled: `Authorization: Bearer <JWT>` is validated RS256 against the
issuer JWKS (cached 10 min); `sub` becomes the analyst ID in audit rows.
No new dependency: `cryptography` + stdlib urllib (both already required).

Keycloak compose profile: see docker-compose `keycloak` service + docs.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from typing import Any

log = logging.getLogger(__name__)

_JWKS: tuple[float, dict[str, Any]] = (0.0, {})


def _b64url(data: str) -> bytes:
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


def _jwks(issuer: str) -> dict[str, Any]:
    global _JWKS
    ts, cached = _JWKS
    if time.time() - ts < 600 and cached:
        return cached
    url = issuer.rstrip("/") + "/.well-known/jwks.json"
    from src.security import fetch_url

    doc = json.loads(fetch_url(url, timeout_s=10).decode())
    _JWKS = (time.time(), doc)
    return doc


def verify_bearer(token: str, issuer: str = "", audience: str = "") -> dict[str, Any]:
    """Validate JWT, return claims. Raises ValueError on any failure (fail-closed)."""
    if not issuer:
        raise ValueError("OIDC disabled")
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.padding import PKCS1v15
        from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicNumbers
        from cryptography.hazmat.primitives.hashes import SHA256
    except ImportError as exc:
        raise ValueError("cryptography lib required for OIDC") from exc
    try:
        h_b64, p_b64, s_b64 = token.split(".")
        header = json.loads(_b64url(h_b64))
        payload = json.loads(_b64url(p_b64))
        sig = _b64url(s_b64)
    except Exception as exc:
        raise ValueError(f"malformed JWT: {exc}") from exc
    if header.get("alg") != "RS256":
        raise ValueError("only RS256 accepted")
    if payload.get("iss", "").rstrip("/") != issuer.rstrip("/"):
        raise ValueError("issuer mismatch")
    if audience and payload.get("aud") != audience:
        raise ValueError("audience mismatch")
    if payload.get("exp", 0) < time.time():
        raise ValueError("token expired")
    keys = [k for k in _jwks(issuer).get("keys", []) if k.get("kid") == header.get("kid")]
    if not keys:
        raise ValueError("signing key not in JWKS")
    jwk = keys[0]
    n = int.from_bytes(_b64url(jwk["n"]), "big")
    e = int.from_bytes(_b64url(jwk["e"]), "big")
    pub = RSAPublicNumbers(e, n).public_key()
    try:
        pub.verify(sig, f"{h_b64}.{p_b64}".encode(), PKCS1v15(), SHA256())
    except InvalidSignature as exc:
        raise ValueError("bad signature") from exc
    return payload


def analyst_id(authorization: str | None = None, fallback: str = "analyst") -> str:
    """Resolve analyst identity: Bearer sub when OIDC is on, else fallback."""
    issuer = os.getenv("OIDC_ISSUER", "")
    if issuer and authorization and authorization.startswith("Bearer "):
        try:
            claims = verify_bearer(authorization[7:], issuer, os.getenv("OIDC_AUDIENCE", ""))
            sub = str(claims.get("preferred_username") or claims.get("sub", fallback))[:128]
            log.info("oidc: authenticated %s", sub)
            return sub
        except ValueError as exc:
            log.warning("oidc: rejecting request (%s)", exc)
            raise PermissionError(f"invalid token: {exc}") from exc
    return (fallback or "analyst")[:128]
