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
_JWKS_FAILED_AT: float = 0.0


def _b64url(data: str) -> bytes:
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


def _jwks(issuer: str) -> dict[str, Any]:
    global _JWKS, _JWKS_FAILED_AT
    now = time.time()
    ts, cached = _JWKS
    if now - ts < 600 and cached:
        return cached
    # Negative cache: a just-failed fetch must not be retried on every request.
    # The fetch has a 10 s timeout, so without this an IdP blip turns into a
    # request-stall stampede (every caller blocks up to 10 s).
    if now - _JWKS_FAILED_AT < 30:
        raise OSError("JWKS fetch failed recently; backing off")
    url = issuer.rstrip("/") + "/.well-known/jwks.json"
    from src.security import fetch_url

    try:
        doc = json.loads(fetch_url(url, timeout_s=10).decode())
    except Exception:
        _JWKS_FAILED_AT = now
        raise
    _JWKS = (now, doc)
    return doc


def verify_bearer(token: str, issuer: str = "", audience: str = "") -> dict[str, Any]:
    """Validate JWT, return claims.

    Raises `ValueError` on a token that fails validation. Note a JWKS fetch
    failure surfaces as `OSError` (after a short negative cache); callers that
    guard on identity must fail closed on *any* exception, not only `ValueError`.
    """
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
    """Resolve analyst identity: Bearer sub when OIDC is on, else fallback.

    With OIDC enabled an unauthenticated request is **rejected**: a
    self-declared `owner`/`X-Analyst`/form field would otherwise be written into
    the case owner and the hash-chained audit trail without verification.
    `OIDC_ALLOW_ANONYMOUS=1` restores the fallback for a trusted single-user or
    lab deployment; it is not a substitute for authenticating the ingress.
    """
    issuer = os.getenv("OIDC_ISSUER", "")
    if issuer and authorization and authorization.startswith("Bearer "):
        try:
            claims = verify_bearer(authorization[7:], issuer, os.getenv("OIDC_AUDIENCE", ""))
            sub = str(claims.get("preferred_username") or claims.get("sub", fallback))[:128]
            log.info("oidc: authenticated %s", sub)
            return sub
        except Exception as exc:
            # Fail closed on *any* failure (bad token, unreachable JWKS, missing
            # lib): an unverifiable identity must not be treated as authenticated.
            log.warning("oidc: rejecting request (%s)", exc)
            raise PermissionError(f"invalid token: {exc}") from exc
    if issuer and os.getenv("OIDC_ALLOW_ANONYMOUS", "0") != "1":
        raise PermissionError(
            "authentication required: OIDC is enabled and no bearer token was sent"
        )
    return (fallback or "analyst")[:128]


def clearance_binding_enabled() -> bool:
    """True when the deployment opts into binding TLP clearance to identity.

    Off by default so field-laptop mode (no IdP, analyst-declared clearance)
    keeps working; on for classified networks, where a self-declared `X-TLP`
    would otherwise let any caller read any record.
    """
    return bool(os.getenv("OIDC_ISSUER")) and os.getenv("OIDC_BIND_CLEARANCE", "0") == "1"


def bound_clearance(authorization: str | None, declared: str) -> str:
    """Cap a declared TLP clearance by the bearer token's clearance claim.

    Fail-closed: with binding enabled, a missing or invalid token — or a token
    with no usable claim — yields the lowest clearance (`TLP:CLEAR`), so a
    caller can never *escalate* above what their identity backs. The claim name
    is configurable via `OIDC_TLP_CLAIM` (default `tlp_clearance`).

    An empty `OIDC_AUDIENCE` also fails closed: without an audience check a
    token minted for any other service at the same issuer (e.g. the dashboard
    client) would be accepted — a token-confusion escalation.
    """
    from src.tlp import clamp_clearance

    if not clearance_binding_enabled():
        return declared
    if not authorization or not authorization.startswith("Bearer "):
        return "TLP:CLEAR"
    issuer = os.getenv("OIDC_ISSUER", "")
    audience = os.getenv("OIDC_AUDIENCE", "")
    if not audience:
        log.warning("oidc: OIDC_AUDIENCE empty with clearance binding; capping to TLP:CLEAR")
        return "TLP:CLEAR"
    claim = os.getenv("OIDC_TLP_CLAIM", "tlp_clearance")
    try:
        claims = verify_bearer(authorization[7:], issuer, audience)
    except Exception as exc:
        # Catch broadly, not just ValueError: an unreachable JWKS raises OSError.
        # A security control must fail closed, never 500 into an open default.
        log.warning("oidc: clearance binding rejecting request (%s)", exc)
        return "TLP:CLEAR"
    return clamp_clearance(declared, str(claims.get(claim, "TLP:CLEAR")))
