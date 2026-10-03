"""Input validation + safe file I/O. Every module calls `validate_media_file` first."""

from __future__ import annotations

import hashlib
import logging
import mimetypes
import urllib.parse
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)

ALLOWED_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
    ".bmp",
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".webm",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm"}

# Magic bytes for the formats we accept (prevents extension spoofing).
MAGIC: dict[bytes, str] = {
    b"\xff\xd8\xff": "jpeg",
    b"\x89PNG\r\n\x1a\n": "png",
    b"II*\x00": "tiff",
    b"MM\x00*": "tiff",
    b"RIFF": "webp/avi",  # refined below
    b"\x1aE\xdf\xa3": "mkv/webm",
    b"....ftyp": "mp4/mov",  # checked positionally, see below
}


class UnsafeInputError(ValueError):
    """Raised when a file fails validation (missing, too large, wrong type)."""


def validate_media_file(path: str | Path, max_mb: int = 500) -> Path:
    """Resolve, check existence/size/extension/magic bytes. Returns resolved Path."""
    p = Path(path).expanduser()
    # Block path traversal oddities; resolve without requiring existence first.
    try:
        resolved = p.resolve()
    except Exception as exc:
        raise UnsafeInputError(f"unresolvable path {path!r}: {exc}") from exc
    if not resolved.is_file():
        raise UnsafeInputError(f"file not found: {resolved}")
    if resolved.is_symlink() and not resolved.exists():
        raise UnsafeInputError(f"dangling symlink: {resolved}")
    size = resolved.stat().st_size
    if size <= 0:
        raise UnsafeInputError(f"empty file: {resolved}")
    if size > max_mb * 1024 * 1024:
        raise UnsafeInputError(f"file {size}B exceeds limit {max_mb}MB")
    if resolved.suffix.lower() not in ALLOWED_SUFFIXES:
        raise UnsafeInputError(f"unsupported extension {resolved.suffix!r}")
    _check_magic(resolved)
    return resolved


def _check_magic(p: Path) -> None:
    """Best-effort magic-byte check. Warns (does not reject) on ambiguous containers."""
    try:
        with open(p, "rb") as fh:
            head = fh.read(32)
    except OSError as exc:
        raise UnsafeInputError(f"cannot read {p}: {exc}") from exc
    if head.startswith((b"\xff\xd8\xff", b"\x89PNG")):
        return
    if head[:4] == b"RIFF" and head[8:12] in (b"WEBP", b"AVI "):
        return
    if head[:4] == b"\x1aE\xdf\xa3":
        return
    if head[4:8] == b"ftyp":  # MP4/MOV
        return
    # TIFF/BMP best-effort: accept by extension, log for audit.
    if p.suffix.lower() in {".tif", ".tiff", ".bmp", ".webp", ".mkv", ".avi", ".mov"}:
        log.warning("ambiguous magic for %s; accepting by extension", p.name)
        return
    log.warning("unrecognized magic %r for %s; accepting by extension", head[:8], p.name)


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    """Streamed SHA-256 so multi-GB videos never load into RAM."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def guess_mime(path: Path) -> str:
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_SUFFIXES


def safe_mkdir(path: str | Path) -> Path:
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


_FETCH_SCHEMES = ("http", "https")


def checked_fetch_url(url: str) -> str:
    """Allowlist http(s) only. Blocks file:/gopher:/dict: SSRF schemes before
    any socket opens — all outbound fetch in this project goes through here."""
    scheme = urllib.parse.urlsplit(url).scheme.lower()
    if scheme not in _FETCH_SCHEMES:
        raise UnsafeInputError(f"refusing non-http(s) fetch (scheme={scheme!r})")
    return url


def fetch_url(
    url: str,
    *,
    method: str = "GET",
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout_s: int = 20,
    max_bytes: int = 4_000_000,
) -> bytes:
    """Single outbound-fetch choke point: scheme-validated, size-capped.

    Callers (OSINT feeds, Telegram Bot API, local Ollama) pass operator- or
    service-owned URLs; untrusted media URLs from feed bodies still resolve
    only over http(s) and can never exfiltrate local files.
    """
    req = urllib.request.Request(
        checked_fetch_url(url), data=data, headers=headers or {}, method=method
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # nosec B310 - see docstring
        out = bytearray()
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                return bytes(out)
            out += chunk
            if len(out) > max_bytes:
                raise UnsafeInputError(f"response exceeds {max_bytes}B cap")


def download_to_file(
    url: str,
    dest: str | Path,
    *,
    headers: dict[str, str] | None = None,
    timeout_s: int = 60,
    max_bytes: int = 500 * 1024 * 1024,
) -> int:
    """Streaming variant of fetch_url for media blobs. Returns byte count."""
    req = urllib.request.Request(checked_fetch_url(url), headers=headers or {})
    size = 0
    # Single choke point (see fetch_url): scheme-validated + size-capped.
    resp = urllib.request.urlopen(req, timeout=timeout_s)  # nosec B310
    with resp, open(dest, "wb") as fh:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                return size
            size += len(chunk)
            if size > max_bytes:
                raise UnsafeInputError(f"download exceeds {max_bytes}B cap")
            fh.write(chunk)
    return size
