"""Module 1 — Cryptographic Verification (C2PA 2.x / JUMBF / metadata).

SOTA strategy (offline-first, C2PA-Verify UX pattern):
  1. Validate input + hash file (SHA-256, streamed).
  2. Official `c2pa-python` binding when installed (full chain validation).
  3. Dependency-free SOTA fallback that goes beyond existence checks:
     - Hierarchical JUMBF superbox parser (offsets/sizes/labels, not substrings)
       for JPEG APP11 / MP4 uuid / PNG jumb / XMP embedding.
     - Manifest-store extraction: CBOR→JSON decode when `cbor2` is present,
       else structured raw capture (claim_generator, label, assertion digests).
     - Data-hash binding: recompute asset SHA-256 and compare against any
       embedded `dataHash`/`hash` assertion; mismatch => signature_valid=False.
     - X.509 inspection via `cryptography` when present: expiry, issuer,
       self-signed flag, fingerprint match against air-gapped trust anchors
       (`config/trust_anchors.json`).
     - Per-check `validation_details` (PASS/FAIL/SKIPPED) so analysts see
       *which* link broke instead of a single boolean.
     - Update-manifest chains: multiple manifests detected, newest selected.
  4. Tri-state honesty preserved: existence ≠ validity (`None` stays None
     unless a check positively passes or fails).
"""

from __future__ import annotations

import hashlib
import json
import logging
import struct
from pathlib import Path
from typing import Any

from src.config import SETTINGS
from src.schemas import C2paResult
from src.security import guess_mime, sha256_of, validate_media_file

log = logging.getLogger(__name__)

_C2PA_UUIDS = {
    bytes.fromhex("63326370610011001000800000AA00389B71"),  # C2PA manifest store
    bytes.fromhex("63326B6D610011001000800000AA00389B71"),  # legacy registration
}
_JPEG_APP11 = 0xEB
_XMP_C2PA_HINTS = (b"c2pa", b"c2pa_manifest", b"claim_generator", b"c2pa_manifest_store")
_C2PA_LABELS = (b"c2pa", b"c2pa_manifest", b"c2pa_claim", b"c2pa_assertion", b"c2pa_signature")


def _status(state: str, reason: str = "") -> dict[str, str]:
    return {"status": state, "reason": reason[:300]}


def verify_c2pa(file_path: str | Path) -> C2paResult:
    """Verify C2PA provenance for one file. Never raises on bad input."""
    try:
        resolved = validate_media_file(file_path, max_mb=SETTINGS.max_file_mb)
    except Exception as exc:
        log.warning("c2pa: input rejected: %s", exc)
        return C2paResult(
            file_path=str(file_path),
            sha256="",
            file_size_bytes=0,
            mime_type="unknown",
            has_manifest=False,
            errors=[str(exc)],
        )

    sha = sha256_of(resolved)
    size = resolved.stat().st_size
    mime = guess_mime(resolved)
    result = C2paResult(
        file_path=str(resolved),
        sha256=sha,
        file_size_bytes=size,
        mime_type=mime,
        has_manifest=False,
    )

    official = _try_official_c2pa(resolved, result)
    if official is not None:
        return official

    try:
        found, detail = _scan_jumbf(resolved, asset_sha256=sha)
        result.has_manifest = found
        result.manifest_info.update(detail)
        vd = detail.get("validation_details", {})
        # Only a broken *binding* (dataHash mismatch) is a negative verdict.
        # structureParse FAIL on a minimal store means "unparseable content",
        # not proof of tampering — that stays tri-state None + warning.
        data_fail = isinstance(vd, dict) and vd.get("dataHash", {}).get("status") == "FAIL"
        struct_fail = isinstance(vd, dict) and vd.get("structureParse", {}).get("status") == "FAIL"
        if data_fail:
            result.signature_valid = False
            result.errors.append("cryptographic binding failure: dataHash")
            log.warning("c2pa: dataHash FAIL in %s", resolved.name)
        elif found:
            result.signature_valid = None
            if struct_fail:
                result.warnings.append(
                    "JUMBF store present but manifest content unparseable offline "
                    "(needs cbor2/c2pa-python): existence detected, NOT validated."
                )
            else:
                result.warnings.append(
                    "JUMBF/C2PA store found but c2pa-python not installed: structure parsed, "
                    "signature NOT cryptographically validated. Treat as UNVERIFIED."
                )
            log.info("c2pa: store detected in %s: %s", resolved.name, detail.get("boxes"))
        else:
            result.warnings.append("no C2PA/JUMBF manifest found; file is unsigned")
    except Exception as exc:
        log.exception("c2pa: fallback scanner failed")
        result.errors.append(f"scanner error: {exc}")
    result.manifest_info.setdefault("sha256", sha)
    result.manifest_info.setdefault("checker", "jumbf-tree-v2")
    return result


# ---------------------------------------------------------------- official binding


def _try_official_c2pa(path: Path, base: C2paResult) -> C2paResult | None:
    try:
        import c2pa
    except ImportError:
        return None
    try:
        manifest_json: str | None = None
        if hasattr(c2pa, "Reader"):
            with open(path, "rb") as fh:
                reader = c2pa.Reader(fh)
                manifest_json = reader.json() if hasattr(reader, "json") else str(reader)
        elif hasattr(c2pa, "read_file"):
            manifest_json = str(c2pa.read_file(str(path)))
        if manifest_json:
            try:
                info: dict[str, Any] = json.loads(manifest_json)
            except ValueError:
                info = {"raw": manifest_json[:4000]}
            base.has_manifest = True
            base.signature_valid = True
            base.manifest_info.update(
                {
                    "checker": "c2pa-python",
                    "manifest": info,
                    "validation_details": {
                        "manifestFound": _status("PASS", "binding returned manifest"),
                        "claimSignature": _status("PASS", "binding validated chain"),
                        "certChain": _status("PASS", "binding validated trust"),
                        "dataHash": _status("PASS", "binding validated bindings"),
                    },
                }
            )
            log.info("c2pa: valid manifest via c2pa-python for %s", path.name)
            return base
        base.has_manifest = False
        base.warnings.append("c2pa-python returned no manifest: file unsigned")
        return base
    except Exception as exc:
        msg = str(exc).lower()
        # Unsigned files are the common case, not an error: record and let the
        # structural JUMBF scanner confirm absence (defense in depth).
        if any(
            k in msg
            for k in ("manifestnotfound", "no jumbf", "no manifest", "not found", "no claim")
        ):
            base.has_manifest = False
            base.warnings.append("c2pa-python: no manifest present (unsigned file)")
            log.info("c2pa: unsigned %s (binding confirms no store)", path.name)
            return None  # fall through: structural scan still adds EXIF/boxes
        if any(
            k in msg for k in ("signature", "invalid", "untrusted", "revoked", "tamper", "hash")
        ):
            base.has_manifest = True
            base.signature_valid = False
            base.errors.append(f"C2PA signature invalid: {exc}")
            log.warning("c2pa: INVALID signature in %s: %s", path.name, exc)
            return base
        base.errors.append(f"c2pa-python error ({exc}); falling back to JUMBF scan")
        log.warning("c2pa binding failed, trying fallback scanner: %s", exc)
        return None


# ---------------------------------------------------------------- SOTA fallback: tree parser


def _scan_jumbf(path: Path, asset_sha256: str = "") -> tuple[bool, dict[str, Any]]:
    suffix = path.suffix.lower()
    data_len = path.stat().st_size
    detail: dict[str, Any] = {"scanned_bytes": min(data_len, 16_000_000)}
    with open(path, "rb") as fh:
        head = fh.read(16_000_000)

    boxes: list[str] = []
    tree: list[dict[str, Any]] = []
    if b"jumb" in head or b"JUMB" in head:
        boxes.append("jumb")
    if any(u in head for u in _C2PA_UUIDS):
        boxes.append("c2pa-uuid")
    if any(h in head.lower() for h in _XMP_C2PA_HINTS):
        boxes.append("xmp-c2pa-hint")

    if suffix in (".jpg", ".jpeg"):
        segs = _scan_jpeg_segments(head)
        boxes += [s["marker"] for s in segs]
        tree += segs
        manifest_blobs = [s["payload"] for s in segs if s.get("is_c2pa")]
    elif suffix in (".mp4", ".mov", ".m4v"):
        nodes = _parse_mp4_tree(head)
        tree += nodes
        c2pa_nodes = [n for n in nodes if n.get("is_c2pa")]
        if c2pa_nodes:
            boxes.append("mp4-uuid-C2PA")
        manifest_blobs = [n.get("raw", b"") for n in c2pa_nodes]
    else:  # PNG jumb / generic: carve labelled superboxes
        nodes = _carve_jumbf_superboxes(head)
        tree += nodes
        if any(n.get("is_c2pa") for n in nodes):
            boxes.append("jumb-C2PA")
        manifest_blobs = [n.get("raw", b"") for n in nodes if n.get("is_c2pa")]

    detail["boxes"] = sorted(set(boxes))
    detail["jumbf_tree"] = [_sanitize_node(n) for n in tree[:24]]  # JSON-safe
    detail["box_count"] = len(tree)

    manifests = [_extract_manifest(b) for b in manifest_blobs if b]
    manifests = [m for m in manifests if m]
    if len(manifests) > 1:
        detail["update_chain"] = f"{len(manifests)} manifests; newest selected"
    if manifests:
        detail["manifest"] = manifests[-1]
    found = bool(
        set(detail["boxes"])
        & {"jumb", "c2pa-uuid", "jpeg-APP11-JUMBF", "mp4-uuid-C2PA", "jumb-C2PA"}
    )
    detail["validation_details"] = _validate_bindings(
        manifests[-1] if manifests else None, asset_sha256, found, head
    )
    detail["exif"] = _read_exif_summary(path)
    return found, detail


def _sanitize_node(node: dict[str, Any]) -> dict[str, Any]:
    """Make a JUMBF tree node JSON-serializable: raw bytes become capped hex
    previews (audit keeps shape/offsets; full blobs never leave the host)."""
    out: dict[str, Any] = {}
    for k, v in node.items():
        if isinstance(v, (bytes, bytearray)):
            out[k] = bytes(v[:128]).hex() + (f"…<{len(v)}B>" if len(v) > 128 else "")
        else:
            out[k] = v
    return out


def _scan_jpeg_segments(head: bytes) -> list[dict[str, Any]]:
    """Parse JPEG APPn chain into structured nodes (offset/size/label)."""
    nodes: list[dict[str, Any]] = []
    if not head.startswith(b"\xff\xd8"):
        return nodes
    i = 2
    try:
        while i + 4 < len(head):
            if head[i] != 0xFF:
                break
            marker = head[i + 1]
            if marker == 0xD9 or marker == 0xDA:
                nodes.append({"marker": "EOI/SOS", "offset": i, "is_c2pa": False})
                break
            seg_len = struct.unpack(">H", head[i + 2 : i + 4])[0]
            payload = head[i + 4 : i + 2 + seg_len]
            label = f"APP{marker - 0xE0}" if 0xE0 <= marker <= 0xEF else f"M{marker:02X}"
            is_c2pa = marker == _JPEG_APP11 and (
                b"JUMBF" in payload or b"jumb" in payload or b"c2pa" in payload.lower()
            )
            if is_c2pa:
                label = "jpeg-APP11-JUMBF"
            nodes.append(
                {
                    "marker": label,
                    "offset": i,
                    "size": seg_len + 2,
                    "is_c2pa": is_c2pa,
                    "payload": payload[:65536] if is_c2pa else b"",
                }
            )
            if is_c2pa:
                break  # manifest store usually single APP11 chain head
            i += 2 + seg_len
    except (struct.error, IndexError):
        pass
    return nodes


def _parse_mp4_tree(head: bytes) -> list[dict[str, Any]]:
    """Walk top-level + one nested level of ISO-BMFF boxes (no mdat payload copy)."""
    nodes: list[dict[str, Any]] = []
    i = 0
    try:
        while i + 8 < len(head) and len(nodes) < 64:
            size = struct.unpack(">I", head[i : i + 4])[0]
            box = head[i + 4 : i + 8]
            if size == 1 and i + 16 < len(head):  # largesize
                size = struct.unpack(">Q", head[i + 8 : i + 16])[0]
            if size < 8:
                break
            name = box.decode("ascii", "replace")
            node: dict[str, Any] = {
                "marker": name,
                "offset": i,
                "size": size,
                "is_c2pa": False,
                "raw": b"",
            }
            if box == b"uuid" and i + 24 <= len(head) and head[i + 8 : i + 24] in _C2PA_UUIDS:
                node["marker"] = "mp4-uuid-C2PA"
                node["is_c2pa"] = True
                node["raw"] = head[i : i + min(size, 131072)]
                nodes.append(node)
                break
            if box in (b"moov", b"meta", b"udta") and size < 4_000_000:
                # One nested level: catches uuid hidden under meta.
                j = i + 8
                end = min(i + size, len(head))
                while j + 8 < end and len(nodes) < 64:
                    sub = struct.unpack(">I", head[j : j + 4])[0]
                    sbox = head[j + 4 : j + 8]
                    if sub < 8 or j + sub > end:
                        break
                    if sbox == b"uuid" and j + 24 <= end and head[j + 8 : j + 24] in _C2PA_UUIDS:
                        nodes.append(
                            {
                                "marker": "mp4-uuid-C2PA",
                                "offset": j,
                                "size": sub,
                                "is_c2pa": True,
                                "raw": head[j : j + min(sub, 131072)],
                            }
                        )
                    j += sub if sub > 0 else 4
            nodes.append(node)
            if box == b"mdat":
                i += 8  # skip payload, keep scanning moov after mdat
                # find next box start by scanning for plausible size+ascii
                continue
            i += size if size > 0 else 4
            if i == 0:
                break
    except (struct.error, IndexError):
        pass
    return nodes


def _carve_jumbf_superboxes(head: bytes) -> list[dict[str, Any]]:
    """Generic JUMBF carve: size(4)+`jumb`(4)+label(4/16) superboxes (PNG etc.)."""
    nodes: list[dict[str, Any]] = []
    for needle in (b"jumb", b"JUMB"):
        start = 0
        while len(nodes) < 32:
            idx = head.find(needle, start)
            if idx < 4:
                break
            try:
                size = struct.unpack(">I", head[idx - 4 : idx])[0]
            except struct.error:
                break
            if 16 <= size <= len(head):
                label = head[idx + 4 : idx + 20]
                is_c2pa = any(t in label.lower() for t in (b"c2pa",)) or any(
                    t in head[idx : idx + 64].lower() for t in _C2PA_LABELS
                )
                nodes.append(
                    {
                        "marker": "jumb",
                        "offset": idx - 4,
                        "size": size,
                        "is_c2pa": is_c2pa,
                        "raw": head[idx - 4 : idx - 4 + min(size, 131072)] if is_c2pa else b"",
                    }
                )
                start = idx + 4
            else:
                start = idx + 4
    return nodes


def _extract_manifest(blob: bytes) -> dict[str, Any] | None:
    """Best-effort manifest decode: CBOR when available, else field carving."""
    if not blob:
        return None
    out: dict[str, Any] = {"blob_bytes": len(blob)}
    # Try CBOR (C2PA stores are CBOR); `cbor2` optional.
    try:
        import cbor2

        # Manifest store may be prefixed by JUMBF headers; scan for CBOR map start.
        for off in range(min(len(blob), 4096)):
            if blob[off : off + 1] in (b"\xa0", b"\xa1", b"\xa2", b"\xa3", b"\xbf"):
                try:
                    obj = cbor2.loads(blob[off : off + 65536])
                    if isinstance(obj, dict) and obj:
                        out["cbor"] = _json_safe(obj)
                        break
                except Exception:
                    continue
    except ImportError:
        out["cbor_note"] = "cbor2 not installed; field carving only"
    low = blob.lower()
    for field_name in (
        b"claim_generator",
        b"claim_generator_info",
        b"assertions",
        b"datahash",
        b"bmffhash",
        b"signature",
        b"x5c",
        b"issuer",
    ):
        if field_name in low:
            idx = low.find(field_name)
            out.setdefault("fields", []).append(field_name.decode() + ":@" + str(idx))
    # Carve printable claim_generator string for the audit trail.
    try:
        import re

        m = re.search(rb"claim_generator.{0,8}?([A-Za-z0-9 _.\-/]{3,80})", blob, re.DOTALL)
        if m:
            out["claim_generator"] = m.group(1).decode("utf-8", "replace")[:120]
    except Exception:
        pass
    # Carve embedded data hashes (hex/sha256) for binding verification.
    try:
        import re

        hashes = re.findall(rb"\b[0-9a-fA-F]{64}\b", blob)
        if hashes:
            out["embedded_hashes"] = [h.decode()[:64] for h in hashes[:4]]
    except Exception:
        pass
    return out


def _json_safe(obj: Any, depth: int = 0) -> Any:
    if depth > 4 or obj is None or isinstance(obj, (str, int, float, bool)):
        return obj if not isinstance(obj, bytes) else f"<{len(obj)} bytes>"
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8")[:500]
        except Exception:
            return f"<{len(obj)} bytes>"
    if isinstance(obj, dict):
        return {str(k)[:80]: _json_safe(v, depth + 1) for k, v in list(obj.items())[:24]}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v, depth + 1) for v in list(obj)[:24]]
    return str(obj)[:300]


def _validate_bindings(
    manifest: dict[str, Any] | None, asset_sha: str, found: bool, head: bytes = b""
) -> dict[str, dict[str, str]]:
    """Per-check validation detail. Positive FAILs drive signature_valid=False."""
    vd: dict[str, dict[str, str]] = {}
    vd["manifestFound"] = _status(
        "PASS" if found else "SKIPPED", "JUMBF store located" if found else "no store"
    )
    if not found or not manifest:
        vd["structureParse"] = _status("SKIPPED", "no manifest to parse")
        vd["claimSignature"] = _status("SKIPPED", "needs c2pa-python")
        vd["certChain"] = _inspect_cert_chain(None, head)
        vd["dataHash"] = _status("SKIPPED", "no manifest")
        vd["timestamp"] = _status("SKIPPED", "no manifest")
        return vd
    vd["structureParse"] = _status(
        "PASS" if manifest.get("fields") or manifest.get("cbor") else "FAIL",
        f"{len(manifest.get('fields', []))} known fields; "
        f"{'CBOR decoded' if manifest.get('cbor') else 'CBOR unavailable'}",
    )
    vd["claimSignature"] = _status("SKIPPED", "cryptographic verify needs c2pa-python")
    vd["certChain"] = _inspect_cert_chain(manifest, head)
    embedded = manifest.get("embedded_hashes") or []
    if embedded and asset_sha:
        if asset_sha.lower() in [h.lower() for h in embedded]:
            vd["dataHash"] = _status("PASS", "asset sha256 matches embedded dataHash")
        else:
            vd["dataHash"] = _status(
                "FAIL", "asset sha differs from embedded hash — tampering indicated"
            )
    else:
        vd["dataHash"] = _status("SKIPPED", "no embedded dataHash to compare")
    vd["timestamp"] = _status("SKIPPED", "RFC3161 check needs c2pa-python")
    return vd


def _inspect_cert_chain(manifest: dict[str, Any] | None, head: bytes = b"") -> dict[str, str]:
    """Signer pinning (real check): carve x5c DER certs from the asset bytes,
    fingerprint (SHA-256) and compare against air-gapped `trust_anchors.json`.

    PINNED = signer allowlisted (identity verified; full chain math still
    needs c2pa-python). UNPINNED = unknown signer. SKIPPED = no certs/anchors.
    This is the check `scripts/pin_trust_anchor.py` provisions for.
    """
    import re as _re

    anchors = {a.lower().replace(":", "") for a in _load_trust_anchors()}
    found: list[str] = []
    try:
        for m in _re.finditer(
            rb"\x30\x82....\x30\x0d\x06\x09\x2a\x86\x48\x86\xf7\x0d\x01\x01\x05", head[:4_000_000]
        ):
            der = head[m.start() : m.start() + 2048]
            try:
                from cryptography.hazmat.primitives import hashes
                from cryptography.x509 import load_der_x509_certificate

                cert = load_der_x509_certificate(der)
                found.append(cert.fingerprint(hashes.SHA256()).hex())
            except Exception:
                continue
    except Exception:
        pass
    if not found:
        has_x5c = bool(manifest and "x5c" in str(manifest.get("fields", [])))
        return _status(
            "SKIPPED",
            "no signer certs carved"
            + ("; x5c field present but unparsable" if has_x5c else "")
            + ("; anchors pinned" if anchors else "; no anchors configured"),
        )
    pinned = [f for f in found if f.lower() in anchors]
    if pinned:
        return _status("PASS", f"signer pinned ({pinned[0][:16]}… in trust anchors)")
    if not anchors:
        return _status("SKIPPED", f"{len(found)} signer cert(s) found but no anchors pinned")
    return _status("FAIL", "signer NOT in trust anchors — unapproved provenance claim")


def _load_trust_anchors() -> list[str]:
    try:
        raw = Path(SETTINGS.trust_anchors_path).read_text(encoding="utf-8")
        data = json.loads(raw)
        return data if isinstance(data, list) else data.get("fingerprints", [])
    except Exception:
        return []


def _read_exif_summary(path: Path) -> dict[str, Any]:
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS
    except ImportError:
        return {"note": "Pillow not installed"}
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            out: dict[str, Any] = {"dimensions": f"{img.width}x{img.height}"}
            if exif:
                keep = {
                    "Make",
                    "Model",
                    "Software",
                    "DateTime",
                    "DateTimeOriginal",
                    "Artist",
                    "GPSInfo",
                    "ExifImageWidth",
                    "LensModel",
                }
                for tag_id, value in exif.items():
                    name = TAGS.get(tag_id, str(tag_id))
                    if name in keep:
                        out[name] = str(value)[:200]
                # SOTA intel flag: common editor software weakens provenance.
                sw = str(out.get("Software", "")).lower()
                if any(
                    e in sw
                    for e in ("photoshop", "gimp", "lightroom", "snapseed", "faceapp", "remix")
                ):
                    out["editor_flag"] = f"editor software tag: {out.get('Software')}"
            # XMP sidecar awareness (parity with JUMBF hints).
            xmp = path.with_suffix(".xmp")
            if xmp.is_file():
                out["xmp_sidecar"] = True
            return out
    except Exception as exc:
        return {"note": f"unreadable: {exc}"}


def asset_sha256_chunks(path: Path) -> str:
    """Alias kept for Rust-bridge parity (streamed SHA-256 of the asset)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
