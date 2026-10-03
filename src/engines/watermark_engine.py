"""AI-watermark & generative-tool metadata detection.

Two tiers, honestly separated:
  metadata (strong)  PNG tEXt/iTXt (ComfyUI workflow JSON, A1111 `parameters`),
                     JPEG COM/APP segments, EXIF Software/UserComment, XMP
                     (`xmp:CreatorTool` Midjourney/DALL-E/Firefly tags).
  SynthID (screening) SynthID v1 embeds in pixels (needs DeepMind weights);
                     without `models/synthid.onnx` we report NEEDS_MODEL and
                     the metadata verdict stands alone — never a fake negative.

Any metadata hit naming a generator flips the file to AI-generated provenance
regardless of pixels: the LLM leg and STIX indicators consume `verdict`.
"""

from __future__ import annotations

import contextlib
import logging
import struct
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_GENERATOR_HINTS = (
    "comfyui",
    "automatic1111",
    "stable diffusion",
    "a1111",
    "invokeai",
    "midjourney",
    "dall-e",
    "dalle",
    "firefly",
    "leonardo",
    "ideogram",
    "synthid",
    "civitai",
    "fooocus",
    "swarmui",
)


@dataclass
class WatermarkResult:
    available: bool = True
    ai_generated: bool | None = None  # True/False/None(unknown)
    generator: str | None = None
    metadata_hits: list[str] = field(default_factory=list)
    synthid: str = "NOT_CHECKED"  # DETECTED | NOT_DETECTED | NEEDS_MODEL | NOT_CHECKED
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_watermark(path: str | Path) -> WatermarkResult:
    res = WatermarkResult()
    p = Path(path)
    try:
        texts = _extract_text_chunks(p)
    except Exception as exc:
        res.errors.append(f"chunk parse failed: {exc}")
        return res
    for key, val in texts:
        kl, vl = key.lower(), val.lower()
        if any(h in kl or h in vl for h in _GENERATOR_HINTS):
            res.metadata_hits.append(f"{key}={val[:160]}")
    # Structural fingerprints (no brand string needed):
    # ComfyUI = prompt+workflow JSON with node class_type graph.
    blob_keys = {k.lower() for k, _ in texts}
    blob_all = "\n".join(f"{k}={v}" for k, v in texts).lower()
    if {"prompt", "workflow"} <= blob_keys and "class_type" in blob_all:
        res.metadata_hits.append("structural:ComfyUI node graph (prompt+workflow+class_type)")
    # A1111 = `parameters` tEXt with sampler/steps grammar.
    if "parameters" in blob_keys and ("steps:" in blob_all or "sampler:" in blob_all):
        res.metadata_hits.append("structural:Automatic1111 parameters grammar")
    if res.metadata_hits:
        res.ai_generated = True
        res.generator = _name_generator(" ".join(res.metadata_hits))
        res.warnings.append(f"generative-tool metadata: probable {res.generator} output")
        log.info("watermark: %s generator=%s", p.name, res.generator)
    # SynthID pixel check: weights-gated, honest status otherwise.
    res.synthid = _synthid_screen(p)
    if res.synthid == "DETECTED":
        res.ai_generated = True
        res.warnings.append("SynthID pixel watermark detected (Google DeepMind scheme)")
    elif res.synthid == "NEEDS_MODEL" and not res.metadata_hits:
        res.warnings.append("SynthID screen unavailable offline (no models/synthid.onnx)")
    if res.ai_generated is None and not res.metadata_hits:
        res.ai_generated = False  # no generator traces found (metadata tier)
    return res


def _name_generator(blob: str) -> str:
    b = blob.lower()
    for name in (
        "comfyui",
        "midjourney",
        "automatic1111",
        "a1111",
        "dall-e",
        "dalle",
        "firefly",
        "stable diffusion",
        "invokeai",
        "ideogram",
        "leonardo",
    ):
        if name in b:
            return "automatic1111" if name == "a1111" else name
    return "unknown-generator"


def _png_text_chunks(raw: bytes, out: list[tuple[str, str]]) -> None:
    """Append PNG tEXt/iTXt/zTXt chunks (mutates out, caps at 32)."""
    i = 8
    while i + 12 < len(raw) and len(out) < 32:
        try:
            ln = struct.unpack(">I", raw[i : i + 4])[0]
            typ = raw[i + 4 : i + 8].decode("ascii", "replace")
        except (struct.error, IndexError):
            break
        body = raw[i + 8 : i + 8 + min(ln, 1_000_000)]
        if typ in ("tEXt", "iTXt", "zTXt"):
            kw, _, rest = body.partition(b"\x00")
            with contextlib.suppress(Exception):
                out.append(
                    (
                        kw.decode("utf-8", "replace")[:80],
                        rest[:2000].decode("utf-8", "replace"),
                    )
                )
        i += 12 + ln
        if typ == "IEND":
            break


def _jpeg_text_chunks(raw: bytes, out: list[tuple[str, str]]) -> None:
    """Append JPEG COM segments (mutates out, caps at 32)."""
    i = 2
    while i + 4 < len(raw) and len(out) < 32:
        if raw[i] != 0xFF:
            break
        marker = raw[i + 1]
        if marker in (0xD9, 0xDA):
            break
        try:
            ln = struct.unpack(">H", raw[i + 2 : i + 4])[0]
        except struct.error:
            break
        if marker == 0xFE:  # COM
            out.append(
                (
                    "JPEG-COM",
                    raw[i + 4 : i + 2 + min(ln, 4000)].decode("utf-8", "replace")[:2000],
                )
            )
        i += 2 + ln


def _exif_text_chunks(p: Path, raw: bytes, out: list[tuple[str, str]]) -> None:
    """Append XMP tag snippets + EXIF software tags (best-effort)."""
    low = raw.lower()
    for tag in (b"xmp:creatortool", b"dc:format", b"c2pa"):
        idx = low.find(tag)
        if idx >= 0:
            out.append((tag.decode(), raw[idx : idx + 200].decode("utf-8", "replace")[:200]))
    try:  # EXIF software tags via Pillow (best-effort).
        from PIL import Image
        from PIL.ExifTags import TAGS

        with Image.open(p) as img:
            for tid, val in (img.getexif() or {}).items():
                if TAGS.get(tid) in ("Software", "UserComment", "ImageDescription", "Artist"):
                    out.append((f"EXIF-{TAGS.get(tid)}", str(val)[:500]))
    except Exception:
        pass


def _extract_text_chunks(p: Path) -> list[tuple[str, str]]:
    """PNG tEXt/iTXt/zTXt + JPEG COM + EXIF Software/UserComment + XMP tags."""
    out: list[tuple[str, str]] = []
    raw = p.read_bytes()[:32_000_000]
    suf = p.suffix.lower()
    if suf == ".png" and raw.startswith(b"\x89PNG"):
        _png_text_chunks(raw, out)
    if suf in (".jpg", ".jpeg") and raw.startswith(b"\xff\xd8"):
        _jpeg_text_chunks(raw, out)
    _exif_text_chunks(p, raw, out)
    return out


def _synthid_screen(p: Path) -> str:
    mp = Path("models/synthid.onnx")
    if not mp.is_file():
        return "NEEDS_MODEL"
    try:
        from security.signature import gate_model_load

        if not gate_model_load(mp):
            return "NEEDS_MODEL"
        import onnxruntime as ort

        from src.device import get_onnx_providers

        sess = ort.InferenceSession(str(mp), providers=get_onnx_providers())
        import numpy as np
        from PIL import Image

        with Image.open(p) as img:
            arr = np.asarray(img.convert("RGB").resize((512, 512)), dtype=np.float32) / 255.0
        out = sess.run(None, {sess.get_inputs()[0].name: arr[None].transpose(0, 3, 1, 2)})
        return "DETECTED" if float(np.asarray(out[0]).ravel()[0]) > 0.5 else "NOT_DETECTED"
    except Exception as exc:
        log.warning("watermark: synthid inference failed (%s)", exc)
        return "NOT_CHECKED"
