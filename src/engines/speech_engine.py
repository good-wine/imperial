"""Local Speech-to-Text + translation (Whisper, fully offline after provisioning).

Crisis-zone audio arrives in local languages/dialects. The engine transcribes
in the source language (auto-detected) and produces an English gloss for the
analyst — both persisted for the LLM leg (claims in speech vs caption claims
are a decontextualization signal) and shown in the dashboard.

Offline posture: models live in `models/whisper/` (fetched once with
`scripts/fetch_whisper.py --model tiny` on the connected station, or
`WHISPER_MODEL=tiny` auto-download on first connected run). Missing model /
lib -> SKIPPED with reason, never a fabricated transcript.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_MODEL = None
_MODEL_NAME = ""


@dataclass
class TranscriptResult:
    available: bool = False
    language: str | None = None
    language_probability: float = 0.0
    text: str = ""
    english_gloss: str = ""
    segments: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load(model_name: str, download_root: str):
    global _MODEL, _MODEL_NAME
    if _MODEL is not None and model_name == _MODEL_NAME:
        return _MODEL
    import whisper

    # Air-gap gate: refuse the Hub round-trip — weights must be provisioned
    # (scripts/fetch_whisper.py) before analysis runs.
    if model_name not in getattr(whisper, "_MODELS", {}):
        raise RuntimeError(f"unknown whisper model '{model_name}'")
    expect = Path(download_root) / f"{model_name}.pt"
    if not expect.is_file():
        raise RuntimeError(f"whisper weights missing: {expect} (provision offline first)")
    from src.device import get_torch_device

    device = get_torch_device()
    if device == "mps":
        device = "cpu"  # whisper MPS kernels are unreliable; documented fallback
    _MODEL = whisper.load_model(model_name, device=device, download_root=download_root)
    _MODEL_NAME = model_name
    log.info("speech: whisper=%s device=%s", model_name, device)
    return _MODEL


def transcribe(
    path: str | Path,
    *,
    model_name: str = "",
    download_root: str = "models/whisper",
    translate_to_english: bool = True,
) -> TranscriptResult:
    """Transcribe + gloss. `model_name` empty -> `WHISPER_MODEL` env (default tiny)."""
    import os

    res = TranscriptResult()
    name = model_name or os.getenv("WHISPER_MODEL", "tiny")
    try:
        model = _load(name, download_root)
    except ImportError:
        res.errors.append("openai-whisper not installed — transcription SKIPPED")
        return res
    except Exception as exc:
        res.errors.append(f"whisper model '{name}' unavailable offline ({exc})")
        res.warnings.append("provision with scripts/fetch_whisper.py on the connected station")
        return res
    try:
        out = model.transcribe(str(path), verbose=False)
        res.available = True
        res.language = str(out.get("language", ""))
        res.language_probability = round(float(out.get("language_probability", 0.0)), 3)
        res.text = str(out.get("text", "")).strip()[:4000]
        for seg in out.get("segments", [])[:60]:
            res.segments.append(
                {
                    "start": round(float(seg.get("start", 0)), 2),
                    "end": round(float(seg.get("end", 0)), 2),
                    "text": str(seg.get("text", ""))[:300],
                }
            )
        if translate_to_english and res.language and res.language != "en" and res.text:
            try:
                tr = model.transcribe(str(path), task="translate", verbose=False)
                res.english_gloss = str(tr.get("text", "")).strip()[:4000]
            except Exception as exc:
                res.warnings.append(f"english gloss failed ({exc}); source transcript kept")
        log.info("speech: %s lang=%s chars=%d", Path(str(path)).name, res.language, len(res.text))
        return res
    except Exception as exc:
        log.warning("speech: transcribe failed (%s)", exc)
        res.errors.append(f"transcription failed: {exc}")
        return res
