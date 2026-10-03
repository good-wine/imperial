"""Offline-first media embeddings: PDQ-lite (DCT 256-bit) + CLIP image vectors.

- PDQ-lite: 16×16 DCT hash rendered as 64 hex chars. Pure NumPy/Pillow, no
  extra dependency; survives crops/resizes where pHash degrades. Threshold
  guidance: ≤48/256 near-duplicate, 49–80 possible derivative.
- CLIP: ONNX `models/clip_image.onnx` (512-d, L2-normalized) when present,
  else lazy `transformers` CLIP when installed, else None (hash-only mode).
  CLIP catches semantic reuse — screenshots, memes, recaptures — that no
  perceptual hash survives. Cosine ≥0.90 ≈ same-scene reuse.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

_CLIP_SESSION = None
_CLIP_LOADED: bool = False


def pdq_lite_hex(gray_64: np.ndarray | None = None, *, pil_image=None) -> str:
    """Compute 256-bit DCT hash. Pass a 64×64 grayscale array or a PIL image."""
    if pil_image is not None:
        arr = np.asarray(pil_image.convert("L").resize((64, 64)), dtype=np.float32)
    elif gray_64 is not None:
        arr = np.asarray(gray_64, dtype=np.float32)
        if arr.shape != (64, 64):
            from PIL import Image as _I

            arr = np.asarray(_I.fromarray(arr.astype(np.uint8)).resize((64, 64)), dtype=np.float32)
    else:
        raise ValueError("pdq_lite_hex needs gray_64 or pil_image")
    dct = _dct2(arr - arr.mean())
    low = dct[:16, :16]
    med = float(np.median(low))
    bits = (low > med).astype(np.uint8).ravel()
    return "".join(f"{int(''.join(map(str, bits[i : i + 4])), 2):x}" for i in range(0, 256, 4))


def _dct2(a: np.ndarray) -> np.ndarray:
    """Separable DCT-II via FFT (no scipy dependency)."""
    n, m = a.shape
    ext = np.concatenate([a, a[::-1, :]], axis=0)
    ext = np.concatenate([ext, ext[:, ::-1]], axis=1)
    f = np.fft.rfft2(ext)[:n, :m].real
    scale = np.ones_like(f)
    scale[0, :] /= 2
    scale[:, 0] /= 2
    return f * scale / (4 * n * m)


def clip_image_embedding(
    pil_image, model_path: str = "models/clip_image.onnx"
) -> list[float] | None:
    """512-d L2-normalized CLIP image vector, or None when unavailable offline."""
    global _CLIP_SESSION, _CLIP_LOADED
    if not _CLIP_LOADED:
        _CLIP_LOADED = True
        mp = Path(model_path)
        if mp.is_file():
            try:
                from security.signature import gate_model_load

                if not gate_model_load(mp):
                    log.warning("embeddings: CLIP refused (bad/missing signature)")
                    _CLIP_SESSION = None
                else:
                    import onnxruntime as ort

                    from src.device import get_onnx_providers

                    _CLIP_SESSION = ort.InferenceSession(str(mp), providers=get_onnx_providers())
                    log.info("embeddings: CLIP ONNX loaded: %s", mp)
            except Exception as exc:
                log.info("embeddings: CLIP ONNX unavailable (%s)", exc)
    if _CLIP_SESSION is not None:
        try:
            arr = np.asarray(pil_image.convert("RGB").resize((224, 224)), dtype=np.float32) / 255.0
            mean = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
            std = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
            blob = ((arr - mean) / std).transpose(2, 0, 1)[None, ...].astype(np.float32)
            out = _CLIP_SESSION.run(None, {_CLIP_SESSION.get_inputs()[0].name: blob})[0]
            vec = np.asarray(out).ravel().astype(float)
            n = float(np.linalg.norm(vec)) + 1e-9
            return (vec / n).tolist()
        except Exception as exc:
            log.warning("embeddings: CLIP inference failed (%s)", exc)
            return None
    # Lazy transformers fallback (pre-downloaded weights for air-gap).
    # local_files_only: NEVER hit the network from the analysis path.
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)  # nosec B615 - pinned offline weights, never downloaded
        proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)  # nosec B615 - see above
        inputs = proc(images=pil_image, return_tensors="pt")
        with torch.no_grad():
            vec = model.get_image_features(**inputs)[0]
            vec = vec / vec.norm()
        return vec.tolist()
    except Exception:
        return None
