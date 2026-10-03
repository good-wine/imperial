"""PII redaction for civilian protection (face + plate blurring).

Policy: TLP:GREEN/CLEAR exports must not expose identifiable civilians.
`redacted_copy()` writes a blurred derivative into the reports dir and
returns its path + redaction log; the ORIGINAL stays sealed in case storage
for the examiner chain-of-custody. Detection is Haar-cascade based (bundled
with OpenCV); plates fall back to an aspect-ratio + edge-density heuristic
when the cascade XML is absent. Misses are possible — the log records
counts, and GREEN exports carry the residual-risk note.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from src.config import SETTINGS

log = logging.getLogger(__name__)

_CASCADES: dict[str, object] = {}


@dataclass
class RedactionLog:
    redacted_path: str | None = None
    faces_blurred: int = 0
    plates_blurred: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def _cascade(name: str):
    if name in _CASCADES:
        return _CASCADES[name]
    try:
        import cv2 as _cv2

        data_dir = getattr(_cv2, "data", None)
        if not (data_dir and hasattr(data_dir, "haarcascades")):
            return None
        from pathlib import Path as _P

        cp = _P(str(data_dir.haarcascades)) / name
        _CASCADES[name] = (
            _cv2.CascadeClassifier(str(cp))  # type: ignore[attr-defined]  # cv2 C++ API
            if cp.is_file()
            else None
        )
        return _CASCADES[name]
    except ImportError:
        return None


def _blur_boxes(img: np.ndarray, boxes, strength: int = 35) -> int:
    try:
        import cv2 as _cv2
    except ImportError:
        return 0
    n = 0
    for x, y, w, h in boxes:
        x, y, w, h = max(int(x), 0), max(int(y), 0), int(w), int(h)
        roi = img[y : y + h, x : x + w]
        if roi.size == 0:
            continue
        k = max(3, (min(w, h) // 3) | 1)  # scale blur to box size
        img[y : y + h, x : x + w] = _cv2.GaussianBlur(roi, (k, k), 0)
        n += 1
    return n


def _plate_candidates(cv2: object, gray: np.ndarray) -> list:
    """Heuristic plates: high edge-density rectangles with 2:1–5:1 aspect."""
    import cv2 as _cv2

    edges = _cv2.Canny(gray, 100, 200)
    contours, _ = _cv2.findContours(edges, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE)
    out = []
    area = gray.shape[0] * gray.shape[1]
    for c in contours:
        x, y, w, h = _cv2.boundingRect(c)
        if h < 12 or w < 40 or not (2.0 <= w / max(h, 1) <= 5.5):
            continue
        if not (0.0005 * area < w * h < 0.02 * area):
            continue
        out.append((x, y, w, h))
    return out[:8]


def redacted_copy(image_path: str | Path, *, out_dir: str | Path | None = None) -> RedactionLog:
    """Blur faces + plates; return log. Original untouched (copy-on-write)."""
    log_out = RedactionLog()
    p = Path(image_path)
    try:
        import cv2 as _cv2
    except ImportError:
        log_out.notes.append("cv2 missing — redaction unavailable, export gated")
        return log_out
    img = _cv2.imread(str(p), _cv2.IMREAD_COLOR)
    if img is None:
        log_out.notes.append("undecodable image — redaction skipped")
        return log_out
    gray = _cv2.cvtColor(img, _cv2.COLOR_BGR2GRAY)
    face_cascade = _cascade("haarcascade_frontalface_default.xml")
    faces = (
        face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(40, 40))
        if face_cascade is not None
        else []
    )
    log_out.faces_blurred = _blur_boxes(img, faces)
    plate_cascade = _cascade("haarcascade_russian_plate_number.xml")
    plates = (
        plate_cascade.detectMultiScale(gray, 1.1, 3, minSize=(40, 12))
        if plate_cascade is not None
        else _plate_candidates(_cv2, gray)
    )
    log_out.plates_blurred = _blur_boxes(img, plates)
    dest = (Path(out_dir) if out_dir else Path(SETTINGS.report_out_dir)) / f"{p.stem}.redacted.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    _cv2.imwrite(str(dest), img)
    log_out.redacted_path = str(dest)
    log_out.notes.append(
        f"blurred {log_out.faces_blurred} face(s), {log_out.plates_blurred} plate(s); "
        "automated detection may miss — examiner review required before release"
    )
    log.info(
        "pii: %s faces=%d plates=%d -> %s",
        p.name,
        log_out.faces_blurred,
        log_out.plates_blurred,
        dest.name,
    )
    return log_out
