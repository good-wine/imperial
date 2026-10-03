"""PRNU sensor fingerprinting (Lukas–Fridrich, screening-grade).

Theory in one paragraph: every sensor leaves a unique Photo-Response
Non-Uniformity pattern (silicon inhomogeneity) in each capture. Averaging
denoising residuals over ~20+ flat-field frames from one camera yields its
fingerprint F; the Peak-to-Correlation-Energy (PCE) between F and a query
residual attributes (or excludes) the source device. Courts accept full
implementations; ours is deliberately screening-grade (Gaussian residual,
no wavelet) — strong exclusion signal, cautious attribution.

Interface:
  fingerprint(images) -> (F, quality)     build from reference stills
  match_score(F, query) -> (pce, verdict)  MATCH / EXCLUDE / INCONCLUSIVE
  save/load F as .npy under data/prnu/<camera_id>.npy
Pipeline use (`PRNU_DB_DIR` set + fingerprints present): query residual is
correlated against every enrolled camera; best hit becomes a finding.
Absent DB -> leg abstains (UNKNOWN), never a false attribution.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

MATCH_PCE = 60.0  # literature threshold for full PRNU; we report, not decide
EXCLUDE_PCE = 10.0


@dataclass
class PrnuResult:
    available: bool = False
    best_camera: str | None = None
    pce: float = 0.0
    verdict: str = "UNKNOWN"  # MATCH | EXCLUDE | INCONCLUSIVE | UNKNOWN
    enrolled: int = 0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def residual(gray: np.ndarray) -> np.ndarray:
    """Denoising residual. Gaussian stand-in for Mihcak wavelet: faster,
    slightly less selective — documented, and the PCE thresholds are kept
    conservative to compensate."""
    try:
        import cv2 as _cv2

        g = gray.astype(np.float32)
        den = _cv2.GaussianBlur(g, (0, 0), 1.2)
    except ImportError:
        from scipy.ndimage import gaussian_filter

        den = gaussian_filter(gray.astype(np.float32), 1.2)
    w = gray.astype(np.float32) - den
    n = float(np.linalg.norm(w))
    return w / (n + 1e-9) if n > 1e-9 else w


def fingerprint(images: list[np.ndarray]) -> tuple[np.ndarray, float]:
    """Average normalized residuals. Quality = mean pairwise correlation
    (low => heterogeneous sources, fingerprint untrustworthy)."""
    res = [residual(_to_gray(im)) for im in images]
    F = np.mean(res, axis=0)
    n = float(np.linalg.norm(F))
    F = F / (n + 1e-9)
    corrs: list[float] = []
    corrs.extend(
        float((res[i] * res[j]).sum()) for i in range(len(res)) for j in range(i + 1, len(res))
    )
    quality = round(float(np.mean(corrs)) if corrs else 0.0, 4)
    return F, quality


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return (0.299 * img[..., 2] + 0.587 * img[..., 1] + 0.114 * img[..., 0]).astype(np.float32)
    return img.astype(np.float32)


def pce_score(F: np.ndarray, query: np.ndarray) -> float:
    """Peak-to-Correlation-Energy via FFT cross-correlation (shift-invariant:
    robust to small crops/translations that kill naive dot products)."""
    h, w = F.shape
    q = query[:h, :w] if query.shape[0] >= h and query.shape[1] >= w else query
    if q.shape != F.shape:
        return 0.0
    corr = np.abs(np.fft.ifft2(np.fft.fft2(F) * np.conj(np.fft.fft2(q))))
    peak = float(corr.max())
    mask = np.ones_like(corr, bool)
    py, px = np.unravel_index(int(np.argmax(corr)), corr.shape)
    y0, y1 = max(py - 5, 0), min(py + 6, corr.shape[0])
    x0, x1 = max(px - 5, 0), min(px + 6, corr.shape[1])
    mask[y0:y1, x0:x1] = False
    floor = float(corr[mask].mean()) + 1e-12
    return round(peak**2 / floor, 2)


def _score_fingerprint(fp: Path, q: np.ndarray) -> tuple[str, float] | None:
    """Score one enrolled fingerprint; None when unreadable (fail-open)."""
    try:
        blk = np.load(fp)
        return fp.stem, pce_score(blk, q)
    except Exception as exc:
        log.warning("prnu: unreadable %s (%s)", fp.name, exc)
        return None


def identify(query_img: np.ndarray, db_dir: str | Path = "data/prnu") -> PrnuResult:
    """Correlate a query frame against every enrolled fingerprint."""
    res = PrnuResult()
    db = Path(db_dir)
    if not db.is_dir():
        return res
    fps = sorted(db.glob("*.npy"))
    res.enrolled = len(fps)
    if not fps:
        return res
    q = residual(_to_gray(query_img))
    best, best_pce = None, -1.0
    for fp in fps:
        scored = _score_fingerprint(fp, q)
        if scored is None:
            continue
        name, s = scored
        if s > best_pce:
            best, best_pce = name, s
    if best is None:
        res.errors.append("no usable fingerprints")
        return res
    res.available, res.best_camera, res.pce = True, best, best_pce
    if best_pce >= MATCH_PCE:
        res.verdict = "MATCH"
        res.warnings.append(f"sensor match {best} (PCE {best_pce}): supports same-device origin")
    elif best_pce <= EXCLUDE_PCE:
        res.verdict = "EXCLUDE"
        res.warnings.append(f"no sensor shares signal (best PCE {best_pce}): device exclusion")
    else:
        res.verdict = "INCONCLUSIVE"
        res.warnings.append(f"PCE {best_pce} in gray zone: no attribution claim")
    log.info("prnu: best=%s pce=%.1f %s", best, best_pce, res.verdict)
    return res
