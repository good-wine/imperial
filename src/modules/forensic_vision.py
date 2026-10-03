"""Module 2 — Forensic Analysis & Computer Vision (SOTA, unsigned-file fallback).

Signal suite (all NumPy/OpenCV baseline, no mandatory torch):
  spatial   ELA (multi-quality max), Laplacian blur, median-residual noise,
            HSV histogram anomaly, ORB copy-move clustering
  spectral  FFT high-frequency artifact energy — GAN/diffusion upsamplers leave
            periodic spectral peaks absent in camera-native images
  compress  8×8 blockiness (BAM) + DCT double-JPEG periodicity — splices and
            re-saves break the original quantization lattice
  semantic  face-aware ELA (Haar cascade bundled with OpenCV; skipped if absent):
            face regions manipulated far more often than backgrounds
  temporal  inter-frame flicker (video): deepfake pipelines jitter frame to frame
  learned   ONNX registry (`models/registry.json`): deepfake + synthetic-image
            heads blended with the handcrafted ensemble, never trusted alone

Ensemble (documented, calibrated for recall on splices/synthesis):
  anomaly = .22·ELA + .18·FFT + .14·blockiness + .10·DCT + .10·noise
          + .08·hist + .06·ORB + .12·face   (+ temporal penalty for video)
  learned heads blend 35% when present. Thresholds: >0.65 high, >0.45 watch.

Edge posture: torch/onnxruntime optional; missing libs degrade to the
handcrafted path with `device_used="cpu-numpy"` and explicit warnings.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.config import SETTINGS
from src.device import get_onnx_providers, get_torch_device
from src.schemas import ForensicResult, FrameMetrics
from src.security import is_video, validate_media_file

log = logging.getLogger(__name__)

_FACE_CASCADE = None  # lazy Haar cascade singleton


def analyze_forensics(file_path: str | Path, max_frames: int | None = None) -> ForensicResult:
    """Run forensic analysis. Never raises on bad input; errors are captured."""
    limit = max_frames or SETTINGS.max_frames
    try:
        resolved = validate_media_file(file_path, max_mb=SETTINGS.max_file_mb)
    except Exception as exc:
        log.warning("forensic: input rejected: %s", exc)
        return ForensicResult(
            file_path=str(file_path),
            device_used="cpu-numpy",
            onnx_used=False,
            num_frames_analyzed=0,
            errors=[str(exc)],
        )

    device = get_torch_device()
    video = is_video(resolved)
    result = ForensicResult(
        file_path=str(resolved),
        device_used=device if _torch_available() else "cpu-numpy",
        onnx_used=False,
        num_frames_analyzed=0,
        is_video=video,
    )
    try:
        import cv2
    except ImportError:
        result.errors.append("opencv (cv2) not installed; forensic analysis unavailable")
        log.error("forensic: cv2 missing")
        return result

    try:
        frames = _extract_keyframes(cv2, resolved, limit)
        if not frames:
            result.errors.append("no decodable frames found")
            return result
        sessions, versions = _load_registry_sessions()
        result.onnx_used = bool(sessions)
        result.model_versions = versions
        prev_small = None
        flickers: list[float] = []
        for idx, (img, ts) in enumerate(frames):
            metrics = _score_frame(cv2, img, idx, ts, sessions, device)
            result.frames.append(metrics)
            small = _small_gray(cv2, img)
            if prev_small is not None:
                flickers.append(float(np.abs(small - prev_small).mean() / 255.0))
            prev_small = small
        result.num_frames_analyzed = len(result.frames)
        result.aggregate_anomaly = float(np.mean([f.anomaly_score for f in result.frames]))
        if video and flickers:
            # Natural video flicker ~0.005-0.02; deepfake jitter often >0.04.
            fl = float(np.mean(flickers))
            result.temporal_consistency = float(
                np.clip(1.0 - max(0.0, fl - 0.015) / 0.06, 0.0, 1.0)
            )
            if result.temporal_consistency < 0.6:
                result.aggregate_anomaly = float(
                    np.clip(
                        result.aggregate_anomaly + 0.10 * (1 - result.temporal_consistency), 0, 1
                    )
                )
                result.warnings.append(
                    f"temporal flicker {fl:.3f}: frame-to-frame instability "
                    "consistent with synthesis"
                )
        if result.aggregate_anomaly > 0.65:
            result.warnings.append("high aggregate anomaly: possible manipulation/synthesis")
        elif result.aggregate_anomaly > 0.45:
            result.warnings.append("moderate anomaly: watch — corroborate with context/provenance")
        if SETTINGS.enable_heatmap and result.frames:
            result.heatmap_path = _write_ela_heatmap(cv2, frames[0][0], resolved)
        log.info(
            "forensic: %s frames=%d anomaly=%.3f temporal=%.2f device=%s onnx=%s",
            resolved.name,
            result.num_frames_analyzed,
            result.aggregate_anomaly,
            result.temporal_consistency,
            result.device_used,
            result.onnx_used,
        )
    except Exception as exc:
        log.exception("forensic: analysis failed")
        result.errors.append(f"analysis error: {exc}")
    return result


# ---------------------------------------------------------------- frame extraction


def _extract_keyframes(
    cv2: object, path: Path, limit: int
) -> list[tuple[np.ndarray, float | None]]:
    import cv2 as _cv2

    if not is_video(path):
        img = _cv2.imread(str(path), _cv2.IMREAD_COLOR)
        return [] if img is None else [(img, None)]
    cap = _cv2.VideoCapture(str(path))
    if not cap.isOpened():
        log.warning("forensic: cannot open video %s", path.name)
        return []
    try:
        total = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = cap.get(_cv2.CAP_PROP_FPS) or 25.0
        idxs = sorted(
            {int(x) for x in np.linspace(0, max(total - 1, 0), min(limit, max(total, 1)))}
        )
        frames: list[tuple[np.ndarray, float | None]] = []
        prev_hist = None
        for fi in idxs:
            cap.set(_cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            hist = _cv2.calcHist([frame], [0], None, [32], [0, 256])
            _cv2.normalize(hist, hist)
            if prev_hist is not None:
                diff = float(_cv2.compareHist(prev_hist, hist, _cv2.HISTCMP_BHATTACHARYYA))
                if diff < 0.15 and len(frames) >= limit:
                    continue
            prev_hist = hist
            frames.append((frame, fi / fps if fps else None))
            if len(frames) >= limit:
                break
        return frames
    finally:
        cap.release()


def _small_gray(cv2: object, img: np.ndarray) -> np.ndarray:
    import cv2 as _cv2

    g = _cv2.cvtColor(img, _cv2.COLOR_BGR2GRAY)
    return _cv2.resize(g, (64, 64)).astype(np.float32)


# ---------------------------------------------------------------- per-frame scoring


def _score_frame(
    cv2: object,
    img: np.ndarray,
    idx: int,
    ts: float | None,
    sessions: dict[str, object],
    device: str,
) -> FrameMetrics:
    import cv2 as _cv2

    ela = _ela_score(_cv2, img)
    gray = _cv2.cvtColor(img, _cv2.COLOR_BGR2GRAY)
    blur_var = float(_cv2.Laplacian(gray, _cv2.CV_64F).var())
    blur_norm = float(np.clip(blur_var / 800.0, 0.0, 1.0))
    noise = _noise_score(_cv2, gray)
    hist_anom = _histogram_anomaly(_cv2, img)
    orb_clusters = _orb_copy_move_hint(_cv2, gray)
    fft_s = _fft_artifact_score(gray)
    block = _blockiness_score(gray)
    dct_hint = _dct_double_jpeg_hint(gray)
    face_count, face_anom = _face_region_anomaly(_cv2, img, gray)

    learned = _registry_scores(sessions, img)
    handcrafted = (
        0.22 * ela
        + 0.18 * fft_s
        + 0.14 * block
        + 0.10 * dct_hint
        + 0.10 * noise
        + 0.08 * hist_anom
        + 0.06 * min(orb_clusters / 4.0, 1.0)
        + 0.12 * face_anom
    )
    handcrafted = float(
        np.clip(
            handcrafted + 0.10 * (1.0 - blur_norm) if blur_norm < 0.15 else handcrafted, 0.0, 1.0
        )
    )
    calib = _calibrated_score(
        {
            "ela_score": ela,
            "fft_score": fft_s,
            "blockiness": block,
            "dct_double_jpeg": dct_hint,
            "noise_score": noise,
            "histogram_anomaly": hist_anom,
            "face_anomaly": face_anom,
        }
    )
    if calib is not None:  # analyst feedback measurably moves scores
        handcrafted = float(np.clip(0.5 * calib + 0.5 * handcrafted, 0.0, 1.0))
    if learned is not None:
        anomaly = float(np.clip(0.65 * learned + 0.35 * handcrafted, 0.0, 1.0))
    else:
        anomaly = float(np.clip(handcrafted, 0.0, 1.0))
    return FrameMetrics(
        frame_index=idx,
        timestamp_sec=ts,
        ela_score=round(ela, 4),
        blur_variance=round(blur_var, 2),
        noise_score=round(noise, 4),
        histogram_anomaly=round(hist_anom, 4),
        orb_clusters=orb_clusters,
        fft_score=round(fft_s, 4),
        blockiness=round(block, 4),
        dct_double_jpeg=round(dct_hint, 4),
        face_count=face_count,
        face_anomaly=round(face_anom, 4),
        anomaly_score=round(anomaly, 4),
    )


def _ela_score(cv2: object, img: np.ndarray) -> float:
    """Multi-quality ELA max (q75/q90/q95): splices diverge at some level."""
    import cv2 as _cv2

    scores = []
    for q in (75, 90, 95):
        ok, buf = _cv2.imencode(".jpg", img, [_cv2.IMWRITE_JPEG_QUALITY, q])
        if not ok:
            continue
        recomp = _cv2.imdecode(buf, _cv2.IMREAD_COLOR)
        if recomp is None:
            continue
        diff = _cv2.absdiff(img, recomp).astype(np.float32).mean() / 255.0
        scores.append(float(np.clip((diff - 0.02) / 0.08, 0.0, 1.0)))
    return max(scores) if scores else 0.0


def _noise_score(cv2: object, gray: np.ndarray) -> float:
    import cv2 as _cv2

    denoised = _cv2.medianBlur(gray, 3)
    residual = _cv2.absdiff(gray, denoised).astype(np.float32)
    return float(np.clip(residual.mean() / 30.0, 0.0, 1.0))


def _histogram_anomaly(cv2: object, img: np.ndarray) -> float:
    import cv2 as _cv2

    hsv = _cv2.cvtColor(img, _cv2.COLOR_BGR2HSV)
    hist = _cv2.calcHist([hsv], [2], None, [64], [0, 256]).ravel().astype(np.float32)
    hist /= hist.sum() + 1e-6
    smooth = np.convolve(hist, np.ones(5) / 5.0, mode="same")
    return float(np.clip(float(np.abs(hist - smooth).sum()) / 0.6, 0.0, 1.0))


def _fft_artifact_score(gray: np.ndarray) -> float:
    """High-frequency spectral energy ratio. Camera PRNU rolls off smoothly;
    GAN/diffusion upsampling (checkerboard) spikes energy above 0.4 Nyquist."""
    try:
        h, w = gray.shape
        small = gray[:: max(1, h // 256), :: max(1, w // 256)].astype(np.float32)
        f = np.fft.rfft2(small - small.mean())
        mag = np.abs(f)
        total = mag.sum() + 1e-9
        # Top 25% frequency bins share of energy; natural ≈ 0.02-0.08.
        cutoff = int(mag.shape[1] * 0.75)
        high = mag[:, cutoff:].sum() / total
        return float(np.clip((high - 0.05) / 0.20, 0.0, 1.0))
    except Exception:
        return 0.0


def _blockiness_score(gray: np.ndarray) -> float:
    """Block Artifact Measure: mean abs discontinuity on 8×8 grid vs off-grid.
    Spliced/recompressed regions break lattice alignment => elevated ratio."""
    try:
        g = gray.astype(np.float32)
        h, w = g.shape
        h8, w8 = (h // 8) * 8, (w // 8) * 8
        g = g[:h8, :w8]
        on_v = np.abs(np.diff(g[:, 7::8], axis=1)).mean()
        on_h = np.abs(np.diff(g[7::8, :], axis=0)).mean()
        off_v = np.abs(np.diff(g[:, 3::8], axis=1)).mean() + 1e-6
        off_h = np.abs(np.diff(g[3::8, :], axis=0)).mean() + 1e-6
        ratio = (on_v / off_v + on_h / off_h) / 2.0  # ~1.0 pristine
        return float(np.clip((ratio - 1.05) / 0.60, 0.0, 1.0))
    except Exception:
        return 0.0


def _dct_double_jpeg_hint(gray: np.ndarray) -> float:
    """Double-JPEG periodicity: variance of the Laplacian histogram's
    second derivative. Single-compressed => smooth; double => periodic peaks."""
    try:
        lap: np.ndarray = gray.astype(np.float32)
        diffs: np.ndarray = np.abs(lap[1:, :] - lap[:-1, :]).ravel()
        if float(diffs.mean()) < 0.3:
            return 0.0  # near-flat frame: no compression lattice to measure
        hist, _ = np.histogram(np.clip(diffs, 0, 64), bins=64)
        hist = hist.astype(np.float32) / (hist.sum() + 1e-9)
        d2 = np.diff(hist, 2)
        periodicity = float(np.abs(d2[8:56]).std())
        return float(np.clip((periodicity - 0.004) / 0.020, 0.0, 1.0))
    except Exception:
        return 0.0


def _orb_copy_move_hint(cv2: object, gray: np.ndarray) -> int:
    import cv2 as _cv2

    try:
        h, w = gray.shape
        small = (
            _cv2.resize(gray, (min(w, 640), max(1, min(h, 640) * h // max(w, 1))))
            if max(h, w) > 640
            else gray
        )
        orb = _cv2.ORB_create(  # type: ignore[attr-defined]  # cv2 C++ API
            nfeatures=300
        )
        kps = orb.detect(small, None)
        if len(kps) < 20:
            return 0
        pts = np.array([k.pt for k in kps], dtype=np.float32)
        gx, gy = (pts / 40.0).astype(int).T
        _, counts = np.unique(list(zip(gx, gy, strict=True)), axis=0, return_counts=True)
        return int((counts >= 6).sum())
    except Exception:
        return 0


def _face_region_anomaly(cv2: object, bgr: np.ndarray, gray: np.ndarray) -> tuple[int, float]:
    """Haar-cascade face boxes + per-face ELA. Returns (count, max anomaly)."""
    global _FACE_CASCADE
    import cv2 as _cv2

    try:
        if _FACE_CASCADE is None:
            data_dir = getattr(_cv2, "data", None)
            cascade_path = (
                Path(str(data_dir.haarcascades)) / "haarcascade_frontalface_default.xml"
                if data_dir and hasattr(data_dir, "haarcascades")
                else None
            )
            if cascade_path and cascade_path.is_file():
                _FACE_CASCADE = _cv2.CascadeClassifier(  # type: ignore[attr-defined]  # cv2 C++ API
                    str(cascade_path)
                )
            else:
                return 0, 0.0
        faces = _FACE_CASCADE.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(48, 48)
        )
        if len(faces) == 0:
            return 0, 0.0
        ok, buf = _cv2.imencode(".jpg", bgr, [_cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            return len(faces), 0.0
        recomp = _cv2.imdecode(buf, _cv2.IMREAD_COLOR)
        if recomp is None:
            return len(faces), 0.0
        peak = 0.0
        for x, y, w, h in faces[:6]:
            roi_o = bgr[y : y + h, x : x + w].astype(np.float32)
            roi_r = recomp[y : y + h, x : x + w].astype(np.float32)
            if roi_o.size == 0:
                continue
            d = float(np.abs(roi_o - roi_r).mean() / 255.0)
            peak = max(peak, float(np.clip((d - 0.02) / 0.08, 0, 1)))
        return len(faces), peak
    except Exception:
        return 0, 0.0


def _write_ela_heatmap(cv2: object, img: np.ndarray, src: Path) -> str | None:
    """Persist ELA heatmap for examiner review. Returns path or None."""
    import cv2 as _cv2

    try:
        ok, buf = _cv2.imencode(".jpg", img, [_cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            return None
        recomp = _cv2.imdecode(buf, _cv2.IMREAD_COLOR)
        if recomp is None:
            return None
        diff = _cv2.absdiff(img, recomp)
        heat = _cv2.applyColorMap(_cv2.cvtColor(diff, _cv2.COLOR_BGR2GRAY), _cv2.COLORMAP_JET)
        out = Path(SETTINGS.report_out_dir) / f"{src.stem}.ela.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        _cv2.imwrite(str(out), heat)
        return str(out)
    except Exception as exc:
        log.warning("forensic: heatmap failed (%s)", exc)
        return None


# ---------------------------------------------------------------- learned registry


_CALIB = None
_CALIB_LOADED = False


def _calibrated_score(feats: dict[str, float]) -> float | None:
    """Analyst-calibrated LR probability (`scripts/calibrate_ensemble.py`).
    Blended 50/50 with literature priors when present; absent by default."""
    global _CALIB, _CALIB_LOADED
    if not _CALIB_LOADED:
        _CALIB_LOADED = True
        try:
            doc = json.loads(Path("models/ensemble_weights.json").read_text(encoding="utf-8"))
            if isinstance(doc.get("weights"), dict) and doc.get("features"):
                _CALIB = doc
                log.info(
                    "forensic: calibrated weights loaded (auc=%.3f n=%d)",
                    doc.get("train_auc", 0),
                    doc.get("n", 0),
                )
        except Exception:
            _CALIB = None
    if not _CALIB:
        return None
    try:
        z = float(_CALIB.get("intercept", 0.0)) + sum(
            float(_CALIB["weights"].get(k, 0.0)) * float(feats.get(k, 0.0))
            for k in _CALIB.get("features", [])
        )
        return float(1.0 / (1.0 + np.exp(-z)))
    except Exception:
        return None


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def _load_registry_sessions() -> tuple[dict[str, object], dict[str, str]]:
    """Load every model in `models/registry.json` that exists on disk.

    Registry entries: {"name","path","task": deepfake|synthetic,"version"}.
    Legacy single-path `ONNX_MODEL_PATH` is honored as `deepfake-legacy`.
    """
    sessions: dict[str, object] = {}
    versions: dict[str, str] = {}
    candidates: list[dict[str, str]] = []
    reg = Path(SETTINGS.model_registry_path)
    if reg.is_file():
        try:
            data = json.loads(reg.read_text(encoding="utf-8"))
            candidates = data.get("models", []) if isinstance(data, dict) else []
        except Exception as exc:
            log.warning("forensic: registry unreadable (%s)", exc)
    legacy = Path(SETTINGS.onnx_model_path)
    if legacy.is_file() and not any(c.get("path") == str(legacy) for c in candidates):
        candidates.append(
            {
                "name": "deepfake-legacy",
                "path": str(legacy),
                "task": "deepfake",
                "version": "legacy",
            }
        )
    for cand in candidates:
        mp = Path(cand.get("path", ""))
        if not mp.is_file():
            continue
        name = cand.get("name", mp.stem)
        if cand.get("task") == "embedding":
            # Representation tower (e.g. CLIP), not a manipulation head:
            # never scored as a detector. Recorded for audit transparency.
            versions[name] = f"{cand.get('version', 'unknown')}+embedding-only"
            continue
        try:
            from security.signature import gate_model_load

            if not gate_model_load(mp):
                log.warning("forensic: model refused (bad/missing signature): %s", mp)
                versions[cand.get("name", mp.stem)] = "refused-bad-signature"
                continue
        except Exception as exc:
            log.warning("forensic: signature gate error (%s); continuing permissive", exc)
        try:
            import onnxruntime as ort

            sess = ort.InferenceSession(str(mp), providers=get_onnx_providers())
            sessions[cand.get("name", mp.stem)] = sess
            versions[cand.get("name", mp.stem)] = cand.get("version", "unknown")
            log.info("forensic: ONNX model loaded: %s (%s)", mp, cand.get("task"))
        except ImportError:
            log.info("forensic: onnxruntime missing, skipping learned heads")
            break
        except Exception as exc:
            log.warning("forensic: ONNX load failed for %s (%s)", mp, exc)
    return sessions, versions


def _registry_scores(sessions: dict[str, object], img: np.ndarray) -> float | None:
    if not sessions:
        return None
    scores = []
    for sess in sessions.values():
        s = _onnx_score(sess, img)
        if s is not None:
            scores.append(s)
    if not scores:
        return None
    # Max-pool: any head firing matters (deepfake vs diffusion heads differ).
    return float(np.clip(max(scores) * 0.7 + float(np.mean(scores)) * 0.3, 0, 1))


def _onnx_score(sess: Any, img: np.ndarray) -> float | None:
    try:
        import cv2 as _cv2

        blob = _cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], np.float32)
        std = np.array([0.229, 0.224, 0.225], np.float32)
        blob = ((blob - mean) / std).transpose(2, 0, 1)[None, ...].astype(np.float32)
        out = sess.run(None, {sess.get_inputs()[0].name: blob})[0]
        logit = float(np.asarray(out).ravel()[0])
        return float(1.0 / (1.0 + np.exp(-logit))) if not (0.0 <= logit <= 1.0) else logit
    except Exception as exc:
        log.warning("forensic: ONNX inference failed (%s)", exc)
        return None
