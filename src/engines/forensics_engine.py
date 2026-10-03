"""Engine 2 — Video & audio forensics (multimodal, device-adaptive).

Pipeline (every leg degrades to SKIPPED with a reason when deps/media absent):
  pre-processing  adversarial hardening: fastNlMeans denoise + illumination
                  normalization on a *copy* (originals never mutated; scores
                  compare raw vs normalized to catch concealment blur).
  visual          delegates to `src/modules/forensic_vision.py` (ELA/FFT/
                  blockiness/DCT/faces/flicker + ONNX registry).
  mp4             full top-level atom inventory + ordering anomalies (faststart,
                  duplicate moov, oversized free/uuid) + container/DCT fusion
                  for the double-compression hint.
  audio           WAV/PCM via stdlib `wave` (+ `librosa` enhancement when
                  installed): neural-vocoder artifact band energy, F0-stability
                  (TTS flag), spectral centroid/rolloff anomalies.
  lip-sync        mouth-region motion envelope vs audio amplitude envelope
                  (Pearson r; needs cv2 + decodable audio, else SKIPPED).
  rPPG            chrominance-based pulse from face ROI time series, 0.7–4 Hz
                  band (42–240 bpm); plausible HR + strong peak => LIVE.

mp4/DCT note: container flags alone never *prove* double compression — the
fused hint is screening-grade and reported as such (honest uncertainty).
"""

from __future__ import annotations

import logging
import struct
import wave
from pathlib import Path
from typing import Any

import numpy as np

from src.schemas import (
    AudioResult,
    ForensicResult,
    LipSyncResult,
    Mp4Result,
    RppgResult,
)
from src.security import is_video

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- pre-processing


def preprocess_frame(img: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Adversarial hardening copy: denoise + normalize. Returns (copy, stats).

    Stats include blur-delta (raw vs denoised Laplacian): large deltas mean the
    input was already smoothed — a concealment indicator, not a filter artifact.
    """
    stats: dict[str, float] = {}
    try:
        import cv2 as _cv2

        gray_raw = _cv2.cvtColor(img, _cv2.COLOR_BGR2GRAY)
        raw_var = float(_cv2.Laplacian(gray_raw, _cv2.CV_64F).var())
        den = _cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 15)
        gray = _cv2.cvtColor(den, _cv2.COLOR_BGR2GRAY)
        stats["raw_blur_var"] = round(raw_var, 2)
        stats["blur_delta"] = round(
            abs(raw_var - float(_cv2.Laplacian(gray, _cv2.CV_64F).var())), 2
        )
        # Min-max illumination normalization in NumPy (avoids cv2.normalize
        # overload ambiguity across OpenCV builds; identical mathematics).
        _gmin, _gmax = float(np.min(gray)), float(np.max(gray))
        norm = ((gray.astype(np.float32) - _gmin) / max(_gmax - _gmin, 1e-9) * 255).astype(np.uint8)
        stats["illum_gain"] = round(float(norm.mean() / (gray.mean() + 1e-6)), 3)
        out = _cv2.cvtColor(norm, _cv2.COLOR_GRAY2BGR)
        return out, stats
    except ImportError:
        g = img.mean(axis=2) if img.ndim == 3 else img.astype(float)
        stats["raw_blur_var"] = round(float(np.var(g)), 2)
        return img.copy(), stats


# ---------------------------------------------------------------- MP4 atoms


def inspect_mp4(path: str | Path, head_cap: int = 4_000_000) -> Mp4Result:
    """Full top-level atom inventory with ordering anomaly flags."""
    res = Mp4Result()
    p = Path(path)
    if p.suffix.lower() not in (".mp4", ".mov", ".m4v"):
        return res
    res.is_mp4 = True
    try:
        data = p.read_bytes()[:head_cap]
    except OSError as exc:
        res.errors.append(f"unreadable: {exc}")
        return res
    atoms: list[dict[str, Any]] = []
    i = 0
    order: list[str] = []
    try:
        while i + 8 <= len(data) and len(atoms) < 128:
            size = struct.unpack(">I", data[i : i + 4])[0]
            name = data[i + 4 : i + 8].decode("ascii", "replace")
            if size == 1 and i + 16 <= len(data):
                size = struct.unpack(">Q", data[i + 8 : i + 16])[0]
            if size < 8:
                break
            atoms.append({"name": name, "offset": i, "size": min(size, len(data) - i)})
            order.append(name)
            i += size if size > 0 else 4
            if i == 0:
                break
    except (struct.error, IndexError) as exc:
        res.errors.append(f"atom walk truncated: {exc}")
    res.atoms = atoms
    names = [a["name"] for a in atoms]
    if "moov" in names and "mdat" in names:
        res.faststart = names.index("moov") < names.index("mdat")
    if names.count("moov") > 1:
        res.anomalies.append("duplicate moov atoms — re-muxed or tampered container")
    if "ftyp" not in names:
        res.anomalies.append("missing ftyp — non-standard muxer")
    for a in atoms:
        if a["name"] in ("free", "skip") and a["size"] > 1_000_000:
            res.anomalies.append(f"oversized {a['name']} ({a['size']}B): possible payload hiding")
    uuid_n = sum(1 for a in names if a == "uuid")
    if uuid_n > 2:
        res.anomalies.append(f"{uuid_n} uuid atoms: unusual metadata stuffing")
    # Container leg of the double-compression hint (visual DCT leg fuses later).
    res.double_compression_hint = round(min(1.0, 0.25 * len(res.anomalies)), 3)
    if res.anomalies:
        res.warnings.append("; ".join(res.anomalies))
    log.info(
        "mp4: %s atoms=%d faststart=%s anomalies=%d",
        p.name,
        len(atoms),
        res.faststart,
        len(res.anomalies),
    )
    return res


# ---------------------------------------------------------------- audio


def _load_audio_mono(path: Path) -> tuple[np.ndarray, int] | None:
    """PCM WAV via stdlib. Returns (mono float32, sr) or None (honest SKIPPED).

    Compressed audio (AAC in MP4) needs ffmpeg/librosa: if `librosa` is present
    we use it as the enhancement path; otherwise the leg reports SKIPPED with
    the exact reason instead of fabricating scores.
    """
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as w:
                sr, n, samp = w.getframerate(), w.getnframes(), w.getsampwidth()
                raw = w.readframes(n)
            dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[samp]
            pcm = np.frombuffer(raw, dtype=dtype).astype(np.float32)
            if w.getnchannels() > 1:
                pcm = pcm.reshape(-1, w.getnchannels()).mean(axis=1)
            scale = float(2 ** (8 * samp - 1))
            return pcm / scale, sr
        except (KeyError, wave.Error, EOFError, OSError) as exc:
            log.warning("audio: wav decode failed (%s)", exc)
            return None
    try:
        import librosa

        audio, rate = librosa.load(str(path), sr=16000, mono=True)
        return np.asarray(audio, dtype=np.float32), int(rate)
    except ImportError:
        log.info("audio: compressed audio needs librosa/ffmpeg (absent) — SKIPPED")
        return None
    except Exception as exc:
        log.warning("audio: decode failed (%s)", exc)
        return None


def _stft_mag(x: np.ndarray, n_fft: int = 1024, hop: int = 256) -> np.ndarray:
    win = np.hanning(n_fft)
    frames = [x[i : i + n_fft] * win for i in range(0, max(len(x) - n_fft, 1), hop)]
    return np.abs(np.fft.rfft(frames, n=n_fft, axis=1))


def analyze_audio(path: str | Path) -> AudioResult:
    """Neural-vocoder artifact screening (numpy STFT, librosa-optional).

    Heuristics from vocoder literature: GAN vocoders (HiFi-GAN/MelGAN) leak
    excess energy into 6–8 kHz with unnatural phase coherence, and TTS prosody
    shows abnormally *stable* F0 tracks. Both are screening flags, not proof.
    """
    res = AudioResult()
    p = Path(path)
    loaded = _load_audio_mono(p)
    if loaded is None:
        res.warnings.append("no decodable audio track (WAV PCM or librosa/ffmpeg required)")
        return res
    y, sr = loaded
    if y.size < sr:
        res.errors.append("audio too short (<1s) for analysis")
        return res
    res.available = True
    res.duration_sec = round(y.size / sr, 2)
    res.sample_rate = sr
    mag = _stft_mag(y[: sr * 10])  # cap 10 s for edge devices
    freqs = np.fft.rfftfreq(1024, 1 / sr)
    frame_e = mag.sum(axis=1) + 1e-9
    centroid = float((mag * freqs).sum(axis=1).mean() / frame_e.mean())
    hi = mag[:, (freqs >= 6000) & (freqs <= 8000)].sum()
    res.vocoder_anomaly = round(float(np.clip((hi / (mag.sum() + 1e-9) - 0.06) / 0.10, 0, 1)), 3)
    # F0 track via autocorrelation per 40 ms frame; low normalized variance => TTS-flat.
    f0s = []
    fl = int(sr * 0.04)
    for i in range(0, min(len(y), sr * 8) - fl, fl):
        seg = y[i : i + fl] - y[i : i + fl].mean()
        ac = np.correlate(seg, seg, "full")[len(seg) - 1 :]
        lo, hi_p = int(sr / 400), int(sr / 60)
        if hi_p < len(ac):
            peak = lo + int(np.argmax(ac[lo:hi_p]))
            if ac[peak] > 0.3 * ac[0]:
                f0s.append(sr / peak)
    if len(f0s) > 10:
        cv = float(np.std(f0s) / (np.mean(f0s) + 1e-9))
        res.pitch_instability = round(float(np.clip((0.06 - cv) / 0.06, 0, 1)), 3)
        res.spectral_notes.append(f"centroid={centroid:.0f}Hz f0_cv={cv:.3f} voiced={len(f0s)}")
    if res.vocoder_anomaly > 0.6:
        res.warnings.append("high-band energy pattern consistent with neural vocoders")
    if res.pitch_instability > 0.7:
        res.warnings.append("abnormally flat prosody — TTS screening flag")
    log.info(
        "audio: %s voc=%.2f pitchflag=%.2f", p.name, res.vocoder_anomaly, res.pitch_instability
    )
    return res


# ---------------------------------------------------------------- lip-sync


def verify_lipsync(video_path: str | Path, audio: AudioResult) -> LipSyncResult:
    """Mouth-motion vs audio-envelope Pearson correlation (heuristic screening)."""
    res = LipSyncResult()
    try:
        import cv2 as _cv2
    except ImportError:
        res.errors.append("cv2 missing — lip-sync SKIPPED")
        return res
    loaded = _load_audio_mono(Path(video_path))
    if loaded is None:
        res.errors.append("no audio track — lip-sync SKIPPED")
        return res
    y, sr = loaded
    cap = _cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        res.errors.append("video undecodable — lip-sync SKIPPED")
        return res
    try:
        fps = cap.get(_cv2.CAP_PROP_FPS) or 25.0
        hop = max(int(sr / fps), 1)
        env = np.array([np.abs(y[i : i + hop]).mean() for i in range(0, len(y) - hop, hop)])
        motion: list[float] = []
        prev = None
        n = 0
        while n < len(env) and n < 200:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            h, w = frame.shape[:2]
            mouth = frame[int(h * 0.62) : int(h * 0.90), int(w * 0.30) : int(w * 0.70)]
            g = _cv2.cvtColor(mouth, _cv2.COLOR_BGR2GRAY).astype(np.float32)
            if prev is not None:
                motion.append(float(np.abs(g - prev).mean()))
            prev = g
            n += 1
        m = min(len(env), len(motion))
        if m < 12:
            res.errors.append("insufficient AV overlap — lip-sync SKIPPED")
            return res
        e, mo = env[:m], np.array(motion)
        r = float(np.corrcoef(e, mo)[0, 1]) if mo.std() > 1e-9 and e.std() > 1e-9 else 0.0
        res.available = True
        res.correlation = round(r, 3)
        res.verdict = "SYNCED" if r > 0.35 else "DIVERGENT"
        if res.verdict == "DIVERGENT":
            res.warnings.append(
                f"AV correlation {r:.2f}: mouth motion diverges from speech — dub/swap flag"
            )
        log.info("lipsync: r=%.2f %s", r, res.verdict)
        return res
    finally:
        cap.release()


# ---------------------------------------------------------------- rPPG


def analyze_rppg(video_path: str | Path, max_seconds: float = 10.0) -> RppgResult:
    """Chrominance rPPG (POS-inspired): green-channel pulse 0.7–4 Hz.

    A living face shows a periodic blood-volume pulse; still images, masks,
    and many face-swaps show none. NO_PULSE on a claimed live capture is a
    strong manipulation flag; quality<0.4 makes the leg abstain (UNKNOWN).
    """
    res = RppgResult()
    try:
        import cv2 as _cv2
    except ImportError:
        res.errors.append("cv2 missing — rPPG SKIPPED")
        return res
    cap = _cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        res.errors.append("video undecodable — rPPG SKIPPED")
        return res
    try:
        fps = cap.get(_cv2.CAP_PROP_FPS) or 25.0
        if fps < 8:
            res.errors.append(f"fps {fps:.1f} too low for pulse extraction")
            return res
        cascade = None
        try:
            data_dir = getattr(_cv2, "data", None)
            if data_dir and hasattr(data_dir, "haarcascades"):
                from pathlib import Path as _P

                cp = _P(str(data_dir.haarcascades)) / "haarcascade_frontalface_default.xml"
                if cp.is_file():
                    cascade = _cv2.CascadeClassifier(  # type: ignore[attr-defined]  # cv2 C++ API
                        str(cp)
                    )
        except Exception:
            cascade = None
        trace: list[float] = []
        frames = 0
        budget = int(min(max_seconds, 12.0) * fps)
        while frames < budget:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            gray = _cv2.cvtColor(frame, _cv2.COLOR_BGR2GRAY)
            roi = None
            if cascade is not None:
                faces = cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
                if len(faces):
                    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
                    fx, fy = x + w // 4, y + h // 5  # forehead: best SNR
                    roi = frame[fy : fy + h // 4, fx : fx + w // 2, 1]
            if roi is None:  # fallback: central cheek band
                h, w = frame.shape[:2]
                roi = frame[int(h * 0.35) : int(h * 0.60), int(w * 0.35) : int(w * 0.65), 1]
            trace.append(float(roi.mean()))
            frames += 1
        if len(trace) < int(3 * fps):
            res.errors.append("clip too short for pulse (<3 s of faces)")
            return res
        sig = np.asarray(trace, dtype=np.float32)
        sig = sig - np.convolve(sig, np.ones(max(int(fps), 1)) / max(int(fps), 1), mode="same")
        sig = (sig - sig.mean()) / (sig.std() + 1e-9)
        spec = np.abs(np.fft.rfft(sig))
        freqs = np.fft.rfftfreq(len(sig), 1 / fps) * 60.0  # bpm axis
        band = (freqs >= 42) & (freqs <= 200)
        if not band.any():
            res.errors.append("unusable spectrum")
            return res
        peak = int(np.argmax(spec * band))
        quality = float(spec[peak] / (spec[band].mean() + 1e-9) / 8.0)
        res.available = True
        res.heart_rate_bpm = round(float(freqs[peak]), 1)
        res.signal_quality = round(float(np.clip(quality, 0, 1)), 3)
        step = max(len(sig) // 120, 1)
        res.waveform = [round(float(v), 3) for v in sig[::step][:120]]
        res.liveness = (
            "LIVE"
            if res.signal_quality >= 0.45 and 45 <= res.heart_rate_bpm <= 180
            else "NO_PULSE"
            if res.signal_quality < 0.25
            else "UNKNOWN"
        )
        if res.liveness == "NO_PULSE":
            res.warnings.append("no physiological pulse — still/mask/swap screening flag")
        log.info("rppg: %.0fbpm q=%.2f %s", res.heart_rate_bpm, res.signal_quality, res.liveness)
        return res
    finally:
        cap.release()


# ---------------------------------------------------------------- orchestrator


def run_forensics_engines(file_path: str | Path) -> dict[str, Any]:
    """Run the full multimodal forensic leg. Returns JSON-serializable dict."""
    from src.modules.forensic_vision import analyze_forensics

    p = Path(file_path)
    visual: ForensicResult = analyze_forensics(p)
    out: dict[str, Any] = {"visual": visual.to_dict()}
    mp4 = inspect_mp4(p)
    out["mp4"] = mp4.to_dict()
    audio = analyze_audio(p)
    out["audio"] = audio.to_dict()
    if is_video(p):
        out["lipsync"] = verify_lipsync(p, audio).to_dict()
        out["rppg"] = analyze_rppg(p).to_dict()
    else:
        out["lipsync"] = LipSyncResult(errors=["image input — lip-sync N/A"]).to_dict()
        out["rppg"] = RppgResult(errors=["image input — rPPG needs video"]).to_dict()
    # Fuse container + visual-DCT legs into the screening hint.
    try:
        dcts = [f.get("dct_double_jpeg", 0) for f in out["visual"].get("frames", [])]
        vdct = max(dcts) if dcts else 0.0
        mp4.double_compression_hint = round(
            min(1.0, 0.5 * mp4.double_compression_hint + 0.5 * float(vdct)), 3
        )
        out["mp4"] = mp4.to_dict()
    except Exception:
        pass
    return out


def multimodal_risk_bump(multi: dict[str, Any]) -> tuple[float, list[str]]:
    """Extra deterministic risk bumps (added to the base score, capped later)."""
    bump, notes = 0.0, []
    au = multi.get("audio", {})
    if au.get("available"):
        if au.get("vocoder_anomaly", 0) > 0.6:
            bump += 12
            notes.append(f"vocoder artifacts {au['vocoder_anomaly']} (+12)")
        if au.get("pitch_instability", 0) > 0.7:
            bump += 8
            notes.append("TTS-flat prosody (+8)")
    ls = multi.get("lipsync", {})
    if ls.get("verdict") == "DIVERGENT":
        bump += 12
        notes.append(f"lip-sync divergent r={ls.get('correlation')} (+12)")
    rp = multi.get("rppg", {})
    if rp.get("liveness") == "NO_PULSE":
        bump += 15
        notes.append("no physiological pulse (+15)")
    mp = multi.get("mp4", {})
    if mp.get("is_mp4") and mp.get("double_compression_hint", 0) > 0.5:
        bump += 8
        notes.append(f"double-compression hint {mp['double_compression_hint']} (+8)")
    wm = multi.get("watermark", {})
    if wm.get("ai_generated") is True:
        bump += 15
        notes.append(f"AI-generator traces ({wm.get('generator')}) (+15)")
    pr = multi.get("prnu", {})
    if pr.get("verdict") == "EXCLUDE":
        bump += 8
        notes.append("sensor exclusion vs enrolled fleet (+8)")
    elif pr.get("verdict") == "MATCH":
        bump -= 8
        notes.append(f"sensor match {pr.get('best_camera')} (−8): supports claimed origin")
    return bump, notes
