"""Train the spectral synthesis head (StandardScaler + LogisticRegression).

Bootstrap corpus with labels FROM CONSTRUCTION (not opinion):
  pristine (y=0)   smooth gradients, flat fields, mild sensor noise, all with
                  a JPEG recompression pass (q75-95) so the head cannot cheat
                  on "never compressed" as the pristine signal.
  manipulated (y=1) noise-patch pastes, spliced blocks from alien gradients,
                  checkerboard high-frequency injection (simulated GAN
                  upsampling lattice), double-JPEG (q60→q90) cycles.

The head consumes the SHARED `spectral_profile()` 32-vector (same function
at train and inference time — no skew by construction) and is exported to
`models/spectral_lr.onnx` with `input: spectral32` registered, so the
forensic ensemble blends it at 35% like any learned head.

HONEST SCOPE: a bootstrap head separates synthetic fixture artifacts, not
production deepfakes. It is a calibrated weak signal (train AUC is printed
and stored in the registry entry); replace with a FaceForensics++/DFDC
trained head for operational use — the registry slot and input contract
are already the right shape.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from src.modules.forensic_vision import spectral_profile

SIZE = 128


def _jpeg(arr: np.ndarray, q: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, q])
    assert ok
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)


def _gradient(rng: np.random.Generator) -> np.ndarray:
    x0, x1 = rng.integers(10, 100), rng.integers(150, 245)
    base = np.tile(np.linspace(x0, x1, SIZE, dtype=np.float32), (SIZE, 1))
    if rng.random() < 0.5:
        base = base.T
    return np.clip(base, 0, 255).astype(np.uint8)


def pristine(rng: np.random.Generator) -> np.ndarray:
    # Cover the operating domain: noisy captures AND clean synthetic
    # backgrounds (memes, gradients), with and without JPEG. The head must
    # learn "smooth = pristine", not "smooth = unseen".
    img = _gradient(rng).astype(np.float32)
    if rng.random() < 0.7:
        img += rng.normal(0, rng.uniform(1.0, 3.0), img.shape)  # sensor-ish noise
        img = np.clip(img, 0, 255).astype(np.uint8)
        return _jpeg(img, int(rng.integers(75, 96)))
    img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        return _jpeg(img, int(rng.integers(75, 96)))
    return img  # never-compressed smooth field


def manipulated(rng: np.random.Generator) -> np.ndarray:
    img = _gradient(rng).astype(np.float32)
    kind = rng.integers(0, 4)
    if kind == 0:  # pasted noise block (splice stand-in)
        x, y = rng.integers(0, 78, 2)
        img[y : y + 50, x : x + 50] = rng.integers(0, 255, (50, 50))
    elif kind == 1:  # alien splice: region from another gradient
        other = _gradient(rng).astype(np.float32)
        x, y = rng.integers(0, 78, 2)
        img[y : y + 50, x : x + 50] = other[y : y + 50, x : x + 50]
    elif kind == 2:  # checkerboard upsampling lattice (GAN fingerprint proxy)
        yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
        img += 18 * np.sin(np.pi * xx) * np.sin(np.pi * yy)
    else:  # double-JPEG cycle
        img = _jpeg(np.clip(img, 0, 255).astype(np.uint8), 60).astype(np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return _jpeg(img, int(rng.integers(85, 96)))


def build_corpus(n_per_class: int = 60, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for _ in range(n_per_class):
        xs.append(spectral_profile(pristine(rng)))
        ys.append(0)
    for _ in range(n_per_class):
        xs.append(spectral_profile(manipulated(rng)))
        ys.append(1)
    return np.stack(xs), np.array(ys)


def main() -> int:
    ap = argparse.ArgumentParser(description="Train spectral synthesis head")
    ap.add_argument("--n-per-class", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="models/spectral_lr.onnx")
    ap.add_argument("--registry", default="models/registry.json")
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = build_corpus(args.n_per_class, args.seed)
    assert np.isfinite(X).all(), "feature leakage: non-finite profile"
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=args.seed, stratify=y)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    clf.fit(Xtr, ytr)
    auc = float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))
    print(f"train n={len(y)} held-out AUC={auc:.3f}")
    if auc < 0.70:
        print("REFUSING: head below 0.70 AUC — corpus or features need work", file=sys.stderr)
        return 1

    from skl2onnx import to_onnx
    from skl2onnx.common.data_types import FloatTensorType

    onx = to_onnx(
        clf,
        Xtr[:1].astype(np.float32),
        initial_types=[("input", FloatTensorType([None, 32]))],
        options={type(clf.steps[-1][1]): {"zipmap": False}},
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(onx.SerializeToString())

    # Verify parity sklearn <-> onnxruntime before registering.
    import onnxruntime as ort

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    out_names = [o.name for o in sess.get_inputs()]
    assert out_names == ["input"], out_names

    def _p_proba(row: np.ndarray) -> float:
        outs = sess.run(None, {"input": row.reshape(1, -1).astype(np.float32)})
        for o in outs:  # probability tensor has width 2 (or 1 for degenerate fits)
            a = np.asarray(o)
            if a.ndim == 2 and a.shape[1] == 2:
                return float(a[0, 1])
        flat = np.asarray(outs[0]).ravel()
        return float(flat[0]) if flat.size else 0.0

    got = np.array([_p_proba(r) for r in Xte.astype(np.float32)])
    ref = clf.predict_proba(Xte.astype(np.float32))[:, 1]
    maxdiff = float(np.abs(ref - got).max())
    print(f"onnx parity max|Δp|={maxdiff:.2e}")
    assert maxdiff < 1e-4, "sklearn/onnxruntime divergence"

    reg = Path(args.registry)
    doc = json.loads(reg.read_text(encoding="utf-8"))
    entry = {
        "name": "spectral-lr",
        "path": args.out,
        "task": "synthetic",
        "input": "spectral32",
        "version": f"bootstrap-v1-auc{auc:.3f}-n{len(y)}",
        "notes": "LR on shared 32-bin radial spectral profile; bootstrap corpus "
        "(labels from construction). Weak signal, blended 35%. Replace "
        "with FaceForensics++/DFDC head for operations.",
    }
    doc["models"] = [m for m in doc.get("models", []) if m.get("name") != "spectral-lr"]
    doc["models"].append(entry)
    reg.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"registered spectral-lr in {reg}; sign with scripts/sign_models.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
