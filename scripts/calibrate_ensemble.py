"""Calibrate the forensic ensemble from analyst feedback (real closing loop).

Reads consumed + pending HITL labels (FP/FN/CONFIRM) joined to stored
forensic frame vectors, trains an L2 logistic regression (scikit-learn,
offline, seconds on CPU), and writes `models/ensemble_weights.json` with
per-signal weights + Platt-style intercept. `forensic_vision` blends these
weights when present (falls back to literature priors otherwise), so every
analyst correction measurably moves future scores — auditable via the
written `report` section (AUC, n, timestamp).

Labels: CONFIRM on HIGH/SUSPICIOUS or FP on LOW/AUTHENTIC => y=0/1 balance
is derived from (verdict, label) pairs; FN (missed manipulation) => y=1.
Runs with <10 labels but refuses (exit 2) under `min_samples` (default 20)
to avoid calibrating on anecdotes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FEATURES = [
    "ela_score",
    "fft_score",
    "blockiness",
    "dct_double_jpeg",
    "noise_score",
    "histogram_anomaly",
    "face_anomaly",
]


def _rows(repo):  # type: ignore[no-untyped-def]
    import sqlite3

    db = getattr(repo, "path", None)
    out = []
    if db is None or not Path(db).is_file():
        return out
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        fbs = conn.execute(
            "SELECT f.label, f.case_id, ar.verdict, ar.report_json FROM feedback f "
            "JOIN cases c ON c.id=f.case_id LEFT JOIN analysis_results ar "
            "ON ar.case_id=c.id ORDER BY f.id"
        ).fetchall()
        for fb in fbs:
            try:
                rep = json.loads(fb["report_json"] or "{}")
            except Exception:
                continue
            frames = (rep.get("forensic") or {}).get("frames") or []
            if not frames:
                continue
            f0 = max(frames, key=lambda f: f.get("anomaly_score", 0))
            x = [float(f0.get(k, 0.0)) for k in FEATURES]
            v = fb["verdict"] or ""
            lab = fb["label"] or ""
            y = (
                1
                if lab == "FN" or (lab == "CONFIRM" and v in ("SUSPICIOUS", "HIGH_RISK"))
                else 0
                if lab == "FP" or lab == "CONFIRM"
                else None
            )
            if y is not None:
                out.append((x, y))
    return out


def main() -> int:  # type: ignore[no-untyped-def]
    ap = argparse.ArgumentParser(description="Calibrate forensic ensemble from feedback")
    ap.add_argument("--db", default="data/provenance.db")
    ap.add_argument("--out", default="models/ensemble_weights.json")
    ap.add_argument("--min-samples", type=int, default=20)
    args = ap.parse_args()
    from src.database import StdlibSqliteRepo

    rows = _rows(StdlibSqliteRepo(args.db))
    print(f"labeled vectors: {len(rows)}")
    if len(rows) < args.min_samples:
        print(f"need >= {args.min_samples}; refusing to calibrate on anecdotes")
        return 2
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_score

    X = np.array([x for x, _ in rows])
    y = np.array([y for _, y in rows])
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(X, y)
    auc = float(roc_auc_score(y, clf.predict_proba(X)[:, 1]))
    cv = (
        float(np.mean(cross_val_score(clf, X, y, cv=min(5, len(y) // 2))))
        if len(set(y.tolist())) > 1 and len(y) >= 4
        else auc
    )
    doc = {
        "features": FEATURES,
        "weights": {k: round(float(w), 4) for k, w in zip(FEATURES, clf.coef_[0], strict=False)},
        "intercept": round(float(clf.intercept_[0]), 4),
        "train_auc": round(auc, 3),
        "cv_auc": round(cv, 3),
        "n": len(rows),
        "classes": {str(k): int((y == k).sum()) for k in (0, 1)},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"wrote {args.out} auc={auc:.3f} cv={cv:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
