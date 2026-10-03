"""Bootstrap the Active Learning loop with ground-truth labels.

A cold-start problem: `calibrate_ensemble.py` refuses under `--min-samples`
labels, and analysts haven't labeled anything yet. This script manufactures
the FIRST labels honestly — from construction ground truth, never from
opinion:

  12 pristine fixtures  (gradients/flats, PNG + JPEG) -> expected LOW risk
  12 manipulated fixtures (noise pastes, alien splices, checkerboard
      injection, double-JPEG — seeds disjoint from train_spectral_head)
      -> expected HIGH risk

Each fixture runs the FULL pipeline (`main.analyze`); the label follows the
calibration mapping against ground truth:
  manipulated scored HIGH  -> CONFIRM (y=1)   manipulated scored LOW -> FN (y=1)
  pristine scored LOW      -> CONFIRM (y=0)   pristine scored HIGH    -> FP (y=0)

Cases+results+feedback land in `--db` (default a scratch file, NOT the dev
database), ready for `calibrate_ensemble.py --db <same>`. Human labels
later supersede these rows (identical schema, `analyst="bootstrap"` marks
provenance of the label itself).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np


def _gradient(rng: np.random.Generator, size: int = 160) -> np.ndarray:
    x0, x1 = rng.integers(10, 100), rng.integers(150, 245)
    base = np.tile(np.linspace(x0, x1, size, dtype=np.float32), (size, 1))
    return np.clip(base.T if rng.random() < 0.5 else base, 0, 255).astype(np.uint8)


def _save(path: Path, img: np.ndarray, jpeg_q: int | None = None) -> None:
    if jpeg_q is None:
        cv2.imwrite(str(path), img)
    else:
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])


def build_fixtures(outdir: Path, seed: int = 99) -> list[tuple[Path, int]]:
    """Return [(path, y)] with y=1 manipulated. Seeds avoid the trainer's."""
    rng = np.random.default_rng(seed)
    outdir.mkdir(parents=True, exist_ok=True)
    fixtures: list[tuple[Path, int]] = []
    for i in range(12):
        img = _gradient(rng).astype(np.float32)
        img += rng.normal(0, 2.0, img.shape)
        img = np.clip(img, 0, 255).astype(np.uint8)
        p = outdir / f"pris_{i:02d}.{'jpg' if i % 2 else 'png'}"
        _save(p, img, 85 if p.suffix == ".jpg" else None)
        fixtures.append((p, 0))
    kinds = ["noise", "splice", "checker", "double"]
    for i in range(12):
        img = _gradient(rng).astype(np.float32)
        kind = kinds[i % 4]
        if kind == "noise":
            x, y = rng.integers(0, 100, 2)
            img[y : y + 60, x : x + 60] = rng.integers(0, 255, (60, 60))
        elif kind == "splice":
            other = _gradient(rng).astype(np.float32)
            x, y = rng.integers(0, 100, 2)
            img[y : y + 60, x : x + 60] = other[y : y + 60, x : x + 60]
        elif kind == "checker":
            yy, xx = np.mgrid[0:160, 0:160].astype(np.float32)
            img += 20 * np.sin(np.pi * xx) * np.sin(np.pi * yy)
        else:
            ok, buf = cv2.imencode(
                ".jpg", np.clip(img, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 55]
            )
            assert ok
            img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        img = np.clip(img, 0, 255).astype(np.uint8)
        p = outdir / f"manip_{i:02d}.jpg"
        _save(p, img, 90)
        fixtures.append((p, 1))
    return fixtures


def main() -> int:
    ap = argparse.ArgumentParser(description="Seed AL loop with ground-truth labels")
    ap.add_argument("--db", default="/tmp/opencode/al_bootstrap.db")
    ap.add_argument("--workdir", default="/tmp/opencode/al_fixtures")
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()

    import main as pipeline
    from src.database import get_repository, reset_repository

    reset_repository()
    repo = get_repository(sqlite_path=args.db)
    fixtures = build_fixtures(Path(args.workdir), args.seed)
    counts = {"CONFIRM": 0, "FP": 0, "FN": 0}
    for path, y in fixtures:
        report = pipeline.analyze(str(path))
        verdict = report["verdict"]
        high = verdict in ("SUSPICIOUS", "HIGH_RISK")
        # Label from construction ground truth: agreement -> CONFIRM,
        # manipulated-but-missed -> FN, clean-but-flagged -> FP.
        label = {(1, True): "CONFIRM", (1, False): "FN", (0, True): "FP", (0, False): "CONFIRM"}[
            (y, high)
        ]
        case = repo.create_case(
            filename=path.name,
            sha256=report["c2pa"]["sha256"],
            file_size=1,
            mime_type="image/jpeg",
            tlp="TLP:CLEAR",
            owner="bootstrap",
            storage_ref=str(path),
        )
        repo.save_result(
            case["case_uuid"],
            pipeline_version="bootstrap",
            risk_score=report["risk_score"],
            verdict=verdict,
            confidence=report.get("analytic_confidence", "Moderate"),
            report_json=json.dumps(report),
        )
        repo.add_feedback(
            case["case_uuid"],
            analyst="bootstrap",
            label=label,
            finding_ref=f"ground-truth y={y}",
            notes=f"constructed fixture ({path.stem}); score={report['risk_score']}",
        )
        counts[label] += 1
    print(f"seeded {len(fixtures)} labeled rows into {args.db}: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
