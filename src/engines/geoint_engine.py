"""Engine 3 — GEOINT solar/shadow analysis (Sun Position Algorithm).

Implements the NREL/NOAA low-precision SPA (Reda & Andreas, ~0.0003° class
inputs simplified to ±0.01° — ample for shadow forensics) in dependency-free
NumPy, then checks *claimed* capture geometry:

  solar elevation/azimuth from (lat, lon, UTC) -> expected shadow azimuth
  (sun + 180°) and shadow/object length ratio (1/tan(elev)).
  measured shadow azimuth: analyst-supplied (preferred) or coarse
  Canny+Hough auto-estimate (cv2 only, low confidence).

Outputs angular error + consistency flag with tolerance (default 12°) and a
confidence that reflects how the measurement was obtained. Auto-measurement is
screening-grade: perspective, slope, and lens distortion are not modeled, and
the report says so explicitly (honest uncertainty beats false precision).
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.schemas import GeointResult

log = logging.getLogger(__name__)

_TOL_DEG = 12.0


def solar_position(lat_deg: float, lon_deg: float, when_utc: datetime) -> tuple[float, float]:
    """Return (elevation_deg, azimuth_deg clockwise from North)."""
    if when_utc.tzinfo is None:
        when_utc = when_utc.replace(tzinfo=timezone.utc)
    jd = when_utc.timestamp() / 86400.0 + 2440587.5
    jc = (jd - 2451545.0) / 36525.0
    # Geometric mean longitude / anomaly (degrees).
    L0 = (280.46646 + jc * (36000.76983 + jc * 0.0003032)) % 360
    M = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    Mr = math.radians(M)
    C = (
        math.sin(Mr) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
        + math.sin(2 * Mr) * (0.019993 - 0.000101 * jc)
        + math.sin(3 * Mr) * 0.000289
    )
    true_long = L0 + C
    omega = 125.04 - 1934.136 * jc
    lam = true_long - 0.00569 - 0.00478 * math.sin(math.radians(omega))
    eps0 = 23 + (26 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60) / 60
    eps = eps0 + 0.00256 * math.cos(math.radians(omega))
    lamr, epsr = math.radians(lam), math.radians(eps)
    dec = math.degrees(math.asin(math.sin(epsr) * math.sin(lamr)))
    y, x = math.cos(epsr) * math.sin(lamr), math.cos(lamr)
    ra = math.degrees(math.atan2(y, x)) % 360
    # Equation of time + solar noon + hour angle.
    eq = 4 * (L0 - 0.0057183 - ra)
    noon_min = 720 - 4 * lon_deg - eq
    mins = when_utc.hour * 60 + when_utc.minute + when_utc.second / 60
    ha = (mins - noon_min) / 4.0  # degrees
    latr, decr, har = map(math.radians, (lat_deg, dec, ha))
    cosz = math.sin(latr) * math.sin(decr) + math.cos(latr) * math.cos(decr) * math.cos(har)
    zen = math.degrees(math.acos(max(-1.0, min(1.0, cosz))))
    elev = 90.0 - zen
    az = (
        math.degrees(
            math.atan2(
                math.sin(har), math.cos(har) * math.sin(latr) - math.tan(decr) * math.cos(latr)
            )
        )
        + 180.0
    ) % 360
    # NOAA refraction correction near horizon.
    if -0.575 < elev < 85:
        te = math.tan(math.radians(elev + 10.3 / (elev + 5.11) if elev > -5 else elev))
        elev += (1.02 / 60) / te if te != 0 else 0
    return round(elev, 3), round(az, 3)


def auto_shadow_angle(image_path: str | Path) -> tuple[float | None, float]:
    """Coarse dominant-line orientation (lower half) via Canny+Hough.

    Returns (azimuth_deg|None, confidence). Confidence caps at 0.45: this is a
    screening estimate, never survey-grade (perspective unmodeled).
    """
    try:
        import cv2 as _cv2
    except ImportError:
        return None, 0.0
    img = _cv2.imread(str(image_path), _cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, 0.0
    h = img.shape[0]
    low = img[int(h * 0.45) :, :]
    edges = _cv2.Canny(low, 60, 160)
    lines = _cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=80, minLineLength=min(low.shape) // 3, maxLineGap=12
    )
    if lines is None or len(lines) < 2:
        return None, 0.0
    angs = []
    for x1, y1, x2, y2 in lines[:, 0]:
        dx, dy = x2 - x1, y2 - y1
        if abs(dx) + abs(dy) < 1:
            continue
        # Image y grows downward; map to compass-ish azimuth of the dark edge.
        angs.append((math.degrees(math.atan2(dx, -dy)) + 360) % 360)
    if not angs:
        return None, 0.0
    circ = np.exp(1j * np.radians([a * 2 for a in angs])).mean()  # axial average
    az = (math.degrees(np.angle(circ)) / 2) % 360
    conf = round(min(0.45, 0.15 + 0.05 * len(angs)), 2)
    return round(az, 1), conf


def verify_geoint(
    image_path: str | Path,
    *,
    latitude: float,
    longitude: float,
    timestamp_utc: str,
    measured_shadow_deg: float | None = None,
    object_height_m: float | None = None,
    shadow_length_m: float | None = None,
    tolerance_deg: float = _TOL_DEG,
) -> GeointResult:
    """Check claimed capture geometry against solar truth."""
    res = GeointResult(latitude=latitude, longitude=longitude, timestamp_utc=timestamp_utc)
    try:
        when = datetime.fromisoformat(timestamp_utc.replace("Z", "+00:00"))
    except ValueError as exc:
        res.errors.append(f"bad timestamp (need ISO-8601 UTC): {exc}")
        return res
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        res.errors.append("coordinates out of range")
        return res
    elev, azim = solar_position(latitude, longitude, when)
    res.solar_elevation_deg, res.solar_azimuth_deg = elev, azim
    res.expected_shadow_azimuth_deg = round((azim + 180) % 360, 2)
    if elev <= 0:
        res.warnings.append(
            f"sun below horizon (elev={elev}°): shadows impossible — "
            "night capture claimed as day, or wrong timestamp"
        )
        res.consistent, res.confidence, res.available = False, 0.85, True
        return res
    if elev > 0.5:
        res.shadow_length_ratio_expected = round(1 / math.tan(math.radians(elev)), 3)
    measured, mconf, method = measured_shadow_deg, 0.9, "analyst"
    if measured is None:
        auto, aconf = auto_shadow_angle(image_path)
        measured, mconf, method = auto, aconf, "auto-coarse"
    if measured is None:
        res.warnings.append("no measurable shadow edge; supply measured_shadow_deg for a verdict")
        res.available = True
        return res
    res.measured_shadow_azimuth_deg = round(measured % 360, 1)
    err = abs((res.measured_shadow_azimuth_deg - res.expected_shadow_azimuth_deg + 180) % 360 - 180)
    res.angular_error_deg = round(err, 2)
    res.consistent = err <= tolerance_deg
    res.confidence = round(mconf * (0.9 if method == "analyst" else 0.6), 2)
    res.available = True
    if object_height_m and shadow_length_m and res.shadow_length_ratio_expected:
        observed = shadow_length_m / max(object_height_m, 1e-6)
        if (
            abs(observed - res.shadow_length_ratio_expected) / res.shadow_length_ratio_expected
            > 0.35
        ):
            res.warnings.append(
                f"length ratio {observed:.2f} vs solar {res.shadow_length_ratio_expected}: "
                "inconsistent sun height — wrong time/place or composited shadows"
            )
            res.consistent = False
    if not res.consistent:
        res.warnings.append(
            f"shadow azimuth off by {err:.1f}° (tol {tolerance_deg}°, via {method}): "
            "timestamp/geolocation claim contradicts solar geometry"
        )
    log.info(
        "geoint: elev=%.1f az=%.1f err=%s consistent=%s",
        elev,
        azim,
        res.angular_error_deg,
        res.consistent,
    )
    return res


def to_dict_safe(res: GeointResult) -> dict[str, Any]:
    return res.to_dict()
