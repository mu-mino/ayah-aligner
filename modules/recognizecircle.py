"""
Kreiserkennung (Ring-Marker-Detektion).

Enthält ausschließlich die geometrische Logik zur Erkennung von
Ring-förmigen Markern mit den für dieses Projekt definierten Merkmalen:
    - Fläche, Breite/Höhe, Seitenverhältnis
    - Kreisförmigkeit (4π·A/P²)
    - Stroke-Ratio (Ringbreite / Gesamtfläche)
    - Hole-Ratio (Innenfläche / Gesamtfläche)

Nicht enthalten: OCR-Validierung, Video-Stream-Verarbeitung, Mapping.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None  # type: ignore[assignment]
import numpy as np

# ---------------------------------------------------------------------------
# Tuning-Parameter (empirische Standardwerte)
# ---------------------------------------------------------------------------

RING_AREA_RANGE: Tuple[float, float] = (1500, 3600)
RING_WIDTH_RANGE: Tuple[float, float] = (45, 82)
RING_HEIGHT_RANGE: Tuple[float, float] = (45, 82)
RING_ASPECT_RANGE: Tuple[float, float] = (0.75, 1.30)
RING_CIRCULARITY_RANGE: Tuple[float, float] = (0.18, 1.00)
RING_STROKE_RATIO_RANGE: Tuple[float, float] = (0.30, 0.70)
RING_HOLE_RATIO_RANGE: Tuple[float, float] = (0.30, 0.70)
RING_STROKE_FRACTION: float = 0.22  # Anteil des Durchmessers für den Erosions-Kernel

CIRCLE_PATCH_SIZE: int = 64
CIRCLE_REPEAT_SIM_THRESHOLD: float = 0.95


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def _require_cv2() -> None:
    if cv2 is None:
        raise RuntimeError(
            "OpenCV (cv2) ist nicht installiert. "
            "Bitte Abhaengigkeiten installieren oder den Run mit '.venv/bin/python' starten."
        )


def _threshold_binary(gray: np.ndarray) -> np.ndarray:
    """Otsu-Schwellwert: Graustufenbild → binäres Bild."""
    _require_cv2()
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def _shrink_range(rng: Tuple[float, float], pct: float) -> Tuple[float, float]:
    span = rng[1] - rng[0]
    pad = span * pct
    return (rng[0] + pad, rng[1] - pad)


# ---------------------------------------------------------------------------
# Datenstrukturen
# ---------------------------------------------------------------------------


@dataclass
class DetectorParams:
    """Parametrisierbarer Satz an Erkennungsgrenzen für den Ring-Detektor."""

    area_range: Tuple[float, float]
    width_range: Tuple[float, float]
    height_range: Tuple[float, float]
    aspect_range: Tuple[float, float]
    circularity_range: Tuple[float, float]
    stroke_ratio_range: Tuple[float, float]
    hole_ratio_range: Tuple[float, float]
    stroke_fraction: float = RING_STROKE_FRACTION

    def clamp(self, bounds: "DetectorParams") -> "DetectorParams":
        """Begrenzt alle Ranges auf die Grenzen von *bounds*."""

        def clamp_range(rng, bound):
            return (max(bound[0], rng[0]), min(bound[1], rng[1]))

        return DetectorParams(
            area_range=clamp_range(self.area_range, bounds.area_range),
            width_range=clamp_range(self.width_range, bounds.width_range),
            height_range=clamp_range(self.height_range, bounds.height_range),
            aspect_range=clamp_range(self.aspect_range, bounds.aspect_range),
            circularity_range=clamp_range(
                self.circularity_range, bounds.circularity_range
            ),
            stroke_ratio_range=clamp_range(
                self.stroke_ratio_range, bounds.stroke_ratio_range
            ),
            hole_ratio_range=clamp_range(
                self.hole_ratio_range, bounds.hole_ratio_range
            ),
            stroke_fraction=self.stroke_fraction,
        )


DEFAULT_DETECTOR_PARAMS = DetectorParams(
    area_range=RING_AREA_RANGE,
    width_range=RING_WIDTH_RANGE,
    height_range=RING_HEIGHT_RANGE,
    aspect_range=RING_ASPECT_RANGE,
    circularity_range=RING_CIRCULARITY_RANGE,
    stroke_ratio_range=RING_STROKE_RATIO_RANGE,
    hole_ratio_range=RING_HOLE_RATIO_RANGE,
    stroke_fraction=RING_STROKE_FRACTION,
)

_EMPIRICAL_BOUNDS: Optional[DetectorParams] = None


@dataclass
class RingCandidate:
    outer_mask: np.ndarray
    inner_mask: np.ndarray
    ring_mask: np.ndarray
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    center: Tuple[float, float]  # (cx, cy)
    radius: float  # max(w, h) / 2
    stroke_ratio: float
    hole_ratio: float
    circularity: float
    stroke_kernel: int
    score: float = 0.0


# ---------------------------------------------------------------------------
# Kern-Detektion
# ---------------------------------------------------------------------------


def _score_candidate(candidate: RingCandidate, params: DetectorParams) -> float:
    """Geometrischer Güte-Score (höher = besser)."""
    w, h = candidate.bbox[2], candidate.bbox[3]
    area = cv2.countNonZero(candidate.outer_mask)
    aspect = w / float(h)

    def center_score(val, rng):
        mid = (rng[0] + rng[1]) / 2.0
        span = max(1e-6, rng[1] - rng[0])
        return max(0.0, 1.0 - abs(val - mid) / span)

    scores = [
        center_score(area, params.area_range),
        center_score(w, params.width_range),
        center_score(h, params.height_range),
        center_score(aspect, params.aspect_range),
        center_score(candidate.circularity, params.circularity_range),
        center_score(candidate.stroke_ratio, params.stroke_ratio_range),
        center_score(candidate.hole_ratio, params.hole_ratio_range),
    ]
    return float(sum(scores) / len(scores))


def _find_ring_candidates(
    binary: np.ndarray,
    params: DetectorParams = DEFAULT_DETECTOR_PARAMS,
    score_candidates: bool = True,
) -> Tuple[List[RingCandidate], np.ndarray]:
    """
    Findet ring-förmige Konturen im Binärbild.

    Filtert nach: Fläche, Breite, Höhe, Seitenverhältnis, Kreisförmigkeit,
    Stroke-Ratio und Hole-Ratio.

    Gibt (candidates, ring_mask) zurück.
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ring_mask = np.zeros_like(binary)
    candidates: List[RingCandidate] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (params.area_range[0] <= area <= params.area_range[1]):
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if not (
            params.width_range[0] <= w <= params.width_range[1]
            and params.height_range[0] <= h <= params.height_range[1]
        ):
            continue

        aspect_ratio = w / float(h)
        if not (params.aspect_range[0] <= aspect_ratio <= params.aspect_range[1]):
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0:
            continue
        circularity = 4.0 * math.pi * area / (perimeter * perimeter)
        if not (
            params.circularity_range[0] <= circularity <= params.circularity_range[1]
        ):
            continue

        outer = np.zeros_like(binary)
        cv2.drawContours(outer, [cnt], -1, 255, thickness=cv2.FILLED)

        ksize = max(3, int(round(max(w, h) * params.stroke_fraction)))
        if ksize % 2 == 0:
            ksize += 1
        inner = cv2.erode(
            outer, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        )
        ring = cv2.subtract(outer, inner)

        outer_area = cv2.countNonZero(outer)
        if outer_area == 0:
            continue
        stroke_ratio = cv2.countNonZero(ring) / float(outer_area)
        hole_ratio = cv2.countNonZero(inner) / float(outer_area)

        if not (
            params.stroke_ratio_range[0] <= stroke_ratio <= params.stroke_ratio_range[1]
        ):
            continue
        if not (params.hole_ratio_range[0] <= hole_ratio <= params.hole_ratio_range[1]):
            continue

        cand = RingCandidate(
            outer_mask=outer,
            inner_mask=inner,
            ring_mask=ring,
            bbox=(x, y, w, h),
            center=(x + w / 2.0, y + h / 2.0),
            radius=max(w, h) / 2.0,
            stroke_ratio=stroke_ratio,
            hole_ratio=hole_ratio,
            circularity=circularity,
            stroke_kernel=ksize,
        )
        cand.score = _score_candidate(cand, params) if score_candidates else 0.0
        candidates.append(cand)
        ring_mask = cv2.bitwise_or(ring_mask, ring)

    return candidates, ring_mask


# ---------------------------------------------------------------------------
# Empirische Parametergrenzen (aus Beispiel-Frames abgeleitet)
# ---------------------------------------------------------------------------


def _derive_empirical_bounds(
    sample_dir: Path = Path("tmp_frames"),
) -> Optional[DetectorParams]:
    """
    Berechnet robuste Grenzen (5–95 Perzentil) aus beschrifteten Beispiel-Frames.
    Gibt None zurück, wenn keine Frames gefunden werden.
    """
    sample_paths = list(sample_dir.glob("frame_*.png"))
    if not sample_paths:
        return None

    metrics: Dict[str, list] = {
        "areas": [],
        "widths": [],
        "heights": [],
        "aspects": [],
        "circularities": [],
        "stroke_ratios": [],
        "hole_ratios": [],
    }

    for path in sample_paths:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        binary = _threshold_binary(gray)
        candidates, _ = _find_ring_candidates(
            binary, DEFAULT_DETECTOR_PARAMS, score_candidates=False
        )
        for cand in candidates:
            x, y, w, h = cand.bbox
            metrics["areas"].append(float(cv2.countNonZero(cand.outer_mask)))
            metrics["widths"].append(float(w))
            metrics["heights"].append(float(h))
            metrics["aspects"].append(w / float(h))
            metrics["stroke_ratios"].append(cand.stroke_ratio)
            metrics["hole_ratios"].append(cand.hole_ratio)
            metrics["circularities"].append(cand.circularity)

    if not metrics["areas"]:
        return None

    def perc_range(values):
        vals = np.array(values, dtype=np.float32)
        lo, hi = np.percentile(vals, [5, 95])
        return (float(lo), float(hi))

    def widen(
        lo: float, hi: float, base: Tuple[float, float], min_pad: float = 0.02
    ) -> Tuple[float, float]:
        span = hi - lo
        pad = max(min_pad, 0.05 * span)
        return (max(base[0], lo - pad), min(base[1], hi + pad))

    return DetectorParams(
        area_range=widen(
            *perc_range(metrics["areas"]),
            DEFAULT_DETECTOR_PARAMS.area_range,
            min_pad=80.0,
        ),
        width_range=widen(
            *perc_range(metrics["widths"]),
            DEFAULT_DETECTOR_PARAMS.width_range,
            min_pad=1.0,
        ),
        height_range=widen(
            *perc_range(metrics["heights"]),
            DEFAULT_DETECTOR_PARAMS.height_range,
            min_pad=1.0,
        ),
        aspect_range=widen(
            *perc_range(metrics["aspects"]),
            DEFAULT_DETECTOR_PARAMS.aspect_range,
            min_pad=0.1,
        ),
        circularity_range=widen(
            *perc_range(metrics["circularities"]),
            DEFAULT_DETECTOR_PARAMS.circularity_range,
            min_pad=0.03,
        ),
        stroke_ratio_range=widen(
            *perc_range(metrics["stroke_ratios"]),
            DEFAULT_DETECTOR_PARAMS.stroke_ratio_range,
            min_pad=0.01,
        ),
        hole_ratio_range=widen(
            *perc_range(metrics["hole_ratios"]),
            DEFAULT_DETECTOR_PARAMS.hole_ratio_range,
            min_pad=0.01,
        ),
        stroke_fraction=RING_STROKE_FRACTION,
    )


def _get_empirical_bounds() -> DetectorParams:
    """Gibt die einmalig berechneten empirischen Grenzen zurück (lazy init)."""
    global _EMPIRICAL_BOUNDS
    if _EMPIRICAL_BOUNDS is None:
        _EMPIRICAL_BOUNDS = _derive_empirical_bounds() or DEFAULT_DETECTOR_PARAMS
    return _EMPIRICAL_BOUNDS


# ---------------------------------------------------------------------------
# Detektor-Presets
# ---------------------------------------------------------------------------


def _build_detector_presets() -> List[Tuple[str, DetectorParams]]:
    """
    Erstellt vier Parametervarianten auf Basis der empirischen Grenzen:
      - "base":          empirische Grenzen
      - "tight":         5 % engere Grenzen
      - "stroke_plus":   dickerer Stroke-Kernel
      - "stroke_minus":  dünnerer Stroke-Kernel
    """
    bounds = _get_empirical_bounds()

    tight = DetectorParams(
        area_range=_shrink_range(bounds.area_range, 0.05),
        width_range=_shrink_range(bounds.width_range, 0.05),
        height_range=_shrink_range(bounds.height_range, 0.05),
        aspect_range=_shrink_range(bounds.aspect_range, 0.05),
        circularity_range=_shrink_range(bounds.circularity_range, 0.05),
        stroke_ratio_range=_shrink_range(bounds.stroke_ratio_range, 0.03),
        hole_ratio_range=_shrink_range(bounds.hole_ratio_range, 0.03),
        stroke_fraction=max(0.18, min(0.26, bounds.stroke_fraction)),
    ).clamp(bounds)

    stroke_plus = DetectorParams(
        area_range=bounds.area_range,
        width_range=bounds.width_range,
        height_range=bounds.height_range,
        aspect_range=bounds.aspect_range,
        circularity_range=bounds.circularity_range,
        stroke_ratio_range=bounds.stroke_ratio_range,
        hole_ratio_range=bounds.hole_ratio_range,
        stroke_fraction=max(0.18, min(0.26, bounds.stroke_fraction + 0.02)),
    )

    stroke_minus = DetectorParams(
        area_range=bounds.area_range,
        width_range=bounds.width_range,
        height_range=bounds.height_range,
        aspect_range=bounds.aspect_range,
        circularity_range=bounds.circularity_range,
        stroke_ratio_range=bounds.stroke_ratio_range,
        hole_ratio_range=bounds.hole_ratio_range,
        stroke_fraction=max(0.18, min(0.26, bounds.stroke_fraction - 0.02)),
    )

    return [
        ("base", bounds),
        ("tight", tight),
        ("stroke_plus", stroke_plus),
        ("stroke_minus", stroke_minus),
    ]


DETECTOR_PRESETS: List[Tuple[str, DetectorParams]] = _build_detector_presets()


# ---------------------------------------------------------------------------
# Patch-Extraktion und Ähnlichkeitsvergleich
# ---------------------------------------------------------------------------


def _extract_circle_patch(
    gray: np.ndarray,
    cand: RingCandidate,
    patch_size: int = CIRCLE_PATCH_SIZE,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Extrahiert den Kreisbereich (Stroke + Inhalt) als normalisiertes Patch.
    Gibt (patch_gray, patch_mask) zurück oder None bei Fehler.
    """
    x, y, w, h = cand.bbox
    pad = max(2, int(round(0.12 * max(w, h))))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(gray.shape[1], x + w + pad)
    y1 = min(gray.shape[0], y + h + pad)
    if x1 <= x0 or y1 <= y0:
        return None

    crop = gray[y0:y1, x0:x1]
    outer = cand.outer_mask[y0:y1, x0:x1]
    if crop.size == 0 or outer.size == 0:
        return None

    crop_r = cv2.resize(crop, (patch_size, patch_size), interpolation=cv2.INTER_AREA)
    outer_r = cv2.resize(
        outer, (patch_size, patch_size), interpolation=cv2.INTER_NEAREST
    )
    mask = outer_r > 0
    if int(mask.sum()) == 0:
        return None

    patch = np.full((patch_size, patch_size), 255, dtype=np.uint8)
    patch[mask] = crop_r[mask]
    return patch, mask.astype(np.uint8)


def _circle_patch_similarity(
    prev_patch: np.ndarray,
    prev_mask: np.ndarray,
    cur_patch: np.ndarray,
    cur_mask: np.ndarray,
) -> float:
    """
    Pixel-Ähnlichkeit zweier Kreis-Patches über die gemeinsame Maske (MAE-basiert).
    1.0 = identisch, 0.0 = maximal verschieden.
    """
    overlap = (prev_mask > 0) & (cur_mask > 0)
    if int(overlap.sum()) == 0:
        return 0.0
    a = prev_patch[overlap].astype(np.float32)
    b = cur_patch[overlap].astype(np.float32)
    mae = float(np.mean(np.abs(a - b)) / 255.0)
    return max(0.0, min(1.0, 1.0 - mae))


def _best_circle_patch(
    gray: np.ndarray,
) -> Tuple[Optional[Tuple[np.ndarray, np.ndarray]], int]:
    """
    Gibt das Patch des am besten bewerteten Kreises und die Gesamtzahl
    erkannter Kandidaten zurück.
    """
    binary = _threshold_binary(gray)
    candidates, _ = _find_ring_candidates(binary, _get_empirical_bounds())
    if not candidates:
        return None, 0
    best = sorted(candidates, key=lambda c: c.score, reverse=True)[0]
    return _extract_circle_patch(gray, best), len(candidates)


# ---------------------------------------------------------------------------
# Normalisierung
# ---------------------------------------------------------------------------


def normalize_ring_mask(gray: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Unterdrückt innere Beschriftungen von Ring-Markern.
    Gibt (normalisierte_Maske, Anzahl_erkannter_Kreise) zurück.
    """
    binary = _threshold_binary(gray)
    candidates, _ = _find_ring_candidates(binary, _get_empirical_bounds())
    if not candidates:
        return binary, 0

    normalized = binary.copy()
    for cand in candidates:
        normalized = cv2.bitwise_and(normalized, cv2.bitwise_not(cand.inner_mask))
        normalized = cv2.bitwise_or(normalized, cand.ring_mask)
    return normalized, len(candidates)


# ---------------------------------------------------------------------------
# Öffentliche Erkennungs-API
# ---------------------------------------------------------------------------


def detect_markers_from_gray(gray: np.ndarray) -> int:
    """
    Erkennt Ring-Marker in einem Graustufenbild.
    Gibt die Anzahl der erkannten Marker zurück.
    """
    binary = _threshold_binary(gray)
    candidates, _ = _find_ring_candidates(binary, _get_empirical_bounds())
    if candidates:
        return len(candidates)

    # Fallback: ältere Marker-Form (einfache Flächen-/Ratio-Prüfung)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h
        area = cv2.contourArea(cnt)
        if 1800 < area < 6000 and 0.85 < aspect_ratio < 1.05:
            count += 1
    return count


def detect_markers(path: str) -> int:
    """Wrapper für detect_markers_from_gray (erwartet BGR- oder Graustufenbild)."""
    frame = cv2.imread(path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return detect_markers_from_gray(frame)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()

    circle_count = detect_markers(args.path)
    print(circle_count)

