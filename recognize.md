Datenstrukturen
  - DetectorParams — parametrisierbare Erkennungsgrenzen mit clamp()
  - RingCandidate — erkannter Ring mit allen geometrischen Metriken
  - DEFAULT_DETECTOR_PARAMS — Standardwerte aus den definierten Konstanten

Erkennungslogik
- _find_ring_candidates() — Kernalgorithmus: Konturen filtern nach Fläche, Breite/Höhe,
Seitenverhältnis, Kreisförmigkeit, Stroke- und Hole-Ratio
- _score_candidate() — geometrischer Güte-Score (Abstand von Range-Mitte)
- _derive_empirical_bounds() / _get_empirical_bounds() — Grenzen aus Beispiel-Frames (5–95
Perzentil)
- DETECTOR_PRESETS — vier Varianten: base, tight, stroke_plus, stroke_minus

Patch-Vergleich (Deduplizierung)
- _extract_circle_patch() — normalisiertes 64×64-Patch
- _circle_patch_similarity() — MAE-basierte Ähnlichkeit
- _best_circle_patch() — bester Kandidat eines Frames

Öffentliche API
- detect_markers_from_gray(gray) / detect_markers(frame) — gibt Anzahl erkannter Marker
