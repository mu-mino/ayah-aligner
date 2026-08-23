"""
Krita-aequivalente "Color to Alpha"-Transformation fuer den arabic_space-Layer.

Referenz-Algorithmus: KisFilterColorToAlpha
  (krita/plugins/filters/colors/kis_color_to_alpha.cpp)

Gegeben eine Ziel-Farbe (aus dem Export-Statement, #RRGGBB) und ein Pixel:

  diff       = DeltaE76(baseColor, pixel) im Lab-Raum (Cap 255)
  newOpacity = 1.0 wenn diff >= threshold, sonst diff / threshold
  nur wenn newOpacity < aktuelle Alpha: Alpha = newOpacity
  inverseOver: pixel[i] = clamp((pixel[i] - base[i]) / opacity + base[i])

Daraus entsteht der "Neon-Invert"-Look: Die dominante Farbe des Layers wird
durch die angegebene Hex-Farbe ersetzt und der Rest in Transparenz ueberfuehrt.

Zusaetzlich gibt es den Neon-Recolor (apply_recolor_image): eine DYNAMISCHE
Umwandlung, die fuer ALLE Hex-Codes funktioniert. Das Bild wird zuerst
analysiert (min/max-Helligkeit aller Nicht-Schwarz-Pixel) und dieser Bereich
dann auf die Ziel-Farbe abgebildet:

  * Helligkeit:  hellster Rotton -> hellste Ziel-Farbe, dunkelster -> dunkelste
  * Saettigung:  Struktur des Originals bleibt erhalten (skaliert auf das Ziel)
  * Schwarz (L ~ 0) bleibt unangetastet.

Rot-Hell wird also Ziel-Hell, Rot-Dunkel Ziel-Dunkel, Rot-gesaettigt
Ziel-gesaettigt - bei jedem Ziel-Hex, auch bei grauen (Silber) oder dunklen
Zielen, ohne feste Konstanten oder Sonderfaelle.
"""

import colorsys

from typing import Tuple

# Krita-Default: threshold=100 (siehe KisFilterColorToAlpha::defaultConfiguration)
DEFAULT_THRESHOLD = 100

# Recolor: Pixel mit Helligkeit (HLS-L) <= diesem Wert gelten als "schwarz"
# und bleiben beim Recolor unangetastet (reines Schwarz hat L=0).
BLACK_HLS_L = 0.02

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Konvertiert #RRGGBB (oder RRGGBB) in ein (r, g, b)-Tupel."""
    cleaned = (hex_color or "").strip().lstrip("#")
    if len(cleaned) != 6 or not all(c in "0123456789abcdefABCDEF" for c in cleaned):
        raise ValueError(f"Ungültige Hex-Farbe: {hex_color!r} (erwartet #RRGGBB)")
    r = int(cleaned[0:2], 16)
    g = int(cleaned[2:4], 16)
    b = int(cleaned[4:6], 16)
    return (r, g, b)


def _srgb_to_lab(r: float, g: float, b: float) -> Tuple[float, float, float]:
    """sRGB (0..1) -> CIELAB (D65, Standard-Referenz)."""

    def _linear(c: float) -> float:
        if c <= 0.04045:
            return c / 12.92
        return ((c + 0.055) / 1.055) ** 2.4

    r_lin, g_lin, b_lin = _linear(r), _linear(g), _linear(b)
    x = 0.4124 * r_lin + 0.3576 * g_lin + 0.1805 * b_lin
    y = 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin
    z = 0.0193 * r_lin + 0.1192 * g_lin + 0.9505 * b_lin

    def _f(t: float) -> float:
        delta = 6.0 / 29.0
        if t > delta ** 3:
            return t ** (1.0 / 3.0)
        return t / (3.0 * delta ** 2) + 4.0 / 29.0

    fx, fy, fz = _f(x / 0.95047), _f(y / 1.0), _f(z / 1.08883)
    l = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b_ = 200.0 * (fy - fz)
    return (l, a, b_)


def delta_e76(c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> float:
    """CIEDE1976 (Lab-Euklidisch) zwischen zwei RGB-Farben, Cap 255."""
    lab1 = _srgb_to_lab(c1[0] / 255.0, c1[1] / 255.0, c1[2] / 255.0)
    lab2 = _srgb_to_lab(c2[0] / 255.0, c2[1] / 255.0, c2[2] / 255.0)
    d = sum((a - b) ** 2 for a, b in zip(lab1, lab2)) ** 0.5
    return min(255.0, d)


def compute_opacity(diff: float, threshold: int = DEFAULT_THRESHOLD) -> float:
    """newOpacity laut Krita: 1.0 ab Threshold, sonst anteilig."""
    if diff >= threshold:
        return 1.0
    return diff / float(threshold)


def inverse_over(
    pixel: Tuple[int, int, int],
    base: Tuple[int, int, int],
    opacity: float,
) -> Tuple[int, int, int]:
    """inverseOver: (pixel - base)/opacity + base, geklemmt auf 0..255."""
    if opacity <= 0:
        return (0, 0, 0)
    out = []
    for p, b in zip(pixel, base):
        v = (p - b) / opacity + b
        v = max(0, min(255, round(v)))
        out.append(v)
    return (out[0], out[1], out[2])


def color_to_alpha_pixel(
    pixel: Tuple[int, int, int],
    base: Tuple[int, int, int],
    threshold: int = DEFAULT_THRESHOLD,
) -> Tuple[int, int, int, int]:
    """Wendet Color-to-Alpha auf einen RGBA-Pixel an; liefert (r, g, b, a)."""
    diff = delta_e76(base, pixel)
    new_opacity = compute_opacity(diff, threshold)
    rgb = inverse_over(pixel, base, new_opacity)
    alpha = int(round(new_opacity * 255))
    return (rgb[0], rgb[1], rgb[2], alpha)


def apply_color_to_alpha_image(
    image,
    base: Tuple[int, int, int],
    threshold: int = DEFAULT_THRESHOLD,
):
    """Wendet Color-to-Alpha auf ein PIL-RGBA-Bild an (Pixelweise).

    Nur Pixel, deren neue Deckkraft UNTER ihrer aktuellen Alpha liegt, werden
    veraendert (identisch zu Krita: `if newOpacity < opacityF(dst)`).
    """
    px = image.load()
    w, h = image.size
    for y in range(h):
        for x in range(w):
            r, g, b, a = px[x, y]
            if a == 0:
                continue
            diff = delta_e76(base, (r, g, b))
            new_opacity = compute_opacity(diff, threshold)
            if new_opacity < a / 255.0:
                nr, ng, nb = inverse_over((r, g, b), base, new_opacity)
                px[x, y] = (nr, ng, nb, int(round(new_opacity * 255)))
    return image


def hex_to_ascii_tag(hex_color: str) -> str:
    """Sanitizer Hex -> Dateinamen-Zusatz (ohne #)."""
    cleaned = (hex_color or "").strip().lstrip("#").upper()
    return cleaned if len(cleaned) == 6 else "default"


def _is_black(l: float) -> bool:
    """True, wenn die Helligkeit (HLS-L) eines Pixels als "schwarz" gilt.

    Reines Schwarz hat L=0; der Schwellenwert beruecksichtigt Rundungsrauschen.
    """
    return l <= BLACK_HLS_L


def _rgb_to_hls_tuple(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """RGB (0..255) -> (hue, lightness, saturation) mit Hue in [0, 1)."""
    return colorsys.rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)


def _hls_to_rgb_tuple(h: float, l: float, s: float) -> Tuple[int, int, int]:
    """(hue, lightness, saturation) -> RGB (0..255)."""
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return (round(r * 255), round(g * 255), round(b * 255))


def _analyze_lightness_range(
    image,
    black_l: float = BLACK_HLS_L,
) -> Tuple[float, float]:
    """Dynamische Analyse: min/max-Helligkeit (HLS-L) aller Nicht-Schwarz-Pixel.

    Liefert den tatsaechlichen Helligkeitsbereich des Bildes, auf den die
    Ziel-Farbe abgebildet wird. Ohne verwertbare Pixel -> (0.0, 1.0).
    """
    px = image.load()
    w, h = image.size
    l_min = 1.0
    l_max = 0.0
    found = False
    for y in range(h):
        for x in range(w):
            r, g, b, a = px[x, y]
            if a == 0:
                continue
            _, l, _ = _rgb_to_hls_tuple(r, g, b)
            if _is_black(l):
                continue
            found = True
            if l < l_min:
                l_min = l
            if l > l_max:
                l_max = l
    if not found:
        return (0.0, 1.0)
    return (l_min, l_max)


def recolor_pixel(
    pixel: Tuple[int, int, int, int],
    target: Tuple[int, int, int],
    l_min: float = 0.0,
    l_max: float = 1.0,
) -> Tuple[int, int, int, int]:
    """Recolor eines RGBA-Pixels: dynamische Abbildung auf die Ziel-Farbe.

    Der Helligkeitsbereich des Bildes [l_min, l_max] wird auf die Helligkeit
    der Ziel-Farbe abgebildet (hellster Rotton -> hellste Ziel-Farbe,
    dunkelster -> dunkelste). Die Saettigung des Originals bleibt erhalten
    (skaliert auf die Ziel-Saettigung). Schwarz (L ~ 0) und transparente
    Pixel bleiben unangetastet; der Alpha-Kanal bleibt erhalten.
    """
    r, g, b, a = pixel
    if a == 0:
        return (0, 0, 0, 0)
    h, l, s = _rgb_to_hls_tuple(r, g, b)
    if _is_black(l):
        return pixel
    target_h, target_l, target_s = _rgb_to_hls_tuple(target[0], target[1], target[2])
    # Dynamische Normalisierung: Quell-Helligkeit [l_min, l_max] -> [0, target_l]
    if l_max > l_min:
        norm = (l - l_min) / (l_max - l_min)
    else:
        norm = 1.0
    norm = max(0.0, min(1.0, norm))
    out_l = max(0.0, min(1.0, target_l * norm))
    # Saettigung: Struktur des Originals erhalten, auf Ziel-Saettigung skaliert
    out_s = max(0.0, min(1.0, s * target_s))
    nr, ng, nb = _hls_to_rgb_tuple(target_h, out_l, out_s)
    return (nr, ng, nb, a)


def apply_recolor_image(
    image,
    target: Tuple[int, int, int],
):
    """Wendet den Neon-Recolor dynamisch auf ein PIL-RGBA-Bild an.

    Analysiert zuerst den Helligkeitsbereich des Bildes (min/max aller
    Nicht-Schwarz-Pixel) und bildet diesen dann auf die Ziel-Farbe ab,
    damit die Helligkeitsstruktur bei jedem Ziel-Hex erhalten bleibt.
    """
    l_min, l_max = _analyze_lightness_range(image)
    px = image.load()
    w, h = image.size
    for y in range(h):
        for x in range(w):
            px[x, y] = recolor_pixel(px[x, y], target, l_min, l_max)
    return image


__all__ = [
    "DEFAULT_THRESHOLD",
    "apply_color_to_alpha_image",
    "apply_recolor_image",
    "color_to_alpha_pixel",
    "compute_opacity",
    "delta_e76",
    "hex_to_ascii_tag",
    "hex_to_rgb",
    "inverse_over",
    "recolor_pixel",
]
