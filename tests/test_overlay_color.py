"""
Test: Krita-äquivalente "Color to Alpha"-Transformation für den
arabic_space-Layer.

Referenz: KisFilterColorToAlpha (plugins/filters/colors/kis_color_to_alpha.cpp)
  diff       = DeltaE76(baseColor, pixel) im Lab-Raum (Cap 255)
  newOpacity = 1.0 wenn diff >= threshold, sonst diff/threshold
  nur wenn newOpacity < aktuelle Alpha: Alpha = newOpacity
  inverseOver: pixel[i] = clamp((pixel[i] - base[i]) / opacity + base[i])

Kernfunktionen (ohne Pillow/PIL-Abhängigkeit) werden hier geprüft:
  delta_e76, compute_opacity, inverse_over, color_to_alpha_pixel
"""

import sys
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from modules.overlay_color import (  # noqa: E402
    compute_opacity,
    delta_e76,
    hex_to_rgb,
    inverse_over,
)


def test_hex_to_rgb():
    assert hex_to_rgb("#114141") == (17, 65, 65)
    assert hex_to_rgb("114141") == (17, 65, 65)
    assert hex_to_rgb("#00FF00") == (0, 255, 0)


def test_hex_to_rgb_invalid():
    for bad in ["#GGGGGG", "#12345", "zzz", ""]:
        with pytest.raises(ValueError):
            hex_to_rgb(bad)


def test_delta_e76_identical():
    assert delta_e76((0, 0, 0), (0, 0, 0)) == pytest.approx(0.0, abs=1e-6)


def test_delta_e76_symmetric():
    a = delta_e76((114, 41, 41), (185, 28, 84))
    b = delta_e76((185, 28, 84), (114, 41, 41))
    assert a == pytest.approx(b)


def test_compute_opacity_unter_threshold():
    # diff < threshold -> anteilige Deckkraft
    assert compute_opacity(50, 100) == pytest.approx(0.5)
    assert compute_opacity(0, 100) == pytest.approx(0.0)


def test_compute_opacity_ab_threshold():
    assert compute_opacity(100, 100) == pytest.approx(1.0)
    assert compute_opacity(200, 100) == pytest.approx(1.0)
    assert compute_opacity(255, 100) == pytest.approx(1.0)


def test_inverse_over_reines_rot():
    # Pixel (185,28,84), base (114,41,41), opacity 0.32
    out = inverse_over((185, 28, 84), (114, 41, 41), 0.32)
    assert out[0] >= 250  # Rot-Kanal wird auf maximiert (neon)
    assert out[1] == 0  # Gruen -> 0
    assert 170 <= out[2] <= 180  # Blau ~ 175


def test_inverse_over_opacity_eins_bleibt():
    out = inverse_over((185, 28, 84), (114, 41, 41), 1.0)
    assert out == (185, 28, 84)


def test_inverse_over_klemmt_auf_null_acht():
    out = inverse_over((0, 0, 0), (255, 255, 255), 0.5)
    assert all(0 <= v <= 255 for v in out)


# --- Neon-Recolor (apply_recolor) ---
# Konzept: Dynamische Umwandlung. Das Bild wird zuerst analysiert (min/max
# Helligkeit der Nicht-Schwarz-Pixel). Dann wird der Helligkeitsbereich des
# Bildes auf die Ziel-Farbe abgebildet: hellster Rotton -> hellste Ziel-Farbe,
# dunkelster -> dunkelste. Sättigung des Originals bleibt erhalten (skaliert
# auf die Ziel-Sättigung). Schwarz bleibt unangetastet. Funktioniert fuer
# ALLE Hex-Codes, ohne feste Konstanten oder Sonderfaelle.

UNIT_LMIN, UNIT_LMAX = 0.02, 0.47  # angenaeherter Bildbereich fuer Unit-Tests


def test_recolor_pixel_schwarz_bleibt_schwarz():
    from modules.overlay_color import recolor_pixel
    out = recolor_pixel((0, 0, 0, 255), (0, 255, 102), UNIT_LMIN, UNIT_LMAX)
    assert out[:3] == (0, 0, 0)
    assert out[3] == 255


def test_recolor_pixel_transparent_bleibt_transparent():
    from modules.overlay_color import recolor_pixel
    out = recolor_pixel((0, 0, 0, 0), (0, 255, 102), UNIT_LMIN, UNIT_LMAX)
    assert out == (0, 0, 0, 0)


def test_recolor_pixel_rot_wird_ziel():
    from modules.overlay_color import recolor_pixel
    # Heller Rotton (185,28,84) -> Gruen #00FF66, dynamisch normalisiert
    out = recolor_pixel((185, 28, 84, 255), (0, 255, 102), UNIT_LMIN, UNIT_LMAX)
    assert out[:3] == (30, 196, 96)
    assert out[3] == 255


def test_recolor_pixel_dunkler_rot_wird_dunkler_ziel():
    from modules.overlay_color import recolor_pixel
    out = recolor_pixel((104, 17, 43, 255), (0, 255, 102), UNIT_LMIN, UNIT_LMAX)
    assert out[:3] == (17, 106, 53)


def test_recolor_pixel_silber_ziel_wird_achromatisch():
    from modules.overlay_color import recolor_pixel
    # Silber #C0C0C0 (S=0) -> graue Stufen, KEIN Rot (kein fester Hue)
    out = recolor_pixel((185, 28, 84, 255), (192, 192, 192), UNIT_LMIN, UNIT_LMAX)
    assert out[0] == out[1] == out[2] == 170
    assert out[3] == 255


def test_recolor_pixel_silber_erhaelt_helligkeit():
    from modules.overlay_color import recolor_pixel
    hell = recolor_pixel((185, 28, 84, 255), (192, 192, 192), UNIT_LMIN, UNIT_LMAX)[:3]
    dunkel = recolor_pixel((104, 17, 43, 255), (192, 192, 192), UNIT_LMIN, UNIT_LMAX)[:3]
    assert hell == (170, 170, 170)
    assert dunkel == (93, 93, 93)
    assert hell[0] > dunkel[0]


def test_recolor_pixel_helligkeit_haengt_von_bildbereich_ab():
    from modules.overlay_color import recolor_pixel
    import colorsys
    # Gleicher Pixel, anderes Bild (anderer l_max) -> andere Helligkeit (dynamisch)
    a = recolor_pixel((185, 28, 84, 255), (0, 255, 102), 0.02, 0.47)[:3]
    b = recolor_pixel((185, 28, 84, 255), (0, 255, 102), 0.02, 0.60)[:3]
    la = colorsys.rgb_to_hls(*(c / 255 for c in a))[1]
    lb = colorsys.rgb_to_hls(*(c / 255 for c in b))[1]
    assert la != lb  # ergebnis ist bildabhaengig, nicht fix


def test_recolor_pixel_ziel_hex_unterscheidbar():
    from modules.overlay_color import recolor_pixel
    a = recolor_pixel((185, 28, 84, 255), (0, 255, 102), UNIT_LMIN, UNIT_LMAX)
    b = recolor_pixel((185, 28, 84, 255), (255, 215, 0), UNIT_LMIN, UNIT_LMAX)
    c = recolor_pixel((185, 28, 84, 255), (0, 0, 54), UNIT_LMIN, UNIT_LMAX)
    assert len({a[:3], b[:3], c[:3]}) == 3


def test_recolor_pixel_alpha_erhalten():
    from modules.overlay_color import recolor_pixel
    out = recolor_pixel((179, 28, 79, 123), (0, 255, 102), UNIT_LMIN, UNIT_LMAX)
    assert out[3] == 123


# --- Integration: apply_recolor_image analysiert das Bild dynamisch ---

from PIL import Image  # noqa: E402


def _make_img(pixels):
    img = Image.new("RGBA", (len(pixels), 1))
    img.putdata(pixels)
    return img


def test_apply_recolor_image_dynamisch():
    from modules.overlay_color import apply_recolor_image
    # Hell (185,28,84), mittel (104,17,43), dunkel (11,0,14), schwarz
    img = _make_img([
        (185, 28, 84, 255), (104, 17, 43, 255), (11, 0, 14, 255), (0, 0, 0, 255),
    ])
    apply_recolor_image(img, (0, 255, 102))
    out = [img.getpixel((i, 0)) for i in range(img.width)]
    assert out[0][:3] == (34, 221, 109)  # hellster -> volles Gruen
    assert out[1][:3] == (19, 118, 59)   # mittel
    assert out[2][:3] == (0, 0, 0)       # dunkelster -> schwarz (Gradient)
    assert out[3] == (0, 0, 0, 255)      # schwarz unangetastet


def test_apply_recolor_image_hellster_erreicht_ziel_helligkeit():
    import colorsys
    from modules.overlay_color import apply_recolor_image
    img = _make_img([(185, 28, 84, 255), (104, 17, 43, 255), (11, 0, 14, 255)])
    apply_recolor_image(img, (0, 255, 102))
    l_out = colorsys.rgb_to_hls(*(c / 255 for c in img.getpixel((0, 0))[:3]))[1]
    l_target = colorsys.rgb_to_hls(0 / 255, 255 / 255, 102 / 255)[1]
    assert l_out == pytest.approx(l_target, abs=1e-3)
