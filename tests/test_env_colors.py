"""
Test: Dynamische ASS-Farben via Export-Statements (Environment).

Drei Kategorien (gott / konstruktiv / destruktiv) koennen per Env-Var
mit einem #RRGGBB-Hexcode ueberschrieben werden. Der Wert muss in das
ASS-BGR-Format (&HBBGGRR&) konvertiert werden; nicht gesetzte Kategorien
behalten ihre Default-Farbe.
"""

import sys
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from modules.text_ass import (  # noqa: E402
    DEFAULT_COLOR_MAP,
    ENV_COLOR_MAP,
    hex_to_ass_bgr,
    resolve_color_map,
)


def test_env_namen_mapping():
    assert ENV_COLOR_MAP["GOD"] == "GOTT"
    assert ENV_COLOR_MAP["CONSTRUCTIVE"] == "KONSTRUKTIV"
    assert ENV_COLOR_MAP["DESTRUCTIVE"] == "DESTRUKTIV"


@pytest.mark.parametrize(
    "hexcolor,expected",
    [
        ("#FADF75", "&H75DFFA&"),  # Gelbgold-Default, RGB->BGR
        ("#B21262", "&H6212B2&"),  # Rot-Default
        ("#003580", "&H803500&"),  # Blau-Default
        ("#00FF00", "&H00FF00&"),  # Reines Gruen
        ("#0000FF", "&HFF0000&"),  # Reines Blau
        ("#FF0000", "&H0000FF&"),  # Reines Rot
        ("00FF00", "&H00FF00&"),  # ohne #-Prefix
        ("#FFFFFF", "&HFFFFFF&"),  # Weiss
    ],
)
def test_hex_to_ass_bgr(hexcolor, expected):
    assert hex_to_ass_bgr(hexcolor) == expected


def test_hex_to_ass_bgr_invalid():
    for bad in ["#GGGGGG", "#12345", "#1234567", "zzz", ""]:
        with pytest.raises(ValueError):
            hex_to_ass_bgr(bad)


def test_resolve_color_map_default_bleibt():
    cm = resolve_color_map({})
    assert cm["GOD"] == DEFAULT_COLOR_MAP["GOD"]
    assert cm["CONSTRUCTIVE"] == DEFAULT_COLOR_MAP["CONSTRUCTIVE"]
    assert cm["DESTRUCTIVE"] == DEFAULT_COLOR_MAP["DESTRUCTIVE"]
    assert cm["NONE"] == DEFAULT_COLOR_MAP["NONE"]


def test_resolve_color_map_gott_override():
    cm = resolve_color_map({"GOTT": "#00FF00"})
    assert cm["GOD"] == "&H00FF00&"
    assert cm["CONSTRUCTIVE"] == DEFAULT_COLOR_MAP["CONSTRUCTIVE"]
    assert cm["DESTRUCTIVE"] == DEFAULT_COLOR_MAP["DESTRUCTIVE"]


def test_resolve_color_map_lowercase_env():
    cm = resolve_color_map({"gott": "#0000FF", "konstruktiv": "#FF0000"})
    assert cm["GOD"] == "&HFF0000&"
    assert cm["CONSTRUCTIVE"] == "&H0000FF&"
    assert cm["DESTRUCTIVE"] == DEFAULT_COLOR_MAP["DESTRUCTIVE"]


def test_resolve_color_map_alle_drei():
    cm = resolve_color_map(
        {
            "GOTT": "#112233",
            "KONSTRUKTIV": "#445566",
            "DESTRUKTIV": "#778899",
        }
    )
    assert cm["GOD"] == "&H332211&"
    assert cm["CONSTRUCTIVE"] == "&H665544&"
    assert cm["DESTRUCTIVE"] == "&H998877&"


def test_resolve_color_map_invalid_raises():
    with pytest.raises(ValueError):
        resolve_color_map({"GOTT": "#bad"})


def test_resolve_sub_bg_inaktiv():
    from modules.text_ass import resolve_sub_bg
    assert resolve_sub_bg({}) is None
    assert resolve_sub_bg({"SUB_BG_ACTIVE": "0"}) is None


def test_resolve_sub_bg_aktiv_default_schwarz():
    from modules.text_ass import resolve_sub_bg
    r = resolve_sub_bg({"SUB_BG_ACTIVE": "1"})
    assert r == ("3", "&H000000&", "&H000000&")


def test_resolve_sub_bg_aktiv_mit_farbe():
    from modules.text_ass import resolve_sub_bg
    r = resolve_sub_bg({"SUB_BG_ACTIVE": "1", "SUB_BG_COLOR": "#00FF00"})
    assert r == ("3", "&H00FF00&", "&H00FF00&")


def test_resolve_sub_bg_aktiv_mit_farbe_bgr():
    from modules.text_ass import resolve_sub_bg
    # #FF0000 (rot) -> BGR &H0000FF&
    r = resolve_sub_bg({"SUB_BG_ACTIVE": "1", "SUB_BG_COLOR": "#FF0000"})
    assert r == ("3", "&H0000FF&", "&H0000FF&")


def test_build_ass_ohne_sub_bg_style_unveraendert():
    import sys
    from pathlib import Path
    from modules.text_ass import build_ass
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from modules.text_ass import Entry
    entries = [Entry(0.0, "test verse")]
    out = build_ass("x", entries, 640, 360, 10.0)
    style = [l for l in out.splitlines() if l.startswith("Style:")]
    assert style and ",1,1,0,5," in style[0]


def test_build_ass_mit_sub_bg_borderstyle3():
    import os
    from modules.text_ass import build_ass, Entry
    os.environ["SUB_BG_ACTIVE"] = "1"
    os.environ["SUB_BG_COLOR"] = "#0000FF"
    try:
        entries = [Entry(0.0, "test verse")]
        out = build_ass("x", entries, 640, 360, 10.0)
    finally:
        os.environ.pop("SUB_BG_ACTIVE", None)
        os.environ.pop("SUB_BG_COLOR", None)
    style = [l for l in out.splitlines() if l.startswith("Style:")]
    assert style and ",3,4,0,5," in style[0]
    assert "&HFF0000&" in style[0]  # #0000FF blau -> BGR &HFF0000&


def test_resolve_font_color_inaktiv():
    from modules.text_ass import resolve_font_color
    assert resolve_font_color({}) is None


def test_resolve_font_color_aktiv_default_schwarz():
    from modules.text_ass import resolve_font_color
    r = resolve_font_color({"FONT_COLOR_ACTIVE": "1"})
    assert r == "&H000000&"


def test_resolve_font_color_mit_farbe():
    from modules.text_ass import resolve_font_color
    r = resolve_font_color({"FONT_COLOR_ACTIVE": "1", "FONT_COLOR": "#FF0000"})
    assert r == "&H0000FF&"  # rot -> BGR


def test_resolve_font_color_mit_farbe_gruen():
    from modules.text_ass import resolve_font_color
    r = resolve_font_color({"FONT_COLOR_ACTIVE": "1", "FONT_COLOR": "#00FF00"})
    assert r == "&H00FF00&"


def test_build_ass_mit_font_color_style():
    import os
    from modules.text_ass import build_ass, Entry
    os.environ["FONT_COLOR_ACTIVE"] = "1"
    os.environ["FONT_COLOR"] = "#0000FF"
    try:
        entries = [Entry(0.0, "test verse")]
        out = build_ass("x", entries, 640, 360, 10.0)
    finally:
        os.environ.pop("FONT_COLOR_ACTIVE", None)
        os.environ.pop("FONT_COLOR", None)
    style = [l for l in out.splitlines() if l.startswith("Style:")]
    assert style and "&HFF0000&" in style[0]  # #0000FF blau -> BGR &HFF0000&
    # PrimaryColour ist das 4. Farbfeld -> sollte &HFF0000& sein
    fields = style[0].split(",")
    assert fields[3] == "&HFF0000&"
