"""
Test: Klammer-/fragment-bewusste Segmentation (5-Zeilen-Split) der ASS-Schicht.

Ein Segment-Schnitt darf NIE einen Klammerblock, ein Bindestrich-Glied oder
eine Versnummer trennen, und kein Segment darf mit einem Satzzeichen-Fragment
beginnen. Eine riesige Klammer (> max_lines Zeilen) bleibt ganz zusammen,
damit ihre Sonderformatierung (grau + klein) erhalten bleibt.
"""

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from modules.text_ass import _split_wrapped_lines, wrap_ass_text  # noqa: E402


def _paren_depth(text: str) -> int:
    depth = 0
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
    return depth


def _check(blocks):
    assert blocks, "mindestens ein Block erwartet"
    for b in blocks:
        assert _paren_depth(b) == 0, f"Block endet mit offener Klammer: {b[:80]}..."
        lines = b.split("\n")
        assert not lines[-1].rstrip().endswith("-"), f"Block endet mit Bindestrich: {b[:80]}..."
        assert not lines[-1].rstrip().endswith(":"), f"Block endet mit einsamer Versnummer: {b[:80]}..."
        stripped = lines[0].lstrip()
        assert not (stripped and stripped[0] in ".,!?;:)-]}\""), f"Block beginnt mit Fragment: {b[:80]}..."


def test_split_laesst_klammer_und_versnummer_zusammen():
    text = (
        "though their plot was a great (one, still) it would never be able to remove "
        "the mountains (real mountains or the Islamic law) from their places (as it is "
        "of no importance) (Tafsir Ibn Kathir, Vol. 2, Page 597). (It is said by some "
        "interpreters regarding this Verse that the Quraish pagans plotted against "
        "Prophet Muhammad SAW to kill him but they failed and were unable to carry out "
        "their plot which they plotted). 47: So think not that Allah"
    )
    lines = wrap_ass_text(text, video_width=1280, font_size=42).split("\n")
    assert len(lines) > 5, "Test braucht mehr als 5 Zeilen"
    blocks = _split_wrapped_lines(lines, max_lines=5)
    assert all(b.count("\n") + 1 <= 5 for b in blocks), "Segment darf max 5 Zeilen haben"
    assert "\n".join(blocks) == "\n".join(lines), "Textverlust durch Splitt"
    _check(blocks)
    # Die Klammer "(It is said by ... plotted)." darf nicht getrennt sein:
    assert not any("plotted)." in b and "(It is said" not in b for b in blocks)
    assert not any("(It is said" in b and "plotted)." not in b for b in blocks)


def test_split_riesige_klammer_bleibt_zusammen():
    giant = "(" + " ".join(f"word{i}" for i in range(200)) + ")"
    lines = wrap_ass_text(giant, video_width=1280, font_size=42).split("\n")
    assert len(lines) > 5, "riesige Klammer braucht mehr als 5 Zeilen"
    blocks = _split_wrapped_lines(lines, max_lines=5)
    assert len(blocks) == 1, "riesige Klammer muss in EINEM Segment bleiben (Formatierung)"
    _check(blocks)


def test_split_hyphen_wort_wird_nicht_getrennt():
    text = "a " * 30 + "Allah is the All-" + " Mighty and the Merciful"
    lines = wrap_ass_text(text, video_width=1280, font_size=42).split("\n")
    blocks = _split_wrapped_lines(lines, max_lines=5)
    _check(blocks)
    for b in blocks:
        assert not b.rstrip().endswith("All-"), "Bindestrich-Wort darf nicht getrennt werden"
