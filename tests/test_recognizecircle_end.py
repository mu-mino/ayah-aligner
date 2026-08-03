"""
Test: end_with_last_verse-Detektion in recognizecircle.

Kriterien (User-Vorgabe):
- detect_markers_from_gray() gibt (Anzahl, end_with_last_verse) zurück.
- end_with_last_verse ist True, wenn links vom am weitesten links liegenden
  Kreis (nach CIRCLE_END_WHITE_GAP_PX, im Kreisband bis zum Bildrand) keine
  weißen Pixel mehr stehen.
- Weiße Pixel links außerhalb des Kreisbands zählen nicht.
"""

import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from modules.recognizecircle import (  # noqa: E402
    RingCandidate,
    _leftmost_circle_ends_text,
    detect_markers_from_gray,
)


def _candidate(x, y, w=50, h=50):
    return RingCandidate(
        outer_mask=np.zeros((1, 1), np.uint8),
        inner_mask=np.zeros((1, 1), np.uint8),
        ring_mask=np.zeros((1, 1), np.uint8),
        bbox=(x, y, w, h),
        center=(x + w / 2.0, y + h / 2.0),
        radius=max(w, h) / 2.0,
        stroke_ratio=0.5,
        hole_ratio=0.5,
        circularity=0.8,
        stroke_kernel=5,
    )


def _binary(h=200, w=200):
    return np.zeros((h, w), np.uint8)


def test_returns_tuple():
    count, end = detect_markers_from_gray(_binary())
    assert isinstance(count, int)
    assert isinstance(end, bool)
    assert count == 0
    assert end is False


def test_no_white_left_is_end():
    binary = _binary()
    assert _leftmost_circle_ends_text(binary, [_candidate(x=20, y=90)]) is True


def test_white_left_is_not_end():
    binary = _binary()
    binary[90:140, 0:10] = 255
    assert _leftmost_circle_ends_text(binary, [_candidate(x=20, y=90)]) is False


def test_white_inside_gap_is_not_counted():
    binary = _binary()
    binary[90:140, 12:19] = 255
    assert _leftmost_circle_ends_text(binary, [_candidate(x=20, y=90)]) is True


def test_white_outside_band_is_not_counted():
    binary = _binary()
    binary[0:30, 0:10] = 255
    assert _leftmost_circle_ends_text(binary, [_candidate(x=20, y=90)]) is True


def test_white_in_band_left_of_non_leftmost_is_end():
    binary = _binary()
    binary[90:140, 0:10] = 255
    leftmost = _candidate(x=20, y=90)
    other = _candidate(x=150, y=90)
    assert _leftmost_circle_ends_text(binary, [leftmost, other]) is False
    assert _leftmost_circle_ends_text(binary, [other, leftmost]) is False
