"""
Test: Subwindow-Überlappung — Circle-Frame kürzen, erstes Subwindow zurücksetzen.

Kriterien (User-Vorgabe):
- Das erste Subwindow wiederholt Circle-Text -> statt doppeltem Vorkommen wird
  der Circle-Frame bis zum nächsten Satzzeichen gekürzt (inkl. Satzzeichen).
- Das erste Subwindow springt zum vorherigen Satzzeichen zurück.
- Der Punkt (.) zählt, außer bei Abkürzungen (i.e., e.g., etc., Vol., No., P.).
"""

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from mapping import (  # noqa: E402
    _find_cut_positions,
    _is_abbreviation_token,
    _is_cut_punctuation,
    _map_mapping_to_content_pos,
)

V5 = (
    'And indeed We sent Moosa (Moses) with Our Ayat (signs, proofs, and evidences) '
    '(saying): "Bring out your people from darkness into light, and make them remember '
    'the annals of Allah. Truly, therein are evidences, proofs and signs for every '
    'patient, thankful (person)."'
)


def test_abbreviation_tokens():
    assert _is_abbreviation_token("i.e.")
    assert _is_abbreviation_token("e.g.")
    assert _is_abbreviation_token("etc.")
    assert _is_abbreviation_token("Vol.")
    assert _is_abbreviation_token("No.")
    assert _is_abbreviation_token("(i.e.")  # führende Klammer wird ignoriert
    assert _is_abbreviation_token('"e.g.')
    assert not _is_abbreviation_token("Allah.")
    assert not _is_abbreviation_token("them.")
    assert not _is_abbreviation_token("darkness")


def test_cut_punctuation_period_exceptions():
    text = "i.e. a test. Allah. etc. Vol. 1, e.g. x"
    assert not _is_cut_punctuation(text, text.index("."))  # i.e.
    assert not _is_cut_punctuation(text, text.index("etc.") + 3)  # etc.
    assert not _is_cut_punctuation(text, text.index("Vol.") + 3)  # Vol.
    # "Allah." ist ein Satzende -> Punkt zählt
    assert _is_cut_punctuation(text, text.index("Allah.") + 5)
    # "(i.e." -> Punkt zählt nicht
    paren = "evidence. (i.e. Prophet)"
    assert not _is_cut_punctuation(paren, paren.index("i.e.") + 3)


def test_find_cut_positions_verse5():
    boundary = V5.index("darkness")
    cut = _find_cut_positions(V5, boundary)
    assert cut is not None
    next_punct, prev_punct = cut
    # Nächstes Satzzeichen vorwärts: Komma nach "light"
    assert V5[next_punct] == ","
    assert V5[next_punct - 5 : next_punct + 1].rstrip() == "light,"
    # Vorheriges Satzzeichen rückwärts: Doppelpunkt nach "(saying)"
    assert V5[prev_punct] == ":"
    assert V5[prev_punct - 8 : prev_punct].endswith("(saying)")


def test_find_cut_positions_none_without_punct():
    # kein Satzzeichen -> None
    assert _find_cut_positions("Hello world darkness", 6) is None


def test_find_cut_positions_without_prev_punct():
    # Grenze in der ersten Satzhälfte: kein vorheriges Satzzeichen ->
    # (next_punct, None); Circle-Trim bleibt möglich.
    v3 = (
        "Those who prefer the life of this world instead of the Hereafter, "
        "and hinder (men) from the Path of Allah"
    )
    boundary = v3.index("of this world")
    cut = _find_cut_positions(v3, boundary)
    assert cut is not None
    next_punct, prev_punct = cut
    assert prev_punct is None
    assert next_punct is not None
    assert v3[next_punct] == ","
    assert v3[: next_punct + 1].endswith("the Hereafter,")


def test_map_content_pos_single_verse():
    verse_ranges = [(5, 0, len(V5), V5)]
    p = V5.index("darkness")
    mapped = _map_mapping_to_content_pos(verse_ranges, p)
    content = f"5: {V5}"
    assert content[mapped] == "d"
    assert mapped == 3 + p


def test_map_content_pos_multi_verse():
    v1 = "First verse text."
    v2 = "Second verse, with comma."
    mapping_text = f"{v1} {v2}"
    verse_ranges = [(5, 0, len(v1), v1), (6, len(v1) + 1, len(mapping_text), v2)]
    p = mapping_text.index("comma")
    mapped = _map_mapping_to_content_pos(verse_ranges, p)
    content = f"5: {v1} 6: {v2}"
    assert content[mapped] == "c"
    # Position im zweiten Vers: offset = len("5: v1 ") + len("6: ")
    assert mapped == len(f"5: {v1} ") + len("6: ") + (p - (len(v1) + 1))
