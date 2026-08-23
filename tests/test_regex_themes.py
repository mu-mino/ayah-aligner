"""
Test: Eindeutige Regex-Klassifikation der Highlight-Kategorien.

Basis: Wortgebrauch in eng_translation/chunked_translation/*.txt.
Kriterium (User): Nur Woerter, die zweifelsfrei in eine Kategorie fallen,
ohne semantisches Matching / LLM. Edge-Cases (False-Positives) sind ausgeschlossen.
"""

import re
import sys
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from modules.text_ass import (  # noqa: E402
    REFERENCE_THEMES,
    _THEME_EXCLUSIONS,
    annotate_highlights,
    normalize,
)


def _classify(word: str):
    clean = normalize(word).lower()
    for cat, pat in REFERENCE_THEMES.items():
        if pat.search(clean):
            if clean in _THEME_EXCLUSIONS.get(cat, ()):
                return None
            return cat
    return None


@pytest.mark.parametrize(
    "word,expected",
    [
        # GOD
        ("Allah", "GOD"), ("Allah's", "GOD"), ("god", "GOD"), ("Lord", "GOD"),
        ("Ar-Rahman", "GOD"), ("Al-Aziz", "GOD"), ("Al-Adl", "GOD"),
        ("Allahu Akbar", "GOD"),
        # DESTRUCTIVE
        ("Hell", "DESTRUCTIVE"), ("Hellfire", "DESTRUCTIVE"), ("torment", "DESTRUCTIVE"),
        ("disbelievers", "DESTRUCTIVE"), ("Al-Mushrikoon", "DESTRUCTIVE"),
        ("punishment", "DESTRUCTIVE"), ("hypocrites", "DESTRUCTIVE"),
        ("Hell", "DESTRUCTIVE"), ("Nay", "DESTRUCTIVE"), ("evil", "DESTRUCTIVE"),
        ("sin", "DESTRUCTIVE"), ("sinners", "DESTRUCTIVE"), ("denied", "DESTRUCTIVE"),
        ("woe", "DESTRUCTIVE"), ("falsehood", "DESTRUCTIVE"), ("injustice", "DESTRUCTIVE"),
        ("kill", "DESTRUCTIVE"), ("stolen", "DESTRUCTIVE"), ("adultery", "DESTRUCTIVE"),
        # CONSTRUCTIVE
        ("Paradise", "CONSTRUCTIVE"), ("Gardens", "CONSTRUCTIVE"), ("righteous", "CONSTRUCTIVE"),
        ("believers", "CONSTRUCTIVE"), ("mercy", "CONSTRUCTIVE"), ("guidance", "CONSTRUCTIVE"),
    ],
)
def test_eindeutige_woerter(word, expected):
    assert _classify(word) == expected, f"{word!r} -> {_classify(word)} (erwartet {expected})"


@pytest.mark.parametrize(
    "word",
    [
        # Edge-Cases: duerfen NICHT markiert werden
        "gods", "lords", "lordship", "alhadid", "assamiri",  # GOD-False-Positives
        "abasa", "deaddestroyed",  # DESTRUCTIVE-False-Positives
        "merchandise", "contents", "lightly", "lightning", "commerce",  # CONSTRUCTIVE-False-Positives
        "the", "and", "book", "quran", "muhammad", "walking", "talking", "thanking",  # neutral
        "hurt", "remain", "persevere", "unrighteous", "impatient",  # Substring-Fallen
    ],
)
def test_edge_cases_nicht_markiert(word):
    assert _classify(word) is None, f"{word!r} faelschlich -> {_classify(word)}"


def test_annotate_highlights_verse():
    verse = (
        "Those who disbelieve from among the people of the Scripture will abide in the "
        "Fire of Hell, and their torment is painful, but Allah is the Most Merciful Lord "
        "and the believers who do righteous deeds enter Paradise."
    )
    annotate_highlights(verse, {})
    tokens = verse.split()
    cats = {}
    for tok in tokens:
        t = tok.strip(".,;:!?\"'()[]{}").lower()
        for c, pat in REFERENCE_THEMES.items():
            if pat.search(normalize(t)):
                cats.setdefault(t, c)
                break
    assert cats.get("allah") == "GOD"
    assert cats.get("lord") == "GOD"
    assert cats.get("hell") == "DESTRUCTIVE"
    assert cats.get("torment") == "DESTRUCTIVE"
    assert cats.get("paradise") == "CONSTRUCTIVE"
    assert cats.get("righteous") == "CONSTRUCTIVE"


def test_paren_font_kein_leak():
    # Sure-82-Fall: Klammer-Schliess-Token hervorgehoben -> fs30 darf nicht leaken
    from modules.text_ass import annotate_highlights

    verse = (
        "Verily, the Abrar (pious and righteous) will be in delight (Paradise); "
        "And verily, the Fujjar (the wicked, disbelievers, sinners and evil-doers) "
        "will be in the blazing Fire (Hell), And they (Al-Fujjar) will not be absent. "
        "And what will make you know what the Day of Recompense is?"
    )
    out = annotate_highlights(verse, {}, base_font_size=56)
    fs30 = out.count(r"\fs30")
    fs56 = out.count(r"\fs56")
    assert fs30 == fs56, f"fs30({fs30}) != fs56({fs56}) -> Klammer-Font leakt"
    assert r"\fs40" not in out, "hartkodiertes \\fs40 darf nicht mehr vorkommen"
    # Text nach der letzten Klammer (Vers 17) ist NICHT innerhalb eines offenen fs30
    idx = out.rfind("make you know")
    before = out[:idx]
    assert before.count(r"\fs30") == before.count(r"\fs56"), "Vers danach ist noch klein"
