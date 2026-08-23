"""
Test: Mapping-Coverage + Sequenzialität für kurze Suren mit Sub-Window-Edge-Case.

Kriterien (User-Vorgabe):
- verse-coverage 100 %: Jede Versnummer der Textdatei (1..N) erscheint im Mapping.
- Mapping sequenziell: Die ERSTE Vorkommensposition jeder Versnummer ist streng
  aufsteigend → keine Lücken/Sprünge in der Versreihenfolge.
- Duplikate sind erlaubt (Circle voll + Sub-Eintrag) und werden nicht als Fehler
  gewertet.

Getestete Suren (kurz, mit 0-Kreis-Sub-Windows – der Edge-Case aus Sure 98):
- 98  Al-Bayyina   (10 Windows, 8 Verse)
- 85  Al-Burooj    (13 Windows, 22 Verse)
- 114 An-Nas       ( 5 Windows,  6 Verse)
"""

import re
import sys
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import modules.whispertranscribe as wt
wt.USE_MODAL = False

from mapping import run as run_mapping  # noqa: E402
from modules.circlelog import parse_text_file  # noqa: E402

HOME = Path.home()
VID_DIR = HOME / "desk/din/quran/maher_workaround/Quran_cropped"
AUD_DIR = HOME / "desk/din/quran/maher_playlist/maher_playlist"
TXT_DIR = HOME / "desk/din/quran/eng_translation/chunked_translation"
OUT_DIR = BASE / "output" / "mapping"

TEST_SURAHS = [98, 85, 114]


def _find(pattern_root: Path, sid: int) -> Path:
    try:
        return next(pattern_root.glob(f"*({sid}) *"))
    except StopIteration:
        pytest.skip(f"Keine Datei fuer Sure {sid} in {pattern_root}")


def _run_mapping(sid: int) -> Path:
    video = _find(VID_DIR, sid)
    audio = _find(AUD_DIR, sid)
    try:
        text = next(TXT_DIR.glob(f"{sid}_*.txt"))
    except StopIteration:
        pytest.skip(f"Keine Textdatei fuer Sure {sid}")
    mapping_path = OUT_DIR / f"{text.stem}.mapping"
    run_mapping(
        video_path=video,
        audio_path=audio,
        text_path=text,
        mapping_path=mapping_path,
        surah=sid,
        whisper_device="cpu",
    )
    return mapping_path


def _verse_numbers(mapping_path: Path):
    nums = []
    for line in mapping_path.read_text(encoding="utf-8").splitlines():
        for m in re.finditer(r"(?:^|\s)(\d+):", line):
            nums.append(int(m.group(1)))
    return nums


def _expected_verse_count(sid: int) -> int:
    text = next(TXT_DIR.glob(f"{sid}_*.txt"))
    _, numbered = parse_text_file(text)
    return len(numbered)


@pytest.mark.parametrize("sid", TEST_SURAHS)
def test_mapping_coverage_und_sequenz(sid: int):
    mapping_path = _run_mapping(sid)
    assert mapping_path.exists(), f"Mapping {mapping_path} fehlt"

    n = _expected_verse_count(sid)
    verse_numbers = _verse_numbers(mapping_path)

    # 1) Coverage 100 %: jede Versnummer 1..n kommt mindestens einmal vor
    present = set(verse_numbers)
    missing = set(range(1, n + 1)) - present
    assert not missing, f"Sure {sid}: Verse fehlen im Mapping: {sorted(missing)}"

    # 2) Keine Versnummer ausserhalb des Bereichs
    assert max(verse_numbers) <= n, f"Sure {sid}: Versnummer > {n} im Mapping"

    # 3) Sequenzialität: erste Vorkommensposition steigt streng (keine Luecken/Sprünge)
    first_pos = {}
    for idx, num in enumerate(verse_numbers):
        first_pos.setdefault(num, idx)
    ordered = sorted(first_pos.items())
    for i in range(1, len(ordered)):
        assert ordered[i][1] > ordered[i - 1][1], (
            f"Sure {sid}: Vers {ordered[i][0]} erscheint vor Vers {ordered[i - 1][0]} "
            f"(erste Positionen {ordered[i - 1][1]} >= {ordered[i][1]}) – Luecke/Sprung"
        )


def test_dedupe_gleicher_timestamp():
    from modules.circlelog import dedupe_mapping_lines

    lines = [
        "[00:00:11] :: 1: a 2: b 3: c\n",
        "[00:00:11] :: 1: a 2: b 3: c\n",
        "[00:00:11] :: 1: a\n",
        "[00:00:38] :: 4: d 5: e\n",
        "[00:00:38] :: 4: d 5: e\n",
        "[00:00:58] :: 6: f\n",
        "[00:00:58] :: f\n",
        "[00:01:06] :: 6: f\n",
    ]
    deduped = dedupe_mapping_lines(lines)
    text = "".join(deduped)
    # identische Zeilen + abgedeckte Zeile entfernt, andere Timestamps bleiben
    assert text.count("[00:00:11] :: 1: a 2: b 3: c") == 1
    assert text.count("[00:00:38] :: 4: d 5: e") == 1
    assert "[00:00:11] :: 1: a\n" not in text      # von Zeile 1 abgedeckt
    assert "[00:00:58] :: f\n" not in text          # von "6: f" abgedeckt
    assert "[00:01:06] :: 6: f\n" not in text       # identisch zu 00:00:58 -> entfernt


def test_balanced_paren_group_verschachtelt():
    from mapping import _balanced_paren_group

    g = _balanced_paren_group("(i.e. Prophet Muhammad (Peace be upon him) and whatever was revealed to him). 5: x", 0)
    assert g == "(i.e. Prophet Muhammad (Peace be upon him) and whatever was revealed to him)"
    g2 = _balanced_paren_group("(in the Oneness of Allah, and in His Messenger Muhammad (Peace be upon him)) including all", 0)
    assert g2 == "(in the Oneness of Allah, and in His Messenger Muhammad (Peace be upon him))"
    g3 = _balanced_paren_group("(biting them from anger) and said", 0)
    assert g3 == "(biting them from anger)"


def test_mapping_zeile_beginnt_nie_mit_klammer():
    # Fuehrender Klammerblock (verschachtelt) wandert ans Ende der vorherigen
    # Zeile; die naechste Zeile startet ohne Klammer-/Kommentar-Ueberreste.
    from mapping import _assemble_mapping_lines

    out = _assemble_mapping_lines([
        (11.0, "[00:00:11] :: 1: a b"),
        (63.0, "[00:01:03] :: (i.e. Prophet Muhammad (Peace be upon him)). 2: c"),
    ])
    assert out[0].endswith("(i.e. Prophet Muhammad (Peace be upon him)).")
    assert out[1] == "[00:01:03] :: 2: c"
    assert not any(" :: (" in l for l in out)


def test_mapping_zeile_endet_nie_mit_halber_klammer():
    # Zeile endet mit ungeschlossener "(": die Klammer-Fortsetzung aus der
    # naechsten Zeile wird herangezogen, damit die naechste Zeile sauber
    # startet (keine Klammer-/Kommentar-Ueberreste). Der Klammer-Ueberrest
    # (', underneath...') ist ein Satzzeichen-Fragment und wandert ebenfalls
    # an das Ende der Vorzeile.
    from mapping import _assemble_mapping_lines

    out = _assemble_mapping_lines([
        (107.0, "[00:01:47] :: 8: Their reward with their Lord is Adn (Eden) Paradise ("),
        (126.0, "[00:02:06] :: Gardens of Eternity), underneath which rivers flow"),
    ])
    assert out == [
        "[00:01:47] :: 8: Their reward with their Lord is Adn (Eden) Paradise (Gardens of Eternity), underneath which rivers flow"
    ]
    assert not any(" :: (" in l for l in out)


def test_mapping_fragment_bindestrich_split():
    # Ein abgeschnittenes Bindestrich-Wort ('All-' / 'Mighty') wird verbunden;
    # ein Vers-Marker in der Fortsetzung bleibt auf seiner eigenen Zeile.
    from mapping import _assemble_mapping_lines

    out = _assemble_mapping_lines([
        (91.0, "[00:01:31] :: And He is the All-"),
        (109.0, "[00:01:49] :: Mighty, the All-Wise. 5: And indeed We sent Moosa"),
    ])
    assert out[0].endswith("And He is the All-Mighty, the All-Wise.")
    assert out[1] == "[00:01:49] :: 5: And indeed We sent Moosa"


def test_mapping_fragment_satzzeichen_anfang():
    # Eine Zeile, die mit einem Satzzeichen beginnt, gehoert an die Vorzeile.
    from mapping import _assemble_mapping_lines

    out = _assemble_mapping_lines([
        (430.0, "[00:07:10] :: we shall drive you out of our land"),
        (447.0, "[00:07:27] :: , or you shall return to our religion."),
    ])
    assert out == [
        "[00:07:10] :: we shall drive you out of our land, or you shall return to our religion."
    ]


def test_mapping_fragment_punkt_vor_versnummer():
    # Steht vor einer Versnummer nur ein Satzzeichen-Fragment, wandert nur das
    # Satzzeichen an die Vorzeile, die Versnummer bleibt auf ihrer Zeile.
    from mapping import _assemble_mapping_lines

    out = _assemble_mapping_lines([
        (63.0, "[00:01:03] :: 4: And the people ... came to them clear evidence"),
        (83.0, "[00:01:23] :: . 5: And they were commanded not"),
    ])
    assert out[0].endswith("clear evidence.")
    assert out[1] == "[00:01:23] :: 5: And they were commanded not"


def test_mapping_halbe_klammer_verschachtelt():
    # Fortsetzung einer halben "(" mit verschachtelter Klammer im Inhalt.
    from mapping import _assemble_mapping_lines

    out = _assemble_mapping_lines([
        (107.0, "[00:01:47] :: 7: those who believe (in the Oneness of Allah, and in His Messenger Muhammad ("),
        (126.0, "[00:02:06] :: in the Oneness of Allah, and in His Messenger Muhammad (Peace be upon him)) and do righteous good deeds"),
    ])
    assert out[0].endswith("Muhammad (in the Oneness of Allah, and in His Messenger Muhammad (Peace be upon him))")
    assert out[1] == "[00:02:06] :: and do righteous good deeds"


def test_circle_override_sure96():
    from modules.circle_overrides import apply_circle_override
    from modules.videowindow import FrameWindow

    # w7 (134.8-155.2s): zeigt Verse 16+17+18, Detektion fand 2 -> n=3
    n = apply_circle_override(96, FrameWindow(134.8, 155.2), 2)
    assert n == 3, f"Sure 96 w7 soll 3 Kreise haben, hat aber {n}"
    # andere Suren/Fenster unveraendert
    assert apply_circle_override(96, FrameWindow(10.8, 26.4), 2) == 2
    assert apply_circle_override(97, FrameWindow(134.8, 155.2), 2) == 2


def test_circle_override_juz30_luecken():
    from modules.circle_overrides import apply_circle_override
    from modules.videowindow import FrameWindow

    # Sure 83 w6 (V13-Marker uebersehen) -> n=2
    assert apply_circle_override(83, FrameWindow(128.8, 146.0), 1) == 2
    # Sure 89 w14 (V27+V28-Rezitation) -> n=2
    assert apply_circle_override(89, FrameWindow(266.0, 287.6), 1) == 2
    # Sure 90 w6 (V16-Marker uebersehen) -> n=3
    assert apply_circle_override(90, FrameWindow(116.0, 133.6), 2) == 3
    # andere unveraendert
    assert apply_circle_override(83, FrameWindow(10.8, 26.4), 2) == 2
    assert apply_circle_override(90, FrameWindow(168.4, 182.8), 2) == 2
