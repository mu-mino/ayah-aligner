"""
Test: WordClassifier (Embedding-gestuetzte Wortklassifikation + if-else-Kaskade).

Kategorien: GOD / DESTRUCTIVE / CONSTRUCTIVE.
if-else-Kaskade schliesst False-Positives aus und markiert verifizierte Woerter.
"""

import sys
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from modules.text_ass import (  # noqa: E402
    REFERENCE_THEMES,
    _THEME_EXCLUSIONS,
    _WORD_CLASSIFIER,
)


@pytest.mark.parametrize(
    "word,expected",
    [
        ("Allah", "GOD"), ("Lord", "GOD"), ("Creator", "GOD"), ("creator", "GOD"),
        ("Divine", "GOD"), ("Originator", "GOD"), ("Omnipotent", "GOD"),
        ("Hell", "DESTRUCTIVE"), ("Nay", "DESTRUCTIVE"), ("evil", "DESTRUCTIVE"),
        ("torment", "DESTRUCTIVE"), ("suffer", "DESTRUCTIVE"), ("suffering", "DESTRUCTIVE"),
        ("distress", "DESTRUCTIVE"), ("prison", "DESTRUCTIVE"), ("betray", "DESTRUCTIVE"),
        ("lust", "DESTRUCTIVE"), ("Woe", "DESTRUCTIVE"), ("sin", "DESTRUCTIVE"),
        ("Paradise", "CONSTRUCTIVE"), ("righteous", "CONSTRUCTIVE"),
        ("believers", "CONSTRUCTIVE"), ("success", "CONSTRUCTIVE"),
        ("successful", "CONSTRUCTIVE"), ("belief", "CONSTRUCTIVE"),
        ("angel", "CONSTRUCTIVE"), ("angels", "CONSTRUCTIVE"),
        # Propheten-Namen (groß geschrieben, wie im Korpus)
        ("Muhammad", "CONSTRUCTIVE"), ("Moosa", "CONSTRUCTIVE"),
        ("Iesa", "CONSTRUCTIVE"), ("Nooh", "CONSTRUCTIVE"),
        ("Yoosuf", "CONSTRUCTIVE"), ("Ibrahim", "CONSTRUCTIVE"),
        ("Ishaque", "CONSTRUCTIVE"), ("Ibraheem", "CONSTRUCTIVE"),
        ("Hood", "CONSTRUCTIVE"), ("Khidr", "CONSTRUCTIVE"),
        ("Messiah", "CONSTRUCTIVE"), ("Dhul-Qarnain", "CONSTRUCTIVE"),
        ("Al-Yasa", "CONSTRUCTIVE"), ("Dhul-Kifl", "CONSTRUCTIVE"),
        # Engel-Namen
        ("Gabriel", "CONSTRUCTIVE"), ("Jibrael", "CONSTRUCTIVE"),
        ("Mikael", "CONSTRUCTIVE"), ("Israfil", "CONSTRUCTIVE"),
        # arabische Glaubensbegriffe
        ("Iman", "CONSTRUCTIVE"), ("Shukr", "CONSTRUCTIVE"),
        ("Hidayah", "CONSTRUCTIVE"), ("Noor", "CONSTRUCTIVE"),
        ("Rahmah", "CONSTRUCTIVE"),
    ],
)
def test_kategorie(word, expected):
    assert _WORD_CLASSIFIER.classify(word) == expected


@pytest.mark.parametrize(
    "word",
    [
        # GOD-False-Positives
        "muhammad", "jesus", "salih", "islamic", "gods", "lords", "deities",
        "fashioned", "shaped", "formed",
        "created", "creators", "tremendous", "enormous", "sacred", "greatest",
        "umrah", "taurah", "bismillah",
        # DESTRUCTIVE-False-Positives
        "leave", "avoid", "forget", "remove", "refuse", "oppose", "vanish",
        "disappear", "cancel", "decision", "warned", "swear", "moved", "therein",
        "consequences",
        # CONSTRUCTIVE-False-Positives
        "benefit", "advice", "towards", "thing", "built", "clouds", "created",
        "lamp", "produce", "planning",
        # neutral
        "the", "and", "book", "quran", "walking", "talking", "thanking",
    ],
)
def test_false_positive_ausgeschlossen(word):
    assert _WORD_CLASSIFIER.classify(word) is None, f"{word!r} faelschlich markiert"


@pytest.mark.parametrize(
    "word,category",
    [
        ("moved", "DESTRUCTIVE"), ("therein", "DESTRUCTIVE"),
        ("decision", "DESTRUCTIVE"), ("thing", "CONSTRUCTIVE"),
        ("built", "CONSTRUCTIVE"), ("clouds", "CONSTRUCTIVE"),
        ("created", "CONSTRUCTIVE"), ("lamp", "CONSTRUCTIVE"),
    ],
)
def test_is_excluded_jsonl(word, category):
    # Wird verwendet, um fehlerhafte jsonl/KI-Markierungen zu entfernen
    assert _WORD_CLASSIFIER.is_excluded(word, category)


@pytest.mark.parametrize(
    "word", ["him", "he", "we", "his", "our", "you", "my", "us", "me", "your", "i"],
)
def test_pronomen_klein_nie_god(word):
    # "Peace be upon him" (Prophet): KLEIN geschriebene Pronomen nie als GOD
    assert _WORD_CLASSIFIER.is_excluded(word, "GOD"), f"{word!r} (klein) darf nicht GOD sein"
    assert _WORD_CLASSIFIER.classify(word) is None


@pytest.mark.parametrize(
    "word", ["He", "We", "Him", "His", "Our", "Me", "You"],
)
def test_pronomen_gross_darf_god_bleiben(word):
    # GROSS geschriebene Pronomen (Mitte-Satz) koennen Allah meinen ->
    # jsonl-Entscheidung bleibt erhalten (nicht pauschal entfernt)
    assert not _WORD_CLASSIFIER.is_excluded(word, "GOD"), f"{word!r} (gross) soll bleiben"


@pytest.mark.parametrize(
    "word", ["Who", "Whom", "Whose", "Which"],
)
def test_relativpronomen_gross_darf_god_bleiben(word):
    # "He it is Who has created you" - grossgeschriebenes Relativpronomen kann Allah meinen
    assert not _WORD_CLASSIFIER.is_excluded(word, "GOD"), f"{word!r} (gross) soll bleiben"


@pytest.mark.parametrize(
    "word", ["who", "whom", "whose", "which"],
)
def test_relativpronomen_klein_nie_god(word):
    assert _WORD_CLASSIFIER.is_excluded(word, "GOD"), f"{word!r} (klein) darf nicht GOD sein"


def test_who_god_rendering():
    # JSONL markiert 'Who' in "He it is Who has created you" als GOD
    from modules.text_ass import annotate_highlights

    out = annotate_highlights(
        "He it is Who has created you from clay",
        {"0": "GOD", "3": "GOD"},
        base_font_size=42,
    )
    assert r"\c&H75DFFA&\b1}Who" in out or r"75DFFA&}Who" in out, "Who nicht GOD gerendert"


def test_fear_ohne_god_kein_constructive():
    # "hearts that Day will shake with fear and anxiety" (Sure 79, V8):
    # Terror der Stunde ohne Gottesbezug -> 'fear' darf NICHT blau bleiben.
    from modules.text_ass import annotate_highlights

    out = annotate_highlights(
        "hearts that Day will shake with fear and anxiety",
        {"6": "CONSTRUCTIVE"},
        base_font_size=42,
    )
    assert "803500" not in out, "fear ohne Gottesbezug faelschlich CONSTRUCTIVE"


def test_fear_mit_god_bleibt_constructive():
    # "so you should fear Him" (Sure 79, V19): Taqwa -> 'fear' bleibt blau,
    # weil im selben Satz ein GOD-Token ('Him') steht.
    from modules.text_ass import annotate_highlights

    out = annotate_highlights(
        "so you should fear Him",
        {"3": "CONSTRUCTIVE", "4": "GOD"},
        base_font_size=42,
    )
    assert "803500&}fear" in out, "fear mit Gottesbezug soll CONSTRUCTIVE bleiben"
    assert r"\c&H75DFFA&\b1}Him" in out or "75DFFA&}Him" in out, "Him soll GOD bleiben"


def test_fear_god_in_anders_satz_kein_constructive():
    # 'fear' (Terror) und 'Allah' (GOD) in VERSCHIEDENEN Saetzen desselben
    # Verses: GOD im selben SATZ, nicht nur im selben Vers, zaehlt.
    from modules.text_ass import annotate_highlights

    out = annotate_highlights(
        "hearts shake with fear. Allah alone is the Knower",
        {"3": "CONSTRUCTIVE"},
        base_font_size=42,
    )
    assert "803500" not in out, "fear darf in Satz ohne Gott nicht blau sein"


def test_angeklebte_klammer_bleibt_ausserehalb_normal():
    # Sure 14:1 "Alif-Lam-Ra.(These ..." - die Klammer klebt am Punkt.
    # "Alif-Lam-Ra." ist NICHT in Klammern und darf nicht klein+grau werden.
    from modules.text_ass import annotate_highlights

    out = annotate_highlights(
        "Alif-Lam-Ra.(These letters are one of the miracles)",
        {"17": "GOD"},
        base_font_size=42,
    )
    assert r"\c&HAAAAAA&}Alif-Lam-Ra" not in out, "Alif-Lam-Ra faelschlich in Klammerstil"
    assert "Alif-Lam-Ra. " + r"{\fs30\c&HAAAAAA&}(These" in out, "(These soll Klammer oeffnen"


def test_most_attribut_compound_god():
    # "Most Merciful" (Sure 14): jsonl/Classify darf CONSTRUCTIVE sein,
    # die REGEX-Kompound-Regel setzt es ausnahmslos auf GOD (Gold).
    from modules.text_ass import annotate_highlights

    out = annotate_highlights(
        "You are indeed Oft-Forgiving, Most Merciful.",
        {"3": "CONSTRUCTIVE", "5": "CONSTRUCTIVE"},
        base_font_size=42,
    )
    assert r"75DFFA&\b1}Most" in out, "Most nicht GOD"
    assert r"75DFFA&\b1}Merciful" in out, "Merciful nicht GOD"
    assert "803500&}Merciful" not in out, "Merciful faelschlich blau"


def test_all_able_und_all_mighty_god():
    # "All-Able" (ein Token) und "All- Mighty" (Umbruch: "All-" + "Mighty")
    from modules.text_ass import annotate_highlights

    out = annotate_highlights(
        "He is All- Mighty, - All-Able of everything",
        {},
        base_font_size=42,
    )
    assert r"75DFFA&\b1}All-" in out, "All- nicht GOD"
    assert r"75DFFA&\b1}Mighty" in out, "Mighty nicht GOD"
    assert r"75DFFA&\b1}All-Able" in out, "All-Able nicht GOD"


def test_most_mit_klammer_compound_god():
    # Sure 34: "(Most Trustworthy)" - die Klammer klebt am Token "(Most".
    from modules.text_ass import annotate_highlights

    out = annotate_highlights(
        "And He is the (Most Trustworthy) AllKnowing Judge.",
        {},
        base_font_size=42,
    )
    assert r"75DFFA&\b1}(Most" in out, "(Most nicht GOD"
    assert r"75DFFA&\b1}Trustworthy" in out, "Trustworthy nicht GOD"
    assert "803500&}Trustworthy" not in out, "Trustworthy faelschlich blau"


def test_ishaque_rendering_blau():
    # Sure 14: "Ishaque (Isaac)" - Ishaque muss blau (CONSTRUCTIVE) sein.
    from modules.text_ass import annotate_highlights

    out = annotate_highlights(
        "He gave me in old age Ismail and Ishaque (Isaac)",
        {},
        base_font_size=42,
    )
    assert r"\c&H803500&}Ishaque" in out, "Ishaque nicht blau gerendert"


def test_name_mit_klammer_erkannt():
    # "(Muhammad SAW)" - Klammer am Namensanfang darf den Case-Check nicht brechen
    out = __import__("modules.text_ass", fromlist=["annotate_highlights"]).annotate_highlights(
        "He (Muhammad SAW) has forged it",
        {},
        base_font_size=42,
    )
    assert r"\c&H803500&}(Muhammad" in out or "803500&}(Muhammad" in out, "(Muhammad nicht blau"


def test_goettliche_pronomen_mittensatz_god():
    # Mid-Sentence grossgeschriebenes "Him"/"He" = Allah-Referenz -> GOD (Gold)
    from modules.text_ass import annotate_highlights

    out = annotate_highlights(
        "for the provision He gives you, on the contrary, you deny Him",
        {},
        base_font_size=42,
    )
    assert r"75DFFA&\b1}He" in out, "He nicht GOD"
    assert r"75DFFA&\b1}Him" in out, "Him nicht GOD"


def test_satzanfang_he_nicht_pauschal_god():
    # Satzanfang "He (Muhammad SAW) said" - grammatikalische Grossschreibung,
    # kann Propheten meinen -> nicht pauschal GOD
    from modules.text_ass import annotate_highlights

    out = annotate_highlights(
        'He (Muhammad SAW) said to his people',
        {},
        base_font_size=42,
    )
    assert "75DFFA&\b1}He" not in out, "Satzanfangs-He faelschlich GOD"


def test_kleines_pronomen_nie_god():
    # "peace be upon him" - klein -> nie GOD
    from modules.text_ass import annotate_highlights

    out = annotate_highlights(
        "Peace be upon him and his family",
        {},
        base_font_size=42,
    )
    assert "75DFFA&\b1}him" not in out and "75DFFA&\b1}his" not in out


def test_split_rest_block_min_2s():
    # Kurzer Rest-Block (niedrige Zeichenzahl) bekommt mindestens 2s;
    # die vorherigen Bloecke erhalten die Restzeit (total - 2s).
    from modules.text_ass import split_overflow_entries, Entry

    text = " ".join(["alpha"] * 16)  # 6 Zeilen -> Kopf + kurzer Rest-Block
    out = split_overflow_entries(
        [Entry(start=10.0, text=text), Entry(start=30.0, text="next")],
        video_width=400, font_size=42, max_lines=5,
    )
    assert len(out) == 3
    head, rest, nxt = out
    # Rest-Block startet so, dass er genau 2s vor der naechsten Zeile endet
    assert abs(30.0 - rest.start - 2.0) < 1e-6, f"Rest-Block {30.0-rest.start:.2f}s != 2s"
    # Kopf bekommt total - 2s (20 - 2 = 18)
    assert abs(rest.start - head.start - 18.0) < 1e-6


def test_split_proportional_ohne_kurzen_rest():
    # Ohne kurzen Rest-Block bleibt die proportionale Verteilung unveraendert.
    from modules.text_ass import split_overflow_entries, Entry

    text = " ".join(["beta"] * 30)  # genug fuer mehrere volle Bloecke
    out = split_overflow_entries(
        [Entry(start=10.0, text=text), Entry(start=30.0, text="next")],
        video_width=400, font_size=42, max_lines=5,
    )
    assert len(out) >= 3
    # alle Bloecke liegen im Intervall [10, 30] und decken es ab
    assert out[0].start == 10.0
    assert abs(out[-2].start + (out[-1].start - out[-2].start) - out[-1].start) < 1e-6
    # letzter echter Block endet bei 30 (naechste Mapping-Zeile)
    assert abs(out[-1].start - 30.0) < 1e-6


def test_pronomen_im_titel_nicht_god():
    # Der ASS-Titel (Header, apply_regex=False) darf "He/Him/His" NICHT als
    # GOD faerben (z.B. Sure 80 "Abasa - He Frowned").
    from modules.text_ass import annotate_highlights

    out = annotate_highlights(
        "Abasa - He Frowned",
        {},
        apply_regex=False,
        base_font_size=42,
    )
    assert r"75DFFA" not in out, "Titel 'He' darf nicht GOD gefaerbt sein"
    assert "He" in out, "Titel-Text muss erhalten bleiben"

    # Im Verstext (apply_regex=True) wird das grossgeschriebene Pronomen
    # MITTEN im Satz weiterhin als GOD markiert.
    out_verse = annotate_highlights(
        "and He it is Who has created you from clay",
        {},
        apply_regex=True,
        base_font_size=42,
    )
    assert r"75DFFA" in out_verse, "Vers-Pronomen soll GOD bleiben"
