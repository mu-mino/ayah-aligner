"""
Embedding-gestützte semantische Wortklassifikation (bge-large-en-v1.5)
+ Fall-für-Fall-Behandlung (if-else-Kaskade) für stabile Markierungen.

Ablauf:
    1. scan_corpus(): schickt ALLE eindeutigen Wörter des Korpus durch das
       grosse englische Embedding-Modell und bewertet die Nähe zu den drei
       Kategorien (GOD / DESTRUCTIVE / CONSTRUCTIVE) anhand von Ankerwörtern.
       Kandidaten ab (score >= SCORE_THRESHOLD, margin >= MARGIN_THRESHOLD)
       landen in einem Dict.
    2. Die Kandidaten wurden danach gegen die Korpus-Kontexte semantisch
       verifiziert; echte Treffer in EXTRA_PATTERNS, False-Positives in der
       if-else-Kaskade der WordClassifier-Klasse einzeln ausgeschlossen.
    3. WordClassifier.classify() = schneller Laufzeitpfad (ohne Embedding):
       if-else-Kaskade -> Regex -> verifizierte Zusatzwörter.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Embedding-Modell (lazy geladen, nur für die Entdeckung/Korpus-Analyse nötig)
# ---------------------------------------------------------------------------

_EMBEDDING_MODEL = None


def _get_embedding_model():
    """Lädt das grosse englische Embedding-Modell (BAAI/bge-large-en-v1.5)."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDING_MODEL = SentenceTransformer("BAAI/bge-large-en-v1.5")
    return _EMBEDDING_MODEL


# Ankerwörter pro Kategorie (Referenzpunkte für die Embedding-Nähe)
CATEGORY_ANCHORS: Dict[str, List[str]] = {
    "GOD": ["Allah", "God", "Lord", "Almighty", "Creator", "divine", "Merciful"],
    "DESTRUCTIVE": ["Hell", "torment", "punishment", "sin", "curse", "destroy", "disbeliever"],
    "CONSTRUCTIVE": ["Paradise", "righteous", "believer", "guidance", "reward", "mercy", "virtue"],
}

# Empirisch bestimmte Grenzen (aus Verteilung: darunter ist das Signal zu schwach)
SCORE_THRESHOLD = 0.60   # Cosinus-Nähe zur besten Kategorie
MARGIN_THRESHOLD = 0.10  # Abstand zur zweitbesten Kategorie


def scan_corpus(
    corpus_dir: Path,
    score_threshold: float = SCORE_THRESHOLD,
    margin_threshold: float = MARGIN_THRESHOLD,
) -> Dict[str, Dict[str, float]]:
    """
    Schiesst alle eindeutigen Wörter des Korpus durch das Embedding-Modell
    und sammelt Kandidaten, deren beste Kategorie über dem Threshold liegt.
    """
    import collections

    from sentence_transformers import util

    model = _get_embedding_model()
    full = "\n".join(
        p.read_text(encoding="utf-8") for p in sorted(corpus_dir.glob("*.txt"))
    )
    tokens = re.findall(r"[A-Za-z][A-Za-z\-']*", full)
    freq = collections.Counter(re.sub(r"[^\w]", "", t).lower() for t in tokens)
    uniq = sorted(freq)

    word_emb = model.encode(uniq, normalize_embeddings=True, batch_size=256)
    anch_emb = {
        cat: model.encode(words, normalize_embeddings=True)
        for cat, words in CATEGORY_ANCHORS.items()
    }

    result = {cat: {} for cat in CATEGORY_ANCHORS}
    for i, w in enumerate(uniq):
        scores = {
            cat: float(util.cos_sim(word_emb[i], ae)[0].max())
            for cat, ae in anch_emb.items()
        }
        best = max(scores, key=scores.get)
        second = sorted(scores.values(), reverse=True)[1]
        if scores[best] >= score_threshold and (scores[best] - second) >= margin_threshold:
            result[best][w] = round(scores[best], 3)
    return result


# ---------------------------------------------------------------------------
# Manuell verifizierte Zusatzwörter (aus Embedding-Kandidaten, gegen die
# Korpus-Kontexte geprüft). Stems verankert (^...$), Konjugation abgedeckt.
# ---------------------------------------------------------------------------

EXTRA_PATTERNS: Dict[str, List[str]] = {
    "GOD": [
        r"^creator$",        # "Allah is the Creator of all things"
        r"^divine$",         # "Divine Laws / Divine Inspiration / Qadar"
        r"^originator$",     # "the Originator of the heavens and the earth"
        r"^omnipotent$",     # "the Omnipotent King (Allah)"
    ],
    "DESTRUCTIVE": [
        r"^suffer\w*$",      # suffer / suffers / suffered / suffering
        r"^distress\w*$",    # distress / distressed / distresses
        r"^anguish\w*$",     # anguish
        r"^prison\w*$",      # prison / prisoners
        r"^betray\w*$",      # betray / betrays / betrayed / betrayal
        r"^lust\w*$",        # lust (vain desires)
    ],
    "CONSTRUCTIVE": [
        r"^success\w*$",     # success / successful ("the supreme success" = Paradise)
        r"^belief\w*$",      # belief / beliefs (Iman)
    ],
}


# ---------------------------------------------------------------------------
# WordClassifier: if-else-Kaskade (jeder Fall einzeln) + Regex + Zusatzwörter
# ---------------------------------------------------------------------------

class WordClassifier:
    """Klassifiziert ein einzelnes Wort in GOD/DESTRUCTIVE/CONSTRUCTIVE.

    Priorität:
        1. if-else-Kaskade (_special_case): entscheidet einzelne Fälle
           definitiv (Kategorie ODER explizit None = nie markieren),
           um False-Positives der Embedding/Regex-Analyse auszuschliessen.
        2. Regex (REFERENCE_THEMES, verankerte Stems).
        3. Verifizierte Embedding-Zusatzwörter (EXTRA_PATTERNS).
    """

    _UNHANDLED = object()

    def __init__(self, regex_themes: Dict[str, "re.Pattern"], exclusions: Dict[str, set]):
        self.regex_themes = regex_themes
        self.exclusions = exclusions
        self.extra = {
            cat: [re.compile(p, re.IGNORECASE) for p in pats]
            for cat, pats in EXTRA_PATTERNS.items()
        }

    # ------------------------------------------------------------------
    # If-else-Kaskade: jeder einzelne Fall, basierend auf Korpus-Kontexten
    # ------------------------------------------------------------------
    def _special_case(self, clean: str):
        # --- GOD: Embedding riecht "göttlich", ist aber KEINE Gottheit ---
        if clean in {
            # Propheten / Personen
            "muhammad", "muhammads", "muhammed", "muhammad's", "jesus", "isa",
            "salih", "ibrahim", "abraham", "moses", "moosa", "maryam", "mary",
            "aaron", "harun", "david", "dawud", "solomon", "sulaiman", "adam",
            "noah", "nuh", "john", "yahya", "joseph", "yusuf", "jacob", "yaqub",
            "elisha", "elyas", "isaac", "ishaq", "idris", "enoch", "ezra",
            "lot", "lut", "jethro", "shuaib", "zachariah", "zakariya", "job", "ayyub",
            # Religion / Quellen (keine Gottheit)
            "islamic", "islam", "monotheism", "monotheist", "monotheists",
            "muslim", "muslims", "religion", "religions", "religious",
            "christian", "christians", "christianity", "jewish", "jews", "judaism",
            "quran", "qurans", "book", "books", "taurah", "torah", "gospel",
            # falsche Gottheiten (Plural / Götzendienst)
            "gods", "lords", "deities", "partnergods", "idols", "idolsworship",
            "idol", "idolaters", "lordship",
            # neutrale Verben/Substantive des Schaffens
            "created", "creates", "creation", "creations", "creators", "makers",
            "maker", "inventing", "invented", "invents", "inventor",
            # Adjektive (keine Gottheit)
            "sacred", "grand", "tremendous", "enormous", "greatest", "wonderful",
            "wondrous", "tremendously", "courteous", "great", "greater", "greatness",
            # Ritus-/Quellen-Begriffe
            "umrah", "hikmah", "fitrah", "bismillah", "salat", "zakat", "sunnah",
            "halal", "haram", "fiqh", "tauheed", "jihad",
        }:
            return None

        # --- GOD: eindeutig göttlich (aus Embedding-Kandidaten verifiziert) ---
        if clean in {"creator", "divine", "originator", "omnipotent"}:
            return "GOD"

        # --- DESTRUCTIVE: neutrale Verben, die Embedding/Regex nicht treffen ---
        if clean in {
            "leave", "leaves", "leaving", "avoid", "avoids", "avoiding",
            "forget", "forgot", "forgotten", "forgetful", "remove", "removes",
            "removed", "refuse", "refused", "refuses", "oppose", "opposed",
            "opposes", "vanish", "vanishs", "disappear", "disappears", "cancel",
            "collapsed", "collapse", "crumble", "dont", "consequences",
            "swear", "swears", "swearing", "swore", "sworn", "argue", "argues",
            "argued", "argument", "conjecture", "replace", "replaces", "disposal",
            "dispose", "disposes", "warn", "warned", "warns", "warning", "warnings",
            "decision", "decide", "decided", "decides", "judgement", "judgment",
            "moved", "therein",
        }:
            return None

        # --- DESTRUCTIVE: verifizierte Zusatzwörter (Leiden/Strafe) ---
        if re.match(r"^(?:suffer|distress|anguish|prison|betray|lust)\w*$", clean):
            return "DESTRUCTIVE"

        # --- CONSTRUCTIVE: neutrale/kommerzielle Wörter ausschliessen ---
        if clean in {
            "towards", "offer", "offers", "offered", "benefit", "benefits",
            "benefited", "earn", "earns", "earned", "profit", "profits",
            "profitable", "pays", "paid", "worth", "worthy", "worthier",
            "palmtrees", "thing", "things", "built", "clouds", "created",
            "abundant", "lamp", "produce", "rainy", "vegetations", "guarding",
            "guardianship", "guard", "guards", "advice", "advices", "advise",
            "advised", "advises", "adviser", "advisers", "advisors", "advising",
            "counsel", "counsels", "counselled", "counselor", "planning", "plan",
            "plans", "planned", "planning", "approach", "approaches", "approaching",
        }:
            return None

        # --- CONSTRUCTIVE: verifizierte Zusatzwörter (Erfolg/Glaube) ---
        if re.match(r"^(?:success|belief)\w*$", clean):
            return "CONSTRUCTIVE"

        return self._UNHANDLED

    def is_excluded(self, token: str) -> bool:
        """True, wenn die if-else-Kaskade das Wort EXPLIZIT als False-Positive
        ausschliesst (Kategorie None ausdruecklich). Wird auch auf jsonl-Eintraege
        angewendet, um fehlerhafte KI-Markierungen zu entfernen."""
        clean = re.sub(r"[^\w]", "", token).lower()
        if not clean:
            return True
        special = self._special_case(clean)
        # expliziter Ausschluss = None; _UNHANDLED/echte Kategorie = kein Ausschluss
        return special is None

    # ------------------------------------------------------------------
    # Öffentliche Klassifikation
    # ------------------------------------------------------------------
    def classify(self, token: str) -> Optional[str]:
        clean = re.sub(r"[^\w]", "", token).lower()
        if not clean:
            return None

        special = self._special_case(clean)
        if special is not self._UNHANDLED:
            return special

        # Regex-Stems (verankert, aus chunked_translations abgeleitet)
        for cat, pat in self.regex_themes.items():
            if pat.search(clean):
                if clean in self.exclusions.get(cat, ()):
                    return None
                return cat

        # Verifizierte Embedding-Zusatzwörter
        for cat, pats in self.extra.items():
            if any(p.match(clean) for p in pats):
                return cat

        return None


def normalize(word: str) -> str:
    return re.sub(r"[^\w]", "", word).strip()
