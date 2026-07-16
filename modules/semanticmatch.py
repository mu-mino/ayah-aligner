"""
Semantisches Matching: arabischer Chunk → Span aus dem englischen Vers-Text.

Ablauf pro ChunkTranscription:
    1. Matcher    : findet den Textspan im Vers, der dem arabischen raw_text
                    am ähnlichsten ist (Cross-Lingual Embeddings, sliding window)

Guard prüft die vom Modell getroffenen Span-Auswahlen:
    1. Kontiguität   : jeder Span ist ein zusammenhängender Substring
    2. Vorwärts      : span[i].start >= span[i-1].start (nie rückwärts)
    3. Vollständigkeit: am Ende keine Lücken – alle Teile des Verses abgedeckt

Wiederholungen sind erlaubt (Anfang, Mitte, Mitte, Ende ist gültig).
Springen und Auslassen sind nicht erlaubt.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Optional, Tuple
from typing import Tuple
import spacy
import regex

from modules.whispertranscribe import ChunkTranscription
from sentence_transformers import SentenceTransformer, util

from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer

from pathlib import Path
import os

os.environ["HF_HUB_OFFLINE"] = "1"
model = SentenceTransformer("symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli")

# ---------------------------------------------------------------------------


def _lcs_length(a_tokens: list, b_tokens: list) -> int:
    """Longest common subsequence length (word-level)."""
    n, m = len(a_tokens), len(b_tokens)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        ai = a_tokens[i - 1]
        for j in range(1, m + 1):
            if ai == b_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = (
                    dp[i - 1][j] if dp[i - 1][j] >= dp[i][j - 1] else dp[i][j - 1]
                )
    return dp[n][m]


def _rouge_l(reference: str, candidate: str) -> float:
    """ROUGE-L F-score (LCS-based, beta=1.2)"""
    ref_tokens = str(reference).split()
    cand_tokens = str(candidate).split()
    if not ref_tokens or not cand_tokens:
        return 0.0
    lcs = _lcs_length(ref_tokens, cand_tokens)
    recall = lcs / len(ref_tokens)
    precision = lcs / len(cand_tokens)
    beta = 1.9
    denom = recall + (beta**2) * precision
    if denom == 0:
        return 0.0
    return ((1 + beta**2) * recall * precision) / denom


# ---------------------------------------------------------------------------
# Arabische Normalisierung
# ---------------------------------------------------------------------------

_AR_DIACRITICS = regex.compile(r"[\p{M}\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u06DF-\u06E8\u06EA-\u06FF]+")
_AR_TATWEEL = "\u0640"
_AR_NON_ARABIC = regex.compile(r"[^\p{Arabic} ]+")
_AR_MULTI_SPACE = regex.compile(r"\s+")
_AR_PREFIX = regex.compile(r"(?<!\S)(و|ف|ب|ك|ل|س)\s+(?=\S)", regex.UNICODE)
_AR_CHAR_MAP = str.maketrans(
    {
        "آ": "ا",
        "ٱ": "ا",
        "أ": "ا",
        "إ": "ا",
        "ى": "ي",
        "ئ": "ي",
        "ؤ": "و",
        "ة": "ه",
        "ء": "",
        "گ": "ك",
        "ڤ": "ف",
        "پ": "ب",
        "چ": "ج",
        "ٓ": "",  # Hamza-oben (Madda-Zeichen)
        "ۭ": "",  # Small High Sign
        "ۙ": "",  # Pause-Zeichen
        "ـ": "",  # Tatweel (falls nicht via separate Zeile entfernt)
    }
)


def _normalize_arabic(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = _AR_DIACRITICS.sub("", text)
    text = text.replace(_AR_TATWEEL, "")
    text = text.translate(_AR_CHAR_MAP)
    text = _AR_NON_ARABIC.sub(" ", text)
    text = _AR_MULTI_SPACE.sub(" ", text).strip()
    text = _AR_PREFIX.sub(r"\1", text)
    return unicodedata.normalize("NFKC", text)


# ---------------------------------------------------------------------------
# Konstanten
# ---------------------------------------------------------------------------

QURAN_API_BASE: str = "https://api.quran.com/api/v4"
QURAN_TRANSLATION_ID: int = 203  # Al-Hilali & Khan
WORD_MATCH_TOLERANCE: float = 0.7

_CACHE_DIR: Path = Path(__file__).resolve().parent.parent / "data" / "api"


# ---------------------------------------------------------------------------
# Datenstrukturen
# ---------------------------------------------------------------------------


@dataclass
class TextSpan:
    """Ein zusammenhängender Textspan aus dem Vers."""

    start: int  # Zeichenposition im Vers-Text (inklusiv)
    end: int  # Zeichenposition im Vers-Text (exklusiv)
    text: str


@dataclass
class MatchResult:
    """Ergebnis des Matchings für einen einzelnen Chunk."""

    chunk: ChunkTranscription
    arabic_text: str
    span: TextSpan
    score: float
    word_alignments: List[Tuple[str, str, int]] = field(default_factory=list)


@dataclass
class GuardReport:
    """Ergebnis der Guard-Prüfung über alle Spans."""

    # order_passed: bool
    completeness_passed: bool
    # order_violations: List[int] = field(default_factory=list)  # result-Indizes
    uncovered_ranges: List[Tuple[int, int]] = field(
        default_factory=list
    )  # Lücken als (start, end)
    correction_hints: List[Tuple[int, str]] = field(
        default_factory=list
    )  # (chunk_idx, hint)
    correction_data: List[Tuple[Tuple[int, int], str]] = field(default_factory=list)

    # @property
    # def passed(self) -> bool:
    # completeness wird nicht verlangt: der nicht abgedeckte Suffix-Teil des Verses
    # wird vom circle_window-Eintrag der nächsten Gruppe übernommen.
    # return self.order_passed


@dataclass
class MatchSession:
    """Gesamtergebnis einer Matching-Sitzung für einen Vers."""

    verse_text: str
    ayah: int = 0
    results: List[MatchResult] = field(default_factory=list)
    guard: Optional[GuardReport] = None


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def extract_verse_text(mapping_line: str) -> str:
    """
    Extrahiert den Vers-Text aus einem mapping_line-Eintrag.
    Entfernt NUR die erste Versnummer, damit '32:'/'33:' als Text erhalten bleiben.
    """
    m = re.match(r"^\[\d{2}:\d{2}:\d{2}\]\s*::\s*(.*)$", mapping_line.strip())
    if not m:
        return ""
    content = m.group(1)
    text = re.sub(r"^\d+:\s*", "", content, count=1)
    return text.strip()


def extract_verse_number(mapping_line: str) -> Optional[int]:
    """Extrahiert die erste Vers-Nummer aus einem mapping_line-Eintrag."""
    m = re.match(r"^\[\d{2}:\d{2}:\d{2}\]\s*::\s*(\d+):", mapping_line.strip())
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Quran API
# ---------------------------------------------------------------------------


def _fetch_verse_words(surah: int, ayah: int) -> list:
    """Holt wortgenaues Alignment für einen Vers von quran.com, mit lokalem Cache."""
    import json
    import requests

    cache_file = _CACHE_DIR / f"{surah}_{ayah}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    resp = requests.get(
        f"{QURAN_API_BASE}/verses/by_key/{surah}:{ayah}",
        params={
            "words": "true",
            "word_fields": "text_uthmani,translation",
            "translations": QURAN_TRANSLATION_ID,
        },
    )
    resp.raise_for_status()
    verse = resp.json()["verse"]
    words = [
        w
        for w in verse["words"]
        if w.get("text_uthmani") and w["char_type_name"] != "end"
    ]
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(
        json.dumps(words, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return words


_CAMEL_DB = None
_CAMEL_ANALYZER = None


def _get_stems(word: str) -> set:
    global _CAMEL_DB, _CAMEL_ANALYZER
    if _CAMEL_ANALYZER is None:
        from camel_tools.morphology.analyzer import Analyzer
        from camel_tools.morphology.database import MorphologyDB

        db_path = "/home/muhammed-emin-eser/.camel_tools/data/morphology_db/calima-msa-r13/morphology.db"
        _CAMEL_DB = MorphologyDB(db_path)
        _CAMEL_ANALYZER = Analyzer(_CAMEL_DB)

    analyses = _CAMEL_ANALYZER.analyze(word)
    stems = set()
    for a in analyses:
        stem = a.get("stem", "")
        if stem:
            stems.add(stem)
    if not stems:
        stems.add(word)
    return stems


def _word_to_translation_from(arabic_word: str, verse_words: list, min_idx: int = 0) -> Tuple[str, int]:
    chunk_stems = _get_stems(arabic_word)

    for idx in range(min_idx, len(verse_words)):
        w = verse_words[idx]
        verse_stems = _get_stems(w["text_uthmani"])
        if chunk_stems & verse_stems:
            t = w.get("translation", "")
            text = t if isinstance(t, str) else ""
            if re.match(r"^\(\d+\)$", text.strip()):
                return ("", -1)
            return (text.strip(), idx)

    norm_word = _normalize_arabic(arabic_word)
    best_score = 0.0
    best_idx = -1
    best_match = None

    for idx in range(min_idx, len(verse_words)):
        w = verse_words[idx]
        norm_verse = _normalize_arabic(w["text_uthmani"])
        score = SequenceMatcher(None, norm_word, norm_verse).ratio()
        if score > best_score:
            best_score = score
            best_idx = idx
            best_match = w

    if best_score < WORD_MATCH_TOLERANCE or best_match is None:
        return ("", -1)

    t = best_match.get("translation", "")
    text = t if isinstance(t, str) else ""
    if re.match(r"^\(\d+\)$", text.strip()):
        return ("", -1)
    return (text.strip(), best_idx)


def _word_to_translation(arabic_word: str, verse_words: list) -> Tuple[str, int]:
    chunk_stems = _get_stems(arabic_word)

    for idx, w in enumerate(verse_words):
        verse_stems = _get_stems(w["text_uthmani"])
        if chunk_stems & verse_stems:
            t = w.get("translation", "")
            text = t if isinstance(t, str) else ""
            if re.match(r"^\(\d+\)$", text.strip()):
                return ("", -1)
            return (text.strip(), idx)

    norm_word = _normalize_arabic(arabic_word)
    best_score = 0.0
    best_idx = -1
    best_match = None

    for idx, w in enumerate(verse_words):
        norm_verse = _normalize_arabic(w["text_uthmani"])
        score = SequenceMatcher(None, norm_word, norm_verse).ratio()
        if score > best_score:
            best_score = score
            best_idx = idx
            best_match = w

    if best_score < WORD_MATCH_TOLERANCE or best_match is None:
        return ("", -1)

    t = best_match.get("translation", "")
    text = t if isinstance(t, str) else ""
    if re.match(r"^\(\d+\)$", text.strip()):
        return ("", -1)
    return (text.strip(), best_idx)


nlp = spacy.blank("en")


def _get_word_similarity(w1: str, w2: str) -> float:
    """Prüft, ob zwei Wörter sich sehr ähnlich sind (z.B. disbelieve vs disbelievers)"""
    # Einfacher, schneller Präfix- und Infix-Vergleich für Wortformen
    w1_clean, w2_clean = w1.lower(), w2.lower()
    if w1_clean == w2_clean:
        return 1.0
    # Erfasst Wortstämme (z.B. "disbeliev")
    min_len = min(len(w1_clean), len(w2_clean))
    if min_len > 4 and w1_clean[: min_len - 2] == w2_clean[: min_len - 2]:
        return 0.8
    return 0.0


def find_semantic_span(
    query: str, full_translation: str
) -> Tuple[int, int, str, float]:
    doc_a = nlp(query)
    doc_b = nlp(full_translation)

    # Bereinige die Query-Wörter (keine Satzzeichen)
    query_words = [t.text for t in doc_a if not t.is_punct]
    if not query_words or len(doc_b) == 0:
        return 0, len(full_translation), full_translation, 0.0

    best_score = -1.0
    best_start_tok = 0
    best_end_tok = len(doc_b)

    # Dynamisches Schiebefenster über die Wörter von Text B
    # Das Fenster orientiert sich an der Länge der Query (mit etwas Puffer)
    min_window = max(1, len(query_words) - 5)
    max_window = min(len(doc_b), len(query_words) + 15)

    for window_size in range(min_window, max_window + 1):
        for i in range(len(doc_b) - window_size + 1):
            sub_span = doc_b[i : i + window_size]

            # Zähhle, wie viele Wörter aus der Query wir in diesem Fenster wiederfinden
            matched_words = 0
            for qw in query_words:
                # Prüfe, ob das Query-Wort (oder ein ähnliches) im Fenster existiert
                if any(_get_word_similarity(qw, bw.text) > 0.7 for bw in sub_span):
                    matched_words += 1

            # Berechne die Dichte (F1-Ähnliche Gewichtung)
            # Wie viel Prozent der Query haben wir gefunden, gestraft durch die Fensterlänge?
            recall = matched_words / len(query_words)
            precision = matched_words / window_size if window_size > 0 else 0

            if recall + precision > 0:
                # F1-Score belohnt Vollständigkeit bei gleichzeitig hoher Dichte
                score = (2 * precision * recall) / (precision + recall)
            else:
                score = 0.0

            # Bei Gleichstand bevorzugen wir das kleinere, präzisere Fenster
            if score > best_score:
                best_score = score
                best_start_tok = i
                best_end_tok = i + window_size
            elif score == best_score and window_size < (best_end_tok - best_start_tok):
                best_start_tok = i
                best_end_tok = i + window_size

    # Konvertiere Token-Indizes zurück zu exakten Zeichen-Indizes
    start_token = doc_b[best_start_tok]
    end_token = doc_b[best_end_tok - 1]

    char_start = start_token.idx
    char_end = end_token.idx + len(end_token.text)
    span_text = full_translation[char_start:char_end]

    return char_start, char_end, span_text, float(best_score)


# ---------------------------------------------------------------------------
# Gap Fill
# ---------------------------------------------------------------------------


def _fill_gaps(results: List[MatchResult], verse_text: str) -> None:
    """
    Füllt den Prefix-Bereich vor dem ersten Span, Lücken zwischen Spans und
    den Suffix nach dem letzten Span.
    Modifiziert results in-place.

    Primäre Entscheidung per Embedding-Cosinus-Ähnlichkeit, gefolgt von
    einer deterministischen Vorwärts-Ausdehnung als Sicherheitsnetz.
    """
    if not results:
        return

    by_start = sorted(results, key=lambda r: r.span.start)

    # Prefix: vor dem ersten Span
    if by_start[0].span.start > 0 and verse_text[: by_start[0].span.start].strip():
        by_start[0].span.start = 0
        by_start[0].span.text = verse_text[0 : by_start[0].span.end]

    # Middle: Lücken zwischen benachbarten Spans
    for i in range(1, len(by_start)):
        prev = by_start[i - 1]
        curr = by_start[i]
        if verse_text[prev.span.end : curr.span.start].strip():
            english_gap_text = verse_text[prev.span.end : curr.span.start].strip()
            arabic_text_prev = prev.chunk.raw_text
            arabic_text_curr = curr.chunk.raw_text

            # Embeddings (Vektoren) für die Texte generieren
            emb_english = model.encode(english_gap_text, convert_to_tensor=True)
            emb_prev = model.encode(arabic_text_prev, convert_to_tensor=True)
            emb_curr = model.encode(arabic_text_curr, convert_to_tensor=True)

            # Ähnlichkeit berechnen
            score_prev = util.cos_sim(emb_english, emb_prev).item()
            score_curr = util.cos_sim(emb_english, emb_curr).item()

            if score_prev > score_curr:
                prev.span.end = curr.span.start
                prev.span.text = verse_text[prev.span.start : prev.span.end]
            else:
                curr.span.start = prev.span.end
                curr.span.text = verse_text[curr.span.start : curr.span.end]

            # --- SATZZEICHEN-KORREKTUR (Waisen-Zeichen verhindern) ---
            if curr.span.text and re.match(r"^[^\w\s]", curr.span.text.strip()):
                symbol = curr.span.text[0]
                curr.span.text = curr.span.text[1:]
                curr.span.start += 1
                prev.span.text = prev.span.text + symbol
                prev.span.end += 1

    # Sicherheit: deterministische Vorwärts-Ausdehnung für verbleibende Lücken
    for i in range(1, len(by_start)):
        prev = by_start[i - 1]
        curr = by_start[i]
        if prev.span.end < curr.span.start:
            prev.span.end = curr.span.start
            prev.span.text = verse_text[prev.span.start : prev.span.end]


# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------


def run_guard(results: List[MatchResult], verse_text: str) -> GuardReport:
    """
    Prüft Vorwärtsrichtung und Vollständigkeit der Span-Auswahlen.

    Vorwärts:
        1. span[i].end >= span[i-1].end  — Ende darf nie zurückgehen
        2. span[i].start >= span[0].start — nie vor den Startpunkt
                                            der ersten Auswahl zurück
    Vollständigkeit:
        Union aller Spans deckt den gesamten Vers-Text lückenlos ab
        (Leerzeichen zwischen Spans werden toleriert).
    """
    if not results:
        return GuardReport(completeness_passed=True)  # , order_passed=True )

    # order_violations: List[int] = []
    # first_start = results[0].span.start
    # prev_end = results[0].span.end

    # for i in range(1, len(results)):
    #     span = results[i].span
    # end_ok = span.end >= prev_end
    # start_ok = span.start >= first_start
    # if not end_ok or not start_ok:
    #     order_violations.append(i)
    # prev_end = max(prev_end, span.end)

    # Vollständigkeit: Lücken im Vers-Text ermitteln (Whitespace toleriert)
    sorted_spans = sorted(
        [(r.span.start, r.span.end) for r in results], key=lambda x: x[0]
    )
    uncovered: List[Tuple[int, int]] = []
    correction_data = []

    if sorted_spans[0][0] > 0:
        prefix = verse_text[: sorted_spans[0][0]]
        if prefix.strip():
            uncovered.append(
                (0, sorted_spans[0][0])
            )  # if uncovered is not last Element

    cursor = sorted_spans[0][0]
    for start, end in sorted_spans:
        if start > cursor:
            gap = verse_text[cursor:start]
            if gap.strip():
                uncovered.append((cursor, start))
        cursor = max(cursor, end)

    if cursor < len(verse_text):
        suffix = verse_text[cursor:]
        if suffix.strip():
            uncovered.append((cursor, len(verse_text)))

    # Static correction hints per violation, explaining why the selection is invalid.
    correction_hints: List[Tuple[int, str]] = []

    # for i in order_violations:
    #     prev = results[i - 1]
    #     curr = results[i]
    #     correction_hints.append(
    #         (
    #             i,
    #             f'Chunk [{i}] goes backward: chunk [{i - 1}] ends at "{prev.span.text}" '
    #             f"(pos {prev.span.end}), but chunk [{i}] was placed at pos {curr.span.start}. "
    #             f"Recitation always moves forward — a later audio chunk cannot match earlier text.",
    #         )
    #     )

    for start, end in uncovered:
        gap_text = verse_text[start:end]
        last_before_gap = max(
            (i for i, r in enumerate(results) if r.span.end <= start),
            default=len(results) - 1,
        )
        correction_hints.append(
            (
                last_before_gap,
                f'Uncovered gap [{start}–{end}]: "{gap_text}". '
                f"Every word of the verse must be assigned to a timestamp.",
            )
        )
        correction_data.append([(start, end), gap_text])

    return GuardReport(
        # order_passed=len(order_violations) == 0,
        completeness_passed=len(uncovered) == 0,
        # order_violations=order_violations,
        uncovered_ranges=uncovered,
        correction_hints=correction_hints,
        correction_data=correction_data,
    )


# ---------------------------------------------------------------------------
# Öffentliche API
# ---------------------------------------------------------------------------


def run_matching(
    chunks: List[ChunkTranscription],
    verse_text: str,
    surah: int,
    ayah: int,
) -> MatchSession:
    """
    Matcht jeden arabischen Chunk gegen den englischen Vers-Text via quran.com API.
    """
    session = MatchSession(verse_text=verse_text, ayah=ayah)

    if not chunks:
        session.guard = GuardReport(completeness_passed=True)
        return session

    verse_words = _fetch_verse_words(surah, ayah)

    for chunk in chunks:
        text = chunk.raw_text.split()
        matched_pairs: List[Tuple[str, str, int]] = []
        for txt in text:
            translation, idx = _word_to_translation(txt, verse_words)
            if translation:
                matched_pairs.append((txt, translation, idx))

        translations = [t for _, t, _ in matched_pairs]
        query = " ".join(translations)
        char_start, char_end, span_text, score = find_semantic_span(
            query, session.verse_text
        )

        wt = chunk.word_timings
        chunk_duration = chunk.window.end_sec - chunk.window.start_sec
        n_words = len(text)
        word_alignments = []
        for i, ar_word in enumerate(text):
            if i < len(wt):
                word_start, word_end = wt[i][1], wt[i][2]
            else:
                word_start = chunk.window.start_sec + (i / n_words) * chunk_duration
                word_end = chunk.window.start_sec + ((i + 1) / n_words) * chunk_duration
            en_word = translations[i] if i < len(translations) else ""
            idx = matched_pairs[i][2] if i < len(matched_pairs) else -1
            word_alignments.append(
                (ar_word, en_word, round(word_start, 3), round(word_end, 3), idx)
            )

        session.results.append(
            MatchResult(
                chunk=chunk,
                arabic_text=text,
                span=TextSpan(start=char_start, end=char_end, text=span_text),
                score=score,
                word_alignments=word_alignments,
            )
        )

    _fill_gaps(session.results, session.verse_text)
    session.guard = run_guard(session.results, session.verse_text)
    return session


# ---------------------------------------------------------------------------
# Circlelog patchen
# ---------------------------------------------------------------------------


def patch_circlelog(
    mapping_path: Path,
    affected_timestamp: str,
    session: MatchSession,
) -> None:
    """
    Fügt Sub-Einträge in die Circlelog-Datei ein.
    """
    from modules.circlelog import seconds_to_timestamp

    if not session.results:
        return

    lines = mapping_path.read_text(encoding="utf-8").splitlines()

    insert_after = None
    for i, line in enumerate(lines):
        if line.startswith(f"[{affected_timestamp}]"):
            insert_after = i
            break

    if insert_after is None:
        raise ValueError(
            f"Timestamp [{affected_timestamp}] nicht in {mapping_path} gefunden."
        )

    sub_entries = []
    seen = set()
    for result in session.results:
        ts = seconds_to_timestamp(result.chunk.window.start_sec)
        text = f"{result.span.text}"
        if text in seen:
            continue
        seen.add(text)
        sub_entries.append(f"[{ts}] :: {text}")

    patched = lines[: insert_after + 1] + sub_entries + lines[insert_after + 1 :]
    mapping_path.write_text("\n".join(patched) + "\n", encoding="utf-8")
