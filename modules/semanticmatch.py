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

from pathlib import Path

_model_path = Path(
    "/home/muhammed-emin-eser/desk/din/ayah-aligner/symanto-model"
).resolve()

model = SentenceTransformer(
    _model_path.as_posix(), local_files_only=True
)
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

_AR_DIACRITICS = regex.compile(r"[\p{M}\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]+")
_AR_TATWEEL = "\u0640"
_AR_NON_ARABIC = regex.compile(r"[^\p{Arabic} ]+")
_AR_MULTI_SPACE = regex.compile(r"\s+")
_AR_PREFIX = regex.compile(r"(?<!\S)(و|ف|ب|ك|ل|س)\s+(?=\S)", regex.UNICODE)
_AR_CHAR_MAP = str.maketrans(
    {
        "آ": "ا",
        "ٱ": "ا",
        "ى": "ي",
        "ئ": "ي",
        "ؤ": "و",
        "ة": "ه",
        "ء": "",
        "گ": "ك",
        "ڤ": "ف",
        "پ": "ب",
        "چ": "ج",
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
WORD_MATCH_TOLERANCE: float = 0.5

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

    mapping_text: str
    verse_ranges: List[Tuple[int, int, int, str]] = field(default_factory=list)
    results: List[MatchResult] = field(default_factory=list)
    guard: Optional[GuardReport] = None


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def mapping_to_per_verse(mapping_line: str) -> Optional[dict]:
    mapping_stamp = re.match(
        r"^\[\d{2}:\d{2}:\d{2}\]\s*::\s*(.*)$", mapping_line.strip()
    )
    if not mapping_stamp:
        return {}

    content = mapping_stamp.group(1)
    verse_markers = list(re.finditer(r"(?:^|\s)(\d+):\s*", content))
    res = {}
    for index, marker in enumerate(verse_markers):
        next_marker = (
            verse_markers[index + 1] if index + 1 < len(verse_markers) else None
        )
        start = marker.end()
        end = next_marker.start() if next_marker else len(content)
        res[int(marker.group(1))] = content[start:end].strip()
    return res


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


def _word_to_translation(arabic_word: str, verse_words: list) -> str:
    """
    Findet das beste Match für ein arabisches Wort in der Vers-Wortliste und
    gibt dessen englische Übersetzung zurück.
    Wirft ValueError wenn kein Match >= WORD_MATCH_TOLERANCE gefunden wird.
    """
    norm_word = _normalize_arabic(arabic_word)
    best_score = 0.0
    best_match = None

    for w in verse_words:
        norm_verse = _normalize_arabic(w["text_uthmani"])
        score = SequenceMatcher(None, norm_word, norm_verse).ratio()
        if score > best_score:
            best_score = score
            best_match = w

    if best_score < WORD_MATCH_TOLERANCE or best_match is None:
        raise ValueError(
            f"Kein Match für '{arabic_word}' (bestes: {best_score:.2f}, Schwelle: {WORD_MATCH_TOLERANCE})"
        )

    t = best_match.get("translation", {})
    text = t.get("text", "") if isinstance(t, dict) else ""
    if re.match(r"^\(\d+\)$", text.strip()):
        return ""
    return text.strip()


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

    last = by_start[-1]
    if last.span.end < len(verse_text) and verse_text[last.span.end :].strip():
        last.span.end = len(verse_text)
        last.span.text = verse_text[last.span.start : last.span.end]


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
    dict_of_verses: dict,
    surah: int,
) -> MatchSession:
    """
    Matcht jeden arabischen Chunk gegen den englischen Vers-Text via quran.com API.

    Ablauf pro Chunk:
        1. quran.com liefert wortgenaues Alignment für surah:ayah
        2. Arabischen Chunk gegen Vers-Wörter matchen → Positionen [i, j]
        3. Wort-Übersetzungen [i..j] konkatenieren → Query
        4. SequenceMatcher findet besten Substring-Match in verse_text
    """
    mapping_parts = []
    verse_ranges: List[Tuple[int, int, int, str]] = []
    cursor = 0
    for ayah, verse_text in dict_of_verses.items():
        if mapping_parts:
            cursor += 1
        start = cursor
        mapping_parts.append(verse_text)
        end = start + len(verse_text)
        verse_ranges.append((ayah, start, end, verse_text))
        cursor = end

    session = MatchSession(
        mapping_text=" ".join(mapping_parts),
        verse_ranges=verse_ranges,
    )

    if not chunks:
        session.guard = GuardReport(completeness_passed=True)
        return session

    verse_words = []
    for ayah in dict_of_verses:
        verse_words.extend(_fetch_verse_words(surah, ayah))

    for chunk in chunks:
        translations = []
        text = chunk.raw_text.split()
        for txt in text:
            translation = _word_to_translation(txt, verse_words)
            if translation:
                translations.append(translation)

        query = " ".join(translations)
        char_start, char_end, span_text, score = find_semantic_span(
            query, session.mapping_text
        )
        session.results.append(
            MatchResult(
                chunk=chunk,
                arabic_text=text,
                span=TextSpan(start=char_start, end=char_end, text=span_text),
                score=score,
            )
        )

    _fill_gaps(session.results, session.mapping_text)
    session.guard = run_guard(session.results, session.mapping_text)
    return session


# ---------------------------------------------------------------------------
# Circlelog patchen
# ---------------------------------------------------------------------------


def _format_span_with_verse_ids(session: MatchSession, start: int, end: int) -> str:
    parts = []
    for ayah, verse_start, verse_end, verse_text in session.verse_ranges:
        overlap_start = max(start, verse_start)
        overlap_end = min(end, verse_end)
        if overlap_start >= overlap_end:
            continue

        text = verse_text[overlap_start - verse_start : overlap_end - verse_start]
        if overlap_start == verse_start:
            text = f"{ayah}: {text}"
        parts.append(text.strip())
    return " ".join(part for part in parts if part)


def patch_circlelog(
    mapping_path: Path,
    affected_timestamp: str,
    session: MatchSession,
) -> None:
    """
    Fügt Sub-Einträge in die Circlelog-Datei ein.

    Für jeden MatchResult aus der MatchSession wird ein Sub-Eintrag erzeugt:
        [HH:MM:SS] :: <gematchter Span-Text>

    Der Timestamp stammt aus dem FrameWindow des zugehörigen Chunks
    (window.start_sec). Die Sub-Einträge werden direkt nach dem betroffenen
    Circlelog-Eintrag eingefügt, vor dem nächsten Eintrag.

    Parameters
    ----------
    mapping_path       : Pfad zur Circlelog-Mapping-Datei.
    affected_timestamp : Timestamp des betroffenen Eintrags, z. B. "00:00:10".
    session            : MatchSession mit den Ergebnissen aus run_matching().
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
    for result in session.results:
        ts = seconds_to_timestamp(result.chunk.window.start_sec)
        text = _format_span_with_verse_ids(session, result.span.start, result.span.end)
        sub_entries.append(f"[{ts}] :: {text}")

    patched = lines[: insert_after + 1] + sub_entries + lines[insert_after + 1 :]
    mapping_path.write_text("\n".join(patched) + "\n", encoding="utf-8")
