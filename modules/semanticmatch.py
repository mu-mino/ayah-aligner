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

import regex

from modules.whispertranscribe import ChunkTranscription

# ---------------------------------------------------------------------------
# Scoring helpers
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

    verse_text: str
    results: List[MatchResult] = field(default_factory=list)
    guard: Optional[GuardReport] = None


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def extract_verse_text(mapping_line: str) -> Optional[str]:
    """
    Extrahiert den Vers-Text aus einer einzelnen circlelog-Zeile.
    Erwartet: '[MM:SS] :: text'
    """
    m = re.match(r"^\[\d{2}:\d{2}\]\s*::\s*(.+)$", mapping_line.strip())
    if not m:
        return None
    text = m.group(1).strip()
    return re.sub(r"^\d+:\s*", "", text)


def extract_verse_number(mapping_line: str) -> Optional[int]:
    """Extrahiert die erste Vers-Nummer aus einer circlelog-Zeile."""
    m = re.match(r"^\[\d{2}:\d{2}\]\s*::\s*(\d+):", mapping_line.strip())
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


def _find_span_by_sequencematcher(
    query: str, full_translation: str
) -> Tuple[int, int, str, float]:
    """
    Findet den Substring in full_translation mit dem höchsten SequenceMatcher-Score
    gegen query. Gleitet wortweise über den vollen Text (O(n²)).
    """
    words = full_translation.split()
    if not words:
        return 0, len(full_translation), full_translation, 0.0

    # Char-Offsets aufbauen
    offsets: List[Tuple[int, int]] = []
    cursor = 0
    for word in words:
        idx = full_translation.index(word, cursor)
        offsets.append((idx, idx + len(word)))
        cursor = idx + len(word)

    n = len(words)
    best_ratio = -1.0
    best_start, best_end = 0, offsets[-1][1]

    for i in range(n):
        for j in range(i + 1, n + 1):
            span_text = full_translation[offsets[i][0] : offsets[j - 1][1]]
            ratio = _rouge_l(query.lower(), span_text.lower())
            if ratio > best_ratio:
                best_ratio = ratio
                best_start = offsets[i][0]
                best_end = offsets[j - 1][1]

    return best_start, best_end, full_translation[best_start:best_end], best_ratio


# ---------------------------------------------------------------------------
# Gap Fill
# ---------------------------------------------------------------------------


def _fill_gaps(results: List[MatchResult], verse_text: str) -> None:
    """
    Füllt den Prefix-Bereich vor dem ersten Span und Lücken zwischen Spans.
    Der Suffix (nach dem letzten Span) wird NICHT angefasst — er wird vom
    circle_window-Eintrag der nächsten Gruppe als Fortsetzung übernommen.
    Modifiziert results in-place.
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
        if (
            curr.span.start > prev.span.end
            and verse_text[prev.span.end : curr.span.start].strip()
        ):
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

    Ablauf pro Chunk:
        1. quran.com liefert wortgenaues Alignment für surah:ayah
        2. Arabischen Chunk gegen Vers-Wörter matchen → Positionen [i, j]
        3. Wort-Übersetzungen [i..j] konkatenieren → Query
        4. SequenceMatcher findet besten Substring-Match in verse_text
    """
    session = MatchSession(verse_text=verse_text)

    if not chunks:
        session.guard = GuardReport(order_passed=True, completeness_passed=True)
        return session

    for chunk in chunks:
        verse_words = _fetch_verse_words(surah, ayah)  # per Chunk, gecacht
        translations = []
        for word in chunk.raw_text.split():
            translation = _word_to_translation(word, verse_words)
            if translation:
                translations.append(translation)

        query = " ".join(translations)
        char_start, char_end, span_text, score = _find_span_by_sequencematcher(
            query, verse_text
        )
        session.results.append(
            MatchResult(
                chunk=chunk,
                arabic_text=chunk.raw_text,
                span=TextSpan(start=char_start, end=char_end, text=span_text),
                score=score,
            )
        )

    _fill_gaps(session.results, verse_text)
    session.guard = run_guard(session.results, verse_text)
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

    Für jeden MatchResult aus der MatchSession wird ein Sub-Eintrag erzeugt:
        [MM:SS] :: <gematchter Span-Text>

    Der Timestamp stammt aus dem FrameWindow des zugehörigen Chunks
    (window.start_sec). Die Sub-Einträge werden direkt nach dem betroffenen
    Circlelog-Eintrag eingefügt, vor dem nächsten Eintrag.

    Parameters
    ----------
    mapping_path       : Pfad zur Circlelog-Mapping-Datei.
    affected_timestamp : Timestamp des betroffenen Eintrags, z. B. "00:10".
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
        sub_entries.append(f"[{ts}] :: {result.span.text}")

    patched = lines[: insert_after + 1] + sub_entries + lines[insert_after + 1 :]
    mapping_path.write_text("\n".join(patched) + "\n", encoding="utf-8")
