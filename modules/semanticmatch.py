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
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Optional, Tuple

from modules.whispertranscribe import ChunkTranscription

# ---------------------------------------------------------------------------
# Konstanten
# ---------------------------------------------------------------------------

QURAN_API_BASE: str = "https://api.quran.com/api/v4"
QURAN_TRANSLATION_ID: int = 203  # Al-Hilali & Khan

MAX_CORRECTION_ATTEMPTS: int = 3  # Max. Korrekturrunden pro Session


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
    correction_requested: bool = False


@dataclass
class GuardReport:
    """Ergebnis der Guard-Prüfung über alle Spans."""

    order_passed: bool
    completeness_passed: bool
    order_violations: List[int] = field(default_factory=list)  # result-Indizes
    uncovered_ranges: List[Tuple[int, int]] = field(
        default_factory=list
    )  # Lücken als (start, end)
    correction_hints: List[Tuple[int, str]] = field(
        default_factory=list
    )  # (chunk_idx, hint)

    @property
    def passed(self) -> bool:
        return self.order_passed and self.completeness_passed


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
    """Holt wortgenaues Alignment für einen Vers von quran.com."""
    import requests

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
    return [w for w in verse["words"] if w.get("text_uthmani") and w["char_type_name"] != "end"]


def _normalize_arabic(text: str) -> str:
    """Entfernt Diakritika und normalisiert Alef-Varianten."""
    text = re.sub(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]", "", text)
    text = re.sub(r"[أإآٱ]", "ا", text)
    text = text.replace("\u0640", "")
    text = re.sub(r"[،؛؟!\u06D4\u06D5]", "", text)
    return text.strip()


def _match_arabic_to_verse_words(arabic_chunk: str, verse_words: list) -> Tuple[int, int, float]:
    """
    Findet den zusammenhängenden Block von Vers-Wörtern der am besten zum
    arabischen Chunk passt. Gibt (start_idx, end_idx, score) zurück (end exklusiv).
    """
    chunk_words = [_normalize_arabic(w) for w in arabic_chunk.split() if _normalize_arabic(w)]
    verse_norm = [_normalize_arabic(w["text_uthmani"]) for w in verse_words]

    n = len(verse_words)
    best_score = -1.0
    best_start, best_end = 0, n

    for start in range(n):
        for end in range(start + 1, n + 1):
            window = verse_norm[start:end]
            hits = sum(1 for w in chunk_words if w in window)
            ratio = SequenceMatcher(None, chunk_words, window).ratio()
            score = 0.5 * (hits / max(len(chunk_words), 1)) + 0.5 * ratio
            if score > best_score:
                best_score = score
                best_start, best_end = start, end

    return best_start, best_end, best_score


def _concat_word_translations(matched_words: list) -> str:
    """Konkateniert die englischen Wort-Übersetzungen der gematchten Vers-Wörter."""
    parts = []
    for w in matched_words:
        t = w.get("translation", {})
        text = t.get("text", "") if isinstance(t, dict) else ""
        if text and not re.match(r"^\(\d+\)$", text.strip()):
            parts.append(text.strip())
    return " ".join(parts)


def _find_span_by_sequencematcher(query: str, full_translation: str) -> Tuple[int, int, str, float]:
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
            span_text = full_translation[offsets[i][0]:offsets[j - 1][1]]
            ratio = SequenceMatcher(None, query.lower(), span_text.lower()).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_start = offsets[i][0]
                best_end = offsets[j - 1][1]

    return best_start, best_end, full_translation[best_start:best_end], best_ratio


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
        return GuardReport(order_passed=True, completeness_passed=True)

    order_violations: List[int] = []
    first_start = results[0].span.start
    prev_end = results[0].span.end

    for i in range(1, len(results)):
        span = results[i].span
        end_ok = span.end >= prev_end
        start_ok = span.start >= first_start
        if not end_ok or not start_ok:
            order_violations.append(i)
        prev_end = max(prev_end, span.end)

    # Vollständigkeit: Lücken im Vers-Text ermitteln (Whitespace toleriert)
    sorted_spans = sorted(
        [(r.span.start, r.span.end) for r in results], key=lambda x: x[0]
    )
    uncovered: List[Tuple[int, int]] = []

    if sorted_spans[0][0] > 0:
        prefix = verse_text[: sorted_spans[0][0]]
        if prefix.strip():
            uncovered.append((0, sorted_spans[0][0]))

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

    for i in order_violations:
        prev = results[i - 1]
        curr = results[i]
        correction_hints.append(
            (
                i,
                f'Chunk [{i}] goes backward: chunk [{i - 1}] ends at "{prev.span.text}" '
                f"(pos {prev.span.end}), but chunk [{i}] was placed at pos {curr.span.start}. "
                f"Recitation always moves forward — a later audio chunk cannot match earlier text.",
            )
        )

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

    return GuardReport(
        order_passed=len(order_violations) == 0,
        completeness_passed=len(uncovered) == 0,
        order_violations=order_violations,
        uncovered_ranges=uncovered,
        correction_hints=correction_hints,
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

    verse_words = _fetch_verse_words(surah, ayah)

    for chunk in chunks:
        arabic_text = chunk.raw_text
        start_idx, end_idx, _ = _match_arabic_to_verse_words(arabic_text, verse_words)
        matched_words = verse_words[start_idx:end_idx]
        query = _concat_word_translations(matched_words)
        char_start, char_end, span_text, score = _find_span_by_sequencematcher(query, verse_text)
        session.results.append(
            MatchResult(
                chunk=chunk,
                arabic_text=arabic_text,
                span=TextSpan(start=char_start, end=char_end, text=span_text),
                score=score,
            )
        )

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
