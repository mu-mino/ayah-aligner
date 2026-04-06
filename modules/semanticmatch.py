"""
Semantisches Matching: übersetzter Chunk → Span aus dem Vers-Text.

Ablauf pro ChunkTranscription:
    1. Übersetzer : arabischer raw_text → englische Übersetzung
    2. Matcher    : findet den Textspan im Vers, der der Übersetzung
                    am ähnlichsten ist (sliding window über Wörter)

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
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from modules.whispertranscribe import ChunkTranscription

# ---------------------------------------------------------------------------
# Modell-Konstanten (austauschbar)
# ---------------------------------------------------------------------------

DEFAULT_TRANSLATION_MODEL: str = "Helsinki-NLP/opus-mt-ar-en"
DEFAULT_EMBEDDING_MODEL: str = (
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

WINDOW_SIZE: int = 10  # Wörter pro Sliding-Window-Kandidat
WINDOW_STEP: int = 3  # Schrittweite des Sliding-Window


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
    translation: str
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
    return m.group(1).strip()


def _sliding_windows(verse_text: str, window_size: int, step: int) -> List[TextSpan]:
    """
    Erzeugt überlappende Wort-Fenster aus dem Vers-Text als TextSpan-Kandidaten.
    """
    words = verse_text.split()
    spans: List[TextSpan] = []
    i = 0
    while i < len(words):
        chunk_words = words[i : i + window_size]
        chunk_text = " ".join(chunk_words)
        start_char = verse_text.index(chunk_text, spans[-1].start if spans else 0)
        end_char = start_char + len(chunk_text)
        spans.append(TextSpan(start=start_char, end=end_char, text=chunk_text))
        if i + window_size >= len(words):
            break
        i += step
    return spans


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

    return GuardReport(
        order_passed=len(order_violations) == 0,
        completeness_passed=len(uncovered) == 0,
        order_violations=order_violations,
        uncovered_ranges=uncovered,
    )


# ---------------------------------------------------------------------------
# Übersetzer
# ---------------------------------------------------------------------------


def build_translator(
    model_name: str = DEFAULT_TRANSLATION_MODEL,
) -> Callable[[str], str]:
    """Lädt einen Übersetzungs-Pipeline (Arabisch → Englisch)."""
    from transformers import pipeline as hf_pipeline

    translator = hf_pipeline("translation", model=model_name)

    def translate(arabic_text: str) -> str:
        if not arabic_text.strip():
            return ""
        result = translator(arabic_text, max_length=512)
        return result[0]["translation_text"]

    return translate


# ---------------------------------------------------------------------------
# Matcher
# ---------------------------------------------------------------------------


def build_matcher(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    window_size: int = WINDOW_SIZE,
    window_step: int = WINDOW_STEP,
) -> Callable[[str, str], Tuple[TextSpan, float]]:
    """
    Lädt ein Sentence-Transformer-Modell.

    Der Matcher erzeugt Sliding-Window-Kandidaten aus dem Vers-Text
    und gibt den Span mit der höchsten Ähnlichkeit zur Übersetzung zurück.
    """
    from sentence_transformers import SentenceTransformer, util

    model = SentenceTransformer(model_name)

    def match(translation: str, verse_text: str) -> Tuple[TextSpan, float]:
        candidates = _sliding_windows(verse_text, window_size, window_step)
        if not candidates:
            fallback = TextSpan(start=0, end=len(verse_text), text=verse_text)
            return fallback, 0.0

        query_emb = model.encode(translation, convert_to_tensor=True)
        cand_embs = model.encode([c.text for c in candidates], convert_to_tensor=True)
        scores = util.cos_sim(query_emb, cand_embs)[0]

        best_idx = int(scores.argmax())
        return candidates[best_idx], float(scores[best_idx])

    return match


# ---------------------------------------------------------------------------
# Öffentliche API
# ---------------------------------------------------------------------------


def run_matching(
    chunks: List[ChunkTranscription],
    verse_text: str,
    translator: Optional[Callable[[str], str]] = None,
    matcher: Optional[Callable[[str, str], Tuple[TextSpan, float]]] = None,
) -> MatchSession:
    """
    Übersetzt jeden Chunk und wählt den passenden Span aus dem Vers-Text.

    Nach allen Chunks wird der Guard ausgeführt. Bei Guard-Verletzungen
    wird correction_requested in den betroffenen MatchResult-Einträgen gesetzt.

    Parameters
    ----------
    chunks     : n=0-Fenster aus whispertranscribe, die zu diesem Vers gehören.
    verse_text : Vollständiger Text des betroffenen circlelog-Eintrags.
    translator : Übersetzungsfunktion ar→en. Bei None: build_translator().
    matcher    : Span-Auswahl-Funktion. Bei None: build_matcher().
    """
    session = MatchSession(verse_text=verse_text)

    if not chunks:
        session.guard = GuardReport(order_passed=True, completeness_passed=True)
        return session

    if translator is None:
        translator = build_translator()
    if matcher is None:
        matcher = build_matcher()

    for chunk in chunks:
        translation = translator(chunk.raw_text)
        span, score = matcher(translation, verse_text)
        session.results.append(
            MatchResult(
                chunk=chunk,
                translation=translation,
                span=span,
                score=score,
            )
        )

    guard = run_guard(session.results, verse_text)
    session.guard = guard

    if not guard.passed:
        for i in guard.order_violations:
            session.results[i].correction_requested = True

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
