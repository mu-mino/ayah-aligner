"""
Orchestrierung der Pipeline.

Ablauf:
    1. videowindow    : Video → List[FrameWindow] (alle Fenster in Reihenfolge)
    2. recognizecircle: pro Fenster Kreise zählen → Gruppen bilden
       - n > 0 → neues Kreis-Fenster, startet eine neue Gruppe
       - n = 0 → gehört zur letzten Gruppe als Sub-Fenster
    3. Verse zuweisen  : pro Gruppe n Verse (n = Kreisanzahl)
    4. whispertranscribe + semanticmatch:
                        ALLE Window-Frames einheitlich transkribieren und gegen
                        die Vers-Texte matchen → jeder Frame erzeugt einen
                        präzisen Alignment-Span als Mapping-Eintrag.
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import dill

import cv2

ENABLE_TMP_WINDOW_FRAMES: bool = True

from modules.videowindow import extract_windows, FrameWindow, run_ffprobe
from modules.recognizecircle import detect_markers_from_gray
from modules.circlelog import (
    parse_text_file,
    build_title_line,
    seconds_to_timestamp,
    write_mapping,
    dedupe_mapping_file,
)
from modules.whispertranscribe import transcribe_chunks, load_model
from modules.semanticmatch import run_matching, _format_span_with_verse_ids

BASE_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Interne Datenstruktur
# ---------------------------------------------------------------------------


import dill


def write_vars(global_vars, local_vars=None):
    # Kombiniere Globals und Locals in ein Dictionary
    all_vars = {}
    all_vars.update(global_vars)
    if local_vars:
        all_vars.update(local_vars)

    # Filtere Systemvariablen, Module und die Funktion selbst heraus,
    # da dill sonst versucht, die offene Datei oder sich selbst zu speichern.
    filtered_vars = {
        k: v
        for k, v in all_vars.items()
        if not k.startswith("__")
        and not hasattr(v, "__package__")
        and k != "f"
        and k != "write_vars"
    }

    with open("main_for_loop.pkl", "wb") as f:
        dill.dump(filtered_vars, f)


@dataclass
class WindowGroup:
    """
    Eine Gruppe bestehend aus einem Kreis-Fenster und seinen Sub-Fenstern.

    circle_window : das Fenster, in dem Kreise erkannt wurden (n > 0)
    circle_count  : Anzahl erkannter Kreise (= Anzahl zugewiesener Verse)
    sub_windows   : aufeinanderfolgende Fenster ohne Kreis (n = 0)
    verses        : die diesem Fenster zugewiesenen (vers_nummer, vers_text)-Paare
    mapping_line  : der erzeugte Circlelog-Eintrag
    """

    circle_window: FrameWindow
    circle_count: int
    sub_windows: List[FrameWindow] = field(default_factory=list)
    verses: List[Tuple[int, str]] = field(default_factory=list)
    mapping_line: str = ""
    mapping_ts: str = ""
    end_with_last_verse: bool = False

    @property
    def all_windows(self) -> List[FrameWindow]:
        """Gibt das Hauptfenster und alle Sub-Fenster als eine gemeinsame Liste zurück."""
        return [self.circle_window] + self.sub_windows


# ---------------------------------------------------------------------------
# Hilfsfunktion
# ---------------------------------------------------------------------------


def _load_gray(video_path: Path, window: FrameWindow):
    """Lädt den mittleren Frame eines FrameWindow als Graustufenbild."""
    info = run_ffprobe(video_path)
    mid_sec = (window.start_sec + window.end_sec) / 2.0
    frame_idx = int(mid_sec * info.fps)
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _verse_at(verse_ranges, pos: int):
    """Liefert die Versnummer, in der die Position *pos* liegt (oder None)."""
    for ayah, start, end, _text in verse_ranges:
        if start <= pos < end:
            return ayah
    return None


def _verses_in_span(verse_ranges, start: int, end: int):
    """Liefert alle Versnummern, die der Span [start, end) überlappt.

    Ein Fenster, dessen Audio das Ende des einen und den Anfang des nächsten
    Verses rezitiert, gehört zu BEIDEN Versen — so geht der nächste
    Vers-Anfang nicht verloren.
    """
    verses = []
    for ayah, vs, ve, _text in verse_ranges:
        if start < ve and end > vs:
            verses.append(ayah)
    return verses


def _verse_abs_range(verse_ranges, verse: int):
    """Liefert (start, end) des Verses *verse* in den Scope-Koordinaten."""
    for ayah, vs, ve, _text in verse_ranges:
        if ayah == verse:
            return vs, ve
    return 0, 0


def _format_verse_span(verse_text: str, verse: int, start: int, end: int) -> str:
    """Formatiert einen Span [start, end) im Vers-Text mit Versnummer-Prefix."""
    text = verse_text[start:end]
    if start == 0:
        text = f"{verse}: {text}"
    return text.strip()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------



def _balanced_paren_group(text: str, start: int) -> str:
    """Liefert das komplette, verschachtelungsbewusste Klammer-Stueck
    "(...)" ab Position start in text (bis zur matchenden schliessenden
    Klammer, inklusive aller enthaltenen Klammern)."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return text[start:]


def _matching_close(content: str, start_depth: int) -> int:
    """Index der schliessenden Klammer in content, die bei start_depth
    offener Klammern die Tiefe auf 0 bringt; -1 wenn keine vorhanden."""
    depth = start_depth
    for j, ch in enumerate(content):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return j
    return -1


def _open_depth(content: str) -> int:
    """Anzahl der am Ende von content noch ungeschlossenen '(' (0 = sauber)."""
    depth = 0
    for ch in content:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
    return depth


def _assemble_mapping_lines(entries) -> list:
    """Baut Mapping-Zeilen aus sortierten '[HH:MM:SS] :: text'-Eintraegen.

    Es darf keine Zeile mit einer Klammer anfangen oder mit einer
    ungeschlossenen "(" enden: der komplette Klammerblock wird an die
    jeweilige Nachbarzeile angehaengt, damit die naechste Zeile mit
    sauberem Text startet (keine Klammer-/Kommentar-Ueberreste).
    """
    lines = []
    for _, line in entries:
        _ts, _content = line.split(" :: ", 1)
        _content = _content.lstrip()
        while _content.startswith("("):
            _group = _balanced_paren_group(_content, 0)
            if not _group.endswith(")") or not lines:
                break
            lines[-1] += _group
            _content = _content[len(_group):].lstrip()
        if _content:
            lines.append(f"{_ts} :: {_content}")

    i = 0
    while i < len(lines) - 1:
        cur = lines[i]
        open_depth = _open_depth(cur)
        if open_depth == 0:
            i += 1
            continue
        _ts, _content = lines[i + 1].split(" :: ", 1)
        # Endet die Zeile mit "(", wird nur die abschliessende Klammer
        # geschlossen (bisheriges Verhalten). Endet sie mitten in einer
        # Klammer, wird die volle offene Tiefe geschlossen.
        start_depth = 1 if cur.rstrip().endswith("(") else open_depth
        cut = _matching_close(_content, start_depth)
        if cut < 0:
            i += 1
            continue
        joiner = (
            ""
            if cur.endswith(" ") or cur.endswith("(") or _content.startswith("(")
            else " "
        )
        lines[i] = cur + joiner + _content[:cut + 1]
        remainder = _content[cut + 1:].lstrip()
        if remainder:
            lines[i + 1] = f"{_ts} :: {remainder}"
        else:
            del lines[i + 1]
            continue
        i += 1

    # Zeilen mit demselben Timestamp zu EINER Zeile zusammenführen: Ein
    # Grenz-Fenster rezitiert das Vers-Ende UND den Vers-Anfang — das muss in
    # einer Mapping-Zeile erscheinen, nicht als zwei Einträge.
    merged = []
    for ln in lines:
        ts = ln.split(" :: ", 1)[0]
        if merged and merged[-1].split(" :: ", 1)[0] == ts:
            merged[-1] += " " + ln.split(" :: ", 1)[1].lstrip()
        else:
            merged.append(ln)
    return merged


def run(
    video_path: Path,
    audio_path: Path,
    text_path: Path,
    mapping_path: Path,
    surah: int,
    whisper_device: Optional[str] = None,
) -> None:
    """
    Führt die vollständige Pipeline aus — einheitlich für alle Window-Frames.

    Jeder Window-Frame (Circle- und Sub-Fenster) wird transkribiert und gegen
    die Vers-Texte gematcht; der präzise Alignment-Span erzeugt genau einen
    Mapping-Eintrag am Timestamp des Frames.

    Parameters
    ----------
    video_path    : Pfad zur Videodatei (für Frame-Analyse).
    audio_path    : Pfad zur Audiodatei (für Transkription).
    text_path     : Pfad zur Textdatei mit Versen.
    mapping_path  : Zieldatei für das Circlelog-Mapping.
    surah         : Sure-Nummer für das semantische Matching.
    whisper_device: 'cuda' oder 'cpu'. Bei None: automatisch.
    """
    # ------------------------------------------------------------------
    # 1. Video segmentieren
    # ------------------------------------------------------------------
    windows = extract_windows(video_path)
    title_window = windows[0] if windows and windows[0].end_sec <= 10.0 else None

    # ------------------------------------------------------------------
    # 2. Gruppen bilden — beim Durchlauf, nicht nachträglich
    #    circle_window (n>0) startet Gruppe; n=0 gehört zur letzten Gruppe
    # ------------------------------------------------------------------
    groups: List[WindowGroup] = []
    current_group: Optional[WindowGroup] = None

    if ENABLE_TMP_WINDOW_FRAMES:
        frames_dir = Path(__file__).parent / "tests" / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

    iter_windows = windows if title_window is None else windows[1:]
    for i, window in enumerate(iter_windows):
        gray = _load_gray(video_path, window)
        if gray is None:
            continue
        if ENABLE_TMP_WINDOW_FRAMES:
            cv2.imwrite(
                str(
                    frames_dir
                    / f"window_{i:04d}_{window.start_sec:.3f}_{window.end_sec:.3f}.png"
                ),
                gray,
            )
        n, end_with_last_verse = detect_markers_from_gray(gray)
        if n > 0:
            current_group = WindowGroup(
                circle_window=window,
                circle_count=n,
                end_with_last_verse=end_with_last_verse,
            )
            groups.append(current_group)
        elif current_group is not None:
            current_group.sub_windows.append(window)
        elif i == 0 and n == 0:
            n += 1
            current_group = WindowGroup(circle_window=window, circle_count=n)
            groups.append(current_group)

    # ------------------------------------------------------------------
    # 3. Verse zuweisen (n Kreise → n Verse pro Gruppe)
    # ------------------------------------------------------------------
    title_lines, numbered_lines = parse_text_file(text_path)
    mapping_lines: List[str] = []

    if title_lines and title_window:
        title_ts = seconds_to_timestamp(title_window.start_sec)
        mapping_lines.append(build_title_line(title_lines, title_ts))

    all_verses: Dict[int, str] = {
        i + 1: text for i, text in enumerate(numbered_lines)
    }

    verse_number = 1
    for group in groups:
        n = group.circle_count
        taken = min(n, len(numbered_lines))
        for i in range(taken):
            group.verses.append((verse_number + i, numbered_lines[i]))
        numbered_lines = numbered_lines[taken:]
        verse_number += taken

    # ------------------------------------------------------------------
    # 4. ALLE Window-Frames einheitlich transkribieren + matchen.
    #    Die Circle-Detektion (Gruppen) dient als Guard: Der Match-Bereich ist
    #    nur das lokale Vers-Fenster (kein ganzer Sure-Text → keine falschen
    #    Positives).
    #
    #    Hybrid:
    #      - Multi-Vers-Gruppen (n>1): gegen die zugewiesenen Verse matchen
    #        mit fill_gaps=True (die Fenster einer Gruppe rezitieren gemeinsam
    #        die n Verse — vollständige Abdeckung).
    #      - Einzel-Vers-Gruppen (n=1): erst rohe Spans gegen [N-1, N+1]
    #        (Kreise erscheinen oft einen Vers zu früh → Off-by-One-Korrektur),
    #        dann die korrekten Pass-1-Spans auf die Vers-Koordinaten
    #        projizieren und Lücken füllen.
    #
    #    Multi-Vers-Verse werden im n=1-Pfad NICHT erneut beansprucht: Ein
    #    Nachbarfenster einer Einzel-Vers-Gruppe, dessen Audio in einen
    #    Multi-Vers-Vers kreuzt, würde sonst denselben Vers ein zweites Mal
    #    füllen (Duplikat, z.B. V16).
    # ------------------------------------------------------------------
    if not any(g.verses for g in groups):
        write_mapping(mapping_lines, mapping_path)
        return
    whisper_model = load_model(device=whisper_device)

    multi_verse_verses = {
        verse for group in groups if len(group.verses) > 1 for verse, _ in group.verses
    }

    entries: List[Tuple[float, str]] = []
    verse_spans: Dict[int, list] = {}

    for group in groups:
        if not group.verses:
            continue
        all_windows = group.all_windows  # Circle + Sub-Fenster einheitlich
        chunks = transcribe_chunks(
            video_path=audio_path,
            windows=all_windows,
            model=whisper_model,
        )
        chunks = [c for c in chunks if c.raw_text.strip()]

        if len(group.verses) > 1:
            id_verse = dict(group.verses)
            session = run_matching(
                chunks=chunks,
                surah=surah,
                dict_of_verses=id_verse,
                fill_gaps=True,
            )
            for result in session.results:
                text = _format_span_with_verse_ids(
                    session, result.span.start, result.span.end
                )
                if text.strip():
                    ts = seconds_to_timestamp(result.chunk.window.start_sec)
                    entries.append(
                        (result.chunk.window.start_sec, f"[{ts}] :: {text}")
                    )
            continue

        # Einzel-Vers-Gruppe: rohe Spans gegen [N-1, N+1]. Die korrekten
        # Pass-1-Positionen werden auf die Vers-Koordinaten projiziert (KEIN
        # Re-Match gegen den Einzelvers — das verursachte bei Grenz-Fenstern
        # falsche Spans, z.B. Vers-17-Schwanz statt -Anfang).
        first_v = group.verses[0][0]
        lo = max(1, first_v - 1)
        hi = min(len(all_verses), first_v + 1)
        scope = {v: all_verses[v] for v in range(lo, hi + 1) if v in all_verses}
        s = run_matching(chunks=chunks, surah=surah, dict_of_verses=scope, fill_gaps=False)
        for result in s.results:
            for verse in _verses_in_span(
                s.verse_ranges, result.span.start, result.span.end
            ):
                if verse in multi_verse_verses:
                    continue
                vs, ve = _verse_abs_range(s.verse_ranges, verse)
                istart = max(result.span.start, vs) - vs
                iend = min(result.span.end, ve) - vs
                if iend > istart:
                    verse_spans.setdefault(verse, []).append(
                        (result.chunk.window, istart, iend)
                    )

    # Vers-Abdeckung (n=1): Lücken füllen UND Überlappungen beschneiden
    # (vorn/mitte/hinten — Whisper ist nie perfekt), ein Eintrag pro Fenster.
    for verse, spans in verse_spans.items():
        spans.sort(key=lambda x: x[1])
        if spans and spans[0][1] > 0:
            w, _st, en = spans[0]
            spans[0] = (w, 0, en)
        for i in range(1, len(spans)):
            prev_w, prev_st, _prev_en = spans[i - 1]
            _cur_w, cur_st, _cur_en = spans[i]
            spans[i - 1] = (prev_w, prev_st, cur_st)
        if spans and spans[-1][2] < len(all_verses[verse]):
            w, st, _en = spans[-1]
            spans[-1] = (w, st, len(all_verses[verse]))
        for window, istart, iend in spans:
            text = _format_verse_span(all_verses[verse], verse, istart, iend)
            if text.strip():
                ts = seconds_to_timestamp(window.start_sec)
                entries.append((window.start_sec, f"[{ts}] :: {text}"))

    entries.sort(key=lambda item: item[0])
    mapping_lines.extend(_assemble_mapping_lines(entries))
    write_mapping(mapping_lines, mapping_path)
    dedupe_mapping_file(mapping_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    BASE_DIR = Path(__file__).parent
    parser = argparse.ArgumentParser(description="Pipeline-Orchestrierung")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--text", required=True, type=Path)
    parser.add_argument(
        "--output",
        default=str(BASE_DIR / "output" / "mapping"),
        type=Path,
        help="Ausgabeverzeichnis (Dateiname wird aus --text abgeleitet)",
    )
    parser.add_argument("--device", default=None, type=str)
    parser.add_argument(
        "--surah", required=True, type=int, help="Surah-Nummer (z.B. 98)"
    )
    args = parser.parse_args()

    mapping_path = args.output / (args.text.stem + ".mapping")

    run(
        video_path=args.video,
        audio_path=args.audio,
        text_path=args.text,
        mapping_path=mapping_path,
        surah=args.surah,
        whisper_device=args.device,
    )


if __name__ == "__main__":
    main()
