"""
Orchestrierung der Pipeline.

Ablauf:
    1. videowindow    : Video → List[FrameWindow] (alle Fenster in Reihenfolge)
    2. recognizecircle: pro Fenster Kreise zählen → Gruppen bilden
       - n > 0 → neues Kreis-Fenster, startet eine neue Gruppe
       - n = 0 → gehört zur letzten Gruppe als Sub-Fenster
    3. circlelog      : pro Gruppe einen Mapping-Eintrag schreiben
                        (n Verse auf einmal bei n Kreisen)
    4. whispertranscribe + semanticmatch:
                        für Gruppen mit Sub-Fenstern (n=0):
                        Audio transkribieren, matchen, Mapping patchen
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
import dill

import cv2

ENABLE_TMP_WINDOW_FRAMES: bool = True

from modules.videowindow import extract_windows, FrameWindow, run_ffprobe
from modules.recognizecircle import detect_markers_from_gray
from modules.circlelog import (
    parse_text_file,
    build_title_line,
    build_verse_line,
    seconds_to_timestamp,
    write_mapping,
    dedupe_mapping_file,
)
from modules.whispertranscribe import transcribe_chunks, load_model
from modules.semanticmatch import (
    run_matching,
    patch_circlelog,
    mapping_to_per_verse,
)

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


# ---------------------------------------------------------------------------
# Subwindow-Überlappung: Circle-Frame kürzen, erstes Subwindow zurücksetzen
# ---------------------------------------------------------------------------

# Satzzeichen, an denen der Circle-Text abgeschnitten wird (inkl. Satzpunkt).
_CUT_PUNCTUATION: set = set(":;!?,")
# Wörter, deren Punkt KEIN Satzende markiert (Abkürzungen aus chunked_translation).
_ABBREV_SINGLE: set = {"etc", "vol", "no", "p"}
_ABBREV_PREFIXES: tuple = ("i.e", "e.g")


def _is_abbreviation_token(token: str) -> bool:
    """
    True, wenn der Punkt dieses Tokens Teil einer Abkürzung ist (z.B. 'i.e.', 'etc.').

    Führende Nicht-Buchstaben (z.B. '(' vor '(i.e.') werden ignoriert.
    """
    word = token.strip()
    while word and not word[0].isalpha():
        word = word[1:]
    t = word.rstrip(".").lower()
    if t in _ABBREV_SINGLE:
        return True
    return t.startswith(_ABBREV_PREFIXES)


def _is_cut_punctuation(text: str, pos: int) -> bool:
    """
    True, wenn das Zeichen an *pos* ein Schnitt-Satzzeichen ist.

    Der Punkt (.) zählt nur, wenn er einen Satz beendet und nicht Teil einer
    Abkürzung ist (i.e., e.g., etc., Vol., No., P.).
    """
    ch = text[pos]
    if ch in _CUT_PUNCTUATION:
        return True
    if ch == ".":
        start = pos
        while start > 0 and not text[start - 1].isspace():
            start -= 1
        end = pos + 1
        while end < len(text) and not text[end].isspace():
            end += 1
        return not _is_abbreviation_token(text[start:end])
    return False


def _find_cut_positions(mapping_text: str, span_start: int):
    """
    Findet um *span_start* herum das nächste (vorwärts) und vorherige
    (rückwärts) Schnitt-Satzzeichen im Mapping-Text.

    Rückgabe: (next_punct_pos, prev_punct_pos) oder None, wenn eines fehlt.
    """
    next_punct = None
    for pos in range(span_start + 1, len(mapping_text)):
        if _is_cut_punctuation(mapping_text, pos):
            next_punct = pos
            break
    prev_punct = None
    for pos in range(span_start - 1, -1, -1):
        if _is_cut_punctuation(mapping_text, pos):
            prev_punct = pos
            break
    if next_punct is None and prev_punct is None:
        return None
    return next_punct, prev_punct


def _map_mapping_to_content_pos(verse_ranges, mapping_pos: int):
    """
    Bildet eine Position im Mapping-Text (verbundene Vers-Texte) auf die
    Position im Circle-Zeilen-Inhalt ('5: text 6: text') ab.

    Gibt None zurück, wenn die Position nicht abbildbar ist.
    """
    offset = 0
    for ayah, start, end, _text in verse_ranges:
        if start <= mapping_pos < end:
            return offset + len(f"{ayah}: ") + (mapping_pos - start)
        offset += len(f"{ayah}: {_text}") + 1
    last_end = verse_ranges[-1][2] if verse_ranges else None
    if last_end is not None and mapping_pos == last_end:
        return offset
    return None


def _replace_mapping_line(mapping_path: Path, timestamp: str, new_content: str) -> None:
    """Ersetzt den Inhalt der Mapping-Zeile mit dem angegebenen Timestamp."""
    lines = mapping_path.read_text(encoding="utf-8").splitlines()
    replaced = False
    out = []
    for line in lines:
        if line.startswith(f"[{timestamp}]"):
            out.append(f"[{timestamp}] :: {new_content}")
            replaced = True
        else:
            out.append(line)
    if not replaced:
        raise ValueError(
            f"Timestamp [{timestamp}] nicht in {mapping_path} gefunden."
        )
    mapping_path.write_text("\n".join(out) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run(
    video_path: Path,
    audio_path: Path,
    text_path: Path,
    mapping_path: Path,
    surah: int,
    whisper_device: Optional[str] = None,
) -> None:
    """
    Führt die vollständige Pipeline aus.

    Parameters
    ----------
    video_path    : Pfad zur Videodatei (für Frame-Analyse).
    audio_path    : Pfad zur Audiodatei (für Transkription).
    text_path     : Pfad zur Textdatei mit Versen.
    mapping_path  : Zieldatei für das Circlelog-Mapping.
    whisper_device: 'cuda' oder 'cpu'. Bei None: automatisch.
    """
    # ------------------------------------------------------------------
    # 1. Video segmentieren
    # ------------------------------------------------------------------
    windows = extract_windows(video_path)

    # ------------------------------------------------------------------
    # 2. Gruppen bilden — beim Durchlauf, nicht nachträglich
    #    circle_window (n>0) startet Gruppe; n=0 gehört zur letzten Gruppe
    # ------------------------------------------------------------------
    title_window = windows[0] if windows and windows[0].end_sec <= 10.0 else None

    groups: List[WindowGroup] = []
    current_group: Optional[WindowGroup] = None
    first_window_has_circle: bool = False

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
        if i == 0 and n > 0:
            first_window_has_circle = True
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
    # 3. Circlelog aufbauen
    #    n Kreise → n Verse auf einmal in einem Eintrag
    #
    #    Fall 1 — erstes Fenster hat Kreis:
    #        Jede Gruppe bekommt ihren eigenen Kreis-Timestamp (kein Shift).
    #    Fall 2 — kein Kreis im ersten Fenster, erste Sichtung count=1:
    #        Vers 1 bei [00:00:10], danach normaler Shift-Flow.
    #    Fall 3/4 — kein Kreis im ersten Fenster, erste Sichtung count>1:
    #        Vers 1 bei [00:00:10], Verse 2..N bei T0 (separate Zeile),
    #        danach normaler Shift-Flow.
    # ------------------------------------------------------------------
    title_lines, numbered_lines = parse_text_file(text_path)
    mapping_lines: List[str] = []

    if title_lines and title_window:
        title_ts = seconds_to_timestamp(title_window.start_sec)
        mapping_lines.append(build_title_line(title_lines, title_ts))

    # Nach einem Fall-3/4-Split haben alle Folge-Gruppen keinen Shift-Vorgänger mehr —
    # sie bekommen ihren eigenen circle_window-Timestamp (wie Fall 1).
    unshifted_mode: bool = first_window_has_circle

    verse_number = 1
    for idx, group in enumerate(groups):
        n = group.circle_count
        taken = min(n, len(numbered_lines))
        for i in range(taken):
            group.verses.append((verse_number + i, numbered_lines[i]))
        numbered_lines = numbered_lines[taken:]
        verse_number += taken

        # end_with_last_verse (Einzel-Kreis, nicht letzte Gruppe): kein neuer
        # Vers im aktuellen Circle-Window-Frame — der Vers wird stattdessen am
        # nächsten Window-Frame gerendert (erster Sub-Fenster bzw. nächste
        # Gruppe). Die letzte Gruppe wird ausgenommen (dort existiert kein
        # sinnvoller Folge-Frame; Vers bleibt am Circle-Frame).
        if (
            group.end_with_last_verse
            and group.circle_count == 1
            and idx + 1 < len(groups)
        ):
            if group.sub_windows:
                frame_start_sec = group.sub_windows[0].start_sec
            elif idx + 1 < len(groups):
                frame_start_sec = groups[idx + 1].circle_window.start_sec
            else:
                frame_start_sec = group.circle_window.start_sec
        else:
            frame_start_sec = group.circle_window.start_sec

        if unshifted_mode:
            # Fall 1 oder nach Fall-3/4-Split: eigener Timestamp, kein Shift
            ts = seconds_to_timestamp(frame_start_sec)
            group.mapping_ts = ts
            line = build_verse_line(ts, group.verses)
            group.mapping_line = line
            mapping_lines.append(line)
        elif idx == 0 and group.circle_count > 1:
            # Fall 3/4: alle Verse bei T0 (gleiches Bild → gleicher Timestamp)
            t0 = seconds_to_timestamp(group.circle_window.start_sec)
            line = build_verse_line(t0, group.verses)
            group.mapping_ts = t0
            group.mapping_line = line
            mapping_lines.append(line)
            unshifted_mode = True
        elif idx == len(groups) - 1 and not group.verses:
            continue
        else:
            # Fall 2 + normaler Shift-Flow
            ts = seconds_to_timestamp(frame_start_sec)
            group.mapping_ts = ts
            line = build_verse_line(ts, group.verses)
            group.mapping_line = line
            mapping_lines.append(line)
    write_mapping(mapping_lines, mapping_path)

    # ------------------------------------------------------------------
    # 4. Sub-Fenster transkribieren + matchen + Mapping patchen
    # ------------------------------------------------------------------
    groups_with_subs = [g for g in groups if g.sub_windows]
    dedupe_mapping_file(mapping_path)
    if not groups_with_subs:
        return
    whisper_model = load_model(device=whisper_device)

    for idx, group in enumerate(groups):
        id_verse: dict[int, str] = mapping_to_per_verse(group.mapping_line)
        if not id_verse:
            continue

        all_windows = group.all_windows if group.sub_windows else [group.circle_window]
        if idx + 1 == len(groups) - 1 and not groups[idx + 1].verses:
            all_windows = all_windows + [groups[idx + 1].circle_window]

        # end_with_last_verse-Verschiebung: Der Vers wurde auf den nächsten
        # Window-Frame verschoben — der Circle-Window selbst wird nicht
        # transkribiert, sonst entstünde dort ein doppelter/fehlgeleiteter
        # Sub-Eintrag (die alten Einträge werden durch die neuen überschrieben).
        if (
            group.end_with_last_verse
            and group.circle_count == 1
            and idx + 1 < len(groups)
        ):
            all_windows = list(group.sub_windows)

        chunks = transcribe_chunks(
            video_path=audio_path,
            windows=all_windows,
            model=whisper_model,
        )

        session = run_matching(
            chunks=chunks,
            surah=surah,
            dict_of_verses=id_verse,
        )

        # Ausnahme: das erste Subwindow wiederholt Circle-Text. Dann wird der
        # Circle-Frame bis zum nächsten Satzzeichen gekürzt (statt dass der
        # Text doppelt vorkommt) und das erste Subwindow springt zum
        # vorherigen Satzzeichen zurück.
        if group.sub_windows and session.results:
            first_sub = next(
                (r for r in session.results if any(r.chunk.window is w for w in group.sub_windows)),
                None,
            )
            if (
                first_sub is not None
                and 0 < first_sub.span.start < len(session.mapping_text)
            ):
                cut = _find_cut_positions(session.mapping_text, first_sub.span.start)
                if cut is not None:
                    next_punct, prev_punct = cut
                    if next_punct is not None and ":: " in group.mapping_line:
                        content = group.mapping_line.split(":: ", 1)[1]
                        content_pos = _map_mapping_to_content_pos(
                            session.verse_ranges, next_punct
                        )
                        if content_pos is not None and 0 <= content_pos < len(content):
                            _replace_mapping_line(
                                mapping_path,
                                group.mapping_ts,
                                content[: content_pos + 1],
                            )
                    if prev_punct is not None:
                        first_sub.span.start = prev_punct + 1
                        while (
                            first_sub.span.start < len(session.mapping_text)
                            and session.mapping_text[first_sub.span.start] in ' \t"'
                        ):
                            first_sub.span.start += 1

        if session.results:
            affected_timestamp = group.mapping_ts
            patch_circlelog(
                mapping_path=mapping_path,
                affected_timestamp=affected_timestamp,
                session=session,
            )

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
