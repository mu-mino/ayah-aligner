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
        n = detect_markers_from_gray(gray)
        if i == 0 and n > 0:
            first_window_has_circle = True
        if n > 0:
            current_group = WindowGroup(circle_window=window, circle_count=n)
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

        if unshifted_mode:
            # Fall 1 oder nach Fall-3/4-Split: eigener Timestamp, kein Shift
            ts = seconds_to_timestamp(group.circle_window.start_sec)
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
        else:
            # Fall 2 + normaler Shift-Flow
            ts = (
                seconds_to_timestamp(groups[idx].circle_window.end_sec)
                if idx == 0
                else seconds_to_timestamp(groups[idx].circle_window.start_sec)
            )
            group.mapping_ts = ts
            line = build_verse_line(ts, group.verses)
            group.mapping_line = line
            mapping_lines.append(line)
    write_mapping(mapping_lines, mapping_path)

    # ------------------------------------------------------------------
    # 4. Sub-Fenster transkribieren + matchen + Mapping patchen
    # ------------------------------------------------------------------
    groups_with_subs = [g for g in groups if g.sub_windows]
    if not groups_with_subs:
        return
    write_vars(globals(), locals())
    whisper_model = load_model(device=whisper_device)

    for group in groups:
        id_verse: dict[int, str] = mapping_to_per_verse(group.mapping_line)
        if not id_verse:
            continue

        all_windows = group.all_windows if group.sub_windows else [group.circle_window]

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

        if session.results:
            affected_timestamp = group.mapping_ts
            patch_circlelog(
                mapping_path=mapping_path,
                affected_timestamp=affected_timestamp,
                session=session,
            )
            # Der Sub-Entry zeigt den Vers-Anfang (verse_text[0:span.end]).
            # Der noch nicht abgedeckte Suffix (verse_text[span.end:]) wird als
            # Fortsetzung in den circle_window-Eintrag der Gruppe geschrieben.
            last_span_end = session.results[-1].span.end
            continuation = session.mapping_text[last_span_end:].strip()
            first_sub_ts = seconds_to_timestamp(
                session.results[0].chunk.window.start_sec
            )
            first_verse_num = group.verses[0][0]
            file_lines = mapping_path.read_text(encoding="utf-8").splitlines()
            verse_num_prepended = False
            cleaned: list = []
            for file_line in file_lines:
                if file_line.startswith(f"[{group.mapping_ts}]"):
                    # Eintrag durch Fortsetzungstext ersetzen (oder entfernen wenn leer)
                    if continuation:
                        cleaned.append(f"[{group.mapping_ts}] :: {continuation}")
                    continue
                if not verse_num_prepended and file_line.startswith(
                    f"[{first_sub_ts}]"
                ):
                    file_line = file_line.replace(
                        f"[{first_sub_ts}] :: ",
                        f"[{first_sub_ts}] :: {first_verse_num}: ",
                        1,
                    )
                    verse_num_prepended = True
                cleaned.append(file_line)
            mapping_path.write_text("\n".join(cleaned) + "\n", encoding="utf-8")


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
        default=str(BASE_DIR / "output"),
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
