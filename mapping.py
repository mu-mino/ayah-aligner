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
import json
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
from modules.whispertranscribe import (
    transcribe_chunks,
    load_model,
    _extract_audio_chunk,
    AUDIO_SAMPLE_RATE,
    USE_MODAL,
    _get_modal_fn,
)
from modules.semanticmatch import (
    run_matching,
    patch_circlelog,
    extract_verse_text,
    extract_verse_number,
    _fetch_verse_words,
    _word_to_translation,
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


# ---------------------------------------------------------------------------
# Hilfsfunktion
# ---------------------------------------------------------------------------


def _transcribe_segments(
    audio_path: Path,
    window: FrameWindow,
    model,
) -> List[Tuple[float, float, str]]:
    """Separate Transkription nur für feine Segment-Grenzen.

    Ruft faster-whisper auf (lokal oder über Modal) mit word_timestamps=True.
    """
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        _extract_audio_chunk(audio_path, window.start_sec, window.end_sec, tmp_path)

        if USE_MODAL:
            import whisperx
            audio = whisperx.load_audio(str(tmp_path))
            modal_fn = _get_modal_fn()
            result = modal_fn.remote(
                audio.tobytes(),
                language="ar",
            )
            segments_raw = result["segments"]
        else:
            segs, _ = model.model.transcribe(
                str(tmp_path),
                language="ar",
                beam_size=5,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 400},
            )
            segments_raw = [
                {
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text.strip(),
                    "words": [
                        {"word": w.word.strip(), "start": w.start, "end": w.end}
                        for w in (seg.words or [])
                    ],
                }
                for seg in segs
            ]
    finally:
        tmp_path.unlink(missing_ok=True)

    offset = window.start_sec
    segments = []
    for seg in segments_raw:
        if seg.get("words"):
            for w in seg["words"]:
                if not w.get("word"):
                    continue
                start = w["start"] + offset
                end = w["end"] + offset
                segments.append((round(start, 3), round(end, 3), w["word"]))
        else:
            start = seg["start"] + offset
            end = seg["end"] + offset
            segments.append((round(start, 3), round(end, 3), seg["text"]))
    return segments


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
        elif idx == len(groups) - 1 and not group.verses:
            continue
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
    # 4. Whisper segments for ALL groups (separate feine Schicht)
    # ------------------------------------------------------------------
    whisper_model = load_model(device=whisper_device)

    word_align_path = BASE_DIR / "output" / "word_align.json"

    # 4a. Circle-window transcribe & word alignment
    circle_word_segs: List[Tuple[WindowGroup, List[Tuple[float, float, str]]]] = []
    all_word_aligns: List[dict] = []

    for group in groups:
        segs = _transcribe_segments(audio_path, group.circle_window, whisper_model)
        circle_word_segs.append((group, segs))

    # 4b. Word-level alignment für ALLE groups (circle windows)
    # Build segment→English text map for segments output
    segment_en_map: Dict[Tuple[float, float], str] = {}
    seen_ayah_idx: set = set()
    seen_seg_key: set = set()

    for group, segs in circle_word_segs:
        for verse_num, _ in group.verses:
            try:
                verse_words = _fetch_verse_words(surah, verse_num)
            except Exception:
                continue
            for start, end, ar_word in segs:
                seg_key = (start, ar_word)
                if seg_key in seen_seg_key:
                    continue
                en_word, idx = _word_to_translation(ar_word, verse_words)
                if idx >= 0 and (verse_num, idx) not in seen_ayah_idx:
                    seen_ayah_idx.add((verse_num, idx))
                    seen_seg_key.add(seg_key)
                    all_word_aligns.append({
                        "start": start,
                        "end": end,
                        "ar": ar_word,
                        "en": en_word,
                        "idx": idx,
                        "ayah": verse_num,
                    })
                # Store first successful English match per segment
                if en_word and (start, end) not in segment_en_map:
                    segment_en_map[(start, end)] = en_word

    word_align_path.write_text(
        json.dumps(all_word_aligns, ensure_ascii=False), encoding="utf-8"
    )

    # 4c. Write segments with English text (circle windows)
    segments_path = BASE_DIR / "output" / "segments" / f"{text_path.stem}.segments"
    segments_path.parent.mkdir(parents=True, exist_ok=True)
    with open(segments_path, "w") as f:
        for _, segs in circle_word_segs:
            for start, end, ar_word in segs:
                en = segment_en_map.get((start, end), "")
                if en:
                    f.write(json.dumps({"start": start, "end": end, "text": en}) + "\n")

    # ------------------------------------------------------------------
    # 5. Sub-Fenster transkribieren + matchen + Mapping patchen
    #    (existing logic, only for groups with sub_windows)
    # ------------------------------------------------------------------
    groups_with_subs = [g for g in groups if g.sub_windows]
    if not groups_with_subs:
        return

    sub_word_aligns: List[dict] = []

    for idx, group in enumerate(groups):
        if not group.sub_windows:
            continue
        if idx + 1 >= len(groups):
            continue

        # Nächste Gruppe → deren Vers-Text ist das Target
        next_group = groups[idx + 1]
        next_verse_text = extract_verse_text(next_group.mapping_line)
        next_verse_num = extract_verse_number(next_group.mapping_line)
        if not next_verse_text or next_verse_num is None:
            continue

        chunks = transcribe_chunks(
            video_path=audio_path,
            windows=group.sub_windows,
            model=whisper_model,
        )

        session = run_matching(
            chunks=chunks,
            verse_text=next_verse_text,
            surah=surah,
            ayah=next_verse_num,
        )

        if session.results:
            # Word alignment aus session.results extrahieren
            for result in session.results:
                for (ar_word, en_word, w_start, w_end, idx) in result.word_alignments:
                    sub_word_aligns.append({
                        "start": w_start,
                        "end": w_end,
                        "ar": ar_word,
                        "en": en_word,
                        "idx": idx,
                        "ayah": next_verse_num,
                    })

            # Sub-window segments mit English text an segments file appenden
            sub_en_map: Dict[Tuple[float, float], str] = {}
            for r in session.results:
                for (_, en_word, ws, we, _) in r.word_alignments:
                    key = (round(ws, 3), round(we, 3))
                    if en_word and key not in sub_en_map:
                        sub_en_map[key] = en_word
            with open(segments_path, "a") as f:
                for chunk in chunks:
                    for seg in chunk.segments:
                        skey = (round(seg.start, 3), round(seg.end, 3))
                        en_text = sub_en_map.get(skey, "")
                        if not en_text:
                            # timenächsten match suchen
                            best = min(
                                sub_en_map.keys(),
                                key=lambda k: abs(k[0] - seg.start),
                                default=None,
                            )
                            if best:
                                en_text = sub_en_map[best]
                        f.write(
                            json.dumps(
                                {"start": skey[0], "end": skey[1], "text": en_text}
                            )
                            + "\n"
                        )

            affected_timestamp = group.mapping_ts
            patch_circlelog(
                mapping_path=mapping_path,
                affected_timestamp=affected_timestamp,
                session=session,
            )

            # continuation: ungedeckter suffix → circle_entry der nächsten gruppe
            last_span_end = session.results[-1].span.end
            continuation = session.verse_text[last_span_end:].strip()
            first_sub_ts = seconds_to_timestamp(
                session.results[0].chunk.window.start_sec
            )
            first_verse_num = next_group.verses[0][0]
            file_lines = mapping_path.read_text(encoding="utf-8").splitlines()
            verse_num_prepended = False
            cleaned: list = []
            for file_line in file_lines:
                if file_line.startswith(f"[{next_group.mapping_ts}]"):
                    if continuation:
                        cleaned.append(f"[{next_group.mapping_ts}] :: {continuation}")
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

    # Sub-window alignments an word_align.json appenden (dedupliziert nach ayah,idx)
    if sub_word_aligns:
        existing = json.loads(word_align_path.read_text(encoding="utf-8"))
        existing_pairs = {(d["ayah"], d["idx"]) for d in existing if d.get("idx", -1) >= 0}
        for wa in sub_word_aligns:
            key = (wa["ayah"], wa["idx"])
            if key not in existing_pairs:
                existing_pairs.add(key)
                existing.append(wa)
        word_align_path.write_text(
            json.dumps(existing, ensure_ascii=False), encoding="utf-8"
        )


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
