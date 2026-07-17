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
from typing import Dict, List, Optional, Tuple
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
)
from modules.semanticmatch import (
    run_matching,
    patch_circlelog,
    extract_verse_text,
    extract_verse_number,
    _fetch_verse_words,
    _word_to_translation,
    _word_to_translation_from,
)
from modules.forced_align import ForcedAligner

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
    model=None,
) -> List[Tuple[float, float, str]]:
    """Separate Transkription nur für feine Segment-Grenzen.

    Ruft faster-whisper direkt auf (nicht den WhisperX-Wrapper)
    mit feinerer VAD-Schwelle und word_timestamps=True.
    Kein Stille-Padding, keine Pausen-Kompression.

    Wenn USE_MODAL=True, wird die GPU-Inferenz auf Modal ausgelagert.
    """
    if USE_MODAL:
        return _transcribe_segments_modal(audio_path, window)
    return _transcribe_segments_local(audio_path, window, model)


def _transcribe_segments_local(
    audio_path: Path,
    window: FrameWindow,
    model,
) -> List[Tuple[float, float, str]]:
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        _extract_audio_chunk(audio_path, window.start_sec, window.end_sec, tmp_path)
        segs, _ = model.model.transcribe(
            str(tmp_path),
            language="ar",
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 400},
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    offset = window.start_sec
    segments = []
    for seg in segs:
        if seg.words:
            for word in seg.words:
                start = word.start + offset
                end = word.end + offset
                segments.append((round(start, 3), round(end, 3), word.word.strip()))
        else:
            start = seg.start + offset
            end = seg.end + offset
            segments.append((round(start, 3), round(end, 3), seg.text.strip()))
    return segments


_MODAL_TRANSCRIBE_FN = None

def _get_modal_transcribe_fn():
    global _MODAL_TRANSCRIBE_FN
    if _MODAL_TRANSCRIBE_FN is None:
        from modal import Function
        _MODAL_TRANSCRIBE_FN = Function.from_name("whispe-ayah-aligner", "transcribe_audio_chunk")
    return _MODAL_TRANSCRIBE_FN


def _transcribe_segments_modal(
    audio_path: Path,
    window: FrameWindow,
) -> List[Tuple[float, float, str]]:
    """Transkribiert ein Audio-Fenster via Modal (GPU serverless)."""
    import tempfile
    import numpy as np

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        _extract_audio_chunk(audio_path, window.start_sec, window.end_sec, tmp_path)
        import soundfile as sf
        audio, sr = sf.read(tmp_path)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        if sr != AUDIO_SAMPLE_RATE:
            import scipy.signal
            audio = scipy.signal.resample(
                audio, int(len(audio) * AUDIO_SAMPLE_RATE / sr)
            )
        audio_bytes = audio.astype(np.float32).tobytes()
    finally:
        tmp_path.unlink(missing_ok=True)

    modal_fn = _get_modal_transcribe_fn()
    result = modal_fn.remote(audio_bytes)

    offset = window.start_sec
    segments = []
    for seg in result["segments"]:
        for w in seg.get("words", []):
            start = w["start"] + offset
            end = w["end"] + offset
            segments.append((round(start, 3), round(end, 3), w["word"]))
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


def _guard_ok(aligns: List[dict], verse_words: list) -> bool:
    if not aligns: return False
    n_verse = len(verse_words)
    if n_verse == 0: return False
    
    if any(a["end"] - a["start"] > 4.0 for a in aligns): return False  # Duration
    if any("score" in a and a["score"] > 0.0 for a in aligns):
        avg_score = sum(a["score"] for a in aligns) / len(aligns)
        if avg_score < 0.5:
            return False
    
    return True


def _interpolate_verse(
    aligns: List[dict],
    verse_words: list,
    window_start: float,
    window_end: float,
    verse_num: int,
) -> List[dict]:
    """Interpoliert zwischen Anchor-Wörtern, deckt alle Vers-Wörter ab."""
    sorted_a = sorted(aligns, key=lambda a: a["idx"])
    n = len(verse_words)
    result = []

    def _make(vw, idx, start, end):
        return {
            "start": round(start, 4), "end": round(end, 4),
            "ar": vw["text_uthmani"], "en": vw.get("translation", ""),
            "idx": idx, "ayah": verse_num, "score": 0.0,
        }

    # Words before first anchor
    first = sorted_a[0]
    if first["idx"] > 0:
        seg_len = first["start"] - window_start
        for i in range(first["idx"]):
            t = window_start + (i / first["idx"]) * seg_len
            t_e = window_start + ((i + 1) / first["idx"]) * seg_len
            result.append(_make(verse_words[i], i, t, t_e))

    # Anchors + gaps between
    for i, a in enumerate(sorted_a):
        result.append(a)
        if i + 1 < len(sorted_a):
            nxt = sorted_a[i + 1]
            gap = nxt["idx"] - a["idx"] - 1
            if gap > 0:
                seg_len = nxt["start"] - a["end"]
                for j in range(gap):
                    vi = a["idx"] + 1 + j
                    t = a["end"] + (j / gap) * seg_len
                    t_e = a["end"] + ((j + 1) / gap) * seg_len
                    result.append(_make(verse_words[vi], vi, t, t_e))

    # Words after last anchor
    last = sorted_a[-1]
    remaining = n - last["idx"] - 1
    if remaining > 0:
        seg_len = window_end - last["end"]
        for j in range(remaining):
            vi = last["idx"] + 1 + j
            t = last["end"] + (j / remaining) * seg_len
            t_e = last["end"] + ((j + 1) / remaining) * seg_len
            result.append(_make(verse_words[vi], vi, t, t_e))

    return result


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

    # 4. Multi-Stage Word Alignment
    # ------------------------------------------------------------------
    word_align_path = BASE_DIR / "output" / "word_align.json"
    all_word_aligns: List[dict] = []
    
    # Initialize Aligners
    forced_aligner = ForcedAligner(device=whisper_device, use_modal=USE_MODAL)
    forced_aligner.global_prompt = " ".join(
        vw["text_uthmani"]
        for group in groups
        for verse_num, _ in group.verses
        for vw in _fetch_verse_words(surah, verse_num)
    )
    if USE_MODAL:
        whisper_model = None  # not needed locally
    else:
        whisper_model = load_model(device=whisper_device)
    
    segment_en_map: Dict[Tuple[float, float], str] = {}
    
    for group in groups:
        # Stage 2 Fallback data
        whisper_segs = _transcribe_segments(audio_path, group.circle_window, whisper_model)
        
        for verse_num, _ in group.verses:
            try:
                verse_words = _fetch_verse_words(surah, verse_num)
                if not verse_words: continue
            except Exception:
                continue

            # --- Stage 1: Forced Alignment (Best Quality) ---
            aligned_words = forced_aligner.align(
                audio_path=audio_path,
                window_start=group.circle_window.start_sec,
                window_end=group.circle_window.end_sec,
            )
            
            temp_aligns = []
            if aligned_words:
                cursor = 0
                for aw in aligned_words:
                    if aw.score < 0.5:
                        continue
                    _, idx = _word_to_translation_from(aw.word, verse_words, cursor)
                    if idx >= 0:
                        en = verse_words[idx].get("translation", "")
                        temp_aligns.append({
                            "start": aw.start, "end": aw.end, "ar": aw.word,
                            "en": en, "idx": idx, "ayah": verse_num, "score": aw.score
                        })
                        cursor = idx + 1

            # Deduplicate anchors: same verse idx -> keep higher score
            seen = {}
            for a in temp_aligns:
                key = a["idx"]
                if key not in seen or a["score"] > seen[key]["score"]:
                    seen[key] = a
            temp_aligns = list(seen.values())
            
            # --- Guard & Fallback Logic ---
            if _guard_ok(temp_aligns, verse_words):
                filled = _interpolate_verse(temp_aligns, verse_words, group.circle_window.start_sec, group.circle_window.end_sec, verse_num)
                all_word_aligns.extend(filled)
                print(f"[V{verse_num}] ✓ Forced Alignment OK ({len(temp_aligns)} anchors, {len(filled)} total)")
                continue

            # --- Stage 2: Whisper Timestamps + Smart Match ---
            print(f"[V{verse_num}] ✗ Forced Alignment failed, trying Whisper Timestamps...")
            
            window_aligns_whisper: List[dict] = []
            cursor = 0
            for start, end, ar_word in whisper_segs:
                en_word, idx = _word_to_translation_from(ar_word, verse_words, cursor)
                if idx >= 0:
                    window_aligns_whisper.append({
                        "start": start, "end": end, "ar": ar_word, "en": en_word,
                        "idx": idx, "ayah": verse_num, "score": 0.0 # No score from this method
                    })
                    cursor = idx + 1
                if en_word and (start, end) not in segment_en_map:
                    segment_en_map[(start, end)] = en_word

            if _guard_ok(window_aligns_whisper, verse_words):
                filled = _interpolate_verse(window_aligns_whisper, verse_words, group.circle_window.start_sec, group.circle_window.end_sec, verse_num)
                all_word_aligns.extend(filled)
                print(f"[V{verse_num}] ✓ Whisper Timestamps OK ({len(window_aligns_whisper)} anchors, {len(filled)} total)")
            else:
                # --- Stage 3: Linear Interpolation (Final Fallback) ---
                print(f"[V{verse_num}] ✗ Whisper Timestamps failed, using Linear Fallback.")
                if whisper_segs:
                    win_start, win_end = whisper_segs[0][0], whisper_segs[-1][1]
                    n, dur = len(verse_words), win_end - win_start
                    if n > 0 and dur > 0:
                        for vi, vw in enumerate(verse_words):
                            t, t_end = win_start + (vi / n) * dur, win_start + ((vi + 1) / n) * dur
                            en = vw.get("translation", "")
                            all_word_aligns.append({
                                "start": round(t, 4), "end": round(t_end, 4),
                                "ar": vw["text_uthmani"], "en": en, "idx": vi, "ayah": verse_num, "score": 0.0
                            })
    
    # 4c. Write final alignments
    word_align_path.write_text(
        json.dumps(all_word_aligns, ensure_ascii=False), encoding="utf-8"
    )
    
    # 4c. Write segments with English text (circle windows)
    segments_path = BASE_DIR / "output" / "segments" / f"{text_path.stem}.segments"
    segments_path.parent.mkdir(parents=True, exist_ok=True)
    with open(segments_path, "w") as f:
        for entry in all_word_aligns:
            # We need the segments of the _original_ whisper transcription,
            # not the aligned words. This means the segment_en_map should be filled from whisper_segs
            # during Stage 2 fallback.
            # However, for simplicity now, we will just take the `en` from all_word_aligns.
            if entry.get("en"):
                f.write(json.dumps({
                    "start": entry["start"],
                    "end": entry["end"],
                    "text": entry["en"],
                    "ayah": entry["ayah"],
                    "idx": entry["idx"],
                }, ensure_ascii=False) + "\n")

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

    # Sub-window alignments an word_align.json appenden
    if sub_word_aligns:
        existing = json.loads(word_align_path.read_text(encoding="utf-8"))
        existing.extend(sub_word_aligns)
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
