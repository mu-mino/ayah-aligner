"""
WhisperX-Transkription für Audio-Häppchen aus erkannten Video-Fenstern.

Dieses Modul wird vom Orchestrierungs-Modul aufgerufen, wenn
recognizecircle innerhalb eines circlelog-Intervalls Frames mit
0 Kreisen findet. In diesem Fall liefert videowindow die Sub-Fenster
(Häppchen) des betroffenen Intervalls als FrameWindow-Objekte.
Dieses Modul:
    1. Extrahiert die Audio jedes Häppchens aus dem Video (via ffmpeg).
    2. Übergibt sie an WhisperX (Sprache: Arabisch, Modell: large-v2).
    3. Gibt Transkript-Segmente mit genauen Zeitstempeln zurück.

Nicht enthalten: Kreiserkennung, Segmenterkennung, Mapping-Schreiben.
"""

import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from modules.videowindow import FrameWindow

_WHISPER_CACHE_DIR: Path = Path(__file__).resolve().parent.parent / "data" / "whisper"

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

WHISPER_MODEL: str = "large-v2"
WHISPER_LANGUAGE: str = "ar"
WHISPER_COMPUTE_TYPE: str = "float16"  # "int8" für CPU ohne VRAM
AUDIO_SAMPLE_RATE: int = 16000  # WhisperX erwartet 16 kHz
SEGMENT_END_TOLERANCE: float = 1.5  # Sekunden, die an segment.end addiert werden

# Quranic recitation (tajweed/melody) is often misclassified as non-speech.
# Disable no_speech filtering and lower VAD onset/offset to capture all audio.
# initial_prompt primes Whisper toward Arabic Quranic vocabulary.
ASR_OPTIONS: dict = {
    "no_speech_threshold": 1.0,
    "initial_prompt": "بسم الله الرحمن الرحيم",
}
VAD_OPTIONS: dict = {
    "vad_onset": 0.1,
    "vad_offset": 0.1,
    "min_silence_duration_ms": 2000,
}

AUDIO_PADDING_SEC: float = (
    1.0  # Stille vor jedem Chunk (hilft WhisperX Spracheinsatz zu erkennen)
)
SILENCE_COMPRESS_MIN_SEC: float = 2.0  # Pausen länger als N Sekunden werden komprimiert
SILENCE_COMPRESS_TARGET_SEC: float = 0.5  # Ziel-Pausenlänge nach Kompression
SILENCE_THRESHOLD: float = 0.005  # Amplitudenschwelle für Stille-Erkennung

MAX_TRANSCRIBE_RETRIES: int = 3  # retries on poor timestamp coverage
COVERAGE_TOLERANCE: float = 2.0  # seconds gap allowed at start/end before retry


# ---------------------------------------------------------------------------
# Datenstrukturen
# ---------------------------------------------------------------------------


@dataclass
class TranscriptSegment:
    """Ein einzelnes Transkript-Segment mit Zeitstempel."""

    start: float  # Sekunden relativ zum Videobeginn
    end: float
    text: str
    stamp: str = ""


@dataclass
class ChunkTranscription:
    """
    Ergebnis der Transkription eines einzelnen Fensters (FrameWindow = Frame).

    window      : das zugehörige FrameWindow [start_sec, end_sec]
    segments    : von WhisperX erkannte Satz-/Wort-Segmente
    raw_text    : vollständiger Transkripttext des Fensters
    """

    window: FrameWindow
    segments: List[TranscriptSegment] = field(default_factory=list)
    raw_text: str = ""


# ---------------------------------------------------------------------------
# Audio-Extraktion
# ---------------------------------------------------------------------------


def _extract_audio_chunk(
    video_path: Path,
    start_sec: float,
    end_sec: float,
    tmp_path: Path,
    sample_rate: int = AUDIO_SAMPLE_RATE,
) -> None:
    """
    Extrahiert einen Zeitabschnitt der Video-Audio als WAV-Datei.

    Nutzt ffmpeg: mono, 16 kHz, PCM – das Format, das WhisperX erwartet.
    """
    duration = max(0.01, end_sec - start_sec)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_sec),
        "-t",
        str(duration),
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "wav",
        str(tmp_path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg Audio-Extraktion fehlgeschlagen "
            f"({start_sec:.2f}s–{end_sec:.2f}s): "
            f"{result.stderr.decode(errors='replace')}"
        )


# ---------------------------------------------------------------------------
# Audio-Vorverarbeitung
# ---------------------------------------------------------------------------


def _preprocess_audio(
    wav_path: Path, sample_rate: int = AUDIO_SAMPLE_RATE
) -> np.ndarray:
    """
    Lädt eine WAV-Datei und bereitet das Audio für WhisperX vor:
      1. Fügt AUDIO_PADDING_SEC Stille am Anfang ein (hilft Spracheinsatz zu erkennen).
      2. Komprimiert Pausen > SILENCE_COMPRESS_MIN_SEC auf SILENCE_COMPRESS_TARGET_SEC.
    """
    import whisperx

    audio = whisperx.load_audio(str(wav_path))

    # 1. Stille-Padding am Anfang
    padding = np.zeros(int(sample_rate * AUDIO_PADDING_SEC), dtype=np.float32)
    audio = np.concatenate([padding, audio])

    # 2. Pausen kürzen
    is_speech = np.abs(audio) > SILENCE_THRESHOLD
    speech_indices = np.where(is_speech)[0]

    if len(speech_indices) > 0:
        min_gap = int(SILENCE_COMPRESS_MIN_SEC * sample_rate)
        target_gap = int(SILENCE_COMPRESS_TARGET_SEC * sample_rate)
        parts = [audio[: speech_indices[0]]]
        for i in range(len(speech_indices) - 1):
            curr, nxt = speech_indices[i], speech_indices[i + 1]
            if nxt - curr > min_gap:
                parts.append(audio[curr : curr + 1])
                parts.append(np.zeros(target_gap, dtype=np.float32))
            else:
                parts.append(audio[curr : curr + 1])
        parts.append(audio[speech_indices[-1] :])
        audio = np.concatenate(parts)

    return audio


# ---------------------------------------------------------------------------
# Modell laden
# ---------------------------------------------------------------------------


def load_model(
    device: Optional[str] = None,
    model_name: str = WHISPER_MODEL,
    compute_type: str = WHISPER_COMPUTE_TYPE,
):
    """
    Lädt das WhisperX-Modell (einmalig, da Laden teuer ist).

    Parameters
    ----------
    device:
        'cuda' oder 'cpu'. Wird automatisch erkannt, wenn None.
    model_name:
        WhisperX-Modell, Standard: 'large-v2'.
    compute_type:
        'float16' (GPU) oder 'int8' (CPU).

    Returns
    -------
    Geladenes WhisperX-Modell-Objekt.
    """
    import torch
    import whisperx

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and compute_type == "float16":
        compute_type = "int8"

    return whisperx.load_model(
        model_name,
        device=device,
        compute_type=compute_type,
        language=WHISPER_LANGUAGE,
        asr_options=ASR_OPTIONS,
    )


# ---------------------------------------------------------------------------
# Transkription
# ---------------------------------------------------------------------------


def _whisper_cache_path(video_path: Path, window: FrameWindow) -> Path:
    return (
        _WHISPER_CACHE_DIR
        / f"{video_path.stem}_{window.start_sec:.3f}_{window.end_sec:.3f}.json"
    )


def _format_segment_stamp(start: float, end: float) -> str:
    return f"[{start:05.2f}s -> {end:05.2f}s]"


def _load_whisper_cache(
    video_path: Path, window: FrameWindow
) -> Optional[ChunkTranscription]:
    cache_file = _whisper_cache_path(video_path, window)
    if not cache_file.exists():
        return None
    data = json.loads(cache_file.read_text(encoding="utf-8"))
    segments = []
    for s in data["segments"]:
        start = s["start"]
        end = s["end"]
        segments.append(
            TranscriptSegment(
                start=start,
                end=end,
                text=s["text"],
                stamp=s.get("stamp", _format_segment_stamp(start, end)),
            )
        )
    return ChunkTranscription(
        window=window, segments=segments, raw_text=data["raw_text"]
    )


def _save_whisper_cache(video_path: Path, result: ChunkTranscription) -> None:
    _WHISPER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _whisper_cache_path(video_path, result.window)
    data = {
        "segments": [
            {"start": s.start, "end": s.end, "text": s.text, "stamp": s.stamp}
            for s in result.segments
        ],
        "raw_text": result.raw_text,
    }
    cache_file.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def stamp_per_segment_transcription(
    audio_path: str,
    time_window: Optional[FrameWindow] = None,
    device: Optional[str] = None,
    model_name: str = "large-v2",
    compute_type: str = "float16",
):
    import torch
    import whisper_timestamped as wt

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and compute_type == "float16":
        compute_type = "int8"

    # 2. Modell laden via whisperx
    model = wt.load_model(
        model_name,
        device=device,
    )

    audio = wt.load_audio(audio_path)

    if time_window is not None:
        sample_rate = AUDIO_SAMPLE_RATE
        start_sample = int(time_window.start_sec * sample_rate)
        end_sample = int(time_window.end_sec * sample_rate)
        audio = audio[start_sample:end_sample]

    result = wt.transcribe(
        model,
        audio,
        language=WHISPER_LANGUAGE,
        beam_size=5,
        no_speech_threshold=ASR_OPTIONS.get("no_speech_threshold", 0.6),
        initial_prompt=ASR_OPTIONS.get("initial_prompt"),
    )

    offset = time_window.start_sec if time_window is not None else 0.0

    print("\n--- Transkription nach Wörtern (whisper-timestamped) ---")
    segments_with_absolute_timestamps: list[TranscriptSegment] = []
    for segment in result["segments"]:
        for word in segment.get("words", []):
            text = word["text"].strip()
            start = word.get("start")
            end = word.get("end")
            if start is not None and end is not None and text:
                absoluter_start = start + offset
                absolutes_ende = end + offset
                stamp = _format_segment_stamp(absoluter_start, absolutes_ende)
                segments_with_absolute_timestamps.append(
                    TranscriptSegment(
                        start=absoluter_start,
                        end=absolutes_ende,
                        text=text,
                        stamp=stamp,
                    )
                )
                print(f"{stamp}: {text}")
    return segments_with_absolute_timestamps


def transcribe_chunk(
    model,
    video_path: Path,
    window: FrameWindow,
) -> ChunkTranscription:
    """
    Transkribiert ein einzelnes videowindow-Häppchen.

    Parameters
    ----------
    model:
        Geladenes WhisperX-Modell (von load_model()).
    video_path:
        Pfad zum Quell-Video.
    window:
        Das FrameWindow, dessen Audio transkribiert werden soll.

    Returns
    -------
    ChunkTranscription mit Segmenten und vollständigem Text.
    """
    cached = _load_whisper_cache(video_path, window)
    if cached is not None:
        return cached

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        _extract_audio_chunk(
            video_path,
            window.start_sec,
            window.end_sec + SEGMENT_END_TOLERANCE,
            tmp_path,
        )
        audio = _preprocess_audio(tmp_path)
        audio_duration = len(audio) / AUDIO_SAMPLE_RATE

        result = model.transcribe(audio, language=WHISPER_LANGUAGE)
        for _ in range(MAX_TRANSCRIBE_RETRIES):
            segs = result.get("segments", [])
            if (
                segs
                and segs[0]["start"] <= COVERAGE_TOLERANCE
                and segs[-1]["end"] >= audio_duration - COVERAGE_TOLERANCE
            ):
                break
            result = model.transcribe(audio, language=WHISPER_LANGUAGE)
    finally:
        tmp_path.unlink(missing_ok=True)

    # Zeitstempel auf absolute Video-Zeit umrechnen (WhisperX gibt Chunk-relative Zeiten zurück)
    offset = window.start_sec
    segments: List[TranscriptSegment] = []
    for seg in result.get("segments", []):
        start = seg["start"] + offset
        end = seg["end"] + offset + SEGMENT_END_TOLERANCE
        segments.append(
            TranscriptSegment(
                start=start,
                end=end,
                text=seg["text"].strip(),
                stamp=_format_segment_stamp(start, end),
            )
        )

    raw_text = " ".join(s.text for s in segments)
    transcription = ChunkTranscription(
        window=window, segments=segments, raw_text=raw_text
    )
    _save_whisper_cache(video_path, transcription)
    return transcription


def transcribe_chunks(
    video_path: Path,
    windows: List[FrameWindow],
    model=None,
    device: Optional[str] = None,
) -> List[ChunkTranscription]:
    """
    Transkribiert eine Liste von videowindow-Häppchen.

    Das Modell wird einmalig geladen, wenn es nicht übergeben wird.

    Parameters
    ----------
    video_path:
        Pfad zum Quell-Video.
    windows:
        Liste von FrameWindow-Objekten (aus videowindow.extract_windows()).
    model:
        Optional bereits geladenes WhisperX-Modell. Wird bei None geladen.
    device:
        'cuda' oder 'cpu'. Nur relevant, wenn model=None.

    Returns
    -------
    Liste von ChunkTranscription – in derselben Reihenfolge wie windows.
    """
    if not windows:
        return []

    if model is None:
        model = load_model(device=device)

    results: List[ChunkTranscription] = []
    for window in windows:
        results.append(transcribe_chunk(model, video_path, window))

    return results
