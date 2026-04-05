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

import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from videowindow import FrameWindow

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

WHISPER_MODEL: str = "large-v2"
WHISPER_LANGUAGE: str = "ar"
WHISPER_COMPUTE_TYPE: str = "float16"  # "int8" für CPU ohne VRAM
AUDIO_SAMPLE_RATE: int = 16000  # WhisperX erwartet 16 kHz


# ---------------------------------------------------------------------------
# Datenstrukturen
# ---------------------------------------------------------------------------


@dataclass
class TranscriptSegment:
    """Ein einzelnes Transkript-Segment mit Zeitstempel."""

    start: float  # Sekunden relativ zum Videobeginn
    end: float
    text: str


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
    )


# ---------------------------------------------------------------------------
# Transkription
# ---------------------------------------------------------------------------


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
    import whisperx

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        _extract_audio_chunk(video_path, window.start_sec, window.end_sec, tmp_path)
        audio = whisperx.load_audio(str(tmp_path))
        result = model.transcribe(audio, language=WHISPER_LANGUAGE)
    finally:
        tmp_path.unlink(missing_ok=True)

    # Zeitstempel auf absolute Video-Zeit umrechnen (WhisperX gibt Chunk-relative Zeiten zurück)
    offset = window.start_sec
    segments: List[TranscriptSegment] = []
    for seg in result.get("segments", []):
        segments.append(
            TranscriptSegment(
                start=seg["start"] + offset,
                end=seg["end"] + offset,
                text=seg["text"].strip(),
            )
        )

    raw_text = " ".join(s.text for s in segments)
    return ChunkTranscription(window=window, segments=segments, raw_text=raw_text)


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
        transcription = transcribe_chunk(model, video_path, window)
        results.append(transcription)

    return results
