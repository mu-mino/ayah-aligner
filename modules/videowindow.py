"""
Video-Fenster-Segmentierung (schwarzer Bildschirm → Inhalt → schwarzer Bildschirm).

Ein *Fenster* (Frame) ist das Intervall zwischen zwei schwarzen Bildschirmen:

    [schwarz] → Inhalt erscheint → [schwarz]

Jedes Fenster ist eine atomare Einheit – es gibt keinen gesonderten
"Peak-Frame", denn das Fenster selbst ist der Frame.

Dieses Modul liefert für jedes Fenster:
    - start_sec : Sekunde, zu der der Bildschirm hell wird
    - end_sec   : Sekunde, zu der der Bildschirm wieder schwarz wird

Nicht enthalten: Kreiserkennung, OCR, Mapping, Rendering.
"""

import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None  # type: ignore[assignment]
import numpy as np

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

MSE_STATIC_THRESHOLD: float = 5.0
MIN_SEGMENT_LEN: int = 10
STATIC_SAFETY_SPLITS: int = 5
BRIGHTNESS_THRESHOLD: int = 1000
FRAME_STRIDE: int = int(os.getenv("FRAME_STRIDE", "10"))
DOWNSAMPLE_SIZE: Tuple[int, int] = (64, 64)
PROGRESS_STEPS: int = int(os.getenv("PROGRESS_STEPS", "4"))

_FEATURE_CACHE: Dict[Tuple[Path, int], List["FrameFeature"]] = {}


# ---------------------------------------------------------------------------
# Datenstrukturen
# ---------------------------------------------------------------------------

@dataclass
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    frames: int


@dataclass
class FrameFeature:
    """Pro-Frame-Metriken für die Schwarzbild-Segmentierung."""
    index: int
    brightness: int       # Anzahl Pixel mit Grauwert > 200
    mse_from_prev: float  # MSE zum vorherigen (downsampled) Frame


@dataclass
class FrameWindow:
    """
    Ein Fenster zwischen zwei schwarzen Bildschirmen – entspricht einem Frame
    im Sinne der Pipeline.

    start_sec : Bildschirm wird hell (Ende des schwarzen Bildschirms davor)
    end_sec   : Bildschirm wird schwarz (Beginn des schwarzen Bildschirms danach)
    """
    start_sec: float
    end_sec: float


# ---------------------------------------------------------------------------
# Video-Metadaten
# ---------------------------------------------------------------------------

def run_ffprobe(path: Path) -> VideoInfo:
    """Liest Video-Metadaten (Auflösung, FPS, Frame-Anzahl) via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
        "-show_entries", "format=duration",
        "-of", "json",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode())
    data = json.loads(proc.stdout.decode())
    stream = data["streams"][0]
    width = int(stream["width"])
    height = int(stream["height"])
    fps_num, fps_den = map(int, stream["r_frame_rate"].split("/"))
    fps = fps_num / fps_den if fps_den else 0.0
    duration = float(data.get("format", {}).get("duration", 0.0))
    if stream.get("nb_frames") and str(stream["nb_frames"]).isdigit():
        frames = int(stream["nb_frames"])
    else:
        frames = int(duration * fps)
    return VideoInfo(path=path, width=width, height=height, fps=fps, frames=frames)


# ---------------------------------------------------------------------------
# Helligkeits-Stream
# ---------------------------------------------------------------------------

def stream_features(info: VideoInfo) -> List[FrameFeature]:
    """
    Liest das Video Frame für Frame und berechnet Helligkeit und MSE.
    Gibt einen Eintrag pro FRAME_STRIDE-tem Frame zurück.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) ist nicht installiert.")
    cap = cv2.VideoCapture(str(info.path))
    if not cap.isOpened():
        raise RuntimeError(f"Video kann nicht geöffnet werden: {info.path}")

    features: List[FrameFeature] = []
    prev_down: Optional[np.ndarray] = None
    frame_idx = 0
    steps = max(1, PROGRESS_STEPS)
    step_interval = max(1, math.ceil(info.frames / steps)) if info.frames else 0
    next_log = step_interval if step_interval else None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        down = cv2.resize(gray, DOWNSAMPLE_SIZE, interpolation=cv2.INTER_AREA)
        if prev_down is None:
            mse_val = 0.0
        else:
            diff = down.astype(np.float32) - prev_down.astype(np.float32)
            mse_val = float(np.mean(diff * diff))
        prev_down = down
        brightness = int((gray > 200).sum())
        features.append(FrameFeature(index=frame_idx, brightness=brightness, mse_from_prev=mse_val))

        if FRAME_STRIDE > 1:
            for _ in range(FRAME_STRIDE - 1):
                if not cap.grab():
                    break
        frame_idx += FRAME_STRIDE

        if next_log is not None and frame_idx >= next_log:
            current = min(frame_idx, info.frames) if info.frames else frame_idx
            pct = (current / info.frames * 100) if info.frames else 0.0
            print(
                f"[{info.path.name}] {current}/{info.frames} Frames ({pct:.1f}%) – stride={FRAME_STRIDE}",
                file=sys.stderr, flush=True,
            )
            while next_log is not None and frame_idx >= next_log:
                next_log += step_interval

    cap.release()
    return features


def get_features(info: VideoInfo) -> List[FrameFeature]:
    """Gecachte Version von stream_features."""
    cache_key = (info.path, FRAME_STRIDE)
    cached = _FEATURE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    feats = stream_features(info)
    _FEATURE_CACHE[cache_key] = feats
    return feats


# ---------------------------------------------------------------------------
# Segment-Erkennung (hell ↔ dunkel)
# ---------------------------------------------------------------------------

def find_segments(features: List[FrameFeature]) -> List[Tuple[int, int]]:
    """
    Findet helle Segmente als Feature-Listen-Indizes (start_idx, end_idx).

    Ein Segment beginnt bei brightness >= BRIGHTNESS_THRESHOLD und endet
    bei einem MSE-Sprung, der eine Szenengrenze signalisiert.
    """
    segments: List[Tuple[int, int]] = []
    in_run = False
    start = 0
    last_split = 0
    tolerance = STATIC_SAFETY_SPLITS
    min_len_feat = max(1, math.ceil(MIN_SEGMENT_LEN / FRAME_STRIDE))
    safety_len_feat = max(1, math.ceil((MIN_SEGMENT_LEN * STATIC_SAFETY_SPLITS) / FRAME_STRIDE))

    for i, feat in enumerate(features):
        if feat.brightness >= BRIGHTNESS_THRESHOLD and not in_run:
            in_run = True
            start = i
            last_split = i
        if in_run:
            if feat.mse_from_prev > MSE_STATIC_THRESHOLD:
                if i - last_split >= min_len_feat:
                    segments.append((start, i - 1))
                    in_run = False
                else:
                    tolerance -= 1
                    if tolerance <= 0:
                        segments.append((start, i - 1))
                        in_run = False
                        tolerance = STATIC_SAFETY_SPLITS
            if in_run and i - start > safety_len_feat and feat.mse_from_prev <= MSE_STATIC_THRESHOLD:
                last_split = i

    if in_run:
        segments.append((start, len(features) - 1))
    return segments


def filter_segments(segments: List[Tuple[int, int]], fps: float) -> List[Tuple[int, int]]:
    """
    Entfernt zu kurze Segmente (< ~2 Sekunden).
    Gibt das Original zurück, wenn nach dem Filtern weniger als 2 übrig bleiben.
    """
    keep_len = int(max(1, math.ceil((fps * 2.0) / FRAME_STRIDE)))
    filtered = [(s, e) for s, e in segments if e >= s and (e - s + 1) >= keep_len]
    if not filtered or len(filtered) <= 1:
        return segments
    return filtered


# ---------------------------------------------------------------------------
# Öffentliche API
# ---------------------------------------------------------------------------

def extract_windows(video_path: Path) -> List[FrameWindow]:
    """
    Segmentiert ein Video in Fenster der Form [schwarz → Inhalt → schwarz].

    Jedes zurückgegebene FrameWindow entspricht einem Frame im Sinne der
    Pipeline – kein weiteres Unterteilen nötig.

    Parameters
    ----------
    video_path : Pfad zur Videodatei.

    Returns
    -------
    Liste von FrameWindow-Objekten, chronologisch sortiert.
    """
    info = run_ffprobe(video_path)
    features = get_features(info)
    segments = filter_segments(find_segments(features), info.fps)

    windows: List[FrameWindow] = []
    for seg_start, seg_end in segments:
        start_sec = features[seg_start].index / info.fps if info.fps else 0.0
        end_sec = features[seg_end].index / info.fps if info.fps else 0.0
        windows.append(FrameWindow(start_sec=start_sec, end_sec=end_sec))

    return windows
