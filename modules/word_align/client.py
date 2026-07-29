"""Client for Modal WhisperX GPU service."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from modal import Function

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "whisperx"


def _ensure_cache():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(surah: int) -> Path:
    return CACHE_DIR / f"surah_{surah:03d}.json"


def _load_cache(surah: int) -> list[dict] | None:
    path = _cache_path(surah)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _save_cache(surah: int, words: list[dict]):
    path = _cache_path(surah)
    path.write_text(json.dumps(words, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Cached {len(words)} words to {path.name}")


def transcribe_surah(
    audio_path: str,
    surah: int,
    force: bool = False,
) -> list[dict]:
    if not force:
        cached = _load_cache(surah)
        if cached is not None:
            print(f"  Using cached WhisperX for surah {surah}")
            return cached

    _ensure_cache()
    import soundfile as sf
    import scipy.signal as ss

    print(f"  Loading audio ({audio_path})...")
    wav, sr = sf.read(audio_path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        target = int(len(wav) * 16000 / sr)
        wav = ss.resample(wav, target).astype(np.float32)
    wav = wav.astype(np.float32)
    duration = len(wav) / 16000
    print(f"  Audio: {duration:.0f}s ({duration/60:.1f}min)")

    fn = Function.from_name("quran-aligner", "transcribe")

    print(f"  Calling Modal transcribe...")
    result = fn.remote(wav.tobytes())
    print(f"  Got {len(result['segments'])} segments")

    words = []
    for seg in result["segments"]:
        for w in seg.get("words", []):
            words.append({
                "text": w["word"],
                "start": w["start"],
                "end": w["end"],
            })

    _save_cache(surah, words)
    return words
