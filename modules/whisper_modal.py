"""
Modal-Integration für WhisperX + whisper-timestamped auf GPU (serverless).

Nur die GPU-intensive Modell-Inferenz wird auf Modal ausgelagert.
VAD und Audio-Extraktion bleiben lokal.
"""

from __future__ import annotations

from typing import Any, Optional

import modal

app = modal.App("whispe-ayah-aligner")

_image = (
    modal.Image.debian_slim()
    .pip_install(
        "whisperx",
        "torch",
        "numpy",
        "whisper-timestamped",
    )
)


@app.local_entrypoint()
def main():
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        print("Usage: modal run modules/whisper_modal.py -- <audio.wav>")
        return
    audio_bytes = open(path, "rb").read()
    result = transcribe_audio_chunk.remote(audio_bytes)
    for seg in result["segments"]:
        print(f'[{seg["start"]:.2f} -> {seg["end"]:.2f}] {seg["text"]}')
        for w in seg.get("words", []):
            print(f'  [{w["start"]:.2f} -> {w["end"]:.2f}] {w["word"]}')


@app.function(
    image=_image,
    gpu="A10G",
    scaledown_window=300,
    timeout=600,
)
def transcribe_audio_chunk(
    audio_bytes: bytes,
    language: str = "ar",
    beam_size: int = 5,
    no_speech_threshold: float = 1.0,
    initial_prompt: str = "بسم الله الرحمن الرحيم",
    word_timestamps: bool = True,
) -> dict[str, Any]:
    import numpy as np
    import whisperx

    audio = np.frombuffer(audio_bytes, dtype=np.float32)

    model = whisperx.load_model(
        "large-v3",
        device="cuda",
        compute_type="float16",
        language=language,
        asr_options={
            "no_speech_threshold": no_speech_threshold,
            "initial_prompt": initial_prompt,
            "word_timestamps": word_timestamps,
        },
    )

    fw_segments, _ = model.model.transcribe(
        audio,
        language=language,
        beam_size=beam_size,
        word_timestamps=word_timestamps,
        vad_filter=False,
        condition_on_previous_text=False,
        no_speech_threshold=no_speech_threshold,
        initial_prompt=initial_prompt,
        max_initial_timestamp=0.0,
    )

    segments = []
    for seg in fw_segments:
        words = []
        if seg.words:
            for w in seg.words:
                wtxt = w.word.strip()
                if not wtxt:
                    continue
                words.append({
                    "word": wtxt,
                    "start": float(w.start),
                    "end": float(w.end),
                })
        segments.append({
            "start": float(seg.start),
            "end": float(seg.end),
            "text": seg.text.strip(),
            "words": words,
        })

    return {"segments": segments}


