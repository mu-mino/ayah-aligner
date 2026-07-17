"""
Forced Alignment via whisper-timestamped für präzise arabische Word-Timestamps.
Unterstützt lokale GPU/CPU sowie Modal (serverless GPU).
"""

from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class AlignedWord:
    word: str
    start: float
    end: float
    score: float


_MODAL_FN = None

def _get_modal_fn():
    global _MODAL_FN
    if _MODAL_FN is None:
        from modal import Function
        _MODAL_FN = Function.from_name("whispe_rayah-aligner", "forced_align_audio")
    return _MODAL_FN


class ForcedAligner:
    def __init__(
        self,
        model_name: str = "large-v2",
        device: Optional[str] = None,
        use_modal: bool = False,
    ):
        self.use_modal = use_modal
        self._cached_audio_path = None
        self._cached_words: List[AlignedWord] = []

        if use_modal:
            print("[ForcedAligner] Using Modal (GPU serverless)")
        else:
            import torch
            import whisper_timestamped as whisper
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            print(f"[ForcedAligner] Loading model '{model_name}' on {self.device}...")
            self._model = whisper.load_model(model_name, device=self.device)
            print("[ForcedAligner] Model loaded.")

    def align(
        self,
        audio_path: Path,
        transcript: Optional[str] = None,
        window_start: float = 0.0,
        window_end: Optional[float] = None,
    ) -> List[AlignedWord]:
        if self.use_modal:
            all_words = self._transcribe_modal(audio_path, transcript=transcript)
        else:
            all_words = self._transcribe_local(audio_path, transcript=transcript)

        if window_end is None:
            return [w for w in all_words if w.start >= window_start]
        return [w for w in all_words if w.start >= window_start and w.end <= window_end]

    def _transcribe_modal(self, audio_path: Path, transcript: Optional[str] = None) -> List[AlignedWord]:
        if self._cached_audio_path == audio_path:
            return self._cached_words

        import numpy as np
        from modules.whispertranscribe import AUDIO_SAMPLE_RATE
        import whisper_timestamped as whisper

        audio = whisper.load_audio(str(audio_path))
        audio_bytes = audio.astype(np.float32).tobytes()

        modal_fn = _get_modal_fn()
        result = modal_fn.remote(audio_bytes, initial_prompt=transcript)

        self._cached_audio_path = audio_path
        self._cached_words = [
            AlignedWord(word=w["text"], start=w["start"], end=w["end"], score=w["confidence"])
            for w in result
        ]
        return self._cached_words

    def _transcribe_local(self, audio_path: Path, transcript: Optional[str] = None) -> List[AlignedWord]:
        import whisper_timestamped as whisper

        audio = whisper.load_audio(str(audio_path))
        result = whisper.transcribe_timestamped(
            self._model,
            audio,
            language="ar",
            initial_prompt=transcript,
            remove_punctuation_from_words=True,
            compute_word_confidence=True,
        )

        words = []
        for segment in result.get("segments", []):
            for word in segment.get("words", []):
                words.append(AlignedWord(
                    word=word["text"],
                    start=word["start"],
                    end=word["end"],
                    score=word["confidence"],
                ))
        return words
