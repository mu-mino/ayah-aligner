"""
Forced Alignment via whisper-timestamped für präzise arabische Word-Timestamps.
"""

import whisper_timestamped as whisper
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class AlignedWord:
    word: str
    start: float
    end: float
    score: float

class ForcedAligner:
    def __init__(self, model_name: str = "large-v2", device: Optional[str] = None):
        import torch
        self.model_name = model_name
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
        
        try:
            audio = whisper.load_audio(str(audio_path))
            
            result = whisper.transcribe_timestamped(
                self._model,
                audio,
                language="ar",
                remove_punctuation_from_words=True,
                compute_word_confidence=True,
            )
            
            aligned_words = []
            for segment in result.get("segments", []):
                for word in segment.get("words", []):
                    start, end = word['start'], word['end']
                    
                    if start >= window_start and (window_end is None or end <= window_end):
                        aligned_words.append(
                            AlignedWord(
                                word=word["text"],
                                start=start,
                                end=end,
                                score=word["confidence"]
                            )
                        )
            
            return aligned_words

        except Exception as e:
            print(f"[ForcedAligner] Alignment with whisper-timestamped failed: {e}")
            return []
