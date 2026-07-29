"""Lightweight pipeline: WhisperX (Modal GPU) + local reference matching.

No local ML model needed — all GPU work is on Modal.
"""

import re

from modules.word_align.client import transcribe_surah
from modules.word_align.quran_ref import get_surah_word_sequence


def _norm(text: str) -> str:
    t = re.sub(r"[\u064B-\u0652\u0670]", "", text)
    t = t.replace("\u0671", "\u0627").replace("\u0640", "")
    t = t.replace("\u0623", "\u0627").replace("\u0625", "\u0627")
    t = t.replace("\u0622", "\u0627").replace("\u0624", "\u0648")
    t = t.replace("\u0626", "\u064A").replace("\u0649", "\u064A")
    return t


def _similar(a: str, b: str) -> float:
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return 0.0
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            tmp = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (0 if a[i - 1] == b[j - 1] else 1))
            prev = tmp
    return 1.0 - dp[m] / max(n, m)


def align_surah(audio_path: str, surah: int, force: bool = False) -> list[dict]:
    words_ref = get_surah_word_sequence(surah)
    words_ref = [w for w in words_ref if w.get("char_type", "word") == "word"]
    ref_texts = [w.get("text_imlaei", w["text_uthmani"]) for w in words_ref]

    whisper_words = transcribe_surah(audio_path, surah, force=force)
    if not whisper_words:
        return []

    whisper_texts = [w["text"] for w in whisper_words]

    n_w, n_r = len(whisper_texts), len(ref_texts)
    sim = [[_similar(whisper_texts[i], ref_texts[j]) for j in range(n_r)] for i in range(n_w)]

    MIN_SIM = 0.4
    dp = [[0.0] * (n_r + 1) for _ in range(n_w + 1)]
    for i in range(n_w + 1):
        dp[i][0] = i * -0.5
    for j in range(n_r + 1):
        dp[0][j] = j * -0.5
    for i in range(1, n_w + 1):
        for j in range(1, n_r + 1):
            match = sim[i - 1][j - 1]
            dp[i][j] = max(
                dp[i - 1][j - 1] + (match if match >= MIN_SIM else -1),
                dp[i - 1][j] - 0.5,
                dp[i][j - 1] - 0.5,
            )

    i, j = n_w, n_r
    aligned = []
    while i > 0 and j > 0:
        match = sim[i - 1][j - 1]
        if dp[i][j] == dp[i - 1][j - 1] + (match if match >= MIN_SIM else -1):
            aligned.append((i - 1, j - 1, match))
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j] - 0.5:
            i -= 1
        else:
            j -= 1
    aligned.reverse()

    result = []
    for wi, ri, score in aligned:
        w = words_ref[ri]
        ww = whisper_words[wi]
        result.append({
            "surah": w["surah"],
            "ayah": w["ayah"],
            "word_index": w["position"],
            "text_uthmani": w["text_uthmani"],
            "text_imlaei": w.get("text_imlaei", w["text_uthmani"]),
            "start_s": round(ww["start"], 3),
            "end_s": round(ww["end"], 3),
            "confidence": round(float(score), 3),
        })

    return result
