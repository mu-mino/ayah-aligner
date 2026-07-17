"""
Quran-Align integration module.

Uses pre-computed word-level alignment data from cpfair/quran-align
to produce word timestamps.

Data covers 12 reciters.  The C++ aligner (bin/quran-align) is also
available for future use should the full training pipeline become
available.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "quran_align"

_LOADED_DATA: Dict[str, dict] = {}


def get_available_reciters() -> List[str]:
    return sorted(f.stem for f in DATA_DIR.glob("*.json"))


def load_reciter_data(reciter: str) -> dict:
    if reciter in _LOADED_DATA:
        return _LOADED_DATA[reciter]

    path = DATA_DIR / f"{reciter}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No quran-align data for reciter '{reciter}'. "
            f"Available: {get_available_reciters()}"
        )

    raw = json.loads(path.read_text(encoding="utf-8"))

    indexed: Dict[Tuple[int, int], dict] = {}
    for entry in raw:
        indexed[(entry["surah"], entry["ayah"])] = entry

    _LOADED_DATA[reciter] = indexed
    return indexed


def pick_closest_reciter(name_hint: str) -> Optional[str]:
    """
    Best-effort matching: if no reciter is specified, pick one whose
    name appears in *name_hint* (case-insensitive), or fall back to
    the first available reciter.
    """
    all_r = get_available_reciters()
    if not all_r:
        return None

    if not name_hint:
        return all_r[0]

    hint_lower = name_hint.lower()
    for r in all_r:
        if hint_lower in r.lower():
            return r
    return all_r[0]


def get_ayah_duration_ms(data: dict, surah: int, ayah: int) -> Optional[int]:
    entry = data.get((surah, ayah))
    if not entry or not entry.get("segments"):
        return None
    return entry["segments"][-1][3]


def get_word_alignments(
    surah: int,
    ayah: int,
    reciter: str,
    window_start: float,
    verse_words: list,
) -> List[dict]:
    """
    Returns per-word timestamps for *ayah* using pre-computed data.

    Parameters
    ----------
    surah, ayah  : surah / verse number
    reciter      : reciter name (must have cached data)
    window_start : absolute start time (seconds) of the containing window
    verse_words  : list of dict from _fetch_verse_words (needed for ar/en)

    Returns a list of dicts compatible with the word_align.json format.
    """
    data = load_reciter_data(reciter)
    entry = data.get((surah, ayah))

    if not entry or not entry.get("segments"):
        return []

    segments = entry["segments"]
    result = []

    for seg_start_idx, seg_end_idx, start_msec, end_msec in segments:
        n_words = seg_end_idx - seg_start_idx
        dur_ms = end_msec - start_msec
        for i in range(n_words):
            w_idx = seg_start_idx + i
            w_start_abs = window_start + (start_msec + (i / n_words) * dur_ms) / 1000.0
            w_end_abs = window_start + (start_msec + ((i + 1) / n_words) * dur_ms) / 1000.0

            ar_text = ""
            en_text = ""
            if w_idx < len(verse_words):
                ar_text = verse_words[w_idx].get("text_uthmani", "")
                t = verse_words[w_idx].get("translation", "")
                en_text = t if isinstance(t, str) else ""

            result.append({
                "start": round(w_start_abs, 4),
                "end": round(w_end_abs, 4),
                "ar": ar_text,
                "en": en_text,
                "idx": w_idx,
                "ayah": ayah,
                "score": 1.0,
            })

    return result


def get_alignments_for_verses(
    surah: int,
    verses: List[Tuple[int, list]],
    reciter: str,
    window_start: float,
) -> List[dict]:
    """
    Chain multiple verses together sequentially.

    *verses* is a list of (ayah_number, verse_words_list).
    Each ayah is placed right after the previous one using
    the quran-align duration as the spacing.
    """
    data = load_reciter_data(reciter)
    all_aligns: List[dict] = []
    cursor = window_start

    for ayah_num, verse_words in verses:
        entry = data.get((surah, ayah_num))
        if not entry or not entry.get("segments"):
            cursor += 3.0
            continue

        aligns = get_word_alignments(surah, ayah_num, reciter, cursor, verse_words)
        all_aligns.extend(aligns)

        ayah_dur = get_ayah_duration_ms(data, surah, ayah_num)
        if ayah_dur is not None:
            cursor += ayah_dur / 1000.0
        elif aligns:
            cursor = aligns[-1]["end"]

    return all_aligns
