#!/usr/bin/env python3
"""
Verify alignment mappings using local Whisper transcription.

Uses exact FrameWindow boundaries from data/windows/{surah}.json to
extract audio spans. For each mapping entry, transcribes the circle_window
audio and compares with expected verse text from data/api/{surah}_{ayah}.json.

Strategy:
    - Start from the LAST entry: if correct, the whole surah is good.
    - If wrong, binary search backward to find the first drift point.
    - Report mismatched entries.
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import whisper_timestamped as wt

from modules.whispertranscribe import (
    AUDIO_SAMPLE_RATE,
    WHISPER_LANGUAGE,
)

LOCAL_AUDIO_DIR = Path("/home/muhammed-emin-eser/desk/din/quran/maher_playlist/maher_playlist/")
OUTPUT_DIR = Path(__file__).parent / "output" / "mapping"
API_DIR = Path(__file__).parent / "data" / "api"
WINDOWS_DIR = Path(__file__).parent / "data" / "windows"

_TASHKEEL = re.compile(r'[\u064b-\u065f]')     # 64B-65F = all tashkeel EXCEPT superscript alef
_SUPERSCRIPT_ALEF = re.compile(r'[\u0670]')    # U+0670 → replace with regular alef
_TATWEEL = re.compile(r'[\u0640]')
_ALEF_NORM = re.compile(r'[آأإٱ]')
_TEH_MARBUTA = re.compile(r'ة')
_ALIF_MAKSURA = re.compile(r'[ى]')
_MULTI_SPACE = re.compile(r'\s+')


_PREFIXES = {'و', 'ف', 'ب', 'ل', 'ك'}

def _strip_prefix(word: str) -> str:
    if len(word) > 1 and word[0] in _PREFIXES:
        return word[1:]
    return word

_ALEF_NO_STRIP = {'الله', 'اللهم', 'الآن', 'الذي', 'التي', 'الذين', 'اللائي', 'اللاتي'}
_TRAILING_ALEF = re.compile(r'[ا]$')

def normalize_arabic(text: str) -> str:
    text = _TASHKEEL.sub('', text)
    text = _SUPERSCRIPT_ALEF.sub('ا', text)
    text = _TATWEEL.sub('', text)
    text = _ALEF_NORM.sub('ا', text)
    text = _TEH_MARBUTA.sub('ه', text)
    text = _ALIF_MAKSURA.sub('ي', text)
    words = []
    for w in text.split():
        if not w:
            continue
        w = _strip_prefix(w)
        if w.startswith('ال') and w not in _ALEF_NO_STRIP:
            w = w[2:]
        w = _TRAILING_ALEF.sub('', w)
        if w:
            words.append(w)
    return " ".join(words)


_COMMON_AR = {'من', 'في', 'على', 'عن', 'مع', 'كان', 'له', 'هم', 'ما', 'لا', 'لم', 'هل', 'قد', 'إن', 'أن', 'إذا', 'إذ', 'أو', 'ثم'}

def arabic_similarity(transcribed: str, expected: str) -> float:
    a = normalize_arabic(transcribed).split()
    b = normalize_arabic(expected).split()
    if not a or not b:
        return 0.0
    set_a, set_b = set(a), set(b)
    intersection = set_a & set_b

    # overlap / max (standard) — penalizes partial captures
    score_max = len(intersection) / max(len(set_a), len(set_b))

    # subset check: if transcribed words (minus common words) are mostly in expected
    key_a = {w for w in set_a if w not in _COMMON_AR}
    if key_a and key_a.issubset(set_b):
        return max(score_max, 0.8)

    return score_max


def timestamp_to_seconds(ts: str) -> int:
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def parse_mapping(path: Path) -> List[Tuple[str, str]]:
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r'^\[(\d{2}:\d{2}:\d{2})\]\s*::\s*(.*)', line)
        if m:
            entries.append((m.group(1), m.group(2)))
    return entries


def extract_verse_numbers(content: str) -> List[int]:
    return [int(x) for x in re.findall(r'(?:^|\s)(\d+):', content)]


def is_title_entry(content: str) -> bool:
    return not re.search(r'(?:^|\s)(\d+):', content)


def get_expected_arabic(surah: int, ayah: int) -> str:
    path = API_DIR / f"{surah}_{ayah}.json"
    if not path.exists():
        return ""
    data = json.loads(path.read_text(encoding="utf-8"))
    words = [w["text_uthmani"] for w in data]
    return " ".join(words)


def get_expected_arabic_for_verses(surah: int, verse_numbers: List[int]) -> str:
    return " ".join(get_expected_arabic(surah, v) for v in verse_numbers)


def load_windows(surah: int) -> List[Tuple[float, float]]:
    path = WINDOWS_DIR / f"{surah}.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [(w[0], w[1]) for w in data["windows"]]


def match_windows_to_entries(
    entries: List[Tuple[str, str]],
    windows: List[Tuple[float, float]],
) -> List[Optional[Tuple[float, float]]]:
    """
    Match each mapping entry to the corresponding window.
    Returns list of (start_sec, end_sec) or None if no match.
    """
    result: List[Optional[Tuple[float, float]]] = []
    window_idx = 0
    for ts_str, content in entries:
        ts_sec = timestamp_to_seconds(ts_str)
        # find next window whose floored start_sec matches
        matched = None
        while window_idx < len(windows):
            w_start, w_end = windows[window_idx]
            if math.floor(w_start) == ts_sec:
                matched = (w_start, w_end)
                window_idx += 1
                break
            window_idx += 1
        result.append(matched)
    return result


class WhisperVerifier:
    def __init__(self, device: Optional[str] = None, use_modal: bool = False):
        self.use_modal = use_modal
        self.sample_rate = AUDIO_SAMPLE_RATE
        self.modal_fn = None
        self.model = None
        if use_modal:
            from modules.whispertranscribe import _get_modal_fn
            print("Using Modal GPU for Whisper transcription...", file=sys.stderr)
            self.modal_fn = _get_modal_fn()
        else:
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading Whisper large-v3 on {device}...", file=sys.stderr)
            self.model = wt.load_model("large-v3", device=device)
            self.device = device

    def transcribe(self, audio_path: Path, start_sec: float, end_sec: float) -> str:
        audio = wt.load_audio(str(audio_path))
        end_sec = max(end_sec, start_sec + 0.5)
        start_sample = int(start_sec * self.sample_rate)
        end_sample = int(end_sec * self.sample_rate)
        chunk = audio[start_sample:end_sample]

        if self.use_modal:
            result = self.modal_fn.remote(
                chunk.astype(np.float32).tobytes(),
                language=WHISPER_LANGUAGE,
            )
            texts = []
            for seg in result.get("segments", []):
                text = seg.get("text", "").strip()
                if text:
                    texts.append(text)
            return " ".join(texts)
        else:
            result = wt.transcribe(
                self.model, chunk, language="ar",
                beam_size=5, no_speech_threshold=1.0,
                initial_prompt="بسم الله الرحمن الرحيم",
            )
            texts = []
            for seg in result.get("segments", []):
                for word in seg.get("words", []):
                    text = word.get("text", "").strip()
                    if text:
                        texts.append(text)
            return " ".join(texts)

    def check_entry(
        self,
        audio_path: Path,
        surah: int,
        span_sec: Tuple[float, float],
        verse_numbers: List[int],
    ) -> Tuple[bool, float, str, str]:
        if not span_sec:
            return True, 1.0, "", ""

        start_sec, end_sec = span_sec
        if not verse_numbers:
            return True, 1.0, "", ""

        expected = get_expected_arabic_for_verses(surah, verse_numbers)
        if not expected:
            return True, 1.0, "", ""

        transcribed = self.transcribe(audio_path, start_sec, end_sec)
        score = arabic_similarity(transcribed, expected)

        ok = score >= 0.2
        return ok, score, transcribed, expected

    def _get_group_spans(
        self,
        entries: List[Tuple[str, str]],
        windows: List[Tuple[float, float]],
    ) -> List[Optional[Tuple[float, float]]]:
        """For each entry, compute full group span = circle window start → next circle window start."""
        matched = match_windows_to_entries(entries, windows)
        # get the actual matched window indices for non-None matches
        matched_indices = []
        wi = 0
        for m in matched:
            if m is not None:
                # find which window index this corresponds to
                while wi < len(windows):
                    ws, we = windows[wi]
                    if (ws, we) == m:
                        matched_indices.append(wi)
                        wi += 1
                        break
                    wi += 1
            else:
                matched_indices.append(-1)

        result: List[Optional[Tuple[float, float]]] = []
        for i, (ts, content) in enumerate(entries):
            if is_title_entry(content):
                result.append(matched[i])  # use circle window for title (if matched)
                continue
            if matched[i] is None:
                result.append(None)
                continue
            start_sec = matched[i][0]
            # find next matched window's start
            next_start = None
            for j in range(i + 1, len(entries)):
                if matched[j] is not None:
                    next_start = matched[j][0]
                    break
            if next_start is not None:
                end_sec = next_start
            else:
                # last entry: use last window's end_sec
                if matched_indices[i] >= 0 and matched_indices[i] < len(windows):
                    end_sec = windows[-1][1]
                else:
                    end_sec = matched[i][1]
            result.append((start_sec, end_sec))
        return result

    def verify_surah(
        self,
        entries: List[Tuple[str, str]],
        windows: List[Tuple[float, float]],
        audio_path: Path,
        surah: int,
        verbose: bool = True,
    ) -> Optional[int]:
        spans = self._get_group_spans(entries, windows)
        N = len(entries)

        if verbose:
            print(f"Verifying {N} entries for surah {surah}...")

        def check(i: int) -> bool:
            ts, content = entries[i]
            if is_title_entry(content):
                return True
            verses = extract_verse_numbers(content)
            ok, score, trans, exp = self.check_entry(
                audio_path, surah, spans[i], verses,
            )
            if verbose:
                span_desc = ""
                if spans[i]:
                    span_desc = f"[{spans[i][0]:.1f}-{spans[i][1]:.1f}s]"
                status = "OK" if ok else "MISMATCH"
                print(f"  {span_desc:25s} [{ts}] #{i} v{verses} score={score:.2f} {status}")
                if not ok and verbose > 1:
                    print(f"    whisper: {trans[:120]}")
                    print(f"    expected: {exp[:120]}")
            return ok

        # check last entry first
        if check(N - 1):
            if verbose:
                print(f"  ✓ Last entry OK → surah verified.")
            return None

        if verbose:
            print(f"  → Last entry bad, binary searching...")
        lo, hi = 0, N - 1
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if check(mid):
                lo = mid
            else:
                hi = mid

        if verbose:
            print(f"  ✗ First bad entry: #{hi} ({entries[hi][0]})")
        return hi


def parse_args():
    parser = argparse.ArgumentParser(description="Verify alignment with local Whisper")
    parser.add_argument("--surah", type=int, nargs="+", help="Surah number(s) to verify")
    parser.add_argument("--all", action="store_true", help="Verify all 114 surahs")
    parser.add_argument("--verbose", "-v", action="count", default=1)
    parser.add_argument("--device", default=None, help="cuda or cpu (local only)")
    parser.add_argument("--modal", action="store_true", help="Use Modal GPU for Whisper")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.all:
        surahs = list(range(1, 115))
    elif args.surah:
        surahs = args.surah
    else:
        print("Specify --surah N or --all")
        sys.exit(1)

    verifier = WhisperVerifier(device=args.device, use_modal=args.modal)
    results: List[Tuple[int, Optional[int], str]] = []

    for sid in surahs:
        mapping_files = list(OUTPUT_DIR.glob(f"{sid}_*.mapping"))
        if not mapping_files:
            results.append((sid, None, "no mapping"))
            continue
        mapping_path = mapping_files[0]
        audio_file = next(LOCAL_AUDIO_DIR.glob(f"*({sid}) *"), None)
        if not audio_file:
            results.append((sid, None, "no audio"))
            continue
        windows = load_windows(sid)
        if not windows:
            results.append((sid, None, "no windows"))
            continue

        print(f"\n{'='*60}")
        print(f"Surah {sid} ({mapping_path.stem})")
        print(f"{'='*60}")
        entries = parse_mapping(mapping_path)
        bad = verifier.verify_surah(entries, windows, audio_file, sid, verbose=args.verbose)
        if bad is None:
            results.append((sid, None, "OK"))
        else:
            results.append((sid, bad, "MISMATCH"))

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    ok_count = 0
    for sid, bad, status in results:
        if bad is not None:
            print(f"  {sid:3d}: {status} at entry #{bad}")
        else:
            print(f"  {sid:3d}: {status}")
            if status == "OK":
                ok_count += 1
    print(f"\nOK: {ok_count}/{len(results)}")


if __name__ == "__main__":
    main()
