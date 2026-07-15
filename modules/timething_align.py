import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.utils.rnn as rnn
import torchaudio
from timething import align, text as timething_text, utils

BASE_DIR = Path(__file__).resolve().parent.parent

QURAN_MODEL = "rabah2026/wav2vec2-large-xlsr-53-arabic-quran-v_final"

NORMALIZE_MAP = {
    '\u0653': '\u0622',  # madd (ٓ) -> alif madd (آ)
    '\u0654': '\u0623',  # hamza above (ٔ) -> alif hamza above (أ)
    '\u0670': '\u0627',  # superscript alif (ٰ) -> alif (ا)
    '\u0671': '\u0627',  # wasla (ٱ) -> alif (ا)
}

QURAN_ANNOTATIONS = re.compile(
    '[\u06D6\u06D7\u06D8\u06D9\u06DA\u06DB\u06DC\u06DD\u06DE'
    '\u06DF\u06E0\u06E1\u06E2\u06E3\u06E4\u06E5\u06E6'
    '\u06E7\u06E8\u06E9\u06EA\u06EB\u06EC\u06ED]'
)


def normalize_uthmani(text: str) -> str:
    text = QURAN_ANNOTATIONS.sub('', text)
    for old, new in NORMALIZE_MAP.items():
        text = text.replace(old, new)
    return text


def arabic_text_allowed(text: str, allowed: set) -> bool:
    chars = set(text)
    chars.discard(' ')
    return chars.issubset(allowed)


def fetch_verse_words(surah: int, ayah: int) -> List[dict]:
    cache_path = BASE_DIR / "data" / "api" / f"{surah}_{ayah}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    import urllib.request
    url = f"https://api.quran.com/api/v4/verses/by_chapter/{surah}?language=en&words=true&page=1&per_page=300&word_fields=text_uthmani,translation"
    req = urllib.request.Request(url, headers={"User-Agent": "ayah-aligner/1.0"})
    resp = json.loads(urllib.request.urlopen(req).read())
    verses = resp.get("verses", [])

    by_ayah = {}
    for v in verses:
        ayah_num = v["verse_key"].split(":")[1]
        word_objs = []
        for w in v.get("words", []):
            word_objs.append({
                "id": w["id"],
                "position": w["position"],
                "char_type_name": w.get("char_type_name", "word"),
                "text_uthmani": w.get("text_uthmani", ""),
                "translation": w.get("translation", {}).get("text", ""),
            })
        by_ayah[int(ayah_num)] = word_objs

    for ayah_num, words in by_ayah.items():
        fname = BASE_DIR / "data" / "api" / f"{surah}_{ayah_num}.json"
        fname.parent.mkdir(parents=True, exist_ok=True)
        fname.write_text(json.dumps(words, ensure_ascii=False), encoding="utf-8")

    return by_ayah.get(ayah, [])


def build_word_alignments(surah: int, audio_path: Path) -> Tuple[List[dict], str]:
    all_words = []
    ayah = 1
    while True:
        words = fetch_verse_words(surah, ayah)
        if not words:
            break
        for w in words:
            if w["char_type_name"] == "word":
                all_words.append({
                    "ayah": ayah,
                    "idx": w["position"] - 1,
                    "ar": w["text_uthmani"],
                    "en": w.get("translation", "") if isinstance(w.get("translation"), str) else w.get("translation", {}).get("text", ""),
                })
        ayah += 1
        if ayah > 500:
            break

    print(f"Fetched {len(all_words)} words for surah {surah} ({ayah-1} verses)", file=sys.stderr)

    full_ar = " ".join(w["ar"] for w in all_words)
    full_en = " ".join(w["en"] for w in all_words)

    clean_ar = normalize_uthmani(full_ar)

    print(f"\n--- Audio ---", file=sys.stderr)
    print(f"Loading: {audio_path}", file=sys.stderr)
    audio, sr = torchaudio.load(str(audio_path))
    print(f"  Sample rate: {sr}, Duration: {audio.shape[-1]/sr:.2f}s", file=sys.stderr)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        audio = resampler(audio)
    audio = torch.mean(audio, 0, keepdim=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", file=sys.stderr)
    cfg = align.Config(
        hugging_model=QURAN_MODEL,
        hugging_pin="main",
        sampling_rate=16000,
        language="ar",
        k_shingles=5,
    )
    aligner = align.Aligner.build(device, cfg)

    vocab = aligner.vocab
    clean_ar_for_model = timething_text.TextCleaner("ar", vocab)(clean_ar)
    y_cleaned = clean_ar_for_model

    print(f"\nText stats:", file=sys.stderr)
    print(f"  Original: {full_ar[:200]}...", file=sys.stderr)
    print(f"  Normalized: {clean_ar[:200]}...", file=sys.stderr)
    print(f"  Cleaned: {y_cleaned[:200]}...", file=sys.stderr)

    xs = [audio]
    ys = [y_cleaned]
    ys_original = [full_ar]
    ids = [f"surah_{surah}"]

    xs_perm = [el.permute(1, 0) for el in xs]
    xs_padded = rnn.pad_sequence(xs_perm, batch_first=True)
    xs_padded = xs_padded.permute(0, 2, 1)

    batch = (xs_padded, ys, ys_original, ids)
    print("\nRunning alignment...", file=sys.stderr)
    alignments = aligner.align(batch)
    al = alignments[0]

    print(f"\nAlignment results:", file=sys.stderr)
    print(f"  Recognised: {al.recognised[:200]}...", file=sys.stderr)
    print(f"  Partition score: {al.partition_score:.4f}", file=sys.stderr)
    print(f"  Words (cleaned): {len(al.words_cleaned)}", file=sys.stderr)
    print(f"  Words (original): {len(al.words)}", file=sys.stderr)

    for i, w in enumerate(al.words_cleaned[:10]):
        start = al.model_frames_to_seconds(w.start)
        end = al.model_frames_to_seconds(w.end)
        print(f"  [{i}] '{w.label}': {start:.4f}-{end:.4f} (score={w.score:.4f})", file=sys.stderr)

    result = []
    # Map words to (ayah, idx) using original (uncleaned) text
    # The original text has words separated by spaces
    orig_words = full_ar.split()
    if len(al.words) != len(orig_words):
        print(f"\nWARNING: Word count mismatch! timething={len(al.words)}, expected={len(orig_words)}", file=sys.stderr)
    n = min(len(al.words), len(orig_words))
    for i in range(n):
        w = al.words[i]
        ar_word = orig_words[i]
        start_sec = al.model_frames_to_seconds(w.start)
        end_sec = al.model_frames_to_seconds(w.end)
        # Find which all_words entry matches
        matched = all_words[i]
        result.append({
            "start": round(start_sec, 4),
            "end": round(end_sec, 4),
            "ar": ar_word,
            "en": matched["en"],
            "idx": matched["idx"],
            "ayah": matched["ayah"],
            "score": round(w.score, 4),
        })

    return result, full_en


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--surah", type=int, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=BASE_DIR / "output" / "word_align.json")
    args = parser.parse_args()

    word_aligns, _ = build_word_alignments(args.surah, args.audio)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(word_aligns, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWritten {len(word_aligns)} alignments to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
