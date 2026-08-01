#!/usr/bin/env python3
import json
import re
import sys
import time
from pathlib import Path
from typing import List, Tuple

from sentence_transformers import SentenceTransformer, util
import torch

from modules.semanticmatch import (
    _fetch_verse_words, _get_stems, _normalize_arabic, _fill_gaps,
    TextSpan, MatchResult,
)
from difflib import SequenceMatcher

WORD_MATCH_TOLERANCE = 0.7
MODEL_NAME = "BAAI/bge-large-en-v1.5"

model = None


def get_model():
    global model
    if model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(MODEL_NAME, device=device)
    return model


def _get_translation_text(w: dict) -> str:
    raw = w.get("translation", "")
    if isinstance(raw, dict):
        return raw.get("text", "") or ""
    return str(raw) if raw else ""


def _word_to_translation_from(arabic_word: str, verse_words: list, min_idx: int = 0) -> tuple[str, int]:
    chunk_stems = _get_stems(arabic_word)
    norm_word = _normalize_arabic(arabic_word)
    for idx in range(min_idx, len(verse_words)):
        w = verse_words[idx]
        verse_stems = _get_stems(w["text_uthmani"])
        if chunk_stems & verse_stems:
            norm_verse = _normalize_arabic(w["text_uthmani"])
            shorter = min(len(norm_word), len(norm_verse))
            if shorter > 0:
                match_len = SequenceMatcher(None, norm_word, norm_verse).find_longest_match(0, len(norm_word), 0, len(norm_verse)).size
                if match_len / shorter < 0.7:
                    continue
            text = _get_translation_text(w)
            return (text.strip(), idx)
    norm_word = _normalize_arabic(arabic_word)
    best_score = 0.0
    best_idx = -1
    best_match = None
    for idx in range(min_idx, len(verse_words)):
        w = verse_words[idx]
        norm_verse = _normalize_arabic(w["text_uthmani"])
        shorter = min(len(norm_word), len(norm_verse))
        if shorter > 0:
            sm = SequenceMatcher(None, norm_word, norm_verse)
            match_len = sm.find_longest_match(0, len(norm_word), 0, len(norm_verse)).size
            if match_len / shorter < 0.7:
                continue
            score = sm.ratio()
        else:
            score = 0.0
        if score > best_score:
            best_score = score
            best_idx = idx
            best_match = w
    if best_score < WORD_MATCH_TOLERANCE or best_match is None:
        return ("", -1)
    text = _get_translation_text(best_match)
    return (text.strip(), best_idx)


def load_chunked_translation(surah: int, translation_dir: Path) -> dict[int, str]:
    files = sorted(translation_dir.iterdir())
    for f in files:
        stem = f.stem
        num_part = stem.split("_")[0]
        if num_part.isdigit() and int(num_part) == surah:
            break
    else:
        raise FileNotFoundError(f"Keine chunked translation für Surah {surah}")
    text = f.read_text(encoding="utf-8")
    verses = {}
    current_ayah = None
    current_lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^(\d+)\.\s*(.*)", line)
        if m:
            if current_ayah is not None:
                verses[current_ayah] = " ".join(current_lines)
            current_ayah = int(m.group(1))
            current_lines = [m.group(2)]
        elif current_ayah is not None:
            current_lines.append(line)
    if current_ayah is not None:
        verses[current_ayah] = " ".join(current_lines)
    return verses


class _MockChunk:
    def __init__(self, raw_text: str):
        self.window = type("W", (), {"start_sec": 0, "end_sec": 0})()
        self.segments = []
        self.raw_text = raw_text


def _tokens_to_span(start_tok: int, end_tok: int, words: list) -> tuple[int, int, str]:
    if start_tok >= len(words):
        return 0, 0, ""
    text = " ".join(words[start_tok:end_tok])
    pre_len = sum(len(w) + 1 for w in words[:start_tok]) if start_tok > 0 else 0
    return pre_len, pre_len + len(text), text


def align_ayah(api_engs: list[str], verse_text: str) -> list[str]:
    if not api_engs or not verse_text:
        return api_engs

    m = get_model()
    words = verse_text.split()
    word_embs = m.encode(words, convert_to_tensor=True)

    cursor = 0
    token_spans = []

    for eng in api_engs:
        if not eng:
            cursor = min(cursor, len(words) - 1) if len(words) > 0 else cursor
            token_spans.append((cursor, cursor + 1))
            cursor += 1
            continue

        if cursor >= len(words):
            token_spans.append((len(words) - 1, len(words)))
            cursor = len(words)
            continue

        qe = m.encode(eng, convert_to_tensor=True)
        nq = max(1, len(eng.split()))
        best_score = -1.0
        best_st = cursor
        best_en = min(cursor + 1, len(words))

        search_end = min(len(words), cursor + nq * 3)
        max_window = min(nq * 2 + 1, len(words) - cursor)
        for size in range(1, max_window + 1):
            for st in range(cursor, min(search_end - size + 1, len(words))):
                en = st + size
                sim = util.cos_sim(qe, word_embs[st:en].mean(dim=0)).item()
                sim *= min(1.0, (nq + 1) / size)
                if sim > best_score:
                    best_score = sim
                    best_st = st
                    best_en = en

        token_spans.append((best_st, best_en))
        cursor = best_en

    # In MatchResult umwandeln (wortgenaue Character-Positionen)
    results: List[MatchResult] = []
    for (st, en), eng in zip(token_spans, api_engs):
        cs, ce, text = _tokens_to_span(st, en, words)
        results.append(MatchResult(
            chunk=_MockChunk(eng),
            arabic_text=eng,
            span=TextSpan(start=cs, end=ce, text=text),
            score=0.0,
        ))

    if results:
        _fill_gaps(results, verse_text)

    return [r.span.text for r in results]


def main():
    word_data_path = Path("output/mapping/98_Al-Bayyinah_word.word_data.json")
    chunked_dir = Path("/home/muhammed-emin-eser/desk/din/quran/eng_translation/chunked_translation")

    with open(word_data_path, encoding="utf-8") as f:
        word_data = json.load(f)

    first_entry = next(iter(word_data.values()))[0]
    surah = first_entry["surah"]
    verses = load_chunked_translation(surah, chunked_dir)

    print(f"Lade {MODEL_NAME} ...")
    t0 = time.time()
    _ = get_model()
    print(f"  Geladen in {time.time()-t0:.1f}s")

    result = {}
    for ayah_key, words in sorted(word_data.items(), key=lambda x: int(x[0])):
        ayah_num = int(ayah_key)
        verse_text = verses.get(ayah_num, "")
        api_words = _fetch_verse_words(surah, ayah_num)
        api_cursor = 0

        api_engs = []
        for w in words:
            eng, idx = _word_to_translation_from(w["text_uthmani"], api_words, api_cursor)
            if eng and idx >= 0:
                api_cursor = idx + 1
            api_engs.append(eng)

        ayah_t0 = time.time()
        aligned = align_ayah(api_engs, verse_text) if verse_text else api_engs
        print(f"  Ayah {ayah_key}: {len(words)} words in {time.time()-ayah_t0:.1f}s")

        result[ayah_key] = [
            {
                "surah": w["surah"],
                "ayah": w["ayah"],
                "word_index": w["word_index"],
                "english": eng.rstrip(),
                "start_s": w["start_s"],
                "end_s": w["end_s"],
                "confidence": w["confidence"],
            }
            for w, eng in zip(words, aligned)
        ]

    out_path = word_data_path.with_name(
        word_data_path.stem.replace("word_data", "word_data_english") + ".json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nGeschrieben: {out_path}")


if __name__ == "__main__":
    main()
