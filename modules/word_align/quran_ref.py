import json
import os
import re
from pathlib import Path

import requests

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "api"


IMLAEI_NORMALIZE = str.maketrans({
    0x0671: 0x0627,  # wasl alif -> regular alif
    0x0670: 0x0627,  # dagger alif (superscript) -> regular alif
    0x0640: None,     # tatweel/kashida -> remove
})


def _ensure_cache_dir():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def fetch_surah_names() -> dict[int, str]:
    resp = requests.get("https://api.quran.com/api/v4/chapters", timeout=10)
    resp.raise_for_status()
    return {ch["id"]: ch["name_simple"] for ch in resp.json()["chapters"]}


def fetch_ayah_text(surah: int, ayah: int) -> str:
    _ensure_cache_dir()
    cache_file = CACHE_DIR / f"{surah}_{ayah}.json"
    if cache_file.exists():
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        return data["verse"].get("text_imlaei") or data["verse"].get("text_uthmani", "")
    url = f"https://api.quran.com/api/v4/verses/by_key/{surah}:{ayah}?words=true&word_fields=text_uthmani,text_imlaei"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    cache_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data["verse"].get("text_imlaei") or data["verse"].get("text_uthmani", "")


def _normalize_imlaei(text: str) -> str:
    return text.translate(IMLAEI_NORMALIZE)


def _normalize_to_model_orthography(text: str) -> str:
    t = text.translate(IMLAEI_NORMALIZE)
    t = t.replace("\u064E\u0640", "")  # fatha + tatweel (sequence before dagger)
    t = t.replace("\u064E", "")  # remove standalone fatha before dagger-alif surrogate
    return t


def fetch_surah_words(surah: int) -> list[dict]:
    _ensure_cache_dir()
    words = []
    ayah_num = 1
    while True:
        cache_file = CACHE_DIR / f"{surah}_{ayah_num}.json"
        if cache_file.exists():
            data = json.loads(cache_file.read_text(encoding="utf-8"))
        else:
            try:
                url = f"https://api.quran.com/api/v4/verses/by_key/{surah}:{ayah_num}?words=true&word_fields=text_uthmani,text_imlaei"
                resp = requests.get(url, timeout=10)
                if resp.status_code != 200:
                    break
                data = resp.json()
                cache_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            except Exception:
                break
        verse = data["verse"]
        for w in verse.get("words", []):
            raw_uthmani = w.get("text_uthmani", "")
            words.append(
                {
                    "surah": surah,
                    "ayah": ayah_num,
                    "word_id": w.get("id"),
                    "position": w.get("position"),
                    "text_uthmani": raw_uthmani,
                    "text_imlaei": _normalize_to_model_orthography(raw_uthmani),
                    "char_type": w.get("char_type_name", "word"),
                }
            )
        ayah_num += 1
    return words


def get_surah_word_sequence(surah: int) -> list[dict]:
    words = fetch_surah_words(surah)
    return [w for w in words if w["char_type"] == "word"]


def get_ayah_word_sequences(surah: int) -> dict[int, list[dict]]:
    words = fetch_surah_words(surah)
    ayah_map: dict[int, list[dict]] = {}
    for w in words:
        if w["char_type"] != "word":
            continue
        ayah_map.setdefault(w["ayah"], []).append(w)
    return ayah_map
