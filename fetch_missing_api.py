#!/usr/bin/env python3
"""Fetch missing quran.com API verse data for surahs without cached files."""
import json
import time
from pathlib import Path
from urllib.request import urlopen, Request

API_DIR = Path(__file__).parent / "data" / "api"
BASE_URL = "https://api.quran.com/api/v4/verses/by_chapter/{surah}?words=true&language=en&word_fields=text_uthmani,translation,transliteration&per_page=300"


def get_surah_verse_count(surah: int) -> int:
    counts = {
        1:7,2:286,3:200,4:176,5:120,6:165,7:206,8:75,9:129,10:109,
        11:123,12:111,13:43,14:52,15:99,16:128,17:111,18:110,19:98,
        20:135,21:112,22:78,23:118,24:64,25:77,26:227,27:93,28:88,
        29:69,30:60,31:34,32:30,33:73,34:54,35:45,36:83,37:182,
        38:88,39:75,40:85,41:54,42:53,43:89,44:59,45:37,46:35,
        47:38,48:29,49:18,50:45,51:60,52:49,53:62,54:55,55:78,
        56:96,57:29,58:22,59:24,60:13,61:14,62:11,63:11,64:18,
        65:12,66:12,67:30,68:52,69:52,70:44,71:28,72:28,73:20,
        74:56,75:40,76:31,77:50,78:40,79:46,80:42,81:29,82:19,
        83:36,84:25,85:22,86:17,87:19,88:26,89:30,90:20,91:15,
        92:21,93:11,94:8,95:8,96:19,97:5,98:8,99:8,100:11,
        101:11,102:8,103:3,104:9,105:5,106:4,107:7,108:3,109:6,
        110:3,111:5,112:4,113:5,114:6,
    }
    return counts.get(surah, 0)


def fetch_surah(surah: int) -> list:
    url = BASE_URL.format(surah=surah)
    req = Request(url, headers={"User-Agent": "ayah-aligner/1.0"})
    with urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())
    verses = data.get("verses", [])
    return verses


def save_verse(surah: int, verse_data: dict):
    ayah = verse_data["verse_number"]
    words = verse_data.get("words", [])
    out_data = []
    for w in words:
        out_data.append({
            "id": w.get("id"),
            "position": w.get("position"),
            "char_type_name": w.get("char_type_name"),
            "text_uthmani": w.get("text_uthmani", ""),
            "page_number": w.get("page_number"),
            "line_number": w.get("line_number"),
            "text": w.get("text", ""),
            "translation": w.get("translation"),
            "transliteration": w.get("transliteration"),
        })
    path = API_DIR / f"{surah}_{ayah}.json"
    path.write_text(json.dumps(out_data, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    API_DIR.mkdir(parents=True, exist_ok=True)
    missing = []
    for sid in range(1, 115):
        existing = list(API_DIR.glob(f"{sid}_*.json"))
        expected = get_surah_verse_count(sid)
        if len(existing) < expected:
            missing.append(sid)

    print(f"Fetching data for {len(missing)} surahs...")
    for sid in missing:
        print(f"  Surah {sid}...", end=" ", flush=True)
        try:
            verses = fetch_surah(sid)
            for v in verses:
                save_verse(sid, v)
            print(f"{len(verses)} verses saved")
            time.sleep(0.5)  # rate limit
        except Exception as e:
            print(f"ERROR: {e}")


if __name__ == "__main__":
    main()
