"""Patches .mapping files with word-level timestamps from Modal Whisper.

Produces:
  - *_word.mapping:  original format (HH:MM:SS integer timestamps)
  - *_word.data.json: companion JSON with per-word timestamps per ayah

Usage:
    python3 -m modules.word_align.patch_mapping output/mapping/98_*.mapping --surah 98
    python3 -m modules.word_align.patch_mapping --all
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_HERE))

from modules.word_align.pipeline import align_surah
from modules.word_align.quran_ref import fetch_surah_names

AUDIO_BASE = Path("/home/muhammed-emin-eser/desk/din/quran/maher_playlist/maher_playlist")
MAPPING_DIR = Path(_HERE) / "output" / "mapping"


def _parse_mapping(path: Path) -> list[dict]:
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(
            r"\[(\d+):(\d+):(\d+(?:\.\d+)?)\]\s*::\s*(\d+(?::\d+)?)\s*:\s*(.*)", line
        )
        if not m:
            continue
        h, mi, s, verse_id, text = m.groups()
        ts = int(h) * 3600 + int(mi) * 60 + float(s)
        entries.append({"timestamp_sec": ts, "verse_id": verse_id, "text": text.strip()})
    return entries


def _find_audio(surah: int) -> str | None:
    for f in AUDIO_BASE.iterdir():
        if f.suffix in (".flac", ".mp3", ".wav") and f"({surah})" in f.stem:
            return str(f)
    return None


def patch_mapping_file(
    mapping_path: str | Path, surah: int | None = None, force: bool = False
) -> Path:
    mp = Path(mapping_path)
    if not mp.exists():
        raise FileNotFoundError(f"Mapping not found: {mp}")
    if surah is None:
        m = re.match(r"(\d+)_", mp.stem)
        surah = int(m.group(1)) if m else None
    if surah is None or not (1 <= surah <= 114):
        raise ValueError(f"Invalid surah number from {mp.name}")

    audio_path = _find_audio(surah)
    if audio_path is None:
        raise FileNotFoundError(f"No audio for surah {surah} in {AUDIO_BASE}")

    entries = _parse_mapping(mp)
    print(f"  {mp.name}: {len(entries)} verses")

    aligned = align_surah(audio_path, surah, force=force)
    print(f"  {len(aligned)} words aligned")

    words_by_ayah: dict[str, list[dict]] = {}
    for w in aligned:
        words_by_ayah.setdefault(str(w["ayah"]), []).append(w)

    out_path = mp.with_name(mp.stem + "_word" + mp.suffix)
    lines = []
    for e in entries:
        ayah_key = e["verse_id"].split(":")[0]
        ts = e["timestamp_sec"]
        h, m = divmod(int(ts), 3600)
        mi, s = divmod(int(m), 60)

        lines.append(f"[{h:02d}:{mi:02d}:{s:02d}] :: {e['verse_id']}: {e['text']}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    data_path = out_path.with_suffix(".word_data.json")
    data_path.write_text(
        json.dumps(
            {str(k): v for k, v in words_by_ayah.items()},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )

    print(f"  Mapping: {out_path}")
    print(f"  Words:   {data_path}")
    return out_path


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("mapping", nargs="?", help="Single .mapping file")
    parser.add_argument("--surah", type=int)
    parser.add_argument("--all", action="store_true", help="Patch all mappings")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.all:
        files = sorted(MAPPING_DIR.glob("*_*.mapping"))
    elif args.mapping:
        files = [Path(args.mapping)]
    else:
        parser.print_help()
        return

    for fp in files:
        if fp.stem.endswith("_word"):
            continue
        try:
            patch_mapping_file(fp, surah=args.surah, force=args.force)
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    main()
