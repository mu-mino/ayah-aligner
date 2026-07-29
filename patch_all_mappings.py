#!/usr/bin/env python3
"""Batch-patch all .mapping files with word-level timestamps.

Usage:
    python3 patch_all_mappings.py [--surah N] [--force]

Without --surah, patches all mapping files found in output/mapping/.
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from modules.word_align.patch_mapping import patch_mapping_file, _surah_from_mapping_name, MAPPING_DIR


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--surah", type=int, help="Single surah to patch")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--audio-base", default=None)
    args = parser.parse_args()

    if args.surah:
        pattern = f"{args.surah:03d}_*.mapping"
        files = list(MAPPING_DIR.glob(pattern))
    else:
        files = sorted(MAPPING_DIR.glob("*_*.mapping"))

    raw_files = [f for f in files if not f.stem.endswith("_word")]
    raw_files.sort()

    if not raw_files:
        print(f"No .mapping files found in {MAPPING_DIR}")
        return

    print(f"Found {len(raw_files)} mapping files to process")
    for f in raw_files:
        surah = _surah_from_mapping_name(f.stem)
        if surah is None:
            print(f"  Skip {f.name}: cannot detect surah")
            continue
        if surah > 114:
            print(f"  Skip {f.name}: surah {surah} out of range")
            continue
        try:
            patch_mapping_file(
                str(f),
                surah=surah,
                audio_base=args.audio_base,
                force=args.force,
            )
        except Exception as e:
            print(f"  Error on {f.name}: {e}")


if __name__ == "__main__":
    main()
