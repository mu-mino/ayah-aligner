#!/usr/bin/env python3
"""
Extract FrameWindow data for all surahs and save to data/windows/{surah_id}.json.
Runs in parallel for speed.
"""
import json
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent))

from modules.videowindow import extract_windows

LOCAL_VIDEO_DIR = Path("/home/muhammed-emin-eser/desk/din/quran/maher_workaround/Quran_cropped/")
WINDOWS_DIR = Path(__file__).parent / "data" / "windows"
WINDOWS_DIR.mkdir(parents=True, exist_ok=True)


def extract_one(surah_id: int) -> str:
    try:
        video_path = next(LOCAL_VIDEO_DIR.glob(f"*({surah_id}) *"))
    except StopIteration:
        return f"[{surah_id}] no video"
    try:
        windows = extract_windows(video_path)
    except Exception as e:
        return f"[{surah_id}] error: {e}"
    data = [[w.start_sec, w.end_sec] for w in windows]
    out_path = WINDOWS_DIR / f"{surah_id}.json"
    out_path.write_text(
        json.dumps({"surah": surah_id, "video": video_path.name, "windows": data}, indent=2),
        encoding="utf-8",
    )
    return f"[{surah_id}] {len(windows)} windows OK"


def main(start=1, end=115):
    print(f"Extracting windows for surahs {start}–{end-1} with 8 workers...")
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(extract_one, sid): sid for sid in range(start, end)}
        for fut in as_completed(futs):
            print(fut.result())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=115)
    args = parser.parse_args()
    main(args.start, args.end)
