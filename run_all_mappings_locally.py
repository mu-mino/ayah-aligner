import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

LOCAL_VIDEO_DIR = Path("/home/muhammed-emin-eser/desk/din/quran/maher_workaround/Quran_cropped/")
LOCAL_AUDIO_DIR = Path("/home/muhammed-emin-eser/desk/din/quran/maher_playlist/maher_playlist/")
LOCAL_TRANSLATION_DIR = Path("/home/muhammed-emin-eser/desk/din/quran/eng_translation/chunked_translation/")
OUTPUT_DIR = Path(__file__).parent / "output" / "mapping"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_WORKERS = 8

def run_one(surah_id: int) -> str:
    from mapping import run as run_mapping
    try:
        video_path = next(LOCAL_VIDEO_DIR.glob(f"*({surah_id}) *"))
        audio_path = next(LOCAL_AUDIO_DIR.glob(f"*({surah_id}) *"))
        text_path = next(LOCAL_TRANSLATION_DIR.glob(f"{surah_id}_*.txt"))
    except StopIteration as e:
        return f"[{surah_id}] FEHLER: Datei nicht gefunden: {e}"
    mapping_path = OUTPUT_DIR / f"{text_path.stem}.mapping"
    print(f"[{surah_id}] Starte Mapping...")
    t0 = time.time()
    try:
        run_mapping(
            video_path=video_path,
            audio_path=audio_path,
            text_path=text_path,
            mapping_path=mapping_path,
            surah=surah_id,
            whisper_device="cpu",
        )
        elapsed = time.time() - t0
        print(f"[{surah_id}] Fertig in {elapsed:.1f}s → {mapping_path.name}")
        return f"[{surah_id}] OK ({elapsed:.1f}s)"
    except Exception as e:
        elapsed = time.time() - t0
        import traceback
        return f"[{surah_id}] FEHLER nach {elapsed:.1f}s: {e}\n{traceback.format_exc()}"

def main(start=1, end=115):
    print(f"Starte Mapping für Suren {start}–{end-1} mit max {MAX_WORKERS} parallel...")
    t_start = time.time()
    results = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        fut_map = {pool.submit(run_one, sid): sid for sid in range(start, end)}
        for fut in as_completed(fut_map):
            sid = fut_map[fut]
            results[sid] = fut.result()
    total = time.time() - t_start
    ok = sum(1 for r in results.values() if r.startswith(f"[") and "OK" in r)
    fail = sum(1 for r in results.values() if "FEHLER" in r)
    print(f"\n=== Gesamt: {total:.0f}s, OK={ok}, Fehler={fail} ===")
    for sid in sorted(results):
        line = results[sid]
        if "FEHLER" in line:
            print(line)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("start", type=int, nargs="?", default=1)
    parser.add_argument("end", type=int, nargs="?", default=115)
    args = parser.parse_args()
    main(args.start, args.end)
