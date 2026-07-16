import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def merge_alignments(
    whisper_path: Path,
    timething_path: Path,
    output_path: Path,
) -> None:
    whisper = json.loads(whisper_path.read_text(encoding="utf-8"))
    timething = json.loads(timething_path.read_text(encoding="utf-8"))

    whisper_idx = {(w["ayah"], w["idx"]): w for w in whisper}
    timething_idx = {(w["ayah"], w["idx"]): w for w in timething}

    all_keys = sorted(set(whisper_idx.keys()) | set(timething_idx.keys()))

    merged = []
    whisper_used = 0
    timething_fallback = 0

    for k in all_keys:
        if k in whisper_idx:
            entry = dict(whisper_idx[k])
            if not entry.get("en") and k in timething_idx:
                entry["en"] = timething_idx[k].get("en", "")
            merged.append(entry)
            whisper_used += 1
        else:
            merged.append(timething_idx[k])
            timething_fallback += 1

    merged.sort(key=lambda w: (w["ayah"], w["idx"]))

    output_path.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Merged: {len(merged)} words total")
    print(f"  Whisper primary: {whisper_used}")
    print(f"  Timething fallback: {timething_fallback}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--whisper", type=Path, default=BASE_DIR / "output" / "word_align_whisper.json")
    parser.add_argument("--timething", type=Path, default=BASE_DIR / "output" / "word_align_timething.json")
    parser.add_argument("--output", type=Path, default=BASE_DIR / "output" / "word_align.json")
    args = parser.parse_args()

    merge_alignments(args.whisper, args.timething, args.output)


if __name__ == "__main__":
    main()
