#!/usr/bin/env python3
import argparse
import re
import subprocess
import sys
from pathlib import Path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_ass_time(t: str) -> float:
    h, m, s = t.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)


def format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def parse_events(ass_path: Path):
    events = []
    for line in ass_path.read_text(encoding='utf-8').splitlines():
        if not line.startswith('Dialogue:'):
            continue
        parts = line.split(',', 9)
        if len(parts) < 10:
            continue
        start_str = parts[1].strip()
        end_str = parts[2].strip()
        text = parts[9]
        start_sec = parse_ass_time(start_str)
        end_sec = parse_ass_time(end_str)
        events.append((start_sec, end_sec, text))
    return events


def text_to_label(text: str, idx: int) -> str:
    label = re.sub(r'\{[^}]*\}', '', text)
    label = re.sub(r'\\N', ' ', label)
    label = re.sub(r'[^a-zA-Z0-9_\-\s]', '', label).strip()
    label = re.sub(r'\s+', '_', label)[:60].strip()
    if not label:
        label = f"event_{idx:03d}"
    return label


def safe_ass_filter_arg(path: Path) -> str:
    p = path.as_posix()
    replacements = {
        "\\": "\\\\",
        ":": r"\:",
        " ": r"\ ",
        "[": r"\[",
        "]": r"\]",
        "(": r"\(",
        ")": r"\)",
        ",": r"\,",
        "'": r"\\'",
    }
    for target, repl in replacements.items():
        p = p.replace(target, repl)
    return p


def extract_frames(
    overlay_video: Path,
    ass_file: Path,
    events: list,
    dest_dir: Path,
    ffmpeg: str = "ffmpeg",
):
    out_dir = ensure_dir(dest_dir)
    ass_arg = f"ass={safe_ass_filter_arg(ass_file)}"

    for i, (start_sec, end_sec, text) in enumerate(events):
        midpoint = start_sec + (end_sec - start_sec) / 2.0
        ts = format_time(midpoint)
        label = text_to_label(text, i)
        filename = f"frame_{i+1:03d}_{label}.png"
        out_path = out_dir / filename

        cmd = [
            ffmpeg,
            "-y",
            "-i", str(overlay_video),
            "-vf", ass_arg,
            "-ss", ts,
            "-frames:v", "1",
            "-update", "1",
            str(out_path),
        ]

        print(f"[{i+1}/{len(events)}] {ts} {filename}")
        subprocess.run(cmd, check=True, capture_output=True)


def main():
    parser = argparse.ArgumentParser(
        description="Extract a frame at the midpoint of each ASS dialogue event"
    )
    parser.add_argument("--video-file", type=Path, required=True)
    parser.add_argument("--ass-file", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("output/frames"))
    parser.add_argument("--ffmpeg-bin", type=str, default="ffmpeg")
    args = parser.parse_args()

    if not args.video_file.exists():
        sys.exit(f"Fehler: Video existiert nicht: {args.video_file}")
    if not args.ass_file.exists():
        sys.exit(f"Fehler: ASS-Datei existiert nicht: {args.ass_file}")

    events = parse_events(args.ass_file)
    if not events:
        sys.exit("Fehler: Keine Dialogue-Events in der ASS-Datei gefunden.")

    print(f"Events gefunden: {len(events)}")
    extract_frames(
        overlay_video=args.video_file,
        ass_file=args.ass_file,
        events=events,
        dest_dir=args.out_dir.resolve(),
        ffmpeg=args.ffmpeg_bin,
    )
    print(f"Fertig: {len(events)} Frames -> {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
