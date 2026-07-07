#!/usr/bin/env python3
"""Convert .segments file (float seconds) → HH:MM:SS format for analysis."""
import json
import sys

def sec_to_hms(s: float) -> str:
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:05.2f}"

in_path = sys.argv[1]
out_path = sys.argv[2] if len(sys.argv) > 2 else in_path + ".hms.txt"

with open(in_path) as f:
    lines = [json.loads(l) for l in f if l.strip()]

with open(out_path, "w") as f:
    for entry in lines:
        start = sec_to_hms(entry["start"])
        end = sec_to_hms(entry["end"])
        text = entry.get("text", "")
        f.write(f"{start} -> {end} | {text}\n")

print(f"Wrote {len(lines)} lines to {out_path}")
