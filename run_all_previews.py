import subprocess
import sys
from pathlib import Path

MAPPING_DIR = Path(__file__).parent / "output" / "mapping"
script = Path(__file__).parent / "preview_frames.py"

mapping_files = sorted(MAPPING_DIR.glob("*.mapping"))

for mf in mapping_files:
    surah_id = mf.stem.split("_")[0]
    print(f"\n=== {surah_id}: {mf.name} ===")
    result = subprocess.run(
        [sys.executable, str(script), "--id", surah_id],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"[FEHLER] Exit {result.returncode}: {result.stderr}")
    else:
        print(result.stdout)
