"""
Upload all surah files to Modal NFS — simple sequential loop.
Run: python upload_all.py
"""
import modal
from pathlib import Path

DIRS = dict(
    video=Path("/home/muhammed-emin-eser/desk/din/quran/maher_workaround/Quran_cropped/"),
    audio=Path("/home/muhammed-emin-eser/desk/din/quran/maher_playlist/maher_playlist/"),
    text=Path("/home/muhammed-emin-eser/desk/din/quran/eng_translation/chunked_translation/"),
)

nfs = modal.NetworkFileSystem.from_name("ayah-aligner-data", create_if_missing=True)

for sid in range(1, 115):
    video = next(DIRS["video"].glob(f"*({sid}) *"), None)
    audio = next(DIRS["audio"].glob(f"*({sid}) *"), None)
    text = next(DIRS["text"].glob(f"{sid}_*.txt"), None)
    if not video or not audio or not text:
        print(f"[{sid:>3}] SKIP")
        continue
    with open(video, "rb") as f:
        nfs.write_file(f"videos/{video.name}", f)
    with open(audio, "rb") as f:
        nfs.write_file(f"audios/{audio.name}", f)
    with open(text, "rb") as f:
        nfs.write_file(f"texts/{text.name}", f)
    print(f"[{sid:>3}] OK ({video.name[:30]}...)")

print("Done")
PYEOF
