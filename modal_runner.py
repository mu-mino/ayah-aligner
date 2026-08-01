from modal import App, Image, NetworkFileSystem, FilePatternMatcher
import sys
from pathlib import Path

REMOTE_BASE_DIR = Path("/remote_data")

# Lokale Quellpfade (werden beim Modal deploy/client-run aufgelöst)
LOCAL_VIDEO_DIR = Path("/home/muhammed-emin-eser/desk/din/quran/maher_workaround/Quran_cropped/")
LOCAL_AUDIO_DIR = Path("/home/muhammed-emin-eser/desk/din/quran/maher_playlist/maher_playlist/")
LOCAL_TRANSLATION_DIR = Path("/home/muhammed-emin-eser/desk/din/quran/eng_translation/chunked_translation/")

image = (
    Image.debian_slim()
    .apt_install("ffmpeg")
    .pip_install([
        "numpy", "opencv-python-headless", "torch",
        "whisperx==3.8.6", "faster-whisper==1.2.1", "ctranslate2==4.7.2",
        "sentence-transformers", "spacy", "regex", "camel-tools",
    ])
    .run_commands(
        # camel_tools Morphologie-Datenbank installieren + hardcodierten Pfad bereitstellen
        "camel_data -i morphology-db-msa-r13",
        "mkdir -p /home/muhammed-emin-eser/.camel_tools/data/morphology_db",
        "ln -s /root/.camel_tools/data/morphology_db/calima-msa-r13 "
        "/home/muhammed-emin-eser/.camel_tools/data/morphology_db/calima-msa-r13",
    )
    .add_local_dir(
        ".",
        "/root",
        ignore=~FilePatternMatcher("mapping.py", "modules/**"),
    )
    # Quellverzeichnisse lokal einbinden (copy=False → kein Einbau ins Image,
    # sondern synct beim Container-Start, Content-hash-gecached)
    .add_local_dir(str(LOCAL_VIDEO_DIR), "/mnt/videos", copy=False)
    .add_local_dir(str(LOCAL_AUDIO_DIR), "/mnt/audios", copy=False)
    .add_local_dir(str(LOCAL_TRANSLATION_DIR), "/mnt/texts", copy=False)
)
app = App(
    "ayah-aligner-mapping",
    image=image,
)

# Erstelle ein geteiltes Netzlaufwerk für den Dateiaustausch
shared_volume = NetworkFileSystem.from_name("ayah-aligner-data", create_if_missing=True)


@app.function(
    network_file_systems={str(REMOTE_BASE_DIR): shared_volume},
    gpu="any",  # WhisperX benötigt eine GPU
    timeout=1800,  # 30 Minuten Timeout
)
def run_mapping_on_modal(surah_id: int):
    """
    Diese Funktion läuft in der Cloud auf Modal.
    Sie führt die Mapping-Logik für eine einzelne Sure aus.
    """
    # PyTorch 2.6 weights_only Fix: pyannote/VAD Model-Checkpoints enthalten
    # omegaconf-Klassen, die von weights_only=True blockiert werden.
    import torch
    _orig_torch_load = torch.load
    torch.load = lambda f, *a, **kw: _orig_torch_load(f, *a, **{**kw, 'weights_only': False})

    from mapping import run as run_mapping_logic

    print(f"Modal job running for Surah {surah_id}...")

    # Pfade für Input (von lokal gemounteten Verzeichnissen)
    video_dir = Path("/mnt/videos")
    audio_dir = Path("/mnt/audios")
    text_dir = Path("/mnt/texts")

    # Output-Verzeichnis auf dem Netzwerklaufwerk (damit wir Ergebnisse herunterladen können)
    mapping_dir = REMOTE_BASE_DIR / "mappings"
    mapping_dir.mkdir(parents=True, exist_ok=True)

    # Finde die richtigen Dateien
    try:
        video_path = next(video_dir.glob(f"*({surah_id}) *"))
        audio_path = next(audio_dir.glob(f"*({surah_id}) *"))
        text_path = next(text_dir.glob(f"{surah_id}_*.txt"))
    except StopIteration as e:
        raise FileNotFoundError(
            f"Konnte nicht alle Input-Dateien für Surah {surah_id} finden. "
            f"videos={list(video_dir.iterdir()) if video_dir.exists() else 'dir missing'}, "
            f"audios={list(audio_dir.iterdir()) if audio_dir.exists() else 'dir missing'}, "
            f"texts={list(text_dir.iterdir()) if text_dir.exists() else 'dir missing'}"
        ) from e

    mapping_path = mapping_dir / f"{text_path.stem}.mapping"

    print(f"  Video: {video_path}")
    print(f"  Audio: {audio_path}")
    print(f"  Text: {text_path}")
    print(f"  Output: {mapping_path}")

    # Führe die eigentliche Mapping-Logik aus
    run_mapping_logic(
        video_path=video_path,
        audio_path=audio_path,
        text_path=text_path,
        mapping_path=mapping_path,
        surah=surah_id,
        whisper_device="cuda",  # In Modal haben wir eine GPU
    )

    print(f"Modal job for Surah {surah_id} finished.")
    return str(mapping_path)

@app.local_entrypoint()
def main():
    print("This script is meant to be called from auto_range_runner.py")


if __name__ == "__main__":
    main()
