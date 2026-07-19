import os
import sys
import subprocess
import asyncio
from pathlib import Path
import modal

import argparse
from modal import Function

# --- Konfiguration ---
# Die lokalen Pfade, wo die Quelldateien liegen
LOCAL_TRANSLATION_DIR = Path("/home/muhammed-emin-eser/desk/din/quran/eng_translation/chunked_translation/")
LOCAL_VIDEO_DIR = Path("/home/muhammed-emin-eser/desk/din/quran/maher_workaround/Quran_cropped/")
LOCAL_AUDIO_DIR = Path("/home/muhammed-emin-eser/desk/din/quran/maher_playlist/maher_playlist/")
LOCAL_OVERLAY_VIDEO_DIR = Path("/home/muhammed-emin-eser/desk/din/quran/maher_workaround/with_overlay/")
BASE_DIR = Path(__file__).parent

# Pfade auf dem Modal Netzlaufwerk (nur für Output)
REMOTE_BASE_DIR = Path("/remote_data")

# --- Hilfsfunktionen ---

def find_local_file(directory: Path, pattern: str) -> Path:
    """Sucht eine Datei in einem lokalen Verzeichnis."""
    try:
        return next(directory.glob(pattern))
    except StopIteration:
        raise FileNotFoundError(f"Keine Datei für Pattern '{pattern}' in {directory} gefunden.")

async def run_ass(mapping_path, video_path, output_path):
    process = await asyncio.create_subprocess_exec(
        sys.executable, "modules/text_ass.py",
        str(mapping_path), str(video_path), str(output_path)
    )
    await process.wait()
    return process

async def run_burn_subs(video_path, ass_path, audio_dir, out_dir):
    process = await asyncio.create_subprocess_exec(
        sys.executable, "burn_subtitles.py",
        "--video-file", str(video_path),
        "--ass-file", str(ass_path),
        "--audio-dir", str(audio_dir),
        "--out-dir", str(out_dir)
    )
    await process.wait()
    return process

# --- Haupt-Pipeline ---

async def process_pipeline(surah_id: int, mapping_func, args):
    """Führt die gesamte Pipeline für eine Sure aus."""
    
    run_all = not (args.run_mapping or args.run_ass or args.run_burn)
    local_mapping_path = None
    ass_path = None

    # 1. Mapping-Prozess
    if run_all or args.run_mapping:
        print(f"[{surah_id}] Starte Mapping-Job auf Modal...")
        call = await mapping_func.spawn.aio(surah_id)
        try:
            result_path = await call.get.aio()
            print(f"[{surah_id}] Mapping-Job beendet. Output: {result_path}")
        except Exception as e:
            print(f"[{surah_id}] Fehler während des Modal-Jobs: {e}", file=sys.stderr)
            return

        local_mapping_dir = BASE_DIR / "output" / "mapping"
        local_mapping_dir.mkdir(parents=True, exist_ok=True)
        nfs = modal.NetworkFileSystem.from_name("ayah-aligner-data")
        
        # result_path kommt als Modal-Mount-Pfad (/remote_data/...),
        # für die lokale NFS-API brauchen wir den Pfad relativ zum NFS-Root
        result_rel_path = str(Path(result_path).relative_to(REMOTE_BASE_DIR))
        remote_mapping_path_obj = Path(result_path)
        local_mapping_path = local_mapping_dir / remote_mapping_path_obj.name
        
        print(f"[{surah_id}] Lade Mapping-Datei herunter nach {local_mapping_path}...")
        chunks = nfs.read_file(result_rel_path)
        data = b"".join(chunks)
        local_mapping_path.write_bytes(data)
    else:
        print(f"[{surah_id}] Überspringe Mapping-Job.")
        try:
            translation_file = find_local_file(LOCAL_TRANSLATION_DIR, f"{surah_id}_*.txt")
            local_mapping_path = find_local_file(BASE_DIR / "output" / "mapping", f"{translation_file.stem}.mapping")
            print(f"[{surah_id}] Verwende existierende Mapping-Datei: {local_mapping_path}")
        except FileNotFoundError as e:
            print(f"[{surah_id}] Fehler: Keine Mapping-Datei für die nächsten Schritte gefunden: {e}", file=sys.stderr)
            return

    # 2. ASS-Generierung
    if run_all or args.run_ass:
        print(f"[{surah_id}] Starte ASS-Generierung...")
        try:
            overlay_video = find_local_file(LOCAL_OVERLAY_VIDEO_DIR, f"*({surah_id})*")
        except FileNotFoundError as e:
            print(f"[{surah_id}] Fehler: Overlay-Video nicht gefunden: {e}", file=sys.stderr)
            return

        ass_output_dir = BASE_DIR / "output" / "ass"
        ass_output_dir.mkdir(parents=True, exist_ok=True)
        ass_path = ass_output_dir / f"{local_mapping_path.stem}.ass"

        p_ass = await run_ass(local_mapping_path, overlay_video, ass_path)
        if p_ass.returncode != 0:
            print(f"[{surah_id}] Fehler bei der ASS-Generierung.", file=sys.stderr)
            return
        print(f"[{surah_id}] ASS-Datei erstellt: {ass_path}")
    elif not run_all and args.run_burn: # Wenn nur burn läuft, brauchen wir den ass path
        print(f"[{surah_id}] Überspringe ASS-Generierung.")
        try:
            translation_file = find_local_file(LOCAL_TRANSLATION_DIR, f"{surah_id}_*.txt")
            ass_path = find_local_file(BASE_DIR / "output" / "ass", f"{translation_file.stem}.ass")
            print(f"[{surah_id}] Verwende existierende ASS-Datei: {ass_path}")
        except FileNotFoundError as e:
            print(f"[{surah_id}] Fehler: Keine ASS-Datei für den Burn-Schritt gefunden: {e}", file=sys.stderr)
            return


    # 3. Untertitel einbrennen
    if run_all or args.run_burn:
        print(f"[{surah_id}] Starte Einbrennen der Untertitel...")
        if ass_path is None:
             print(f"[{surah_id}] Fehler: Keine ASS-Datei zum Einbrennen vorhanden.", file=sys.stderr)
             return
        try:
            overlay_video = find_local_file(LOCAL_OVERLAY_VIDEO_DIR, f"*({surah_id})*")
        except FileNotFoundError as e:
            print(f"[{surah_id}] Fehler: Overlay-Video nicht gefunden: {e}", file=sys.stderr)
            return

        final_output_dir = BASE_DIR / "output" / "final"
        p_burn = await run_burn_subs(overlay_video, ass_path, LOCAL_AUDIO_DIR, final_output_dir)
        if p_burn.returncode != 0:
            print(f"[{surah_id}] Fehler beim Einbrennen der Untertitel.", file=sys.stderr)
            return
        print(f"[{surah_id}] Video erfolgreich erstellt.")
    
    print(f"[{surah_id}] Pipeline-Durchlauf beendet.")


MAX_CONCURRENT = 8  # Maximale parallele Suren-Verarbeitung

async def main():
    parser = argparse.ArgumentParser(description="Führt die Ayah-Aligner-Pipeline für einen Bereich von Suren aus.")
    parser.add_argument("start_id", type=int, help="Start-ID der Sure.")
    parser.add_argument("end_id", type=int, help="End-ID der Sure (exklusiv).")
    parser.add_argument("--run-mapping", action="store_true", help="Führt nur den Mapping-Prozess aus.")
    parser.add_argument("--run-ass", action="store_true", help="Führt nur die ASS-Generierung aus.")
    parser.add_argument("--run-burn", action="store_true", help="Führt nur das Einbrennen der Untertitel aus.")
    args = parser.parse_args()

    # Deployed Modal-Function referenzieren
    mapping_func = Function.from_name("ayah-aligner-mapping", "run_mapping_on_modal")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async def run_with_semaphore(surah_id):
        async with semaphore:
            await process_pipeline(surah_id, mapping_func, args)

    tasks = [run_with_semaphore(i) for i in range(args.start_id, args.end_id)]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    # Nötig für Windows, unter Linux/macOS kann direkt asyncio.run(main()) verwendet werden
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main())

