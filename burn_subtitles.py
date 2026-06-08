#!/usr/bin/env python3
"""
Isolated Video Renderer:
Nimmt fertig generierte ASS-Untertitel, Overlay-Videos und Audio-Dateien
und brennt diese via FFmpeg zu einem finalen Video zusammen.
"""
import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".opus"}

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def _normalize_stem(name: str) -> str:
    """Hilfsfunktion für den flexiblen Namensvergleich (alles klein, ohne Sonderzeichen)."""
    import re
    return re.sub(r"[^0-9a-z]+", "", name.lower())

def video_id_from_path(path: Path) -> Optional[int]:
    """Extrahiert eine ID in Klammern (z.B. '74') aus dem Dateinamen."""
    import re
    m = re.search(r"\((\d+)\)", path.name)
    return int(m.group(1)) if m else None

def cleaned_output_stem(path: Path) -> str:
    """Liefert einen sauberen Dateinamen ohne Pfad und Endung."""
    # Falls das spezifische 'clean_output_stem' fehlt, nutzen wir den Standard-Stem
    return path.stem

def safe_ass_filter_arg(path: Path) -> str:
    """Escapt Sonderzeichen im ASS-Pfad, damit FFmpeg den Filter fehlerfrei liest."""
    p = path.as_posix()
    replacements = {
        "\\": "\\\\",  # Backslash zuerst escapen
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

def collect_audio_files(audio_dir: Path) -> List[Path]:
    """Sammelt alle unterstützten Audiodateien aus einem Verzeichnis."""
    return sorted(
        [p for p in audio_dir.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS],
        key=lambda p: p.name,
    )

def select_audio_for_video(video: Path, audio_candidates: List[Path]) -> Tuple[Optional[Path], Optional[str]]:
    """
    Findet die passende Audiodatei zum Video anhand der ID oder des Namens.
    Gibt (audio_path, warning_message) zurück.
    """
    if not audio_candidates:
        return None, "Keine Audio-Dateien im Audio-Ordner gefunden."

    vid_id = video_id_from_path(video)
    if vid_id is not None:
        id_matches = [p for p in audio_candidates if video_id_from_path(p) == vid_id]
        if len(id_matches) == 1:
            return id_matches[0], None
        if len(id_matches) > 1:
            return id_matches[0], f"Mehrere Audios mit ID ({vid_id}) gefunden; verwende {id_matches[0].name}."

    exact = [p for p in audio_candidates if p.stem == video.stem]
    if len(exact) == 1:
        return exact[0], None
    if len(exact) > 1:
        return exact[0], f"Mehrere Audios mit exakt gleichem Stem '{video.stem}' gefunden; verwende {exact[0].name}."

    video_norm = _normalize_stem(video.stem)
    norm_matches = [p for p in audio_candidates if _normalize_stem(p.stem) == video_norm]
    if len(norm_matches) == 1:
        return norm_matches[0], None
    if len(norm_matches) > 1:
        return norm_matches[0], f"Mehrere Audios passen nach Normalisierung zu '{video.stem}'; verwende {norm_matches[0].name}."

    return audio_candidates[0], f"Keine passende Audio-Datei für '{video.stem}' gefunden; Fallback: {audio_candidates[0].name}."

def burn_subs(
    overlay_video: Path, ass: Path, audio_file: Path, dest_dir: Path, ffmpeg: str = "ffmpeg"
) -> Path:
    """Führt Video, Audio und Untertitel per FFmpeg im Lossless-Modus zusammen."""
    path_original = ensure_dir(dest_dir / "original") / f"{cleaned_output_stem(overlay_video)}_original.mp4"
    ass_arg = f"ass={safe_ass_filter_arg(ass)}"
    
    cmd = [
        ffmpeg,
        "-y",
        "-i", str(overlay_video),
        "-i", str(audio_file),
        "-vf", ass_arg,
        "-c:v:0", "libx264",
        "-crf:v:0", "0",
        "-preset:v:0", "veryslow",
        "-c:a:0", "pcm_s16le",
        "-map", "0:v:0", "-map", "1:a:0",
        str(path_original),
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return path_original

def main():
    parser = argparse.ArgumentParser(description="Isoliertes Einbrennen von ASS-Untertiteln in Videos")
    parser.add_argument("--video-file", type=Path, required=True, help="Pfad zum Overlay-Video (.mp4/.mkv)")
    parser.add_argument("--ass-file", type=Path, required=True, help="Pfad zur fertigen .ass Untertiteldatei")
    parser.add_argument("--audio-dir", type=Path, required=True, help="Ordner, der die Audiodateien enthält")
    parser.add_argument("--out-dir", type=Path, default=Path("output"), help="Zielordner für das fertige Video")
    parser.add_argument("--ffmpeg-bin", type=str, default="ffmpeg", help="Pfad zur ffmpeg Binary")
    
    args = parser.parse_args()

    if not args.video_file.exists():
        sys.exit(f"Fehler: Video-Datei existiert nicht: {args.video_file}")
    if not args.ass_file.exists():
        sys.exit(f"Fehler: ASS-Datei existiert nicht: {args.ass_file}")
    if not args.audio_dir.exists():
        sys.exit(f"Fehler: Audio-Verzeichnis existiert nicht: {args.audio_dir}")

    # 1. Passendes Audio automatisch zuordnen
    audio_candidates = collect_audio_files(args.audio_dir)
    audio_file, warning = select_audio_for_video(args.video_file, audio_candidates)
    
    if warning:
        print(f"[Ankündigung] {warning}")
    if not audio_file:
        sys.exit("Fehler: Es konnte keine passende Audiodatei gefunden werden.")

    print(f"Verarbeite: {args.video_file.name}")
    print(f"Nutze Audio: {audio_file.name}")
    print(f"Brenne ein: {args.ass_file.name}")

    # 2. Untertitel einbrennen und zusammenfügen
    try:
        output_path = burn_subs(
            overlay_video=args.video_file,
            ass=args.ass_file,
            audio_file=audio_file,
            dest_dir=args.out_dir,
            ffmpeg=args.ffmpeg_bin
        )
        print(f"\n[Erfolg] Fertiges Video erstellt unter: {output_path}")
    except subprocess.CalledProcessError as e:
        sys.exit(f"\n[Fehler] FFmpeg-Prozess abgebrochen: {e}")

if __name__ == "__main__":
    main()
