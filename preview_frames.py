#!/usr/bin/env python3
"""
Generiert für jeden Eintrag in einer .mapping-Datei ein einzelnes
Vorschaubild. Das Bild zeigt exakt den Frame zum gegebenen Zeitpunkt,
inklusive des eingebrannten Untertitels.

Dies ermöglicht eine schnelle visuelle Prüfung der Synchronisation,
ohne das gesamte Video rendern zu müssen.
"""
import argparse
import subprocess
import sys
import re
from pathlib import Path

# --- Hilfsfunktionen (aus text_ass.py und burn_subtitles.py extrahiert) ---

def sec_to_ass_time(t: float) -> str:
    """Konvertiert Sekunden in das ASS-Zeitformat h:mm:ss.cs."""
    cs_total = int(round(t * 100))
    h, rem = divmod(cs_total, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

def safe_ass_filter_arg(path: Path) -> str:
    """Escapt Sonderzeichen im ASS-Pfad für FFmpeg."""
    p = path.as_posix()
    return "'" + p.replace("'", "'\\''") + "'"

def ffprobe_get_resolution(video: Path) -> (int, int):
    """Ermittelt die Videoauflösung."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=s=x:p=0", str(video)
    ]
    out = subprocess.check_output(cmd).decode().strip()
    width, height = map(int, out.split('x'))
    return width, height

def build_single_entry_ass(text: str, width: int, height: int) -> str:
    """Erstellt eine minimale .ass-Datei für einen einzigen Texteintrag."""
    font_size = max(42, int(height * 0.052))
    style = f"Style: Default,Cormorant Garamond,{font_size},&H00FFFFFF,&HFF000000,&HFF000000,&HFF000000,0,0,0,0,100,100,0,0,1,1,0,5,60,60,40,1"
    
    # Simples Wrapping, um den Text im Bild zu halten
    avg_char_width = font_size * 0.45
    max_chars = max(10, int(width / avg_char_width * 0.9)) # 90% der Breite
    wrapped_lines = []
    for line in text.split('\\n'):
        words = line.split(' ')
        current_line = ""
        for word in words:
            if len(current_line) + len(word) > max_chars:
                wrapped_lines.append(current_line)
                current_line = word
            else:
                current_line += ' ' + word
        wrapped_lines.append(current_line.strip())
    
    ass_text = '\\N'.join(wrapped_lines)
    
    # Positionierung in der unteren Hälfte
    y_pos = int(height * 0.8)
    
    dialogue = f"Dialogue: 0,0:00:00.00,0:00:05.00,Default,,0,0,0,,{{\\an5\\pos({width//2},{y_pos})}}{ass_text}"

    return "\n".join([
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "",
        "[V4+ Styles]",
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding",
        style,
        "",
        "[Events]",
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text",
        dialogue
    ])

def extract_preview_frame(
    ffmpeg_bin: str,
    video_file: Path,
    ass_file: Path,
    timestamp_sec: float,
    output_image: Path
):
    """Extrahiert einen einzelnen Frame und brennt den Untertitel darauf."""
    ass_arg = f"ass={safe_ass_filter_arg(ass_file)}"
    
    cmd = [
        ffmpeg_bin,
        "-y",
        "-ss", str(timestamp_sec),
        "-i", str(video_file),
        "-vf", ass_arg,
        "-frames:v", "1",
        str(output_image)
    ]
    # print(f"DEBUG CMD: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, capture_output=True, text=True)


# --- Hauptlogik ---

TS_RE = re.compile(r"^\[(?P<time>\d{2}:\d{2}:\d{2})\]\s*::\s*(?P<txt>.*)$")

def parse_mapping_line(line: str) -> (float, str):
    """Parst eine Zeile aus der .mapping-Datei."""
    match = TS_RE.match(line.strip())
    if not match:
        return None, None
    
    time_str, text = match.groups()
    h, m, s = map(int, time_str.split(':'))
    seconds = h * 3600 + m * 60 + s
    
    return float(seconds), text.strip()


def find_files(surah_id: int) -> (Path, Path):
    """Findet Video- und Mapping-Dateien basierend auf der Suren-ID."""
    try:
        video_file = subprocess.check_output(
            f"find /home/muhammed-emin-eser/desk/din/quran/maher_workaround/with_overlay/ -type f -name '*({surah_id})*'",
            shell=True, text=True
        ).strip().splitlines()[0]

        mapping_file = subprocess.check_output(
            f"find /home/muhammed-emin-eser/desk/din/ayah-aligner/output/mapping/ -type f -name '{surah_id}_*'",
            shell=True, text=True
        ).strip().splitlines()[0]
        
        return Path(video_file), Path(mapping_file)
    except (subprocess.CalledProcessError, IndexError) as e:
        sys.exit(f"Fehler: Konnte die benötigten Dateien für ID {surah_id} nicht finden. Details: {e}")


def main():
    parser = argparse.ArgumentParser(description="Vorschaubilder für Mapping-Einträge generieren")
    parser.add_argument("--id", type=int, required=True, help="Suren-ID, um Video- und Mapping-Datei automatisch zu finden")
    parser.add_argument("--out-dir", type=Path, default=Path("output/previews"), help="Zielordner für die Vorschaubilder")
    parser.add_argument("--ffmpeg-bin", type=str, default="ffmpeg", help="Pfad zur ffmpeg Binary")
    args = parser.parse_args()

    video_file, mapping_file = find_files(args.id)
    args.out_dir = args.out_dir / str(args.id)

    if not video_file.exists():
        sys.exit(f"Fehler: Video-Datei existiert nicht: {video_file}")
    if not mapping_file.exists():
        sys.exit(f"Fehler: Mapping-Datei existiert nicht: {mapping_file}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    temp_ass_dir = args.out_dir / "temp_ass"
    temp_ass_dir.mkdir(exist_ok=True)

    print(f"Verwende Video: {video_file.name}")
    print(f"Verarbeite Mappings aus: {mapping_file.name}")
    print(f"Speichere Previews in: {args.out_dir}")

    width, height = ffprobe_get_resolution(video_file)
    
    lines = mapping_file.read_text(encoding="utf-8").splitlines()
    
    for i, line in enumerate(lines):
        timestamp_sec, text = parse_mapping_line(line)
        if timestamp_sec is None:
            continue

        print(f"  - Verarbeite Frame für Zeitstempel {sec_to_ass_time(timestamp_sec)}...")

        # 1. Temporäre ASS-Datei für diesen einen Eintrag erstellen
        ass_content = build_single_entry_ass(text, width, height)
        temp_ass_path = temp_ass_dir / f"preview_{i:04d}.ass"
        temp_ass_path.write_text(ass_content, encoding="utf-8")

        # 2. Frame extrahieren und Untertitel einbrennen
        output_image_path = args.out_dir / f"preview_{sec_to_ass_time(timestamp_sec).replace(':', '-')}.png"
        try:
            extract_preview_frame(
                args.ffmpeg_bin,
                video_file,
                temp_ass_path,
                timestamp_sec,
                output_image_path
            )
        except subprocess.CalledProcessError as e:
            print(f"    [FEHLER] bei Zeitstempel {sec_to_ass_time(timestamp_sec)}: {e.stderr}")
            continue

    print("\n[Erfolg] Alle Vorschaubilder wurden erstellt.")
    import shutil
    shutil.rmtree(temp_ass_dir)


if __name__ == "__main__":
    main()
