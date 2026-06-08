#!/usr/bin/env python3
import argparse
import re
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

TS_RE = re.compile(
    r"^\[(?:(?P<h>\d{2}):)?(?P<m>\d{2}):(?P<s>\d{2})\]\s*::\s*(?P<txt>.*)\s*$"
    r"|^\[(?P<m2>\d{2}):(?P<s2>\d{2})\]\s*::\s*(?P<txt2>.*)\s*$"
)


@dataclass
class Entry:
    start: float
    text: str


def ffprobe_get(video: Path) -> Tuple[int, int, float]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nw=1:nk=1",
        str(video),
    ]
    out = (
        subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        .decode()
        .strip()
        .splitlines()
    )
    width = int(out[0])
    height = int(out[1])
    duration = float(out[2])
    return width, height, duration


def parse_timestamp(line: str) -> Optional[Entry]:
    m = TS_RE.match(line.strip())
    if not m:
        return None

    if m.group("m2") is not None:
        mm = int(m.group("m2"))
        ss = int(m.group("s2"))
        txt = m.group("txt2") or ""
        start = mm * 60 + ss
    else:
        hh = int(m.group("h") or 0)
        mm = int(m.group("m"))
        ss = int(m.group("s"))
        txt = m.group("txt") or ""
        start = hh * 3600 + mm * 60 + ss

    # ---- TEXT CLEANUP ----
    # first convert explicit \"\\n\" sequences into actual newlines so header line breaks are preserved
    txt = txt.replace("\\n", "\n")
    txt = txt.replace("\\", "")  # Backslashes komplett entfernen (sonst ASS-Escape)
    txt = txt.replace("{", "")  # ASS-Tags verhindern
    txt = txt.replace("}", "")
    txt = txt.replace('"', '"')  # escaped quotes normalisieren
    txt = txt.replace("\r", " ")

    # Kein Splitten nach Versnummern: Mapping-Text strikt 1:1 übernehmen.
    txt = txt.replace('"', "")
    lines = [line.strip() for line in txt.split("\n")]
    txt = "\n".join(lines)

    return Entry(float(start), txt)


def spread_same_timestamps(entries: List[Entry], step: float = 0.18) -> List[Entry]:
    out: List[Entry] = []
    i = 0
    while i < len(entries):
        j = i + 1
        while j < len(entries) and abs(entries[j].start - entries[i].start) < 1e-9:
            j += 1
        group = entries[i:j]
        base = group[0].start
        for k, e in enumerate(group):
            out.append(Entry(base + k * step, e.text))
        i = j
    return out


def sec_to_ass_time(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    cs = int(round((t - int(t)) * 100))
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def wrap_ass_text(text: str, video_width: int, font_size: int) -> str:
    # 1. Maximale sichtbare Zeichen pro Zeile berechnen
    avg_char_width = font_size * 0.44
    max_chars = max(10, int(video_width / avg_char_width))

    wrapped_lines = []

    # 2. Wir gehen Zeile für Zeile durch (falls schon Umbrüche da sind)
    for raw_line in text.split("\n"):
        if not raw_line.strip():
            wrapped_lines.append("")
            continue

        current_line = []
        current_visible_length = 0
        words = raw_line.split(" ")

        for word in words:
            # HIER IST DER TRICK: Wir entfernen alle {\...} Tags NUR für die Längenmessung
            sichtbares_wort = re.sub(r"\{.*?\}", "", word)
            word_len = len(sichtbares_wort)

            # Leerzeichen einrechnen, falls die Zeile nicht leer ist
            space_padding = 1 if current_visible_length > 0 else 0

            if current_visible_length + word_len + space_padding <= max_chars:
                current_line.append(word)
                current_visible_length += word_len + space_padding
            else:
                # Zeile voll! Speichern und neue Zeile anfangen
                if current_line:
                    wrapped_lines.append(" ".join(current_line))
                current_line = [word]
                current_visible_length = word_len

        if current_line:
            wrapped_lines.append(" ".join(current_line))

    return "\n".join(wrapped_lines)


def split_overflow_entries(
    entries: List[Entry],
    video_width: int,
    font_size: int,
    max_lines: int = 5,
    video_duration: Optional[float] = None,
) -> List[Entry]:
    """
    Enforce the 5-line rule before ASS-Erzeugung:
    - Zeilenzahl nach finalem Wrap prüfen
    - Überlange Einträge in max_lines-Gruppen aufteilen
    - Zeit proportional zur Zeichenlänge (Lesezeit) gewichten
    - Letzter Eintrag erhält die verbleibende Videodauer für den Split
    """
    if not entries:
        return []

    out: List[Entry] = []

    for idx, entry in enumerate(entries):
        # Zeilen umbrechen und analysieren
        lines = wrap_ass_text(
            entry.text, video_width=video_width, font_size=font_size
        ).split("\n")

        # Wenn der Text passt, einfach unverändert übernehmen
        if len(lines) <= max_lines:
            out.append(entry)
            continue

        # 1. Blöcke erstellen
        remainder = len(lines) % max_lines
        groups = len(lines) // max_lines
        segment_count = groups if remainder == 0 else groups + 1

        blocks = []
        for g in range(segment_count):
            start_idx = g * max_lines
            end_idx = start_idx + max_lines
            block_lines = lines[start_idx:end_idx]
            if block_lines:
                blocks.append("\n".join(block_lines))

        # 2. Verfügbare Gesamtlaufzeit ermitteln
        if idx < len(entries) - 1:
            # Normaler Eintrag: Zeit bis zum nächsten Vers
            total_time = float(entries[idx + 1].start) - float(entry.start)
        else:
            # Sonderbehandlung letzter Eintrag mit exakter Videodauer
            if video_duration and video_duration > float(entry.start):
                total_time = video_duration - float(entry.start)
            else:
                # Schätzungs-Fallback, falls keine duration übergeben wurde
                char_count = len(entry.text)
                total_time = max(5.0, (char_count / 100.0) * 4.0)

        # 3. Zeit proportional zur Zeichenlänge gewichten
        total_chars = sum(len(b) for b in blocks)

        # Sicherheits-Fallback, falls alle Blöcke leer sein sollten
        if total_chars == 0:
            total_chars = 1

        # 4. Blöcke mit gewichteten Timestamps abspeichern
        current_start = float(entry.start)
        for block_text in blocks:
            # Anteil dieses Blocks an der Gesamtzeichenzahl
            weight = len(block_text) / total_chars
            block_duration = total_time * weight

            out.append(Entry(start=current_start, text=block_text))

            # Der nächste Block startet exakt, wenn dieser endet
            current_start += block_duration

    return out


def karaoke_reveal_words(text: str) -> str:
    lines = text.split("\n")
    rendered_lines = []

    for line in lines:
        words = [w for w in re.split(r"(\s+)", line) if w]
        cs_per = 2
        out = []
        for token in words:
            if token.isspace():
                out.append(token)
            else:
                out.append(r"{\k" + str(cs_per) + "}" + token)
        rendered_lines.append("".join(out))

    return r"\N".join(rendered_lines)


# Modell-Initialisierung
MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# Liste von Wörtern, die die KI ignorieren MUSS (verhindert [KI-FAIL] bei "the", "him" etc.)
# fmt: off
STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "from", 
    "by", "he", "him", "his", "they", "them", "who", "whom", "whose", 
    "made", "make", "me", "i", "you", "your", "it", "its", "with", "for", "of", "revelations", "revelation"
}

# 1. HARDCODED REGEX (Exakte Treffer ohne KI)
_GOD_KEYS = [
    "allah", "god", "lord", "albarr", "all mighty", "allmighty", "all-mighty",
    "all knower", "allknower", "all-knower", "all wise", "allwise", "all-wise",
    "all hearer", "allhearer", "all-hearer", "all seer", "allseer", "all-seer",
    "all aware", "all-aware", "all provider", "all-provider", "all strong", "all-strong",
    "all knowing", "all-knowing", "all capable", "all-capable", "most beneficent",
    "most merciful", "most great", "most high", "most generous", "most kind",
    "most just", "most near", "oft forgiving", "oft-forgiving", "omnipotent king",
    "supreme creator", "real bestower", "owner of power"
]
# fmt:on

REFERENCE_THEMES = {
    "GOD": re.compile(
        r"^(?:the\s+)?("
        + "|".join(k.replace(" ", r"\s*").replace("-", r"-?") for k in _GOD_KEYS)
        + r")$",
        re.IGNORECASE,
    ),
    "PUNISHMENT": re.compile(
        r"^("
        r"hell(?:fire)?|blaze|torment\w*|chastis\w*|punish\w*|tortur\w*|scourge\w*|doom\w*|disgrace\w*|"
        r"strik\w*|seiz\w*|destroy\w*|destruct\w*|curs\w*|reject\w*|belying|belied|"
        r"painful|severe|grievous|miserable|awful|dreadful|terrible|heavy|wretched|wicked|"
        r"disbeliev\w*|wrongdoer\w*|polytheist\w*|sinner\w*|arrogant\w*|transgressor\w*"
        r")$",
        re.IGNORECASE,
    ),
    "LIGHT": re.compile(
        r"^("
        r"paradise|garden\w*|light|peace|bliss|mercy|grace|glor\w*|victory|victor\w*|"
        r"forgiv\w*|pardon\w*|guid\w*|purif\w*|save\w*|admit\w*|reward\w*|rejoic\w*|triumph\w*|bless\w*|"
        r"beautiful|pure|eternal|successful|radiant|noble|trustworthy|truth|"
        r"believer\w*|righteous|pious|humble|muttaqoon"
        r")$",
        re.IGNORECASE,
    ),
}

# 2. KI-REFERENZ-PHRASEN (Als begriffliche Anker für die Vektoren)
REFERENCE_PHRASES = {
    "GOD": ["divine attributes", "the creator", "the almighty lord", "holy deity"],
    "PUNISHMENT": [
        "hellfire torment",
        "painful punishment",
        "destruction doom",
        "severe chastisement",
    ],
    "LIGHT": ["paradise gardens", "eternal bliss", "divine light", "righteous success"],
}

# Vorab-Berechnung der Embeddings
REFERENCE_EMBEDDINGS = {
    category: MODEL.encode(phrases, convert_to_tensor=True)
    for category, phrases in REFERENCE_PHRASES.items()
}

THRESHOLDS = {
    "GOD": 0.65,
    "PUNISHMENT": 0.50,
    "LIGHT": 0.55,
}


def cosine_similarity_matrix(a, b):
    """Berechnet die Cosine Similarity zwischen zwei Embedding-Matrizen."""
    a_norm = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.dot(a_norm, b_norm.T)


def semantic_highlighter(text: str) -> str:
    """
    Analysiert Text-Tokens. Nutzt primär hocheffiziente Regex-Suchen.
    Nur wenn kein Treffer erzielt wird, entscheidet die KI via Embeddings.
    """
    # 1. Zitate direkt auf nativer String-Ebene in Kursiv wandeln
    text = re.sub(r'["\'](.*?)["\']', r"{\\i1}\1{\\i0}", text)

    # 2. "O ..."-Prinzip (Direkte Ansprachen am Satzanfang)
    def call_to_action_replacer(match):
        o_part = match.group(1)
        subject = match.group(2)
        return rf"{{\b1\c&HFFFFFF&\fs55}}{o_part}{{\\b0\fs45}}{subject.upper()}"

    text = re.sub(r"\b(O\s+)(.+)", call_to_action_replacer, text)

    # 3. Ausrufezeichen-Nachdruck (Macht den Satz bis zum ! fett)
    text = re.sub(r"CN_START([^.!?]*?!)", r"{\\b1}\1{\\b0}", text)

    # 4. Klammern dezent formatieren
    text = re.sub(r"\((.*?)\)", r"{\\fs30\\c&HAAAAAA&}(\1){\\fs40\\c&HFFFFFF&}", text)

    # 5. HYBRIDE FILTERUNG (Regex vs. KI)
    tokens = re.split(r"(\{.*?\})", text)

    for i in range(len(tokens)):
        if (
            tokens[i].startswith("{")
            and tokens[i].endswith("}")
            or not tokens[i].strip()
        ):
            continue

        words = tokens[i].split(" ")
        processed_words = []

        for word in words:
            # 1. Bereinige das Wort von Satzzeichen UND wandle es direkt in Kleinbuchstaben um!
            clean_word = re.sub(r"[^\w\s-]", "", word).strip().lower()

            # Falls das Wort nach der Bereinigung leer ist
            if not clean_word:
                processed_words.append(word)
                continue

            # 2. SCHUTZSCHILD: Stopwörter und alles unter 3 Zeichen lautlos ignorieren
            if clean_word in STOP_WORDS or len(clean_word) <= 2:
                processed_words.append(word)
                # print(f"[SKIP] '{word}' wurde als Stopwort ignoriert.") # Optional zum Testen
                continue

            highlighted = False

            # --- STUFE 1: REGEX CHECK ---
            for category, pattern in REFERENCE_THEMES.items():
                # Da clean_word jetzt klein ist, matcht es perfekt auf deine verankerten Muster
                if pattern.search(clean_word):
                    print(f"[REGEX] '{word.strip()}' -> Kategorie: {category}")
                    # Tausche die alten Farb-Tags in deiner highlighter-Schleife gegen diese aus:
                    if category == "GOD":
                        word = (
                            f"{{\\c&H75DFFA&}}{word}{{\\c}}"  # Majestätisches Gelbgold
                        )
                    elif category == "PUNISHMENT":
                        word = f"{{\\c&H6212B2&}}{word}{{\\c}}"  # Alarmierendes Hellrot (Strafe)
                    elif category == "DISBELIEVER":
                        word = (
                            f"{{\\c&H33105E&}}{word}{{\\c}}"  # Tiefes, düsteres Weinrot
                        )
                    elif category == "BELIEVER":
                        word = f"{{\\c&HD069BE&}}{word}{{\\c}}"  # Edles lila
                    elif category == "REWARD":
                        word = f"{{\\c&HA137FF&}}{word}{{\\c}}"  # Leuchtendes Pink (Belohnung)
                    highlighted = True
                    break

            # --- STUFE 2: KI EMBEDDING CHECK ---
            if not highlighted:
                # Wir schicken das saubere, kleingeschriebene Wort zur KI
                word_embedding = MODEL.encode(clean_word)

                for category, ref_embeds in REFERENCE_EMBEDDINGS.items():
                    sims = cosine_similarity_matrix(
                        word_embedding, ref_embeds.cpu().numpy()
                    )
                    max_sim = np.max(sims)

                    if max_sim >= THRESHOLDS[category]:
                        print(
                            f"[KI-MATCH] '{word.strip()}' -> Kategorie: {category} (Score: {max_sim:.4f} >= Threshold: {THRESHOLDS[category]})"
                        )
                        if category == "GOD":
                            word = f"{{\\c&H00C5A0&}}{word}{{\\c}}"
                        elif category == "PUNISHMENT":
                            word = f"{{\\c&H444444&}}{word}{{\\c}}"
                        elif category == "LIGHT":
                            word = f"{{\\c&H50C850&}}{word}{{\\c}}"
                        highlighted = True
                        break
                    else:
                        if max_sim > 0.40:
                            print(
                                f"[KI-FAIL]  '{word.strip()}' -> Kategorie: {category} (Score: {max_sim:.4f} < Threshold: {THRESHOLDS[category]})"
                            )

            processed_words.append(word)

        tokens[i] = " ".join(processed_words)

    return "".join(tokens)


def build_ass(
    entries,
    width,
    height,
    duration,
    font_name="Cormorant Garamond",
    treat_first_as_header: bool = True,
):
    font_size = max(42, int(height * 0.052))
    line_height = int(round(font_size * 1.25))
    events = []

    entries_sorted = sorted(entries, key=lambda e: e.start)

    entries_sorted = split_overflow_entries(
        entries_sorted,
        video_width=width,
        font_size=font_size,
        max_lines=5,
        video_duration=duration,
    )

    only_header = len(entries_sorted) == 1

    for i, e in enumerate(entries_sorted):
        start = e.start
        end = entries_sorted[i + 1].start if i + 1 < len(entries_sorted) else duration
        if only_header:
            # Fallback: Titel nicht das ganze Video zeigen, sondern nur kurz einblenden
            end = min(duration, start + 6.0)
        if end <= start:
            end = start + 0.25

        # 1. Semantische KI-Hervorhebung und Interpunktions-Styling anwenden
        styled_text = semantic_highlighter(e.text)

        # 2. Erst danach Zeilenumbruch und Karaoke-Effekte generieren
        txt = wrap_ass_text(styled_text, width, font_size)
        txt = karaoke_reveal_words(txt)

        x = int(width * 0.5)
        y = int(height * 0.75)

        # --- HEADER (erste Zeile) ---
        if i == 0 and treat_first_as_header:
            header_font_size = int(font_size * 1.25)

            tags = (
                rf"\an5\pos({x},{int(height * 0.7)})"
                rf"\fad(200,200)"
                rf"\fs{header_font_size}"
                rf"\b1"
                rf"\c&H00FFFFFF&"
                rf"\3c&H00000000&"
                rf"\bord1\blur0"
            )

            line = f"Dialogue: 0,{sec_to_ass_time(start)},{sec_to_ass_time(end)},Overlay,,0,0,0,,{{{tags}}}{txt}"
            events.append(line)
            continue

        # --- NORMALER TEXT ---
        # Nutzen der echten Zeilenumbrüche (\n) für die korrekte 5-Zeilen-Positionsverschiebung
        line_count = txt.count("\n") + 1
        y_adjusted = y + (line_height // 2 if line_count >= 5 else 0)
        tags = rf"\an5\pos({x},{y_adjusted})\fad(120,150)\bord1\blur0"
        line = f"Dialogue: 0,{sec_to_ass_time(start)},{sec_to_ass_time(end)},Overlay,,0,0,0,,{{{tags}}}{txt}"
        events.append(line)

    # Basis-Style (Schmaler Outline)
    primary = "&H00FFFFFF"  # Weiß als neutrale Grundfarbe
    secondary = "&HFF000000"  # Unsichtbar
    outline = (
        "&HFF000000"  # Schwarz für perfekte Lesbarkeit über dem Ornament-Hintergrund
    )
    back = "&HFF000000"

    ass = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "WrapStyle: 2",
        "",
        "[V4+ Styles]",
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,"
        "Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,"
        "Alignment,MarginL,MarginR,MarginV,Encoding",
        f"Style: Overlay,{font_name},{font_size},{primary},{secondary},{outline},{back},"
        "0,0,0,0,100,100,0,0,1,1,0,5,60,60,40,1",
        "",
        "[Events]",
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text",
        *events,
    ]
    return "\n".join(ass)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ASS subtitles from mapping + overlay video."
    )
    parser.add_argument("mapping", type=Path, help="Pfad zur Mapping-Datei")
    parser.add_argument("video", type=Path, help="Pfad zum Overlay-Video")
    parser.add_argument("output", type=Path, help="Zielpfad für erzeugte ASS-Datei")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)
    mapping_path = args.mapping
    video_path = args.video
    out_ass = args.output

    width, height, duration = ffprobe_get(video_path)

    entries = []
    for line in mapping_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        e = parse_timestamp(line)
        if e:
            entries.append(e)

    if not entries:
        raise RuntimeError("Keine gültigen Zeilen gefunden.")

    if len(entries) <= 1:
        raise RuntimeError("Nur Titel gefunden. Mapping enthält keine Vers-Zeilen.")

    ass_text = build_ass(entries, width, height, duration)
    out_ass.write_text(ass_text, encoding="utf-8")
    print(out_ass)


if __name__ == "__main__":
    main()
