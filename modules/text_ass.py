#!/usr/bin/env python3
import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from llama_cpp import Llama

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
    return int(out[0]), int(out[1]), float(out[2])


def parse_timestamp(line: str) -> Optional[Entry]:
    m = TS_RE.match(line.strip())
    if not m:
        return None

    if m.group("m2") is not None:
        start = int(m.group("m2")) * 60 + int(m.group("s2"))
        txt = m.group("txt2") or ""
    else:
        hh = int(m.group("h") or 0)
        mm = int(m.group("m"))
        ss = int(m.group("s"))
        start = hh * 3600 + mm * 60 + ss
        txt = m.group("txt") or ""

    txt = (
        txt.replace("\\n", "\n")
        .replace("\\", "")
        .replace("{", "")
        .replace("}", "")
        .replace("\r", " ")
    )
    txt = txt.replace('"', "")
    lines = [l.strip() for l in txt.split("\n")]
    return Entry(float(start), "\n".join(lines))


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
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    cs = int(round((s - int(s)) * 100))
    return f"{int(h)}:{int(m):02d}:{int(s):02d}.{cs:02d}"


def wrap_ass_text(text: str, video_width: int, font_size: int) -> str:
    avg_char_width = font_size * 0.44
    max_chars = max(10, int(video_width / avg_char_width))
    wrapped_lines = []

    for raw_line in text.split("\n"):
        if not raw_line.strip():
            wrapped_lines.append("")
            continue

        current_line = []
        current_visible_length = 0
        words = raw_line.split(" ")

        for word in words:
            sichtbares_wort = re.sub(r"\{.*?\}", "", word)
            word_len = len(sichtbares_wort)
            space_padding = 1 if current_visible_length > 0 else 0

            if current_visible_length + word_len + space_padding <= max_chars:
                current_line.append(word)
                current_visible_length += word_len + space_padding
            else:
                if current_line:
                    wrapped_lines.append(" ".join(current_line))
                current_line = [word]
                current_visible_length = word_len

        if current_line:
            wrapped_lines.append(" ".join(current_line))
        if current_line and wrapped_lines and re.match(r"^[^\w\s]", current_line[0]):
            symbol = current_line[0]
            current_line[0] = current_line[0][1:]
            wrapped_lines[-1] += " " + symbol if symbol else ""

    return "\n".join(wrapped_lines)


def split_overflow_entries(
    entries: List[Entry],
    video_width: int,
    font_size: int,
    max_lines: int = 5,
    video_duration: Optional[float] = None,
) -> List[Entry]:
    if not entries:
        return []
    out: List[Entry] = []

    for idx, entry in enumerate(entries):
        lines = wrap_ass_text(
            entry.text, video_width=video_width, font_size=font_size
        ).split("\n")
        if len(lines) <= max_lines:
            out.append(entry)
            continue

        remainder = len(lines) % max_lines
        segment_count = (len(lines) // max_lines) + (1 if remainder != 0 else 0)
        blocks = [
            "\n".join(lines[g * max_lines : (g + 1) * max_lines])
            for g in range(segment_count)
            if lines[g * max_lines : (g + 1) * max_lines]
        ]

        if idx < len(entries) - 1:
            total_time = float(entries[idx + 1].start) - float(entry.start)
        else:
            total_time = (
                (video_duration - float(entry.start))
                if video_duration and video_duration > float(entry.start)
                else max(5.0, (len(entry.text) / 100.0) * 4.0)
            )

        total_chars = max(1, sum(len(b) for b in blocks))
        current_start = float(entry.start)
        for block_text in blocks:
            block_duration = total_time * (len(block_text) / total_chars)
            out.append(Entry(start=current_start, text=block_text))
            current_start += block_duration
    return out


def karaoke_reveal_words(text: str) -> str:
    lines = text.split("\n")
    rendered_lines = []
    for line in lines:
        words = [w for w in re.split(r"(\s+)", line) if w]
        out = [token if token.isspace() else rf"{{\k2}}{token}" for token in words]
        rendered_lines.append("".join(out))
    return r"\N".join(rendered_lines)


# ==========================================
# LLM INITIALISIERUNG & PROMPT CONFIG
# ==========================================
LLM = Llama(
    model_path="/home/muhammed-emin-eser/.cache/huggingface/hub/models--bartowski--Qwen2.5-14B-Instruct-GGUF/snapshots/05244aa5d871c661c80082a15d3bce44714d068d/Qwen2.5-14B-Instruct-Q4_K_M.gguf",
    n_ctx=1000,
    n_threads=8,
    n_gpu_layers=-1,
    verbose=True,
)

ALLOWED_LABELS = ["GOD", "DESTRUCTIVE", "CONSTRUCTIVE"]

COLOR_MAP = {
    "GOD": "&H75DFFA&",  # Majestätisches Gelbgold
    "DESTRUCTIVE": "&H6212B2&",  # Alarmierendes Hellrot
    "CONSTRUCTIVE": "&H803500&",  # Blau
}

SYSTEM_PROMPT = """You are a strict, conservative annotation selector.CRITICAL RULE:- Word extraction and categorization is COMPLETELY OPTIONAL.- It is significantly better to return an EMPTY list than to include a doubtful or incorrect word. - Never force a word into a category just to fill the JSON. If a text contains NO clear matches for the categories, you MUST return an empty labels array.TASK:Analyze the input text and select ONLY words that clearly, directly, and without ambiguity belong to one of the specified categories.CATEGORIES & CRITERIA:- GOD: Direct names or explicit synonyms for the deity (e.g., Allah, God, Lord). Do NOT include chapter names, placeholders, pronouns, or surrounding nouns.- DESTRUCTIVE: Words expressing explicit destruction, damage, sin, or severe negativity (e.g., destroy, death, torment, cursed).- CONSTRUCTIVE: Words expressing creation, purification, reward, or explicit positivity (e.g., create, purify, patience, blessings).OUTPUT RULES:- You DO NOT modify, capitalize, or rewrite text.- You only select exact WORDS from the input text.- Do NOT output any conversational text, explanations, thoughts, or formatting blocks before or after the JSON.- Return ONLY valid JSON adhering strictly to this format:{  "labels": [    {"word": "exact_word_from_text", "category": "GOD|DESTRUCTIVE|CONSTRUCTIVE"}  ]}NEGATIVE EXAMPLE (How to handle empty cases):Input: "The cloaked one refers to the Prophet Muhammad who used to pray in a cave."Output:{  "labels": []}"""
# fmt: off
_GOD_KEYS=["allah","god","lord","albarr","allmighty","allmighty","all-mighty","allknower","allknower","all-knower","allwise","allwise","all-wise","allhearer","allhearer","all-hearer","allseer","allseer","all-seer","allaware","all-aware","allprovider","all-provider","allstrong","all-strong","allknowing","all-knowing","allcapable","all-capable","mostbeneficent","mostmerciful","mostgreat","mosthigh","mostgenerous","mostkind","mostjust","mostnear","oftforgiving","oft-forgiving","omnipotentking","supremecreator","realbestower","ownerofpower",]
# Unumstößliche Regex-Listen (allgemeingültig vor der KI)
REFERENCE_THEMES = {
    "GOD": re.compile(
        r"^(?:the\s+)?("
        + "|".join(k.replace(" ", r"\s*").replace("-", r"-?") for k in _GOD_KEYS)
        + r")$",
        re.IGNORECASE,
    ),
}

mock_json="""{"labels":[{"word":"Al-Muddathir","category":"GOD"},{"word":"ARISE","category":"CONSTRUCTIVE"},{"word":"WARN","category":"DESTRUCTIVE"},{"word":"ALLAH","category":"GOD"},{"word":"purify","category":"CONSTRUCTIVE"},{"word":"Ar-Rujz","category":"DESTRUCTIVE"},{"word":"give","category":"CONSTRUCTIVE"},{"word":"deeds","category":"CONSTRUCTIVE"},{"word":"obedience","category":"CONSTRUCTIVE"},{"word":"patient","category":"CONSTRUCTIVE"},{"word":"transgressors","category":"DESTRUCTIVE"},{"word":"Allah","category":"GOD"},{"word":"Trumpet","category":"DESTRUCTIVE"},{"word":"gather","category":"CONSTRUCTIVE"},{"word":"Hard","category":"DESTRUCTIVE"},{"word":"disbelievers","category":"DESTRUCTIVE"},{"word":"Al-Waleed","category":"GOD"},{"word":"bin","category":"GOD"},{"word":"Al-Mugheerah","category":"GOD"},{"word":"Al-Makhzoomee","category":"GOD"},{"word":"granted","category":"CONSTRUCTIVE"},{"word":"abundance","category":"CONSTRUCTIVE"},{"word":"smooth","category":"CONSTRUCTIVE"},{"word":"comfortable","category":"CONSTRUCTIVE"},{"word":"God","category":"GOD"},{"word":"strength","category":"CONSTRUCTIVE"},{"word":"rock","category":"CONSTRUCTIVE"},{"word":"hills","category":"CONSTRUCTIVE"},{"word":"understanding","category":"CONSTRUCTIVE"},{"word":"formed","category":"CONSTRUCTIVE"},{"word":"stubborn","category":"DESTRUCTIVE"},{"word":"opposing","category":"DESTRUCTIVE"},{"word":"torment","category":"DESTRUCTIVE"},{"word":"Verily","category":"GOD"},{"word":"cursed","category":"DESTRUCTIVE"},{"word":"plotted","category":"DESTRUCTIVE"},{"word":"plot","category":"DESTRUCTIVE"},{"word":"god","category":"GOD"},{"word":"create","category":"CONSTRUCTIVE"},{"word":"destroy","category":"DESTRUCTIVE"},{"word":"life","category":"CONSTRUCTIVE"},{"word":"death","category":"DESTRUCTIVE"},{"word":"frowned","category":"DESTRUCTIVE"}]}"""
# fmt: on


def classify_sentence(sentence: str):
    # prompt = f"{SYSTEM_PROMPT}\n\nTEXT:\n{sentence}\n"
    # output = LLM(prompt, max_tokens=256, temperature=0.0)
    # text = output["choices"][0]["text"]
    # print(text)

    try:
        return json.loads(mock_json).get("labels", [])
    except Exception:
        RuntimeError(Exeption)


def split_into_sentences(text: str):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def normalize(word: str) -> str:
    return re.sub(r"[^\w]", "", word).strip()


# ==========================================
# ANNOTATION ENGINE (DEINE ORIGINAL-STRUKTUR)
# ==========================================
def annotate_text(text: str) -> str:
    # 1. Zitate direkt auf nativer String-Ebene in Kursiv wandeln
    text = re.sub(r'["\'](.*?)["\']', r"{\\i1}\1{\\i0}", text)

    # 2. "O ..."-Prinzip (Direkte Ansprachen am Satzanfang)
    def call_to_action_replacer(match):
        o_part = match.group(1)
        subject = match.group(2)
        return (
            rf"{{\b1\c&HFFFFFF&\fs55}}{o_part}{{\\b0\c&HFFFFFF&\fs45}}{subject.upper()}"
        )

    text = re.sub(r"\b(O\s+)(.+)", call_to_action_replacer, text)

    # 3. Ausrufezeichen-Nachdruck (Macht den Satz bis zum ! fett)
    text = re.sub(r"CN_START([^.!?]*?!)", r"{\\b1}\1{\\b0}", text)

    # 4. Klammern dezent formatieren
    text = re.sub(r"\((.*?)\)", r"{\\fs30\\c&HAAAAAA&}(\1){\\fs40\\c&HFFFFFF&}", text)

    # 5. Sätze splitten für LLM-Verarbeitung
    sentences = split_into_sentences(text)
    output_sentences = []

    for sentence in sentences:
        labels = classify_sentence(sentence)

        # Build deterministic lookup: word -> category
        highlight_map = {}
        for item in labels:
            word = item.get("word", "")
            cat = item.get("category", "")
            if cat in ALLOWED_LABELS:
                highlight_map[word] = cat

        words = sentence.split(" ")
        processed = []

        for w in words:
            clean = normalize(w)
            clean_lower = clean.lower()
            highlighted = False

            # --- STUFE 1: REGEX CHECK (Allgemeingültig vor dem LLM) ---
            for category, pattern in REFERENCE_THEMES.items():
                if pattern.search(clean_lower):
                    print(f"[REGEX] '{w.strip()}' -> Kategorie: {category}")
                    if category in COLOR_MAP:
                        w = f"{{\\c{COLOR_MAP[category]}}}{w}{{\\c}}"
                    highlighted = True
                    break

            # --- STUFE 2: LLM LOOKUP MATCH ---
            if not highlighted and clean in highlight_map:
                cat = highlight_map[clean]
                color = COLOR_MAP.get(cat)
                if color:
                    w = f"{{\\c{color}}}{w}{{\\c}}"

            processed.append(w)

        output_sentences.append(" ".join(processed))

    return " ".join(output_sentences)


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

    entries_sorted = split_overflow_entries(
        sorted(entries, key=lambda e: e.start),
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
            end = min(duration, start + 6.0)
        if end <= start:
            end = start + 0.25

        # LLM-basierte Annotation & Formatierungen anwenden
        styled_text = annotate_text(e.text)
        txt = wrap_ass_text(styled_text, width, font_size)
        txt = karaoke_reveal_words(txt)

        x, y = (
            int(width * 0.5),
            int(height * (0.72 if (txt.count(r"\N") + 1) == 5 else 0.75)),
        )

        if i == 0 and treat_first_as_header:
            header_font_size = int(font_size * 1.25)
            tags = rf"\an5\pos({x},{int(height * 0.7)})\fad(200,200)\fs{header_font_size}\b1\c&H00FFFFFF&\3c&H00000000&\bord1\blur0"
            events.append(
                f"Dialogue: 0,{sec_to_ass_time(start)},{sec_to_ass_time(end)},Overlay,,0,0,0,,{{{tags}}}{txt}"
            )
            continue

        line_count = txt.count(r"\N") + 1
        y_adjusted = y + (line_height // 2 if line_count >= 5 else 0)
        tags = rf"\an5\pos({x},{y_adjusted})\fad(120,150)\bord1\blur0"
        events.append(
            f"Dialogue: 0,{sec_to_ass_time(start)},{sec_to_ass_time(end)},Overlay,,0,0,0,,{{{tags}}}{txt}"
        )

    ass = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "WrapStyle: 2",
        "",
        "[V4+ Styles]",
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding",
        f"Style: Overlay,{font_name},{font_size},&H00FFFFFF,&HFF000000,&HFF000000,&HFF000000,0,0,0,0,100,100,0,0,1,1,0,5,60,60,40,1",
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
    parser.add_argument("mapping", type=Path)
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)
    width, height, duration = ffprobe_get(args.video)

    entries = []
    for line in args.mapping.read_text(encoding="utf-8").splitlines():
        if line.strip():
            e = parse_timestamp(line)
            if e:
                entries.append(e)

    if not entries:
        raise RuntimeError("Keine gültigen Zeilen gefunden.")

    ass_text = build_ass(entries, width, height, duration)
    args.output.write_text(ass_text, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
