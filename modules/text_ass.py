#!/usr/bin/env python3
import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
from llama_cpp import Llama
from pyparsing import originalTextFor, nestedExpr, CharsNotIn

BASE_DIR = Path(__file__).parent.parent

LLM = Llama(
    model_path="/home/muhammed-emin-eser/.cache/huggingface/hub/models--bartowski--Qwen2.5-14B-Instruct-GGUF/snapshots/05244aa5d871c661c80082a15d3bce44714d068d/Qwen2.5-14B-Instruct-Q4_K_M.gguf",
    n_ctx=6000,
    n_threads=12,
    n_gpu_layers=-1,
    verbose=True,
)


def seconds_to_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


TS_RE = re.compile(
    r"^\[(?:(?P<h>\d{2}):)?(?P<m>\d{2}):(?P<s>\d{2})\]\s*::\s*(?P<txt>.*)\s*$"
    r"|^\[(?P<m2>\d{2}):(?P<s2>\d{2})\]\s*::\s*(?P<txt2>.*)\s*$"
)
COLOR_MAP = {
    "GOD": "&H75DFFA&",  # Majestätisches Gelbgold
    "DESTRUCTIVE": "&H6212B2&",  # Alarmierendes Hellrot
    "CONSTRUCTIVE": "&H803500&",  # Blau
    "NONE": "&H00FFFFFF",
}


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


def _progressive_word_lines(
    text: str,
    entry_start: float,
    entry_end: float,
    word_aligns: List[dict],
    font_size: int = 42,
    video_width: int = 1356,
    video_height: int = 638,
) -> List[Tuple[str, float, float]]:
    aligns_sorted = sorted(
        [wa for wa in word_aligns if wa.get("en")],
        key=lambda x: (x.get("ayah", 0), x.get("idx", 0)),
    )

    wrapped_lines = text.split("\n")
    word_entries = []
    for li, raw_line in enumerate(wrapped_lines):
        if not raw_line.strip():
            continue
        tokens = raw_line.split()
        current_verse = None
        pos_in_verse = 0
        for t in tokens:
            m = re.match(r"^(\d+):$", t)
            if m:
                current_verse = int(m.group(1))
                pos_in_verse = 0
            else:
                word_entries.append((t, li, current_verse, pos_in_verse))
                pos_in_verse += 1

    if not word_entries:
        return []

    num_words = len(word_entries)

    if not aligns_sorted:
        lines_text = []
        for li in range(len(wrapped_lines)):
            words_in_line = [w for w, l, v, p in word_entries if l == li]
            if words_in_line:
                lines_text.append(" ".join(rf"{{\k2}}{w}" for w in words_in_line))
        full_text = r"\N".join(lines_text)
        return [(full_text, entry_start, entry_end)]

    timing_lookup = {}
    for wa in aligns_sorted:
        timing_lookup[(wa["ayah"], wa["idx"])] = (wa["start"], wa["end"])

    word_timings: List[Optional[Tuple[float, float]]] = []
    for word, li, verse, pos in word_entries:
        key = (verse, pos) if verse is not None else None
        if key and key in timing_lookup:
            word_timings.append(timing_lookup[key])
        else:
            word_timings.append(None)

    i = 0
    while i < num_words:
        if word_timings[i] is not None:
            i += 1
            continue
        j = i
        while j < num_words and word_timings[j] is None:
            j += 1
        prev_end = entry_start
        if i > 0 and word_timings[i - 1] is not None:
            prev_end = word_timings[i - 1][1]
        next_start = entry_end
        if j < num_words and word_timings[j] is not None:
            next_start = word_timings[j][0]
        gap = next_start - prev_end
        count = j - i
        for k in range(count):
            t_start = prev_end + (gap * k) / count
            t_end = prev_end + (gap * (k + 1)) / count
            word_timings[i + k] = (t_start, t_end)
        i = j

    for i in range(1, num_words):
        prev_start = word_timings[i - 1][0]
        cur_start, cur_end = word_timings[i]
        if cur_start < prev_start:
            word_timings[i] = (prev_start, max(cur_end, prev_start))

    result: List[Tuple[str, float, float]] = []
    for i, (word, li, verse, pos) in enumerate(word_entries):
        cur_start, cur_end = word_timings[i]
        line_end = max(word_timings[i + 1][0], cur_start) if i + 1 < num_words else max(entry_end, cur_start)

        word_dur_cs = max(1, int((cur_end - cur_start) * 100))

        lines_visible = []
        for li_w in range(len(wrapped_lines)):
            words_in_this_line = [
                (j, w)
                for j, (w, l, v, p) in enumerate(word_entries[: i + 1])
                if l == li_w
            ]
            if not words_in_this_line:
                continue
            parts = []
            for j, w in words_in_this_line:
                if j < i:
                    parts.append(w)
                else:
                    parts.append(rf"{{\K{word_dur_cs}}}{w}")
            lines_visible.append(" ".join(parts))

        line_text = r"\N".join(lines_visible)
        line_text = rf"{{\2c&H222222&}}" + line_text
        result.append((line_text, cur_start, line_end))

    return result


def normalize(word: str) -> str:
    return re.sub(r"[^\w]", "", word).strip()


# ==========================================
# ANNOTATION ENGINE (DEINE ORIGINAL-STRUKTUR)
# ==========================================
def get_annotated_text(file_name: str) -> Dict:
    m = re.match(r"^(\d+)", file_name)
    result_path = None
    if m:
        suffix = m.group(1)
        base = Path("/home/muhammed-emin-eser/desk/din/quran/qwen_final_jsonl/")

        p = base / f"{suffix}.jsonl"
        if p.exists():
            result_path = p

    if result_path is None:
        return {}

    with open(result_path, "r") as f:
        lines = f.readlines()
    verses = {}
    for line in lines:
        parsed = json.loads(line)
        cid = parsed["custom_id"]
        if ":" in cid:
            verse_num = int(cid.split(":")[-1])
        else:
            verse_num = int(cid)
        try:
            verses[verse_num] = json.loads(
                parsed["response"]["body"]["choices"][0]["message"]["content"]
            )
        except json.JSONDecodeError:
            print(f"  [WARN] Malformed JSON for verse {verse_num}, skipping")
    return verses


def annotate_highlights(verse, highlights: Dict):
    # 1. Zitate direkt auf nativer String-Ebene in Kursiv wandeln
    matches = re.sub(r'["\'](.*?)["\']', r"{\\i1}\1{\\i0}", verse)

    # 2. "O ..."-Prinzip (Direkte Ansprachen am Satzanfang)
    def call_to_action_replacer(match):
        o_part = match.group(1)  # z. B. "O "
        full_text_after_o = match.group(2)  # z. B. "you Messenger, deliver the message"
        prompt = f"""Analyze the following text for a vocative address ("O-<subject>" / "Oh ...").
    Identify the true grammatical subject/entity being called upon or addressed after the particle "O".

    Examples for your understanding:
    - Input: "O you who believe, eat of the good things" -> Output: "you who believe"
    - Input: "O Prophet, fight the disbelievers" -> Output: "Prophet"
    - Input: "O you Messenger, deliver what has been revealed" -> Output: "you Messenger"
    - Input": "O People of the Scripture, why do you disbelieve" -> Output: "People of the Scripture"

    Task: Extract ONLY the full entity being addressed.
    YOUR RESPONSE SHOULD BE ONLY THE ENTITY BEING ADDRESSED: ONLY ONLY ONLY: NOT ONE MORE WORD.
    Your output should ONLY contain the name of the entity. DO NOT RETURN ANYTHING ELSE THAN THAT.
    Output format: Respond with EXACTLY that extracted phrase/word. No punctuation, no quotes, no explanations.

    Text: {o_part}{full_text_after_o}
    Addressed Entity:"""

        output = LLM(prompt=prompt, max_tokens=10, temperature=0.0, stop="\n\n")
        text = output["choices"][0]["text"].strip()

        # Wir schneiden das Subjekt aus dem Text aus, der NACH dem "O" kommt
        rest_of_text = full_text_after_o.replace(text, "", 1)

        return (
            (
                rf"{{\b1\c&HFFFFFF&\fs55}}{o_part}{text.upper()}"
                rf"{{\b0\c&HFFFFFF&\fs45}}{rest_of_text}"
            )
            if text.lower() in full_text_after_o.lower()
            else (f"{o_part}{full_text_after_o}")
        )

    matches = re.sub(r"(\s*[oO]\s+)(.+)", call_to_action_replacer, verse)

    # 3. Ausrufezeichen-Nachdruck (Macht den Satz bis zum ! fett)
    matches = re.sub(r"CN_START([^.!?]*?!)", r"{\\b1}\1{\\b0}", verse)

    # 4. Klammern dezent formatieren

    detected_matches = []

    def format_nested_parentheses_with_pyparsing(verse):
        # 1. CharsNotIn (großes C) definiert Text außerhalb der Klammern
        text_outside = CharsNotIn("()")

        # 2. Findet alles zwischen ( und ), egal wie tief verschachtelt
        parentheses_content = originalTextFor(nestedExpr("(", ")"))

        # 3. Der gesamte Parser sucht entweder nach Text ODER nach einer Klammer
        parser = (text_outside | parentheses_content)[...]

        # 4. Den Vers parsen (gibt eine Liste von Textbausteinen zurück)
        parsed_tokens = parser.parseString(verse).asList()

        # 5. Die Bausteine wieder zusammensetzen und die Klammern dabei formatieren
        result = []
        for token in parsed_tokens:
            if token.startswith("(") and token.endswith(")"):
                # Es ist ein Klammerblock -> ASS-Formatierung anwenden
                # token[1:-1] schneidet die äußeren Klammern ab
                inner_content = token[1:-1]
                result.append(
                    rf"{{\fs30\c&HAAAAAA&}}({inner_content}){{\fs40\c&HFFFFFF&}}"
                )
            else:
                # Es ist normaler Text außerhalb der Klammern
                result.append(token)

        return "".join(result)

    matches = format_nested_parentheses_with_pyparsing(verse)

    verse_tokens = matches.split()
    for i, w in highlights.items():
        cat = normalize(w).strip()
        if cat in COLOR_MAP:
            detected_matches.append(
                {
                    "index": i,
                    "word": verse_tokens[i] if i < len(verse_tokens) else "",
                    "category": cat,
                    "color": COLOR_MAP[cat],
                }
            )
        else:
            cat = "NONE"
            detected_matches.append(
                {
                    "index": i,
                    "word": verse_tokens[i] if i < len(verse_tokens) else "",
                    "category": cat,
                    "color": COLOR_MAP[cat],
                }
            )
    ass_lines = []
    highlighted_ids = [el["index"] for el in detected_matches]
    for i, token in enumerate(verse_tokens):
        if i in highlighted_ids:
            w = next((match for match in detected_matches if match["index"] == i), None)
            if w is None:
                w = token
            elif r"\c&HAAAAAA&" in token:
                colored = token.replace(r"\c&HAAAAAA&", rf"\c{w['color']}")
                w = rf"{{\b1}}{colored}{{\b0}}" if w["category"] == "GOD" else colored
            elif w["category"] == "GOD":
                w = f"{{\\c{w['color']}}}{{\\b1}}{token}{{\\b0}}{{\\c}}"
            else:
                w = f"{{\\c{w['color']}}}{token}{{\\c}}"
        else:
            w = token
        ass_lines.append(w)
    return " ".join(ass_lines)


def build_ass(
    file_name,
    entries,
    width,
    height,
    duration,
    font_name="Cormorant Garamond",
    treat_first_as_header: bool = True,
    word_alignments: Optional[List[dict]] = None,
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
    semantic_indexes: Dict = get_annotated_text(file_name)  # verse_id: content
    last_verse_nums: set = set()

    for i, e in enumerate(entries_sorted):
        start = e.start
        end = entries_sorted[i + 1].start if i + 1 < len(entries_sorted) else duration
        if only_header:
            end = min(duration, start + 6.0)
        if end <= start:
            end = start + 0.25

        # LLM-basierte Annotation & Formatierungen anwenden
        semantic_content: Dict = {}
        tokens = e.text.split()
        verse_nums = set()
        for t in tokens:
            m = re.match(r"^(\d+):$", t)
            if m:
                verse_nums.add(int(m.group(1)))
        if verse_nums:
            last_verse_nums = verse_nums
        elif last_verse_nums:
            verse_nums = last_verse_nums
        for vn in verse_nums:
            ann = semantic_indexes.get(vn, {})
            if not ann:
                continue
            prefix = f"{vn}:"
            if prefix in tokens:
                offset = tokens.index(prefix)
                for k, v in ann.items():
                    semantic_content[int(k) + offset + 1] = v

        highlighted = annotate_highlights(e.text, semantic_content)
        wrapped = wrap_ass_text(
            highlighted if highlighted else e.text, width, font_size
        )

        wa_for_entry = (
            [wa for wa in word_alignments if wa.get("ayah") in verse_nums]
            if word_alignments
            else []
        )

        line_count = wrapped.count("\n") + 1
        x = int(width * 0.5)
        y = int(height * (0.72 if line_count == 5 else 0.75))
        y_adjusted = y + (line_height // 2 if line_count >= 5 else 0)

        if i == 0 and treat_first_as_header:
            header_text = karaoke_reveal_words(wrapped)
            header_font_size = int(font_size * 1.25)
            tags = rf"\an5\pos({x},{int(height * 0.7)})\fad(200,200)\fs{header_font_size}\b1\c&H00FFFFFF&\3c&H00000000&\bord1\blur0"
            events.append(
                f"Dialogue: 0,{sec_to_ass_time(start)},{sec_to_ass_time(end)},Overlay,,0,0,0,,{{{tags}}}{header_text}"
            )
            continue

        word_lines = _progressive_word_lines(
            wrapped, start, end, wa_for_entry,
            font_size=font_size, video_width=width, video_height=height,
        )
        for line_text, w_start, w_end in word_lines:
            line_tags = rf"\an5\pos({x},{y_adjusted})\bord1\blur0"
            events.append(
                f"Dialogue: 0,{sec_to_ass_time(w_start)},{sec_to_ass_time(w_end)},Overlay,,0,0,0,,{{{line_tags}}}{line_text}"
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
    parser.add_argument(
        "mapping",
        type=Path,
    )
    parser.add_argument(
        "video",
        type=Path,
    )
    parser.add_argument("output", type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)
    file_name = args.output.stem
    width, height, duration = ffprobe_get(args.video)

    entries = []
    for line in args.mapping.read_text(encoding="utf-8").splitlines():
        if line.strip():
            e = parse_timestamp(line)
            if e:
                entries.append(e)

    if not entries:
        raise RuntimeError("Keine gültigen Zeilen gefunden.")

    word_alignments = []
    word_align_path = BASE_DIR / "output" / "word_align.json"
    if word_align_path.exists():
        word_alignments = json.loads(word_align_path.read_text(encoding="utf-8"))

    ass_text = build_ass(
        file_name, entries, width, height, duration, word_alignments=word_alignments
    )
    args.output.write_text(ass_text, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
