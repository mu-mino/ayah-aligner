#!/usr/bin/env python3
import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np


BASE_DIR = Path(__file__).parent.parent

# LLM ist ungenutzt (Annotationen kommen aus get_annotated_text/Regex) –
# deshalb lazy laden, damit der Modul-Import nicht blockiert.
_LLM = None


def _get_llm():
    global _LLM
    if _LLM is None:
        from llama_cpp import Llama

        _LLM = Llama(
            model_path=str(
                Path.home()
                / ".cache/huggingface/hub/models--bartowski--Qwen2.5-14B-Instruct-GGUF"
                / "snapshots/05244aa5d871c661c80082a15d3bce44714d068d"
                / "Qwen2.5-14B-Instruct-Q4_K_M.gguf"
            ),
            n_ctx=6000,
            n_threads=12,
            n_gpu_layers=-1,
            verbose=True,
        )
    return _LLM



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
    cs_total = int(round(t * 100))
    h, rem = divmod(cs_total, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _ts2sec(t: str) -> float:
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


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


DIVINE_ATTRIBUTES = {
    "beneficent", "merciful", "great", "high", "generous", "kind", "just",
    "near", "forgiving", "wise", "knower", "hearer", "seer", "aware",
    "provider", "strong", "knowing", "capable", "holy", "compassionate",
    "majestic", "glorious", "living", "eternal", "subduer", "avenger",
    "sustainer", "gracious", "faithful", "forbearing", "magnificent",
    "appreciative", "preserver", "reckoner", "watchful", "responsive",
    "embracing", "loving", "resurrector", "witness", "truth", "trustee",
    "firm", "friend", "praiseworthy", "originator", "restorer",
    "pardoner", "subtle", "beautiful", "powerful", "courteous", "ready",
}

KNOWN_DIVINE_COMPOUNDS = {
    "all-mighty", "all-knowing", "all-hearer", "all-seer", "all-aware",
    "all-wise", "all-provider", "all-strong", "all-capable", "all-just",
    "all-forgiving", "all-merciful", "all-compassionate", "all-majestic",
    "all-glorious", "all-subtle", "all-powerful",
    "allmighty", "allknowing", "allhearer", "allseer", "allaware",
    "allwise", "allprovider", "allstrong", "allcapable", "alljust",
    "allforgiving", "allmerciful", "allcompassionate", "allmajestic",
    "allglorious", "allsubtle", "allpowerful",
    "oft-forgiving", "oftforgiving",
}


# ---------------------------------------------------------------------------
# Eindeutige Regex-Klassifikation (ohne semantisches Matching / LLM)
#
# Basis: tatsaechlicher Wortgebrauch in eng_translation/chunked_translation/*.txt.
# Stems sind am Token-Anfang verankert, um Substring-False-Positives zu vermeiden
# (z.B. "commerce" != "merc", "disgrace" != "grace", "persevere" != "severe").
# _THEME_ARABIC = alternative Schreibweisen / arabische Transkriptionen (Fallback).
# _THEME_EXCLUSIONS entfernt nachweisliche False-Positives.
# ---------------------------------------------------------------------------

_THEME_STEMS = {
    "GOD": ["allah", "god", "lord"],
    "DESTRUCTIVE": [
        "torment", "hell", "fire", "disbeliev", "disbelief", "punish", "chastis",
        "doom", "curse", "wrath", "ruin", "perish", "destroy", "destruct",
        "arrogant", "polytheist", "idolater", "hypocri", "sinner", "wrongdoer",
        "transgress", "evildoer", "wicked", "disobedien", "rebel", "calamit",
        "afflict", "tortur", "scourge", "grievous", "painful", "severe", "misguid",
        "stray", "blasphem", "mock", "ridicul", "slander", "backbit", "corrupt",
        "oppress", "tyrann", "envy", "deceiv", "greed", "miser", "boast", "haught",
        "shirk", "disgrace", "abas", "humiliat", "despair", "regret", "remorse",
        "scorch", "blaze", "burn", "venge", "retribution", "penalty", "seiz",
        "dreadful", "awful", "terrible", "miserab", "wretch", "insolent", "ingrate",
        "ungrateful", "belied", "belying", "reject",
    ],
    "CONSTRUCTIVE": [
        "paradis", "garden", "eden", "jannah", "firdaus", "light", "peace", "bliss",
        "merc(?:y|iful|ies)", "grac", "glor", "victor", "bount", "favor", "salv",
        "prosper", "forgiv", "pardon", "guid", "purif", "sav", "reward", "rejoic",
        "triumph", "bless", "believ", "righteous", "pious", "humbl", "muslim",
        "submit", "patient", "patience", "steadfast", "devout", "repent", "truthful",
        "sincere", "taqwa", "sabr", "tawakkul", "content(?:ed|ment)?", "eternal",
        "radiant", "nobl", "trustworth", "truth", "houri", "salsabil", "kawthar",
        "siddiq", "martyr", "shahid", "good[- ]?deed", "righteous[- ]?deed", "pure",
        "excellent", "grateful", "thankful", "admit",
    ],
}

_THEME_ARABIC = {
    "GOD": [
        "rahman", "raheem", "malik", "quddus", "salam", "mumin", "aziz",
        "jabbar", "mutakabbir", "khaliq", "bari", "musawwir", "ghaffar", "qahhar",
        "wahhab", "razzaq", "fattah", "alim", "qabid", "basit", "khafid", "rafi",
        "muzill", "sami", "basir", "hakam", "adl", "latif", "khabir", "halim",
        "azim", "ghafur", "shakur", "ali", "kabir", "hafiz", "muqit", "hasib",
        "jalil", "karim", "raqib", "mujib", "wasi", "hakim", "wadud", "majid",
        "baith", "shahid", "haqq", "wakil", "qawiy", "matin", "wali", "hayy",
        "qayyum", "wahid", "ahad", "samad", "qadir", "muqtadir", "zahir", "batin",
        "barr", "tawwab", "muntaqim", "afuww", "rauf", "hadi", "baqi", "warith",
        "rashid", "sabur",
    ],
    "DESTRUCTIVE": [
        "mushrik", "mufsid", "munafiq", "nifaq", "rijs", "rijz", "fitnah",
        "jahannam", "gehenna", "saqar", "saqr", "hutamah", "lahab", "zaqqum",
        "ghislin", "hamim", "sijjin", "samum", "dukhan", "pharaoh", "firawn",
        "thamud", "madyan", "midian", "qarun", "sodom", "gomorrah", "nimrod",
        "tubba",
    ],
    "CONSTRUCTIVE": [
        "tasnim", "illiyyin", "ridwan", "sakinah", "sakina", "sidratulmuntaha",
        "zanjeebil", "zanjabil", "muqarrab", "qanitin",
    ],
}

_THEME_EXCLUSIONS = {
    "GOD": {"gods", "lords", "lordship", "alhadid", "assamiri", "assamit",
            "assalamu", "illallah"},
    "DESTRUCTIVE": {"abasa", "deaddestroyed"},
    "CONSTRUCTIVE": {"merchandise", "contents", "lightly", "lightning",
                     "commerce", "commercial", "enlightenment", "flight"},
}


def _compile_theme(cat: str) -> "re.Pattern":
    arabic = "|".join(_THEME_ARABIC[cat])
    prefix = r"(?:al|ar|as|at|az|ad|ah|ak|aq|am|an|aw)[- ]?"
    if cat == "GOD":
        # "god"/"lord" exakt (Plural = falsche Gottheiten), göttl. Namen mit
        # al-/ar-Praefix muessen das ganze Token abdecken (vermeidet "alhadid").
        stems = r"allah\w*|god|lord"
        return re.compile(rf"(?i)^(?:{stems}|{prefix}(?:{arabic}))$", re.IGNORECASE)
    stems = "|".join(_THEME_STEMS[cat])
    return re.compile(rf"(?i)^(?:(?:{stems})\w*|(?:{prefix})?(?:{arabic})\w*)", re.IGNORECASE)


REFERENCE_THEMES = {cat: _compile_theme(cat) for cat in _THEME_STEMS}

def annotate_highlights(verse, highlights, apply_regex=True):
    tokens = verse.split()
    n = len(tokens)

    # Step 1: normalize AI annotations to int keys
    proc = {}
    for k, v in highlights.items():
        proc[int(k)] = v.upper() if isinstance(v, str) else v

    # Step 2: "Most" → attribute shift
    for idx in sorted(proc.keys()):
        word = normalize(tokens[idx]) if idx < n else ""
        if word.lower() == "most" and proc[idx] == "GOD":
            if idx + 1 < n:
                next_word = normalize(tokens[idx + 1]).lower()
                if next_word in DIVINE_ATTRIBUTES:
                    del proc[idx]
                    proc[idx + 1] = "GOD"

    # Step 3: add missing unambiguous words
    for i, token in enumerate(tokens):
        clean = normalize(token).lower()
        normed = normalize(token)
        if clean in {"allah", "allâh"} and i not in proc:
            proc[i] = "GOD"
        if clean == "lord" and normed and normed[0].isupper() and i not in proc:
            proc[i] = "GOD"
        raw_clean = token.strip(".,;:!?\"'()[]{}").lower()
        if raw_clean in KNOWN_DIVINE_COMPOUNDS and i not in proc:
            proc[i] = "GOD"

    # Step 3a: eindeutige Regex-Klassifikation (ohne semantisches Matching).
    # Basis: Wortgebrauch in eng_translation/chunked_translation. GOD höchste
    # Priorität; DESTRUCTIVE/CONSTRUCTIVE füllen nur nicht-klassifizierte Token.
    # _THEME_EXCLUSIONS entfernt nachweisliche False-Positives.
    if apply_regex:
        for i, token in enumerate(tokens):
            clean_lower = normalize(token).lower()
            if not clean_lower:
                continue
            cat = None
            for c, pat in REFERENCE_THEMES.items():
                if pat.search(clean_lower):
                    cat = c
                    break
            if cat is None or clean_lower in _THEME_EXCLUSIONS.get(cat, ()):
                continue
            if cat == "GOD":
                proc[i] = "GOD"
            elif i not in proc:
                proc[i] = cat

    # Step 3b: override misclassifications for unambiguous divine words
    for i, token in enumerate(tokens):
        clean = normalize(token).lower()
        normed = normalize(token)
        cat = proc.get(i)
        if cat and cat != "GOD":
            if clean in {"allah", "allâh"}:
                proc[i] = "GOD"
            if clean == "lord" and normed and normed[0].isupper():
                proc[i] = "GOD"

    # Step 3c: strip GOD from known false-positive categories
    NEVER_GOD = {
        "the", "of", "in", "from", "to", "for", "with", "by", "at", "on",
        "and", "but", "or", "nor", "yet", "so", "if", "then", "than", "as",
        "is", "are", "was", "were", "be", "been", "being",
        "has", "have", "had", "do", "does", "did",
        "will", "would", "shall", "should", "can", "could", "may", "might",
        "this", "that", "these", "those",
        "it", "its", "itself", "they", "them", "their", "theirs",
        "who", "whom", "whose", "which", "what",
        "only", "just", "even", "also", "too", "very", "still", "already",
        "never", "always", "often", "sometimes",
        "muhammad", "moses", "moosa", "ibrahim", "maryam",
        "angel", "angels", "prophet", "prophets",
        "quran", "book", "books", "ayat", "verses",
        "heaven", "heavens", "earth", "sky", "throne",
        "day", "hour", "resurrection",
        "worship", "salat", "zakat",
        "paradise", "hell", "fire", "torment", "recompense",
        "etc", "ie", "v",
    }
    for i, token in enumerate(tokens):
        clean = normalize(token).lower()
        if clean in NEVER_GOD:
            if i in proc:
                del proc[i]

    # Step 4: track parentheses depth
    paren_depth = [0] * n
    depth = 0
    for i, token in enumerate(tokens):
        depth += token.count("(")
        paren_depth[i] = depth
        depth -= token.count(")")

    # Step 5: single-pass token formatting
    ass_lines = []
    for i, token in enumerate(tokens):
        in_parens = paren_depth[i] > 0
        hl = proc.get(i)

        if hl:
            color = COLOR_MAP.get(hl, COLOR_MAP["NONE"])
            if hl == "GOD":
                ass_lines.append(f"{{\\c{color}}}{{\\b1}}{token}{{\\b0}}{{\\c}}")
            else:
                ass_lines.append(f"{{\\c{color}}}{token}{{\\c}}")
        elif in_parens:
            is_first = i == 0 or paren_depth[i - 1] == 0
            is_last = i + 1 >= n or paren_depth[i + 1] == 0
            if is_first and not is_last:
                ass_lines.append(f"{{\\fs30\\c&HAAAAAA&}}{token}")
            elif is_last and not is_first:
                ass_lines.append(f"{{\\c&HAAAAAA&}}{token}{{\\fs40\\c&HFFFFFF&}}")
            elif is_first and is_last:
                ass_lines.append(f"{{\\fs30\\c&HAAAAAA&}}{token}{{\\fs40\\c&HFFFFFF&}}")
            else:
                ass_lines.append(f"{{\\c&HAAAAAA&}}{token}")
        else:
            ass_lines.append(token)

    return " ".join(ass_lines)


def build_ass(
    file_name,
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

        is_header = i == 0 and treat_first_as_header
        highlighted = annotate_highlights(
            e.text, semantic_content, apply_regex=not is_header
        )
        wrapped = wrap_ass_text(
            highlighted if highlighted else e.text, width, font_size
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

        txt = karaoke_reveal_words(wrapped)
        line_tags = rf"\an5\pos({x},{y_adjusted})\bord1\blur0\fad(120,150)"
        events.append(
            f"Dialogue: 0,{sec_to_ass_time(start)},{sec_to_ass_time(end)},Overlay,,0,0,0,,{{{line_tags}}}{txt}"
        )

    # Post-processing: fill render pauses > 0.5s and resolve overlaps
    ts_pairs = []
    for ev in events:
        m = re.match(r"Dialogue: \d,([^,]+),([^,]+),", ev)
        if m:
            ts_pairs.append((_ts2sec(m.group(1)), _ts2sec(m.group(2)), ev))
    ts_pairs.sort(key=lambda x: x[0])

    # Phase 1 — fill pauses > 0.5s by extending previous event end
    for i in range(1, len(ts_pairs)):
        gap = ts_pairs[i][0] - ts_pairs[i - 1][1]
        if gap > 0.5:
            new_end = ts_pairs[i][0]
            ts_pairs[i - 1] = (
                ts_pairs[i - 1][0],
                new_end,
                re.sub(
                    r"(Dialogue: \d,[^,]+),([^,]+)",
                    rf"\1,{sec_to_ass_time(new_end)}",
                    ts_pairs[i - 1][2],
                    count=1,
                ),
            )

    # Phase 2 — resolve overlaps: shift event start forward if it begins before previous ends
    for i in range(1, len(ts_pairs)):
        prev_end = ts_pairs[i - 1][1]
        cur_start, cur_end, cur_ev = ts_pairs[i]
        if cur_start < prev_end:
            new_start = prev_end
            new_end = max(cur_end, new_start + 0.01)
            new_start_str = sec_to_ass_time(new_start)
            new_end_str = sec_to_ass_time(new_end)
            ts_pairs[i] = (
                new_start,
                new_end,
                re.sub(
                    r"(Dialogue: \d,)[^,]+,[^,]+",
                    rf"\g<1>{new_start_str},{new_end_str}",
                    cur_ev,
                    count=1,
                ),
            )

    events = [t[2] for t in ts_pairs]

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

    ass_text = build_ass(file_name, entries, width, height, duration)
    args.output.write_text(ass_text, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
