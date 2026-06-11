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

# fmt: off

REFERENCE_THEMES = {
   "GOD": re.compile(r"(?i)(?:(?:al|ar|as|at|az|ad|ah|ak|aq|am|an|aw)[- ]?)?(?:ra[hk]?m[ae]?n|ra[hk]?[yi]m|malik|qudd[u]?s|sala[ae]?m|m[u']?min|'?az[yi]z|jabb[a]?r|mutakabbir|khal[i]?q|b[a]?ri[']?|mu[s]?awwir|ghaff[a]?r|qahh[a]?r|wahh[a]?b|razz[a]?q|fatta[eh]|'?ali[ae]?m|q[a]?bi[zd]|b[a]?si[tdt]|kh[a]?fi[dz]|r[a]?fi[']?|m[u']?izz|mudhill|s[a]?mi[']?|b[a]?si[ry]|h[a]?kam|'?adl|la[td]i[yf]|khab[yi]r|h[a]?li[ym]|'?a[z]?i[ym]|ghaf[uo]r|shak[uo]r|'?a[l]?i[']?|kab[yi]r|h[a]?fi[z]|muq[yi]t|h[a]?s[yi]b|jal[yi]l|kar[yi]m|r[a]?q[yi]b|muj[yi]b|w[a]?si[']?|h[a]?k[yi]m|wad[uo]d|maj[yi]d|b[a]?[']?ith|shah[yi]d|h[a]?qq|wak[yi]l|qaw[wi]|mat[yi]n|wal[yi]|h[a]?mi[zd]|muh[s]?[yi]|mubd[yi]|mu[']?[yi]d|muh[yi]|mum[yi]t|h[ae]yy|qayy[uo]m|w[a]?jid|w[a]?hid|'?a[h]?ad|s[a]?mad|q[a]?dir|muqtadir|muqaddim|mu[']?akhkhir|'?awwal|'?akhir|z[a]?hir|b[a]?tin|muta[']?ali|barr|taww[a]?b|muntaqim|'?afuww|ra[']?uf|n[uy]r|h[au]d[ae]|'?a[z]?h[ae]r|b[au]t[ae]?n|'?aww[ae]l|'?a[hk]i[r]|wali[ae]?|mawla[ae]?|nas[yi]r|q[ae]ri[bd]|m[au]ji[d]|shak[uo]r|'?al[yi]y[ae]?|'?a[zd]?h[ae]?m|'?a[kl]?r[ae]?m|'?a[hk]?s[ae]?n|'?aj[ae]?m[ae]?l|'?aj[ae]?w[ae]?d|'?ak[ae]?r[ae]?m|'?a[kl]?b[ae]?r|'?a'?l[ae]?|'?a'?z[ae]?|'?a'?z[ae]?m|'?a[hk]?f[ae]?d|'?a[hk]?f[ae]?z|'?a[hk]?k[ae]?m|'?a[hk]?l[ae]?q|'?a[hk]?m[ae]?d|'?a[hk]?n[ae]?|'?a[hk]?n[ae]?m|'?a[hk]?n[ae]?t|'?a[hk]?n[ae]?y|'?a[hk]?q[ae]?r|'?a[hk]?r[ae]?m|'?a[hk]?s[ae]?b|'?a[hk]?s[ae]?n|'?a[hk]?s[ae]?r|'?a[hk]?t[ae]?n|'?a[hk]?t[ae]?r|'?a[hk]?y[ae]?r|'?a[hk]?z[ae]?|'?a[hk]?z[ae]?m)(?:\s+(?:the\s+)?(?:all|most|oft)[- ]?(?:mighty|merciful|beneficent|great|high|generous|kind|just|near|forgiving|wise|knower|hearer|seer|aware|provider|strong|knowing|capable|holy|compassionate|majestic|glorious|living|eternal|subduer|avenger|sustainer|gracious|faithful|forbearing|magnificent|appreciative|preserver|reckoner|watchful|responsive|embracing|loving|resurrector|witness|truth|trustee|firm|friend|praiseworthy|originator|restorer|giver\s+of\s+life|taker\s+of\s+life|self[- ]subsisting|finder|one|unique|able|determined|expediter|delayer|first|last|manifest|hidden|patron|self[- ]exalted|source\s+of\s+goodness|acceptor\s+of\s+repentance|pardoner|kind|compeller|opener|withholder|expander|abaser|exalter|honorer|dishonorer|judge|subtle|all-just|all-forgiving|all-merciful|all-compassionate|all-majestic|all-glorious|mostholy|mostcompassionate|thebeneficent|themerciful|theforgiving|thejust|themighty|thewise|theliving|theeternal|thesubduer|theavenger))?(?:(?:allah|god|lord|albarr|omnipotentking|supremecreator|realbestower|ownerofpower|king|sovereign|master|creator|maker|fashioner|subduer|bestower|provider|opener|judge|avenger|pardoner|compeller|expander|abaser|exalter|honorer|dishonorer|withholder|subtle|gracious|faithful|forbearing|magnificent|appreciative|preserver|reckoner|watchful|responsive|loving|resurrector|witness|trustee|friend|praiseworthy|originator|restorer|finder|determined|expediter|delayer|manifest|patron|source\s+of\s+goodness|acceptor\s+of\s+repentance))", re.IGNORECASE), 
    "DESTRUCTIVE": re.compile(
        r"^("
r"hell(?:fire)?|blaze|blazing\s+fire|burning\s+fire|torment\w*|chastis\w*|punish\w*|tortur\w*|scourge\w*|doom\w*|disgrace\w*|"r"gehenna|jahannam|abyss|furnace|boiling\s+water|fetters|chains|"r"strik\w*|seiz\w*|destroy\w*|destruct\w*|curs\w*|reject\w*|belying|belied|wrath|anger|retribution|vengeance|penalty|calamity|ruin\w*|perish\w*|overthrow\w*|"r"painful|severe|grievous|miserable|awful|dreadful|terrible|heavy|wretched|wicked|scorching|"r"disbeliev\w*|unbeliev\w*|wrongdoer\w*|polytheist\w*|sinner\w*|arrogant\w*|transgress\w*|hypocri\w*|evildoer\w*|disobedien\w*|rebel\w*|insolent\w*|infidel\w*|idolater\w*|misguid\w*|stray\w*|"r"blasphem\w*|envy|envious|hatred|malice|regret\w*|remorse\w*|despair\w*"r")$"r"|saqar|saqr|hutamah|laz[aā]\w*|jah[iī]m|sa[iī]r|haawiyah|h[aaā]wiyah|"r"zaqqum|zaqq[ou]*m|ghisl[iī]n|ghass[aa]*q|ham[iī]m|"r"sijjin|sijji[n]|al-sijjin|"r"dari\w*|dhar[iī]\w*|samum\w*|samoom\w*|dukhan\w*|"r"fir['`]?[aā]wn\w*|pharaoh\w*|"r"tham[uū]d\w*|samood\w*|"r"(?:['`]?)?aad\w*|"r"madyan\w*|midianit\w*|"r"q[aaā]r[uū]n\w*|korah\w*|"r"ab[uū]\s+lahab|lahab\w*|"r"nimrod\w*|"r"sodom\w*|gomorrah\w*|tubba\w*|"r"sting\w*|miser\w*|miserl\w*|"r"boast\w*|vainglor\w*|"r"slander\w*|backbit\w*|"r"mock\w*|ridicul\w*|"r"corrupt\w*|corrupter\w*|"r"oppress\w*|oppressor\w*|"r"tyrann\w*|"r"ingrat\w*|ungrateful\w*|"r"deceiv\w*|decepti\w*|"r"heedless\w*|"r"greed\w*|avaric\w*|covet\w*|"r"nif[aaā]q\w*|mun[aaā]fiq\w*|"r"shirk\w*|mushrik\w*|mushrik[ou]n|"r"bagh[yī]\w*|"r"fas[aaā]d\w*|mufsid\w*|"r"rijs\w*|rijz\w*|"r"najis\w*|"r"fitnah\w*",re.IGNORECASE,),
    "CONSTRUCTIVE": re.compile(
        r"^("
r"paradise|garden\w*|eden|jannah|firdaus|abode\s+of\s+peace|"r"light|peace|bliss|mercy|grace|glor\w*|victory|victor\w*|bount\w*|favor\w*|salvation|prosperity|"r"forgiv\w*|pardon\w*|guid\w*|purif\w*|save\w*|admit\w*|reward\w*|rejoic\w*|triumph\w*|bless\w*|"r"beautiful|pure|eternal|successful|radiant|noble|trustworthy|truth|glorio\w*|content\w*|"r"believer\w*|righteous\w*|pious|humble|muttaqoon|muslim\w*|submitter\w*|patient|steadfast\w*|devout\w*|repent\w*|good-doer\w*|truthful|"r"qanit\w*|sabir\w*|tawwab\w*|"r"tawakkul\w*|sabr\w*|rid[aaā]\w*|contentment\w*"r")$"r"|kawthar\w*|"r"tasn[iī]m\w*|"r"salsab[iī]l\w*|"r"ma['`]?i[nī]?n\w*|zanjab[iī]l\w*|kaf[uū]r\w*|salhab\w*|"r"israfil\w*|azra['`]?il\w*|harut\w*|marut\w*|"r"houri\w*|hur\w*|"r"sidrat\w*|sidrah\w*|"r"illiyy[iī]n\w*|"r"ridw[aaā]n\w*|"r"sak[iī]nah\w*|"r"shukr\w*|shakir\w*|shak[uu]r\w*|grateful\w*|thankful\w*|"r"ikhl[aaā]s\w*|sincere\w*|sincerity|"r"taqw[aaā]\w*|god[-]fearing|god[-]conscious|"r"ihs[aaā]n\w*|muhsin\w*|excellent\w*|"r"salih\w*|s[aā]lih\w*|"r"aww[aaā]b\w*|"r"martyr\w*|shah[iī]d\w*|shuhad[aa]*\w*|"r"muqarrab\w*|near[-]brought|nearest\s+to\s+allah|"r"jibr[iī]l\w*|gabrie?l\w*|"r"m[iī]k[aaā]l\w*|michae?l\w*|"r"ridwan\w*|"r"guardian[-]angel\w*|"r"keepers?\s+of\s+paradise|"r"companions?\s+of\s+the\s+right|"r"people\s+of\s+the\s+right|"r"right[-]handed|"r"sidd[iī]q\w*|truthful[-]one\w*",re.IGNORECASE,),
}

# mock_json="""{"labels":[{"word":"Al-Muddathir","category":"GOD"},{"word":"ARISE","category":"CONSTRUCTIVE"},{"word":"WARN","category":"DESTRUCTIVE"},{"word":"ALLAH","category":"GOD"},{"word":"purify","category":"CONSTRUCTIVE"},{"word":"Ar-Rujz","category":"DESTRUCTIVE"},{"word":"give","category":"CONSTRUCTIVE"},{"word":"deeds","category":"CONSTRUCTIVE"},{"word":"obedience","category":"CONSTRUCTIVE"},{"word":"patient","category":"CONSTRUCTIVE"},{"word":"transgressors","category":"DESTRUCTIVE"},{"word":"Allah","category":"GOD"},{"word":"Trumpet","category":"DESTRUCTIVE"},{"word":"gather","category":"CONSTRUCTIVE"},{"word":"Hard","category":"DESTRUCTIVE"},{"word":"disbelievers","category":"DESTRUCTIVE"},{"word":"Al-Waleed","category":"GOD"},{"word":"bin","category":"GOD"},{"word":"Al-Mugheerah","category":"GOD"},{"word":"Al-Makhzoomee","category":"GOD"},{"word":"granted","category":"CONSTRUCTIVE"},{"word":"abundance","category":"CONSTRUCTIVE"},{"word":"smooth","category":"CONSTRUCTIVE"},{"word":"comfortable","category":"CONSTRUCTIVE"},{"word":"God","category":"GOD"},{"word":"strength","category":"CONSTRUCTIVE"},{"word":"rock","category":"CONSTRUCTIVE"},{"word":"hills","category":"CONSTRUCTIVE"},{"word":"understanding","category":"CONSTRUCTIVE"},{"word":"formed","category":"CONSTRUCTIVE"},{"word":"stubborn","category":"DESTRUCTIVE"},{"word":"opposing","category":"DESTRUCTIVE"},{"word":"torment","category":"DESTRUCTIVE"},{"word":"Verily","category":"GOD"},{"word":"cursed","category":"DESTRUCTIVE"},{"word":"plotted","category":"DESTRUCTIVE"},{"word":"plot","category":"DESTRUCTIVE"},{"word":"god","category":"GOD"},{"word":"create","category":"CONSTRUCTIVE"},{"word":"destroy","category":"DESTRUCTIVE"},{"word":"life","category":"CONSTRUCTIVE"},{"word":"death","category":"DESTRUCTIVE"},{"word":"frowned","category":"DESTRUCTIVE"}]}"""
# fmt: on


import json


import json


def check_semantics(sentence, matches):
    """
    matches: Liste von Dicts/Tuples, z.B.:
             [{"index": 3, "word": "Gott", "category": "GOD"}, ...]
    """
    # Erstelle eine lesbare Liste der Zielwörter für den Prompt
    targets_string = "\n".join(
        [
            f'- Index {m["index"]}: "{m["word"]}" (Kategorie: {m["category"]})'
            for m in matches
        ]
    )

    prompt = f"""You are a precise semantic validation engine.
    TASK:
    Analyze the sentence below and determine for each listed target word if it truly refers to its assigned category in this exact semantic context.

    CRITERIA FOR CATEGORIES:
    1. GOD (Deity & Divine Attributes)
    - MUST refer ONLY to the Supreme Deity/Creator (e.g., Allah, Lord, the Merciful).
    - DO NOT match if it refers to creation

    2. DESTRUCTIVE (Sin, Disbelief, Punishment, and Cosmic Destruction)
    - MUST match explicit acts of destruction, divine punishment, torment (e.g., Hell, punishment, doom).
    - MUST match terms of ultimate spiritual failure, rebellion, or sin: This explicitly includes "disbelievers" (Kafir), "sinners/criminals" (Mujrim), hypocrites, arrogance against God, and major sins.
    - CRITERIA: If the word embodies spiritual ruin, hostility to truth, or physical destruction, it is DESTRUCTIVE.

    3. CONSTRUCTIVE (Faith, Virtue, Divine Reward, and Creation)
    - MUST match explicit acts of creation, guidance, purification, and virtue (e.g., patience, charity).
    - MUST match terms of spiritual success and obedience: This explicitly includes "believers" (Mu'min), "the righteous", "angels" (as agents of divine good), and divine rewards (e.g., Paradise, blessings).
    - CRITERIA: If the word embodies spiritual success, divine alignment, moral virtue, or creation, it is CONSTRUCTIVE.

    SENTENCE:
    \"\"\"{sentence}\"\"\"

    TARGET WORDS TO VERIFY:
    {targets_string}

    OUTPUT RULE:
    Your response must be a valid JSON object where the keys are the string representation of the indexes, and values are booleans (true/false).
    Do NOT include any markdown formatting, backticks, or explanation.
    
    Example Output Format:
    {{
        "3": true,
        "7": false
    }}"""

    output = LLM(prompt, max_tokens=512, temperature=0.0)
    text = output["choices"][0]["text"].strip()

    try:
        # Konvertiert JSON-Keys ("3") zu Integern (3) für einfachere Handhabung
        decisions = {int(k): v for k, v in json.loads(text).items()}
    except Exception:
        decisions = {m["index"]: False for m in matches}

    # Konsolen-Validierung (bunter Satz)
    GRUEN = "\033[32m"
    RESET = "\033[0m"
    worte = sentence.split()
    for m in matches:
        idx = m["index"]
        if idx < len(worte) and decisions.get(idx, False):
            worte[idx] = f"{GRUEN}{worte[idx]}{RESET}"
    bunter_satz = " ".join(worte)

    ausgabe = f"{'#' * 55}\nSENTENCE: \t {bunter_satz}\n{'#' * 55}\nDECISIONS: \t {decisions}\n\n\n"
    print(ausgabe)

    return decisions


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
        # Build deterministic lookup: word -> category
        words = sentence.split(" ")
        processed = []

        # 1. Schritt: Alle Regex-Treffer im Satz sammeln
        detected_matches = []
        for i, w in enumerate(words):
            clean = normalize(w)
            clean_lower = clean.lower()

            for category, pattern in REFERENCE_THEMES.items():
                if pattern.search(clean_lower):
                    print(f"[REGEX] '{w.strip()}' -> Kategorie: {category}")
                    if category in COLOR_MAP:
                        detected_matches.append(
                            {
                                "index": i,
                                "word": w,
                                "category": category,
                                "color": COLOR_MAP[category],
                            }
                        )
                        break

        # 2. Schritt: Einmalige LLM-Validierung für den ganzen Satz
        llm_decisions = {}
        if detected_matches:
            llm_decisions = check_semantics(sentence, detected_matches)

        # 3. Schritt: Satz mit den validierten Treffern final verarbeiten
        for i, w in enumerate(words):
            match_info = next((m for m in detected_matches if m["index"] == i), None)

            if match_info and llm_decisions.get(i, False):
                if match_info["category"] == "GOD":
                    w = f"{{\\c{match_info['color']}}}{{\\b1}}{w}{{\\b0}}{{\\c}}"
                else:
                    w = f"{{\\c{match_info['color']}}}{w}{{\\c}}"

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
