import requests
import shlex
import paramiko
import sys
import time
import json
import os
import glob
import re
from pathlib import Path
from typing import Optional, List, Tuple
import stanza

stanza.download("en", processors="tokenize,pos")
nlp = stanza.Pipeline("en", processors="tokenize,pos", use_gpu=True, quiet=True)

# --- Hilfsfunktionen für das Text-Parsing und Splitten ---


def parse_text_file(path: Path) -> Tuple[List[str], List[str]]:
    """
    Liest eine Textdatei und gibt (title_lines, numbered_lines) zurück.

    Nummerierte Zeilen (z. B. "1. Text" oder "1) Text") werden als Verse
    erkannt, sofern die Nummerierung sequenziell ist. Nicht-nummerierte
    Zeilen vor dem ersten Vers werden als Titelzeilen behandelt.
    Fortsetzungszeilen ohne eigene Nummer werden an die vorherige Zeile
    angehängt.
    """
    title_lines: List[str] = []
    numbered_lines: List[str] = []
    numbered = re.compile(r"^(\d+)[\.)]\s*(.*)")
    expected_next: Optional[int] = None

    with path.open(encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            m = numbered.match(stripped)
            if m:
                idx = int(m.group(1))
                text = m.group(2).strip()
                if expected_next is None or idx == expected_next:
                    expected_next = idx + 1
                    numbered_lines.append(text if text else stripped)
                    continue
            if numbered_lines:
                numbered_lines[-1] = f"{numbered_lines[-1]} {stripped}"
            else:
                title_lines.append(stripped)

    return title_lines, numbered_lines


def split_into_sentences(text: str) -> list:
    """
    Trennt einen Text in Sätze auf.
    Hier simpel gelöst durch Satzzeichen. Bei Bedarf durch NLTK/Spacy ersetzen.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s]


def split_into_tokens(sentence: str) -> list:
    """Trennt einen Satz in einzelne Wörter (Tokens)."""
    return re.findall(r"\b\w+\b|[^\w\s]", sentence)


# --- Die Haupt-Generator-Funktion ---


def process_directory(directory_path: str):
    """
    Liest alle .txt Dateien über parse_text_file ein und verarbeitet die nummerierten Verse.
    Yieldet pro Datei: (filepath, file_structure)
    """
    search_pattern = os.path.join(directory_path, "*.txt")

    for filepath in glob.iglob(search_pattern):
        file_structure = []  # Hält alle Verse dieser Datei
        path_obj = Path(filepath)

        # Verwende deine neue sequenzielle Parsing-Logik
        _, numbered_lines = parse_text_file(path_obj)

        # Verarbeite die extrahierten, nummerierten Verse
        # Da numbered_lines sequenziell ab Vers 1 befüllt wird, nutzen wir enumerate(..., start=1)
        for verse_idx, verse_text in enumerate(numbered_lines, start=1):
            if verse_text:
                verse_structure = {"verse_number": verse_idx, "sentences": []}

                sentences = split_into_sentences(verse_text)

                for sentence in sentences:
                    tokens = split_into_tokens(sentence)
                    matches = extract_meaningful_tokens(sentence)

                    # LLM semantische Prüfung aufrufen (gibt Dict zurück: {index: "KATEGORIE"})
                    decisions = check_semantics(sentences, sentence, matches)

                    # Gefundene Kategorien in die Matches-Struktur integrieren
                    for m in matches:
                        idx = m["index"]
                        if idx in decisions:
                            m["category"] = decisions[idx]

                    verse_structure["sentences"].append(
                        {"sentence_context": sentence, "tokens": matches}
                    )

                file_structure.append(verse_structure)

        yield filepath, file_structure


def extract_meaningful_tokens(sentence: str) -> list:
    """
    Nutzt Stanza, um den Satz zu analysieren und nur Wörter mit echter Bedeutung
    (Nomen, Verben, Adjektive, Eigennamen) inklusive ihres originalen Wort-Indexes zurückzugeben.
    """
    doc = nlp(sentence)
    meaningful_matches = []

    FORBIDDEN_POS = {"PUNCT", "DET", "CCONJ", "SCONJ", "PRON", "ADP", "PART"}

    token_index = 0
    for stanza_sentence in doc.sentences:
        for word in stanza_sentence.words:
            if word.upos not in FORBIDDEN_POS:
                meaningful_matches.append(
                    {
                        "index": token_index,
                        "word": word.text,
                    }
                )
            token_index += 1

    return meaningful_matches


# --- Semantische Analyse ---


def run_llama(prompt):
    url = "http://127.0.0.1:8080/completion"

    headers = {"Content-Type": "application/json"}

    payload = {
        "prompt": prompt,
        "temperature": 0.0,
        "n_predict": 256,
        "stop": ["}", "<|im_end|>", "<|endoftext|>"],
    }
    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        result_json = response.json()
        ki_text_output = result_json.get("content", "")
        if not ki_text_output.endswith("}"):
            ki_text_output = ki_text_output.replace("'", "") + "}"
        print("Gefiltertes Ergebnis:")
        print(json.dumps(ki_text_output, indent=2))
        return ki_text_output
    else:
        print(f"Fehler beim Server-Aufruf: {response.status_code}")
        bereinigtes_ergebnis = {}
        return bereinigtes_ergebnis


def check_semantics(verse, sentence, tokens):
    """
    matches: Liste von Dicts/Tuples, z.B.:
             [{"index": 3, "word": "Gott"}, ...]
    """
    targets_string = "\n".join(
        [f'- Index {m["index"]}: "{m["word"]}" )' for m in tokens]
    )
    prompt = f"""
    You are a precise semantic validation engine.
    TASK:
    Analyze the sentence and determine for each target word its semantic category based STRICTLY on the definitions below. Use the context of the whole sentence.

    CATEGORIES (only these four strings are allowed):
    - "GOD"
    - "DESTRUCTIVE"
    - "CONSTRUCTIVE"
    - "NONE"

    CRITERIA:

    1. GOD (Deity & Divine Attributes)
    - MUST refer ONLY to the Supreme Creator, His names, His attributes, and His revealed Word (the Quran as a whole, but NOT single letters or disjointed letters like "Ha-Meem").
    - Examples of divine attributes (always GOD, never CONSTRUCTIVE): All-Mighty, All-Knowing, All-Hearing, Oft-Forgiving, Most Merciful, All-Wise, the Irresistible, the Seer, etc.
    - Also includes: Allah, Lord (when referring to God), the Quran (as a book), His Throne, His Command (when divine).
    - DO NOT match if it refers to creation (e.g., "lord" as a human master) or to ordinary attributes.

    2. DESTRUCTIVE (Sin, Disbelief, Punishment, Cosmic Destruction)
    - Explicit acts of divine punishment, retribution, torment, hellfire, cosmic destruction.
    - Terms of spiritual failure: disbelievers, sinners, criminals, hypocrites, wrongdoers, polytheists, evildoers, rebellious, arrogant, transgressors.
    - Specific sins: backbiting, slandering, mocking, spying, envy, hatred, deceit, oppression, tyranny, ingratitude.
    - Names of Hell and its torments: Hellfire, blazing fire, Saqar, Hutamah, Laza, Jahim, Hawiyah, Zaqqum, Ghislin, Hamim, fetters, chains.
    - Cosmic destruction / historical punishments: 'Aad, Thamud, Pharaoh, Qarun, flood, earthquake, etc. – when mentioned as destroyed or punished.
    - Verbs of threatening, seizing, punishing, humiliating, warning of doom.
    - **Note:** "fear" when it is fear of other than God or worldly fear is DESTRUCTIVE; fear of God is CONSTRUCTIVE (see below).

    3. CONSTRUCTIVE (Faith, Virtue, Divine Reward, Cosmic Creation)
    - Acts of creation, life-giving, divine reward, ultimate bliss, salvation.
    - Terms of spiritual success: believers, righteous, pious, submitters, truthful, martyrs, those brought near (Muqarrabun).
    - Virtues: patience, humility, repentance, gratitude, sincerity, God-consciousness (Taqwa), excellence (Ihsan), trust in God, contentment.
    - Names of Paradise: Jannah, Gardens of Eden, Firdaws, Na‘im, Darus-Salam, Illiyyin.
    - Heavenly delights: Tuba-tree, Sidrah, Kawthar, Tasnim, Salsabil, pure milk, honey, non-intoxicating wine.
    - Acts of divine grace: guiding, forgiving, pardoning, embracing, admitting to Paradise, bringing glad tidings, loving.
    - **Fear of God (and only that)** falls under CONSTRUCTIVE. Example: "fear Allah" → fear = CONSTRUCTIVE.

    4. NONE (use this for words that do not fit any of the above)
    - Grammatical particles (and, but, so, then, etc.)
    - Interjections / emphasizers: "Verily", "indeed", "surely", "O" (when calling), "nay".
    - Disjointed letters (Al-Muqatta'at): "Ha-Meem", "Alif-Laam-Meem", etc.
    - Numbers, page references, footnote markers.
    - Verbs of saying that do not carry constructive/destructive meaning by themselves (e.g., "say", "tell", "call") – unless they are part of a command to do good or to avoid evil.
    - Common nouns that are neutral (e.g., "people", "men", "day", "time") when not explicitly tied to one of the three categories.
    - If a word belongs clearly to GOD/DESTRUCTIVE/CONSTRUCTIVE, do NOT use NONE.

    CONTEXT:
    "{verse}"

    SENTENCE:
    "{sentence}"

    TARGET WORDS TO VERIFY:
    {targets_string}

    OUTPUT RULE:
    Your response must be a valid JSON object where the keys are the string representation of the indexes, and values are the assigned category strings.
    Do NOT include any markdown formatting, backticks, or explanation.
    
    Example Output Format:
    {{
        "3": "GOD",
        "7": "CONSTRUCTIVE"
    }}
    """

    print(f"{'#' * 55} \nTOKENS: \n\t{tokens} \n{'#' * 55}")
    safe = shlex.quote(prompt)

    decisions = run_llama(safe)
    try:
        decisions = {int(k): v for k, v in json.loads(decisions).items()}

    except Exception:
        decisions = {}

    # Konsolen-Validierung (bunter Satz)
    GRUEN = "\033[32m"
    RESET = "\033[0m"
    worte = sentence.split()
    for m in tokens:
        idx = m["index"]
        if idx < len(worte) and decisions.get(idx):
            worte[idx] = f"{GRUEN}{worte[idx]}{RESET}"
    bunter_satz = " ".join(worte)

    ausgabe = f"{'#' * 55}\nSENTENCE: \t {bunter_satz}\n{'#' * 55}\nDECISIONS: \t {decisions}\n\n\n"
    print(ausgabe)

    return decisions


# --- Ausführung ---

if __name__ == "__main__":
    # Konsolen-Farben für die Statusmeldungen (ANSI-Codes)
    ANSI_GRUEN = "\033[32m"
    ANSI_ROT = "\033[31m"
    ANSI_GELB = "\033[33m"
    ANSI_RESET = "\033[0m"

    ORDNER_PFAD = (
        "/home/muhammed-emin-eser/desk/din/quran/eng_translation/chunked_translation"
    )

    COLOR_MAP = {
        "GOD": "&H75DFFA&",  # Majestätisches Gelbgold
        "DESTRUCTIVE": "&H6212B2&",  # Alarmierendes Hellrot
        "CONSTRUCTIVE": "&H803500&",  # Blau
    }

    print(
        f"{ANSI_GRUEN}[INFO] Starte Verarbeitung im Ordner: {ORDNER_PFAD}{ANSI_RESET}"
    )

    if not os.path.exists(ORDNER_PFAD):
        print(
            f"{ANSI_ROT}[FEHLER] Der Pfad '{ORDNER_PFAD}' existiert nicht! Bitte anpassen.{ANSI_RESET}"
        )
        exit(1)

    file_generator = process_directory(ORDNER_PFAD)
    dateien_zaehler = 0

    for file_path, file_data in file_generator:
        dateien_zaehler += 1
        print(f"\nVerarbeite Datei: {file_path}")

        detected_matches = []

        # Hier flachen wir die Struktur ab, um das finale, persistente JSON-Format zu füllen
        for verse in file_data:
            v_num = verse["verse_number"]

            for sentence_node in verse["sentences"]:
                context = sentence_node["sentence_context"]

                for token in sentence_node["tokens"]:
                    category = token.get("category")

                    # Abgleich mit der Farbtabelle und persistentes Abspeichern
                    if category in COLOR_MAP:
                        detected_matches.append(
                            {
                                "file_reference": file_path,
                                "verse_number": v_num,
                                "sentence_context": context,
                                "word_index": token["index"],
                                "word": token["word"],
                                "category": category,
                                "color": COLOR_MAP[category],
                            }
                        )

        # Persistent abspeichern (Erstellt ein valides JSON direkt neben der .txt-Datei)
        if detected_matches:
            output_json_path = (
                "/home/muhammed-emin-eser/desk/din/quran/llm_json/"
                + os.path.splitext(os.path.basename(file_path))[0]
                + "_output.json"
            )
            with open(output_json_path, "w", encoding="utf-8") as json_file:
                json.dump(detected_matches, json_file, ensure_ascii=False, indent=4)
            print(
                f"{ANSI_GRUEN}--> {len(detected_matches)} Treffer persistent gespeichert in: {output_json_path}{ANSI_RESET}"
            )
        else:
            print(
                f"{ANSI_GELB}--> Keine relevanten Wörter in dieser Datei gefunden.{ANSI_RESET}"
            )

        print("-" * 50)

    if dateien_zaehler == 0:
        print(f"{ANSI_GELB}\n[WARNUNG] Keine .txt-Dateien gefunden.{ANSI_RESET}")
    else:
        print(
            f"{ANSI_GRUEN}\n[FERTIG] Insgesamt {dateien_zaehler} Datei(en) erfolgreich verarbeitet.{ANSI_RESET}"
        )
