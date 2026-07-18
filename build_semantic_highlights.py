import json
import os
import glob
import re
from pathlib import Path
from typing import Optional, List, Tuple

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
    numbered_lines: List[list] = []
    numbered = re.compile(r"^(\d+)[\.)]\s*(.*)")
    expected_next: Optional[int] = None

    with Path(path).open(encoding="utf-8") as f:
        verse_number = 0
        for line in f:
            stripped = line.replace("\\n", " ").replace("\\", " ").strip()
            if not stripped:
                continue
            match = numbered.match(stripped)
            if match:
                verse_number = int(match.group(1))
                numbered_lines.append([verse_number, match.group(2)])
            else:
                if numbered_lines:
                    numbered_lines[-1][1] = numbered_lines[-1][1] + " " + stripped
                else:
                    title_lines.append(stripped)

    return title_lines, numbered_lines


# --- Die Haupt-Generator-Funktion ---


def process_directory(directory_path: str):
    """
    Liest alle .txt Dateien über parse_text_file ein und verarbeitet die nummerierten Verse.
    Yieldet pro Datei: (filepath, file_structure)
    """
    search_pattern = "/home/muhammed-emin-eser/desk/din/quran/eng_translation/chunked_translation/*.txt"
    alle_dateien = list(glob.iglob(search_pattern))

    json_pattern = os.path.join(
        "/home/muhammed-emin-eser/desk/din/quran/llm_json/", "*.json"
    )
    existing_json_names = {
        os.path.splitext(os.path.basename(f))[0].replace("_output", "")
        for f in glob.glob(json_pattern)
    }

    def extrahiere_zahl(pfad):
        zahlen = re.findall(r"\d+", pfad)
        return int(zahlen[-1]) if zahlen else 0

    alle_dateien.sort(key=extrahiere_zahl, reverse=True)

    for file_path in alle_dateien:
        print(f"Verarbeite in REIHENFOLGE: {file_path}")
        txt_name = os.path.splitext(os.path.basename(file_path))[0]
        if True:
            file_structure = []
            path_obj = Path(file_path)

            # Verwende deine neue sequenzielle Parsing-Logik
            _, numbered_lines = parse_text_file(path_obj)

            # Verarbeite die extrahierten, nummerierten Verse
            for verse_text in numbered_lines:
                if not verse_text:
                    continue
                verse_number = verse_text[0]
                verse_structure = {"verse_number": verse_number, "sentences": []}

                # LLM semantische Prüfung aufrufen (gibt Dict zurück: {index: "KATEGORIE"})
                decisions = check_semantics(verse_text, txt_name, verse_number)
                continue

                    # Gefundene Kategorien in die Matches-Struktur integrieren
                file_structure.append(verse_structure)
        else:
            continue

        yield file_path, file_structure


# --- Semantische Analyse ---


MAX_TOKENS = 4096


def run_llama(verse, file_name, verse_idx):

    verse_text = verse[1]  # raw verse text
    words = verse_text.split()

    # Format each word on its own line with 0-based index
    numbered_lines = "\n".join(f"{i}: {w}" for i, w in enumerate(words))

    prompt = f"""
    You are a precise semantic validation engine.
    TASK:
    Analyze the sentence and determine for each target word its semantic category based STRICTLY on the definitions below. Use the context of the whole sentence.

    IMPORTANT CONTEXT: These words will be color-coded in a video:
    - RED (DESTRUCTIVE)  → warning, punishment, sin — Allah WARNS the reader
    - BLUE (CONSTRUCTIVE) → good news, reward, virtue — Allah gives glad tidings
    - GOLD (GOD)          → direct reference to Allah, His names, His attributes
    Ask yourself: is Allah warning the reader here (RED), giving good news (BLUE), or referring to Himself (GOLD)?

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
    - Names of Paradise: Jannah, Gardens of Eden, Firdaws, Na'im, Darus-Salam, Illiyyin.
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

    OUTPUT RULE:
    - Return a valid JSON object where keys are 0-based word indices (shown before each word in the VERSE listing below) and values are the assigned category strings.
    - ONLY include words that are GOD, DESTRUCTIVE, or CONSTRUCTIVE. OMIT words that are NONE — do not list them.
    - Do NOT include any markdown formatting, backticks, or explanation.
    
    Example Output Format:
    {{
        "3": "GOD",
        "7": "CONSTRUCTIVE"
    }}
    """

    user_content = (
        f"VERSE (0-based word indices):\n{numbered_lines}\n\n"
        f"Return a JSON object with only non-NONE words (GOD, DESTRUCTIVE, CONSTRUCTIVE)."
    )

    try:
        request = {
            "custom_id": f"{file_name}_{verse_idx}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "qwen-max",
                "max_tokens": MAX_TOKENS,
                "messages": [
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": user_content,
                    },
                ],
                "response_format": {"type": "json_object"},
            },
        }
        with open(
            f"/home/muhammed-emin-eser/desk/din/quran/prompts_jsonl/{file_name}.jsonl",
            "a",
            encoding="utf-8",
        ) as f:
            f.write(json.dumps(request, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Error message: {e}")
        RuntimeError("Non ok Status Code")


def check_semantics(verse, file_name, verse_idx):
    run_llama(verse, file_name, verse_idx)


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

        for verse in file_data:
            v_num = verse["verse_number"]

            for sentence_node in verse["sentences"]:
                context = sentence_node["sentence_context"]

                for token_entry in sentence_node["tokens"]:
                    if isinstance(token_entry, list):
                        sub_tokens = token_entry
                    else:
                        sub_tokens = [token_entry]

                    for token in sub_tokens:
                        category = token.get(
                            "category"
                        )  # abgleich mit der farbtabelle und persistentes abspeichern
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
