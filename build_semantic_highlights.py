import json
import os
import glob
import re
from pathlib import Path
from typing import Optional, List, Tuple
import stanza
from openai import OpenAI

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
    numbered_lines: List[list] = []
    numbered = re.compile(r"^(\d+)[\.)]\s*(.*)")
    expected_next: Optional[int] = None

    with Path(
        f"/home/muhammed-emin-eser/desk/din/ayah-aligner/output/mapping/{path.name.replace('txt', 'mapping')}"
    ).open(encoding="utf-8") as f:
        verse_number = 0
        for line in f:
            stripped = line.replace("\\n", " ").replace("\\", " ").strip()
            if not stripped:
                continue
            if re.match(r"^\[\d+:\d+\] :: ", line):
                stripped = re.sub(r"^\[\d+:\d+\] :: ", "", stripped)
                numbered_lines.append([verse_number, stripped])
                verse_number += 1
            else:
                numbered_lines[-1][1] = numbered_lines[-1][1] + stripped

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
            # Da numbered_lines sequenziell ab Vers 1 befüllt wird, nutzen wir enumerate(..., start=1)
            for verse_idx, verse_text in enumerate(numbered_lines, start=0):
                if verse_idx == 0:
                    continue
                if verse_text:
                    verse_structure = {"verse_number": verse_idx, "sentences": []}

                    # LLM semantische Prüfung aufrufen (gibt Dict zurück: {index: "KATEGORIE"})
                    decisions = check_semantics(verse_text, txt_name, verse_idx)
                    continue

                    # Gefundene Kategorien in die Matches-Struktur integrieren
                file_structure.append(verse_structure)
        else:
            continue

        yield file_path, file_structure


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


def run_llama(verse, file_name, verse_idx):

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


    OUTPUT RULE:
    Your response must be a valid JSON object where the keys are the string representation of the word indexes, and values are the assigned category strings.
    Do NOT include any markdown formatting, backticks, or explanation.
    
    Example Output Format:
    {{
        "3": "GOD",
        "7": "CONSTRUCTIVE"
    }}
    """

    try:
        # client = OpenAI(
        #     api_key="sk-ws-H.IEDDEX.RqQn.MEYCIQDXufYP1QkVhsy8CYvgtWKM6itnCPzplRhHpsXmo1DAxAIhAN0P_Na2bxg43pfEekm7S58u0IeeJfU5jmAxXLT4beRv",
        #     base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        # )
        #
        #        {
        #            "custom_id": "req-1",
        #            "method": "POST",
        #            "url": "/v1/chat/completions",
        #            "body": {
        #                "model": "qwen-plus",
        #                "messages": [
        #                    {
        #                        "role": "user",
        #                        "content": "Summarize quantum computing in two sentences.",
        #                    }
        #                ],
        #            },
        #        }
        #
        #        {
        #            "custom_id": "req-2",
        #            "method": "POST",
        #            "url": "/v1/chat/completions",
        #            "body": {
        #                "model": "qwen-plus",
        #                "messages": [{"role": "user", "content": "What is 2+2?"}],
        #            },
        #        }
        #
        request = {
            "custom_id": f"{verse_idx}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "qwen-max",
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
                        "content": f"VERSE:\n'{verse[1]}'\n\n",
                    },
                ],
            },
        }
        with open(
            f"/home/muhammed-emin-eser/desk/din/quran/prompts_jsonl/{file_name}.jsonl",
            "a",
            encoding="utf-8",
        ) as f:
            # json.dumps konvertiert das Dictionary in einen einzeiligen String
            f.write(json.dumps(request, ensure_ascii=False) + "\n")
    #     response = client.chat.completions.create(
    #         model="qwen3.7-max",
    #         temperature=0.0,
    #         messages=[
    #             {
    #                 "role": "system",
    #                 "content": [
    #                     {
    #                         "type": "text",
    #                         "text": f"{prompt}",
    #                         "cache_control": {"type": "ephemeral"},
    #                     }
    #                 ],
    #             },
    #             {
    #                 "role": "user",
    #                 "content": f"""
    #             VERSE:
    #             "{verse}"
    #
    #             TARGET WORDS TO VERIFY:
    #             {targets_string}
    #         """,
    #             },
    #         ],
    #     )
    #     ki_text_output = response.choices[0].message.content
    #     print(
    #         f"Cache created: {response.usage.prompt_tokens_details.cache_creation_input_tokens}"
    #     )
    #     print(f"Cache hit: {response.usage.prompt_tokens_details.cached_tokens}")
    #     print(ki_text_output)
    except Exception as e:
        print(f"Error message: {e}")
        RuntimeError("Non ok Status Code")
    #
    # return ki_text_output


def check_semantics(verse, file_name, verse_idx):
    """
    matches: Liste von Dicts/Tuples, z.B.:
             [{"index": 3, "word": "Gott"}, ...]
    """

    decisions = run_llama(verse, file_name, verse_idx)
    return None
    try:
        decisions = {int(k): v for k, v in json.loads(decisions).items()}

    except Exception:
        decisions = {}

    # Konsolen-Validierung (bunter Satz)
    GRUEN = "\033[32m"
    RESET = "\033[0m"
    worte = verse.split()
    for v in tokens:
        for m in v:
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
