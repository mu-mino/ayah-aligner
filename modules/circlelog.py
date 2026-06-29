"""
Zeitdokumentation erkannter Kreise (Mapping-Format).

Dieses Modul enthält ausschließlich die Logik, um Zeitpunkte, an denen
Kreise (Ring-Marker) identifiziert wurden, im projekteigenen Format zu
dokumentieren und zu schreiben:

    [HH:MM:SS] :: inhalt

Nicht enthalten: Kreiserkennung, Video-Analyse, OCR, Rendering.
"""

import re
from pathlib import Path
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Zeitformatierung
# ---------------------------------------------------------------------------


def seconds_to_timestamp(seconds: float) -> str:
    """Wandelt Sekunden in das Zeitstempel-Format HH:MM:SS um."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


# ---------------------------------------------------------------------------
# Textformatierung
# ---------------------------------------------------------------------------


def format_title_block(title_lines: List[str]) -> str:
    """
    Formatiert Titelzeilen für die erste Mapping-Zeile.

    Fügt bei Bedarf einen Trennstrich zwischen erstem Wort und Rest ein
    (z. B. "Al-Bayyina The Clear Proof" → "Al-Bayyina - The Clear Proof").
    Mehrere Zeilen werden mit dem Literal '\\n' verbunden,
    damit sie die Mapping-Serialisierung überstehen.
    """
    if not title_lines:
        return ""

    formatted: List[str] = []
    for idx, line in enumerate(title_lines):
        if idx == 0 and " - " not in line and " " in line:
            first, rest = line.split(" ", 1)
            line = f"{first} - {rest}"
        formatted.append(line)
    return "\\n".join(formatted)


# ---------------------------------------------------------------------------
# Textdatei-Parser
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Mapping-Zeilen bauen
# ---------------------------------------------------------------------------


def build_title_line(title_lines: List[str]) -> str:
    """Erstellt die Titelzeile im Mapping-Format: '[00:00:00] :: Titel'."""
    return f"[00:00:00] :: {format_title_block(title_lines)}"


def build_verse_line(timestamp: str, verse_entries: List[Tuple[int, str]]) -> str:
    """
    Erstellt eine Vers-Zeile im Mapping-Format.

    Parameters
    ----------
    timestamp:
        Zeitstempel im Format 'MM:SS'.
    verse_entries:
        Liste von (vers_nummer, vers_text)-Paaren.

    Beispiel
    --------
    >>> build_verse_line("00:42", [(3, "Text A"), (4, "Text B")])
    '[00:00:42] :: 3: Text A 4: Text B'
    """
    content = " ".join(f"{num}: {text}" for num, text in verse_entries)
    return f"[{timestamp}] :: {content}"


# ---------------------------------------------------------------------------
# Mapping schreiben und lesen
# ---------------------------------------------------------------------------


def write_mapping(lines: List[str], dest: Path) -> Path:
    """
    Schreibt ein Mapping in die Zieldatei.

    Format pro Zeile: '[HH:MM:SS] :: inhalt'

    Wirft RuntimeError, wenn keine Vers-Zeilen vorhanden sind (nur
    Titelzeile würde eine leere Datei implizieren).
    """
    if len(lines) <= 1:
        raise RuntimeError("Mapping enthält keine Vers-Zeilen – nichts geschrieben.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines) + "\n"
    dest.write_text(content, encoding="utf-8")
    return dest


def read_mapping(path: Path) -> List[str]:
    """
    Liest eine Mapping-Datei und gibt die nicht-leeren Zeilen zurück.
    Kommentarzeilen (beginnend mit '#') werden übersprungen.
    """
    lines = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            stripped = line.rstrip("\n")
            if stripped and not stripped.startswith("#"):
                lines.append(stripped)
    return lines


def parse_mapping_line(line: str) -> Optional[Tuple[str, str]]:
    """
    Parst eine Mapping-Zeile und gibt (timestamp, inhalt) zurück.

    Erwartet das Format '[HH:MM:SS] :: inhalt'.
    Gibt None zurück, wenn das Format nicht erkannt wird.
    """
    m = re.match(r"^\[(\d{2}:\d{2})\]\s*::\s*(.*)", line)
    if not m:
        return None
    return m.group(1), m.group(2)
