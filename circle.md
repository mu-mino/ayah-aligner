Zeitformatierung
  - seconds_to_timestamp(seconds) → "MM:SS"

Textformatierung
- format_title_block(title_lines) — Titelzeilen zusammenführen, Trennstrich einfügen
- parse_text_file(path) — Textdatei in (title_lines, numbered_lines) aufteilen

Mapping-Zeilen bauen
- build_title_line(title_lines) → "[00:00] :: Titel"
- build_verse_line(timestamp, verse_entries) → "[MM:SS] :: 1: Text 2: Text"

Mapping I/O
- write_mapping(lines, dest) — schreibt die Mapping-Datei
- read_mapping(path) — liest eine Mapping-Datei (überspringt Kommentare)
- parse_mapping_line(line) — parst eine einzelne Zeile → (timestamp, inhalt)