# Pipeline-Orchestrierung

## Übersicht

```
Video + Textdatei
       │
       ▼
┌─────────────────┐
│  videowindow    │  Segmentiert das Video anhand schwarzer Bildschirme
│  extract_windows│  Jedes Fenster = ein Frame [start_sec, end_sec]
└────────┬────────┘
         │  List[FrameWindow]
         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Für jedes FrameWindow                    │
│                                                             │
│  ┌──────────────────┐                                       │
│  │ recognizecircle  │  detect_markers(frame)                │
│  │                  │  → Anzahl erkannter Kreise            │
│  └────────┬─────────┘                                       │
│           │                                                 │
│    ┌──────┴──────┐                                          │
│    │             │                                          │
│  n > 0         n = 0                                        │
│  Kreis(e)    kein Kreis                ┌────────────────┐   │
│  erkannt     erkannt ──────────────────► whispertranscribe  │
│                                        └────────────────┘   │
└────┬──────────────────────────────────────────────────────── ┘
     │
     ▼
┌────────────┐
│ circlelog  │  build_verse_line(ts, n)
│            │  → [MM:SS] :: vers_id : text
│            │
│            │  write_mapping(lines)
│            │  → *_mapping.txt
└────────────┘
```

---

## Transkription (n = 0)

Fenster ohne Kreis werden an
`whispertranscribe` übergeben. 

```
FrameWindow [start_sec ──── end_sec]
         │
         ▼
┌─────────────────────┐
│  whispertranscribe  │
│                     │
│  1. Audio-Extraktion│  ffmpeg: WAV, mono, 16 kHz
│     pro Fenster     │  [start_sec, end_sec] aus dem Video
│                     │
│  2. WhisperX        │  Modell  : large-v2
│     Transkription   │  Sprache : Arabisch (ar)
│                     │  Device  : CUDA / CPU (auto)
│                     │
│  3. Zeitstempel-    │  WhisperX zählt ab 0 (Clip-Anfang).          │
│     Korrektur       │  segment.start += window.start_sec           │
│                     │  → Zeitstempel werden absolut im Video       │
└──────────┬──────────┘

  Beispiel:
  FrameWindow liegt bei 30s–45s im Video.
  WhisperX findet ein Segment bei t=2s im Clip.
  → 2s + 30s = 32s  (absolute Position im Video)
           │
           ▼
  List[ChunkTranscription]
  ┌─────────────────────────────────────────┐
  │ window   : FrameWindow [start, end]     │
  │ segments : [TranscriptSegment, ...]     │
  │   └─ start, end, text  (absolut)        │
  │ raw_text : vollständiger Text           │
  └─────────────────────────────────────────┘
```

---

## Modul-Übersicht

| Modul                  | Zweck                                                        | Eingabe                                            | Ausgabe                          |
|------------------------|--------------------------------------------------------------|----------------------------------------------------|----------------------------------|
| `videowindow.py`       | Video → Fenster zwischen schwarzen Bildschirmen              | Videopfad                                          | `List[FrameWindow]`              |
| `recognizecircle.py`   | Kreise in einem Frame erkennen (geometrisch)                 | Graustufenbild                                     | Anzahl erkannter Kreise          |
| `circlelog.py`         | Zeitstempel erkannter Kreise im Mapping-Format dokumentieren | Timestamps + Vers-Texte                            | `[MM:SS] :: vers_id : text`-Datei|
| `whispertranscribe.py` | Audio-Häppchen mit WhisperX transkribieren                   | Videopfad + `List[FrameWindow]`                    | `List[ChunkTranscription]`       |
| `semanticmatch.py`     | Arabischen Text übersetzen + gegen Teile EINES Verses matchen| `List[ChunkTranscription]` + Vers-Text (ein String)| `MatchSession`                   |

---

## Datenfluss (kompakt)

```
Video
 │
 ▼
videowindow ──► List[FrameWindow]
                       │
               pro Fenster: recognizecircle
                       │
              ┌────────┴────────┐
           n > 0             n = 0
              │                 │
              ▼                 ▼
          circlelog      whispertranscribe
     build_verse_line()  transcribe_chunks()
     write_mapping()     → List[ChunkTranscription]
     → _mapping.txt              │
              │                  ▼
              │           semanticmatch
              │           1. Übersetzer : raw_text (ar) → Englisch
              └──────────►2. Matcher    : Ähnlichkeit Übersetzung ↔ Vers-Text
                             (Vers-Text = vollständiger String aus circlelog-Eintrag)
                          3. Guard      : Vollständigkeit (alle Chunks haben Ergebnis), Sequenzeinhaltung
                          → MatchSession
```

### Guard-Detail

```
Chunk 1 → Übersetzung berechnet, Score = 0.82  ✓
Chunk 2 → Übersetzung berechnet, Score = 0.74  ✓
Chunk 3 → Fehler bei Übersetzung              ✗  → correction_requested = True
                                                    missing_chunks += [Chunk 3]

Nach allen Chunks:
  missing_chunks nicht leer → completeness_passed = False
```
