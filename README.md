# Pipeline Orchestration

## Overview

```
Video + Text file
       │
       ▼
┌─────────────────┐
│  videowindow    │  Segments video by black screens
│  extract_windows│  Each window = one frame [start_sec, end_sec]
└────────┬────────┘
         │  List[FrameWindow]
         ▼
┌─────────────────────────────────────────────────────────────┐
│                    For each FrameWindow                     │
│                                                             │
│  ┌──────────────────┐                                       │
│  │ recognizecircle  │  detect_markers(frame)                │
│  │                  │  → (circles, end_with_last_verse)     │
│  └────────┬─────────┘                                       │
│           │                                                 │
│    ┌──────┴──────┐                                          │
│    │             │                                          │
│  n > 0         n = 0                                        │
│  circle(s)   no circle             ┌────────────────┐       │
│  detected    detected ─────────────► whispertranscribe      │
│                                    └────────────────┘       │
└────┬──────────────────────────────────────────────────────── ┘
     │
     ▼
┌────────────┐
│ circlelog  │  build_verse_line(ts, n)
│            │  → [MM:SS] :: verse_id : text
│            │
│            │  write_mapping(lines)
│            │  → *_mapping.txt
└────────────┘
```

---

## Transcription (n = 0)

Windows without a circle marker are passed to `whispertranscribe`.

```
FrameWindow [start_sec ──── end_sec]
         │
         ▼
┌─────────────────────┐
│  whispertranscribe  │
│                     │
│  1. Audio extraction│  ffmpeg: WAV, mono, 16 kHz
│     per window      │  [start_sec, end_sec] sliced from video
│                     │
│  2. WhisperX        │  Model   : large-v2
│     transcription   │  Language: Arabic (ar)
│                     │  Device  : CUDA / CPU (auto)
│                     │
│  3. Timestamp       │  WhisperX counts from 0 (clip start).
│     correction      │  segment.start += window.start_sec
│                     │  → timestamps become absolute in video
└──────────┬──────────┘

  Example:
  FrameWindow spans 30s–45s in the video.
  WhisperX finds a segment at t=2s in the clip.
  → 2s + 30s = 32s  (absolute position in video)
           │
           ▼
  List[ChunkTranscription]
  ┌─────────────────────────────────────────┐
  │ window   : FrameWindow [start, end]     │
  │ segments : [TranscriptSegment, ...]     │
  │   └─ start, end, text  (absolute)       │
  │ raw_text : full transcript text         │
  └─────────────────────────────────────────┘
```

---

## Module Overview

| Module                 | Purpose                                                      | Input                                              | Output                           |
|------------------------|--------------------------------------------------------------|----------------------------------------------------|----------------------------------|
| `videowindow.py`       | Video → windows between black screens                        | Video path                                         | `List[FrameWindow]`              |
| `recognizecircle.py`   | Detect circles in a frame (geometric)                        | Grayscale image                                    | Number of detected circles       |
| `circlelog.py`         | Document detected circle timestamps in mapping format        | Timestamps + verse texts                           | `[MM:SS] :: verse_id : text` file|
| `whispertranscribe.py` | Transcribe audio chunks with WhisperX                        | Video path + `List[FrameWindow]`                   | `List[ChunkTranscription]`       |
| `semanticmatch.py`     | Translate Arabic text + match against verse span             | `List[ChunkTranscription]` + verse text (string)   | `MatchSession`                   |

---

## Data Flow (compact)

```
Video
 │
 ▼
videowindow ──► List[FrameWindow]
                       │
               per window: recognizecircle
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
              │           1. Translator : raw_text (ar) → English
              └──────────►2. Matcher    : similarity translation ↔ verse text
                             (verse text = full string from circlelog entry)
                          3. Guard      : completeness (all chunks have result), sequence order
                          → MatchSession
```

### Guard Detail

```
Chunk 1 → translation computed, score = 0.82  ✓
Chunk 2 → translation computed, score = 0.74  ✓
Chunk 3 → translation error                   ✗  → correction_requested = True
                                                    missing_chunks += [Chunk 3]

After all chunks:
  missing_chunks not empty → completeness_passed = False
```
