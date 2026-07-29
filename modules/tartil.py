"""
Tartil/Tarteel-Echtzeit-Worterkennung via Waydroid/ADB.

Ablauf:
    1. Audio der Rezitation wird in Waydroid-Mikrofon eingespeist
    2. Tarteel-App verarbeitet das Audio in Echtzeit und hebt
       das aktuell erkannte Wort blau hervor
    3. Wir scrapen per ADB (uiautomator dump) den Bildschirm und
       extrahieren das hervorgehobene Wort mitsamt Zeitstempel
    4. Ergebnis: Liste von (zeitsekunde, arabisches_wort)-Paaren

Integration in Pipeline:
    5. Jedes arabische Wort wird via quran.com-API in die englische
       Übersetzung aufgelöst (wie in semanticmatch._word_to_translation)
    6. Die mapping-Datei wird um Sub-Einträge pro Wort erweitert
"""

import json
import re
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

ADB_POLL_INTERVAL: float = 0.15
TARTIL_PACKAGE: str = "com.tarteel.ui"
TARTIL_WORD_CONTAINER_ID: str = "com.tarteel.ui:id/rvWords"
TARTIL_CURRENT_WORD_ID: str = "com.tarteel.ui:id/tvWord"

AUDIO_SAMPLE_RATE: int = 16000

# ---------------------------------------------------------------------------
# Datenstruktur
# ---------------------------------------------------------------------------


@dataclass
class TartilWord:
    """
    Ein von Tarteel erkanntes Wort mit Zeitstempel.

    timestamp_sec : Sekunde relativ zum Capture-Start (time.monotonic())
    arabic_word   : das erkannte arabische Wort (Uthmani-Text)
    is_highlighted: ob es zum Capture-Zeitpunkt blau markiert war
    """

    timestamp_sec: float
    arabic_word: str
    is_highlighted: bool = True


@dataclass
class TartilCaptureSession:
    """
    Ergebnis einer vollständigen Capture-Sitzung.

    words        : chronologische Liste erkannter Wörter
    surah        : Sure (falls bekannt)
    start_time   : time.monotonic() bei Aufnahmestart
    end_time     : time.monotonic() bei Aufnahmeende
    """

    words: List[TartilWord] = field(default_factory=list)
    surah: Optional[int] = None
    start_time: float = 0.0
    end_time: float = 0.0


# ---------------------------------------------------------------------------
# ADB-Interface
# ---------------------------------------------------------------------------

_ADB_PATH: str = "adb"


def _adb_exec(args: List[str]) -> Tuple[int, str, str]:
    """Führt ein ADB-Kommando aus und gibt (returncode, stdout, stderr) zurück."""
    cmd = [_ADB_PATH] + args
    proc = subprocess.run(cmd, capture_output=True, timeout=30)
    return (
        proc.returncode,
        proc.stdout.decode(errors="replace"),
        proc.stderr.decode(errors="replace"),
    )


def adb_available() -> bool:
    """Prüft, ob ADB installiert ist und ein Waydroid-Gerät verbunden ist."""
    rc, out, _ = _adb_exec(["devices"])
    if rc != 0:
        return False
    for line in out.splitlines():
        if "\tdevice" in line and "emulator" not in line:
            return True
    return False


def _dump_ui_xml() -> Optional[str]:
    """
    Ruft den aktuellen UI-XML-Baum von Waydroid ab.
    Liefert None bei Fehler oder wenn kein Gerät verbunden ist.
    """
    rc, out, err = _adb_exec([
        "exec-out", "uiautomator", "dump", "/dev/stdout"
    ])
    if rc != 0:
        return None
    match = re.search(
        r'(<\?xml version="1\.0" encoding="utf-8"\?>.*?)(</hierarchy>|</dump>)',
        out,
        re.DOTALL,
    )
    if match:
        return match.group(1) + match.group(2)
    return out if "<node" in out else None


# ---------------------------------------------------------------------------
# UI-Parsing
# ---------------------------------------------------------------------------


def _extract_words_from_xml(xml_str: str) -> List[Tuple[str, bool]]:
    """
    Extrahiert Wörter aus dem uiautomator-XML.

    Unterscheidet zwischen hervorgehobenem (blau) und normalem Wort,
    sofern Tarteel den State/die Farbe im Accessibility-Tree exponiert.

    Returns
    -------
    Liste von (arabic_text, is_highlighted)
    """
    if not xml_str:
        return []

    words: List[Tuple[str, bool]] = []

    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError:
        return []

    for node in root.iter("node"):
        text = node.get("text", "")
        if not text or len(text.strip()) < 1:
            continue
        if not _is_arabic(text.strip()):
            continue

        is_selected = node.get("selected", "false") == "true"
        is_focused = node.get("focused", "false") == "true"

        resource_id = node.get("resource-id", "")
        if TARTIL_WORD_CONTAINER_ID in resource_id:
            for child in node:
                child_text = child.get("text", "")
                if child_text and _is_arabic(child_text.strip()):
                    child_selected = (
                        child.get("selected", "false") == "true"
                        or child.get("focused", "false") == "true"
                    )
                    words.append((child_text.strip(), child_selected or is_selected or is_focused))
            continue

        words.append((text.strip(), is_selected or is_focused))

    return words


def _extract_single_highlighted_word(
    words: List[Tuple[str, bool]],
) -> Optional[str]:
    """
    Nimmt eine Liste von (text, highlighted)-Paaren und gibt das aktuell
    hervorgehobene Wort zurück (oder None, falls keins hervorgehoben ist).
    """
    highlighted = [w for w, h in words if h]
    if highlighted:
        return highlighted[-1]
    return None


_AR_CHAR_RANGE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+")


def _is_arabic(text: str) -> bool:
    """Prüft, ob Text arabische Zeichen enthält."""
    return bool(_AR_CHAR_RANGE.search(text))


# ---------------------------------------------------------------------------
# Audio-Injection (PulseAudio/PipeWire)
# ---------------------------------------------------------------------------


def _pactl(args: List[str]) -> Tuple[int, str]:
    """Führt pactl mit args aus."""
    cmd = ["pactl"] + args
    proc = subprocess.run(cmd, capture_output=True, timeout=15)
    return proc.returncode, (proc.stdout + proc.stderr).decode(errors="replace")


def _pw_link(source_port: str, sink_input: str) -> bool:
    """Erstellt einen PipeWire-Link zwischen Source-Port und Sink-Input."""
    cmd = [
        "pw-link",
        source_port,
        sink_input,
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=10)
    return proc.returncode == 0


def setup_audio_injection(
    mic_source_name: str = "tartil_mic",
) -> Optional[str]:
    """
    Richtet eine virtuelle Audio-Quelle ein, in die wir die Rezitation
    abspielen können, und verbindet sie mit Waydroid's Mikrofon-Eingang.

    Erstellt einen PipeWire-Null-Sink + -Source-Knoten (``), der als
    Mikrofon für Waydroid fungiert.

    Gibt den Namen der virtuellen Source zurück, oder None bei Fehler.
    """
    rc, out = _pactl([
        "load-module",
        "module-null-sink",
        f"sink_name={mic_source_name}",
        "sink_properties=device.description=TartilMic",
    ])
    if rc != 0:
        return None

    mic_source = f"{mic_source_name}.monitor"
    return mic_source


def teardown_audio_injection(
    mic_source_name: str = "tartil_mic",
) -> None:
    """Räumt die virtuelle Audio-Quelle wieder auf."""
    _pactl(["unload-module", f"module-null-sink"])


def _find_waydroid_sink_input() -> Optional[str]:
    """Findet den Waydroid-Sink-Input (PulseAudio-Index)."""
    rc, out = _pactl(["list-sink-inputs"])
    if rc != 0:
        return None
    for line in out.splitlines():
        if "application.process.binary" in line and "waydroid" in line.lower():
            return line.split("=")[-1].strip().strip('"')
    return None


def _find_waydroid_source_output() -> Optional[str]:
    """Findet den Waydroid-Audio-Eingang (Source-Output)."""
    rc, out = _pactl(["list-source-outputs"])
    if rc != 0:
        return None
    entries = re.split(r"Source Output #(\d+)", out)[1:]
    for i in range(0, len(entries) - 1, 2):
        block = entries[i + 1]
        if "waydroid" in block.lower() or "android" in block.lower():
            return entries[i]
    return None


def route_audio_to_waydroid(
    source_monitor: str,
    target_app: str = "Tarteel",
) -> bool:
    """
    Verbindet eine Audio-Quelle mit Waydroid's Mikrofon-Eingang.
    Dies erlaubt es, die Rezitation direkt in Tarteel einzuspeisen.
    """
    waydroid_source_output = _find_waydroid_source_output()
    if not waydroid_source_output:
        return False

    rc, out = _pactl([
        "move-source-output",
        waydroid_source_output,
        source_monitor,
    ])
    return rc == 0


# ---------------------------------------------------------------------------
# Capture-Loop
# ---------------------------------------------------------------------------


class TartilCapture:
    """
    Hauptklasse für die Echtzeit-Erkennung via Tarteel.

    Verwendung:
        capture = TartilCapture()
        capture.start()
        # ... Audio abspielen/injizieren ...
        capture.stop()
        session = capture.session  # TartilCaptureSession

    Oder als Kontextmanager:
        with TartilCapture() as capture:
            # ... Audio abspielen ...
            pass
    """

    def __init__(
        self,
        surah: Optional[int] = None,
        poll_interval: float = ADB_POLL_INTERVAL,
        word_callback: Optional[Callable[[TartilWord], None]] = None,
        audio_playback_cmd: Optional[List[str]] = None,
    ):
        self.surah = surah
        self.poll_interval = poll_interval
        self.word_callback = word_callback
        self.audio_playback_cmd = audio_playback_cmd
        self.session = TartilCaptureSession(surah=surah)
        self._running = False
        self._last_word: Optional[str] = None
        self._mono_start: float = 0.0

    def start(self) -> None:
        """Startet den Capture-Loop."""
        if not adb_available():
            raise RuntimeError(
                "Kein Waydroid-Gerät per ADB verbunden. "
                "Starte Waydroid und verbinde ADB."
            )
        self._mono_start = time.monotonic()
        self.session.start_time = self._mono_start
        self._running = True

        if self.audio_playback_cmd:
            self._audio_proc = subprocess.Popen(
                self.audio_playback_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    def stop(self) -> TartilCaptureSession:
        """Stoppt den Capture-Loop und gibt die Session zurück."""
        self._running = False
        self.session.end_time = time.monotonic() - self._mono_start

        if hasattr(self, "_audio_proc"):
            self._audio_proc.terminate()
            try:
                self._audio_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._audio_proc.kill()

        return self.session

    def poll_once(self) -> Optional[TartilWord]:
        """
        Einmaliger Poll-Zyklus: UI dumpen, Wort extrahieren, Zeitstempel.
        Gibt TartilWord zurück, wenn ein neues (oder weiterhin
        hervorgehobenes) Wort erkannt wurde, sonst None.
        """
        xml_str = _dump_ui_xml()
        if xml_str is None:
            return None

        words = _extract_words_from_xml(xml_str)
        highlighted = _extract_single_highlighted_word(words)

        if highlighted is None:
            self._last_word = None
            return None

        now = time.monotonic() - self._mono_start
        word = TartilWord(
            timestamp_sec=now,
            arabic_word=highlighted,
            is_highlighted=True,
        )
        self.session.words.append(word)

        if self.word_callback:
            self.word_callback(word)

        self._last_word = highlighted
        return word

    def run(self) -> TartilCaptureSession:
        """
        Führt den Polling-Loop aus, bis stop() aufgerufen wird
        (von einem anderen Thread/Signal-Handler).
        """
        self.start()
        try:
            while self._running:
                self.poll_once()
                time.sleep(self.poll_interval)
        finally:
            self.stop()
        return self.session

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


# ---------------------------------------------------------------------------
# Export: Tartil-Wörter → JSONL speichern
# ---------------------------------------------------------------------------


def save_session_to_jsonl(
    session: TartilCaptureSession, path: Path
) -> Path:
    """
    Speichert eine TartilCaptureSession als JSONL-Datei.
    Eine Zeile pro erkanntem Wort.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for word in session.words:
            record = {
                "timestamp_sec": round(word.timestamp_sec, 3),
                "arabic_word": word.arabic_word,
                "is_highlighted": word.is_highlighted,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def load_session_from_jsonl(path: Path) -> TartilCaptureSession:
    """Lädt eine TartilCaptureSession aus einer JSONL-Datei."""
    session = TartilCaptureSession()
    words: List[TartilWord] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            words.append(
                TartilWord(
                    timestamp_sec=data["timestamp_sec"],
                    arabic_word=data["arabic_word"],
                    is_highlighted=data.get("is_highlighted", True),
                )
            )
    session.words = words
    return session


# ---------------------------------------------------------------------------
# Integration: Tartil-Wörter → Mapping patchen
# ---------------------------------------------------------------------------


def _extract_translation_from_verse_word(w: dict) -> str:
    """Extrahiert den Übersetzungstext aus einem verse_word-Dict.
    Die API liefert translation als Dict {'text': '...', 'language_name': '...'}
    oder als plain string.
    """
    t = w.get("translation", "")
    if isinstance(t, dict):
        return t.get("text", "")
    if isinstance(t, str):
        return t
    return ""


def resolve_tartil_words_to_english(
    session: TartilCaptureSession,
    surah: int,
    ayah: int,
    ayah_start_word: int = 1,
) -> List[Tuple[TartilWord, str, float]]:
    """
    Übersetzt jedes arabische Wort aus der Session via quran.com API.

    Parameters
    ----------
    session            : TartilCaptureSession mit arabischen Wörtern
    surah, ayah        : Sure und Vers für die API-Abfrage
    ayah_start_word    : Index des ersten Wortes im Vers (default 1)

    Returns
    -------
    Liste von (tartil_word, english_translation, confidence_score)
    """
    from modules.semanticmatch import _fetch_verse_words, _word_to_translation

    verse_words = _fetch_verse_words(surah, ayah)
    result: List[Tuple[TartilWord, str, float]] = []

    for word in session.words:
        _, idx = _word_to_translation(word.arabic_word, verse_words)
        translation = ""
        if 0 <= idx < len(verse_words):
            translation = _extract_translation_from_verse_word(verse_words[idx])
        result.append((word, translation, 1.0 if idx >= 0 else 0.0))

    return result


def patch_mapping_with_tartil(
    mapping_path: Path,
    surah: int,
    ayah: int,
    session: TartilCaptureSession,
) -> None:
    """
    Erweitert eine bestehende mapping-Datei um Wort-für-Wort-Einträge
    aus der Tartil-Capture-Session.

    Format pro Sub-Eintrag:
        [HH:MM:SS] :: ar: بِسْمِ | en: In (the) name

    Der Eintrag wird unter dem zugehörigen Vers-Eintrag eingefügt,
    basierend auf dem Zeitstempel des Wortes.
    """
    from modules.circlelog import seconds_to_timestamp
    from modules.semanticmatch import _fetch_verse_words, _word_to_translation

    if not session.words:
        return

    verse_words = _fetch_verse_words(surah, ayah)

    lines = mapping_path.read_text(encoding="utf-8").splitlines()
    patched: List[str] = []

    word_idx = 0
    for line in lines:
        patched.append(line)
        if not line.startswith("["):
            continue

        ts_match = re.match(r"^\[(\d{2}:\d{2}:\d{2})\]", line)
        if not ts_match:
            continue

        line_ts = ts_match.group(1)
        line_sec = _timestamp_to_seconds(line_ts)

        sub_lines: List[str] = []
        while word_idx < len(session.words):
            word = session.words[word_idx]
            word_ts = seconds_to_timestamp(word.timestamp_sec)

            if _timestamp_to_seconds(word_ts) > line_sec + 120:
                break

            translation, _ = _word_to_translation(word.arabic_word, verse_words)
            sub_lines.append(
                f"[{word_ts}] :: ar: {word.arabic_word} | en: {translation}"
            )
            word_idx += 1

        if sub_lines:
            patched.extend(sub_lines)
            patched.append("")

    mapping_path.write_text("\n".join(patched).strip() + "\n", encoding="utf-8")


def _timestamp_to_seconds(ts: str) -> float:
    """Wandelt 'HH:MM:SS' in Sekunden um."""
    if not ts:
        return 0.0
    parts = list(map(int, ts.split(":")))
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return float(parts[0]) if parts else 0.0
