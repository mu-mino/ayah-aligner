"""
CLI-Tool: Tarteel-Echtzeit-Worterkennung via Waydroid/ADB.

Verwendung:
    # Rezitation live capturen (parallele Audio-Wiedergabe und Injection)
    python tartil_capture.py capture --surah 98 --play-audio recitation.wav \\
        -o data/tartil/98_live.jsonl

    # Capture-Session nachträglich ins Mapping einspielen
    python tartil_capture.py patch \\
        --mapping output/mapping/98_Al-Bayyinah.mapping \\
        --capture data/tartil/98_live.jsonl \\
        --surah 98 --ayah 1

    # Capture-Session analysieren (Wörter + Timestamps anzeigen)
    python tartil_capture.py show data/tartil/98_live.jsonl
"""

import argparse
import json
import signal
import sys
from pathlib import Path

from modules.tartil import (
    TartilCapture,
    TartilCaptureSession,
    TartilWord,
    load_session_from_jsonl,
    patch_mapping_with_tartil,
    resolve_tartil_words_to_english,
    save_session_to_jsonl,
    adb_available,
)


def cmd_capture(args: argparse.Namespace) -> None:
    """Live-Capture aus Waydroid/Tarteel."""
    if not adb_available():
        print(
            "Fehler: Kein Waydroid-Gerät per ADB verbunden.\n"
            "Starte Waydroid und verbinde ADB:\n"
            "  waydroid session start\n"
            "  adb connect 127.0.0.1:5555",
            file=sys.stderr,
        )
        sys.exit(1)

    output_path = Path(args.output)
    if output_path.exists():
        if not args.force:
            print(
                f"Datei {output_path} existiert bereits. "
                "Verwende --force zum Überschreiben.",
                file=sys.stderr,
            )
            sys.exit(1)
        output_path.unlink()

    playback_cmd = None
    if args.play_audio:
        import shlex
        playback_cmd = shlex.split(args.play_audio)

    capture = TartilCapture(
        surah=args.surah,
        poll_interval=args.interval,
        audio_playback_cmd=playback_cmd,
    )

    def on_word(word: TartilWord) -> None:
        print(
            f"[{word.timestamp_sec:07.3f}s] {word.arabic_word}",
            flush=True,
        )

    if args.verbose:
        capture.word_callback = on_word

    def handle_signal(signum, frame):
        print("\nCapture wird gestoppt...", file=sys.stderr)
        capture.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print(
        "Tarteel-Capture gestartet. Drücke Ctrl+C zum Beenden.",
        file=sys.stderr,
    )
    print(
        f"Ziel: {output_path}",
        file=sys.stderr,
    )

    if args.dry_run:
        print("[TROCKENLAUF] Capture würde starten...", file=sys.stderr)
        return

    session = capture.run()
    _apply_time_offset(session, args.time_offset)
    save_session_to_jsonl(session, output_path)

    print(
        f"\nCapture beendet. {len(session.words)} Wörter erkannt.",
        file=sys.stderr,
    )
    print(f"Gespeichert: {output_path}", file=sys.stderr)


def _apply_time_offset(
    session: TartilCaptureSession, offset: float
) -> None:
    if offset == 0.0:
        return
    for word in session.words:
        word.timestamp_sec += offset


def cmd_patch(args: argparse.Namespace) -> None:
    """Tartil-Capture in bestehendes Mapping einspielen."""
    mapping_path = Path(args.mapping)
    if not mapping_path.exists():
        print(f"Mapping-Datei nicht gefunden: {mapping_path}", file=sys.stderr)
        sys.exit(1)

    capture_path = Path(args.capture)
    if not capture_path.exists():
        print(f"Capture-Datei nicht gefunden: {capture_path}", file=sys.stderr)
        sys.exit(1)

    session = load_session_from_jsonl(capture_path)
    _apply_time_offset(session, args.time_offset)

    patch_mapping_with_tartil(
        mapping_path=mapping_path,
        surah=args.surah,
        ayah=args.ayah,
        session=session,
    )

    print(
        f"Mapping gepatcht: {len(session.words)} Wörter eingefügt.",
        file=sys.stderr,
    )


def cmd_show(args: argparse.Namespace) -> None:
    """Tartil-Capture-Session anzeigen/analysieren."""
    capture_path = Path(args.capture)
    if not capture_path.exists():
        print(f"Datei nicht gefunden: {capture_path}", file=sys.stderr)
        sys.exit(1)

    session = load_session_from_jsonl(capture_path)
    _apply_time_offset(session, args.time_offset)

    print(f"Session: {len(session.words)} Wörter")
    if session.words:
        print(f"Zeitraum: {session.words[0].timestamp_sec:.3f}s – "
              f"{session.words[-1].timestamp_sec:.3f}s")
        print()

    for word in session.words:
        print(f"  [{word.timestamp_sec:07.3f}s] {word.arabic_word}")


def cmd_resolve(args: argparse.Namespace) -> None:
    """Tartil-Wörter via quran.com-API in Englisch auflösen."""
    capture_path = Path(args.capture)
    if not capture_path.exists():
        print(f"Datei nicht gefunden: {capture_path}", file=sys.stderr)
        sys.exit(1)

    session = load_session_from_jsonl(capture_path)
    _apply_time_offset(session, args.time_offset)
    word_data = resolve_tartil_words_to_english(
        session=session,
        surah=args.surah,
        ayah=args.ayah,
        ayah_start_word=args.start_word,
    )

    print(f"Wortauflösung für {args.surah}:{args.ayah}")
    print(f"{'Zeit':>10s}  {'Arabisch':<20s}  {'Englisch':<40s}  {'Score':>5s}")
    print("-" * 80)
    for word, translation, score in word_data:
        print(
            f"  {word.timestamp_sec:>7.3f}s  "
            f"{word.arabic_word:<20s}  "
            f"{translation:<40s}  "
            f"{score:.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tarteel-Echtzeit-Worterkennung via Waydroid/ADB"
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--force", "-f", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    # capture
    p_capture = sub.add_parser("capture", help="Live-Capture starten")
    p_capture.add_argument("--surah", type=int, default=None)
    p_capture.add_argument("--output", "-o", required=True)
    p_capture.add_argument("--interval", type=float, default=0.15)
    p_capture.add_argument("--play-audio", type=str, default=None,
                           help="Audio-Datei (oder Shell-Kommando) zum Abspielen "
                                "in Waydroid-Mikrofon")
    p_capture.add_argument("--dry-run", action="store_true")
    p_capture.add_argument("--time-offset", type=float, default=0.0,
                           help="Zeitversatz in Sekunden (negativ bei früherem Start)")

    # patch
    p_patch = sub.add_parser("patch", help="Capture ins Mapping einspielen")
    p_patch.add_argument("--mapping", required=True)
    p_patch.add_argument("--capture", required=True)
    p_patch.add_argument("--surah", type=int, required=True)
    p_patch.add_argument("--ayah", type=int, required=True)
    p_patch.add_argument("--time-offset", type=float, default=0.0,
                         help="Zeitversatz in Sekunden")

    # show
    p_show = sub.add_parser("show", help="Capture-Session anzeigen")
    p_show.add_argument("capture")
    p_show.add_argument("--time-offset", type=float, default=0.0,
                        help="Zeitversatz in Sekunden")

    # resolve
    p_resolve = sub.add_parser("resolve", help="Wörter in Englisch auflösen")
    p_resolve.add_argument("--capture", required=True)
    p_resolve.add_argument("--surah", type=int, required=True)
    p_resolve.add_argument("--ayah", type=int, required=True)
    p_resolve.add_argument("--start-word", type=int, default=1)
    p_resolve.add_argument("--time-offset", type=float, default=0.0,
                           help="Zeitversatz in Sekunden")

    args = parser.parse_args()

    if args.command == "capture":
        cmd_capture(args)
    elif args.command == "patch":
        cmd_patch(args)
    elif args.command == "show":
        cmd_show(args)
    elif args.command == "resolve":
        cmd_resolve(args)


if __name__ == "__main__":
    main()
