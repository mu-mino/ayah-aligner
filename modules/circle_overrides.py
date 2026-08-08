"""
Künstliche Kreis-Korrekturen für bekannte Fehl-/Über-Erkennungen.

Die Circle-Detektion (recognizecircle) kann an einzelnen Stellen versagen:

  - Übersehener Kreis  → der Vers-Shift geht NACH HINTEN (Vers kommt zu spät /
    fehlt ganz)  → Korrektur  n += 1
  - Falscher Kreis     → der Vers-Shift geht NACH VORNE (Vers kommt zu früh /
    wird doppelt zugeordnet)  → Korrektur  n -= 1

Diese Tabelle überschreibt die erkannte Kreisanzahl deterministisch pro
(Sure, Window.start_sec). Die Einträge sind durch Whisper-Stichproben
(vorne/mitte/hinten) verifiziert — nicht geraten.
"""


def apply_circle_override(surah: int, window, detected_n: int) -> int:
    """Liefert die korrigierte Kreisanzahl für (surah, window).

    window: FrameWindow mit .start_sec (aus modules.videowindow).
    detected_n: die von detect_markers_from_gray erkannte Anzahl.
    """
    n = detected_n
    start = round(window.start_sec, 1)

    # ------------------------------------------------------------------
    # Sure 96 (Al-Alaq): Window 134.8-155.2s zeigt Verse 16+17+18, die
    # Detektion findet aber nur 2 Kreise (der 3. Kreis ist defekt).
    # Folge: Vers 19 fehlt im Mapping (Shift nach hinten) → n += 1.
    # Verifiziert per Whisper: w7 rezitiert "...فندعو الزبانية" (V18).
    # ------------------------------------------------------------------
    if surah == 96 and 134.8 <= start <= 155.2:
        n = 3

    return n
