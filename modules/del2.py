import dill
from text_ass import build_ass
import sys
import os

# 1. Daten laden
with open("./alle_lokalen_daten.pkl", "rb") as f:
    geladene_box = dill.load(f)

# 2. Variablen in den globalen Speicher schütten
globals().update(geladene_box)

# 3. Funktion ganz normal ausführen lassen (OHNE breakpoint)
ass_text = build_ass(file_name, entries, width, height, duration)
print("🎯 Berechnung erfolgreich abgeschlossen!")
