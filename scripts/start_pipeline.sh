#!/bin/bash
set -e

PROJECT_DIR="/home/muhammed-emin-eser/desk/din/ayah-aligner"
MAHER_DIR="/home/muhammed-emin-eser/desk/din/quran/maher_playlist"
REC_DIR="/storage/emulated/0/Android/data/com.mmmoussa.iqra/files/Recordings"
RECITATION="$MAHER_DIR/سورة البينة (98) بصوت القارئ الشيخ ماهر المعيقلي [6WJ0icZq6ms]_clean.wav"

echo "=== 1. Waydroid UI starten ==="
nohup env WAYLAND_DISPLAY=wayland-1 waydroid show-full-ui > /tmp/waydroid.log 2>&1 &
WAYDROID_PID=$!

for i in $(seq 1 30); do
    sleep 2
    adb connect 192.168.240.112:5555 2>/dev/null
    if adb shell "echo ok" 2>/dev/null; then
        echo "ADB ready after ${i}s"
        break
    fi
done

echo "=== 2. Tarteel starten (damit Pfad erzeugt wird) ==="
adb shell "am start -n com.mmmoussa.iqra/.MainActivity"
sleep 15

echo "=== 3. Audio-Datei pushen ==="
adb shell "mkdir -p $REC_DIR"
adb push "$RECITATION" "$REC_DIR/recitation.wav"
adb shell "echo 'recitation' > $REC_DIR/phantom.txt"
adb shell "chmod 666 $REC_DIR/*"

echo "=== 4. Tarteel neustarten mit PhantomMic ==="
adb shell "am force-stop com.mmmoussa.iqra" 2>/dev/null
sleep 3
adb shell "am start -n com.mmmoussa.iqra/.MainActivity"

echo ""
echo "=== BEREIT ==="
echo "Waydroid PID: $WAYDROID_PID"
echo "Tarteel läuft im Waydroid-Fenster"
echo ""
echo "Manuell: SAF schließen -> Welcome -> Al-Bayyinah -> Mikrofon"
echo ""
echo "Stop: kill $WAYDROID_PID"
