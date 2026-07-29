#!/bin/bash
set -e

ADB="adb"
MAHER="/home/muhammed-emin-eser/desk/din/quran/maher_playlist"
REC_DIR="/storage/emulated/0/Android/data/com.mmmoussa.iqra/files/Recordings"
WAV="$MAHER/سورة البينة (98) بصوت القارئ الشيخ ماهر المعيقلي [6WJ0icZq6ms]_clean.wav"

echo "[PIPELINE] Starte Waydroid..."
WAYLAND_DISPLAY=wayland-1 waydroid session start 2>&1 &
for i in $(seq 1 30); do
    sleep 2
    $ADB connect 192.168.240.112:5555 2>/dev/null
    if $ADB shell "echo ok" 2>/dev/null; then break; fi
done

echo "[PIPELINE] Installiere patched Tarteel..."
$ADB install-multiple -r /tmp/patched_new/*.apk 2>&1

echo "[PIPELINE] Starte Tarteel (Pfad anlegen)..."
$ADB shell "am start -n com.mmmoussa.iqra/.MainActivity"
sleep 15

echo "[PIPELINE] Pushe Audio + phantom.txt..."
$ADB shell "mkdir -p $REC_DIR"
$ADB push "$WAV" "$REC_DIR/recitation.wav"
$ADB shell "echo 'recitation' > $REC_DIR/phantom.txt"
$ADB shell "chmod 666 $REC_DIR/*"

echo "[PIPELINE] Tarteel neustarten mit Phantom..."
$ADB shell "am force-stop com.mmmoussa.iqra" 2>/dev/null
sleep 3
$ADB shell "am start -n com.mmmoussa.iqra/.MainActivity"
sleep 15

echo "[PIPELINE] Warte auf SAF Dialog..."
sleep 5
# SAF schließen falls vorhanden
$ADB shell "uiautomator dump /sdcard/ui_check_saf.xml" 2>/dev/null
$ADB pull /sdcard/ui_check_saf.xml /tmp/ui_check_saf.xml 2>/dev/null
if grep -q "Can.t use this folder" /tmp/ui_check_saf.xml 2>/dev/null; then
    echo "[PIPELINE] SAF erkannt -> schließen"
    $ADB shell input keyevent 4
    sleep 3
fi

echo "[PIPELINE] Setze extreme Auflösung..."
$ADB shell "wm size 412x3072"
sleep 5

echo "[PIPELINE] Warte auf User-Input..."
echo "Bitte im Waydroid-Fenster:"
echo "1. Welcome-Screen durchklicken"
echo "2. Zu Al-Bayyinah navigieren"
echo "3. Mic-Button drücken"
echo "Dann Enter drücken..."
read -p ""

echo "[PIPELINE] Prüfe PhantomMic Logs..."
$ADB logcat -d -s PHANTOM_MIC 2>/dev/null | tail -10

echo "[PIPELINE] Capture starten..."
# Hier Screencap Loop
for i in $(seq 1 160); do
    $ADB shell "screencap -p /sdcard/frame_$i.png" 2>/dev/null
    sleep 0.1
done
echo "[PIPELINE] Fertig"
