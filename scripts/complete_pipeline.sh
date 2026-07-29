#!/bin/bash
# Complete Pipeline: Tarteel + PhantomMic Audio Injection
# Usage: bash scripts/complete_pipeline.sh

ADB="adb"
MAHER="/home/muhammed-emin-eser/desk/din/quran/maher_playlist"
PROJECT="/home/muhammed-emin-eser/desk/din/ayah-aligner"
WAV="$MAHER/سورة البينة (98) بصوت القارئ الشيخ ماهر المعيقلي [6WJ0icZq6ms]_clean.wav"

echo "=== 1. Waydroid starten ==="
WAYLAND_DISPLAY=wayland-1 waydroid session start 2>&1
for i in $(seq 1 30); do
    sleep 2; $ADB connect 192.168.240.112:5555 2>/dev/null
    $ADB shell "echo ok" 2>/dev/null && break
done

echo "=== 2. Tarteel patchen + installieren ==="
$ADB uninstall com.mmmoussa.iqra 2>/dev/null
FOR f in /tmp/patched_new/*.apk; do [ -f "$f" ] || break; done
if [ ! -f /tmp/patched_new/tarteel_base_new-398-lspatched.apk ]; then
    BASE=$($ADB shell "pm path com.mmmoussa.iqra" | grep base | cut -d: -f2)
    EN=$($ADB shell "pm path com.mmmoussa.iqra" | grep config.en | cut -d: -f2)
    TV=$($ADB shell "pm path com.mmmoussa.iqra" | grep config.tvdpi | cut -d: -f2)
    X8=$($ADB shell "pm path com.mmmoussa.iqra" | grep config.x86_64 | cut -d: -f2)
    $ADB pull "$BASE" /tmp/tarteel_base.apk
    $ADB pull "$EN" /tmp/split_en.apk; $ADB pull "$TV" /tmp/split_tvdpi.apk; $ADB pull "$X8" /tmp/split_x86.apk
    mkdir -p /tmp/patched_final
    java -jar /tmp/lspatch.jar -m /tmp/PhantomMic.apk \
      -k /tmp/my.keystore 123456 key0 123456 -f -o /tmp/patched_final \
      /tmp/tarteel_base.apk /tmp/split_en.apk /tmp/split_tvdpi.apk /tmp/split_x86.apk
    $ADB install-multiple -r /tmp/patched_final/*.apk
else
    $ADB install-multiple -r /tmp/patched_new/*.apk
fi

echo "=== 3. Audio-Dateien pushen ==="
REC_DIR="/storage/emulated/0/Android/data/com.mmmoussa.iqra/files/Recordings"
$ADB shell "am start -n com.mmmoussa.iqra/.MainActivity"; sleep 15
$ADB shell "mkdir -p $REC_DIR"
$ADB push "$WAV" "$REC_DIR/recitation.wav"
$ADB shell "echo 'recitation' > $REC_DIR/phantom.txt && chmod 666 $REC_DIR/*"

echo "=== 4. Tarteel neustarten ==="
$ADB shell "am force-stop com.mmmoussa.iqra" 2>/dev/null; sleep 3
$ADB shell "am start -n com.mmmoussa.iqra/.MainActivity"; sleep 15

echo "=== 5. SAF schließen ==="
$ADB shell input keyevent 4; sleep 3

echo "=== 6. Extreme Auflösung + GET STARTED ==="
$ADB shell "wm size 412x3072"; sleep 5
# GET STARTED Button bei y~2830
$ADB shell input tap 206 2830; sleep 12

echo "=== 7. Grünen Button tippen ==="
$ADB shell input tap 114 736; sleep 8

echo "=== 8. Recording starten (Text antippen) ==="
$ADB shell input tap 206 1700; sleep 5

echo "=== 9. PhantomMic prüfen ==="
$ADB logcat -d -s PHANTOM_MIC 2>/dev/null | tail -10

echo "=== 10. Capture Loop ==="
for i in $(seq 1 200); do
    $ADB shell "screencap -p /sdcard/cap_$i.png" 2>/dev/null
    sleep 0.5
done
echo "Fertig. Bilder unter /sdcard/cap_*.png"
