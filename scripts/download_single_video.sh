#!/usr/bin/env bash
# =============================================================================
# Einzelvideo-Downloader
#
# Nutzt die yt-dlp-Konfiguration aus ~/desk/projects/yt-downloader
# (selenium_scroll.py) – aber nur fuer EINE Video-URL, nicht Playlist/Kanal.
#
# Download-Ziel: ~/desk/din/yt-videos/
#
# Nutzung:
#   download_single_video.sh <youtube-url>
#   download_single_video.sh --out DIR <youtube-url>
#   download_single_video.sh --HH:MM:SS [--HH:MM:SS] <youtube-url>
#
# Cut-Flag:
#   --HH:MM:SS             Startzeitpunkt; schneidet von dort bis zum Video-Ende
#   --HH:MM:SS --HH:MM:SS  Start- und Endzeitpunkt; schneidet das Intervall
#
# Env-Overrides: YT_VIDEOS_DIR (Zielordner, Default din/yt-videos)
# =============================================================================
set -euo pipefail

YT_DOWNLOADER_DIR="$HOME/desk/projects/yt-downloader"
DEFAULT_YT_VIDEOS_DIR="$HOME/desk/din/yt-videos"

usage() {
  echo "Usage:"
  echo "  $0 [--out DIR] [--HH:MM:SS [--HH:MM:SS]] <youtube_url>"
  echo
  echo "Cut-Flag:"
  echo "  --HH:MM:SS             Start; schneidet bis zum Video-Ende"
  echo "  --HH:MM:SS --HH:MM:SS  Start und Ende; schneidet das Intervall"
  exit 1
}

require_tools() {
  command -v yt-dlp >/dev/null 2>&1 || {
    echo "yt-dlp not found"
    exit 1
  }
  command -v ffmpeg >/dev/null 2>&1 || {
    echo "ffmpeg not found"
    exit 1
  }
}

# Prüft, ob ein Argument ein gültiges Zeitformat (HH:MM:SS oder MM:SS) ist
is_time_arg() {
  [[ "$1" =~ ^([0-9]{1,2}):([0-9]{2}):([0-9]{2})$ || "$1" =~ ^([0-9]{1,2}):([0-9]{2})$ ]]
}

# Schneidet das Video von start bis end (end leer = bis Video-Ende)
cut_video() {
  local in_file="$1"
  local start="$2"
  local end="$3"

  local stem="${in_file%.*}"
  local ext="${in_file##*.}"

  local tag="${start//:/_}"
  if [[ -n "$end" ]]; then
    tag="${tag}_${end//:/_}"
  else
    tag="${tag}_end"
  fi

  local out_file="${stem}_cut_${tag}.${ext}"

  echo "Cut: $in_file"
  echo "  Start: $start   Ende: ${end:-<Video-Ende>}"
  echo "  Ausgabe: $out_file"

  if [[ -n "$end" ]]; then
    ffmpeg -nostdin -y -v error \
      -ss "$start" -i "$in_file" \
      -to "$end" -c copy \
      "$out_file"
  else
    ffmpeg -nostdin -y -v error \
      -ss "$start" -i "$in_file" \
      -c copy \
      "$out_file"
  fi

  echo "Cut fertig: $out_file"
}

main() {
  require_tools

  local out_dir="${YT_VIDEOS_DIR:-$DEFAULT_YT_VIDEOS_DIR}"
  local url=""
  local start=""
  local end=""

  while [[ $# -gt 0 ]]; do
    case "$1" in
    --out)
      out_dir="$2"
      shift 2
      ;;
    -h | --help)
      usage
      ;;
    --*)
      if is_time_arg "${1#--}"; then
        if [[ -z "$start" ]]; then
          start="${1#--}"
        elif [[ -z "$end" ]]; then
          end="${1#--}"
        else
          echo "Zu viele Zeit-Argumente (max. Start + Ende)." >&2
          usage
        fi
        shift
      else
        echo "Unbekannte Option: $1" >&2
        usage
      fi
      ;;
    *)
      url="$1"
      shift
      ;;
    esac
  done

  [[ -n "$url" ]] || usage

  mkdir -p "$out_dir"

  echo "Download: $url"
  echo "Ziel:     $out_dir"
  if [[ -n "$start" ]]; then
    echo "Cut:      start=$start end=${end:-<Video-Ende>}"
  fi

  # Basis-Flags identisch zu yt-downloader/selenium_scroll.py (Einzelvideo-Pfad):
  #   --no-playlist  -> nur das eine Video, nie Playlist/Kanal
  #   --restrict-filenames, Android-Client, Edge-Cookies
  local dl_path
  dl_path="$(yt-dlp \
    --no-warnings \
    --no-progress \
    --no-playlist \
    --ignore-errors \
    --no-overwrites \
    --concurrent-fragments 4 \
    --cookies-from-browser edge \
    --restrict-filenames \
    --extractor-args "youtube:player_client=android" \
    -f "bv*+ba/b" \
    --merge-output-format mp4 \
    --recode-video mp4 \
    --print after_move:filepath \
    -o "$out_dir/%(title).200B [%(id)s].%(ext)s" \
    "$url" | tail -n 1)"

  [[ -n "$dl_path" && -f "$dl_path" ]] || {
    echo "Download fehlgeschlagen oder Datei nicht gefunden." >&2
    exit 1
  }

  echo "Downloadet: $dl_path"

  if [[ -n "$start" ]]; then
    cut_video "$dl_path" "$start" "$end"
  else
    echo "Fertig. Datei liegt in: $out_dir"
  fi
}

main "$@"
