#!/usr/bin/env bash
# video_setup.sh — Bootstrap the video generation environment.
#
# Usage:
#   chmod +x video_setup.sh
#   ./video_setup.sh
#
# What this script does:
#   1. Installs video pipeline pip dependencies (whisperx, Pillow, opencv, etc.)
#   2. Downloads WhisperX large-v2 model weights via the whisperx Python API
#   3. Downloads Noto Serif Regular + Bold .ttf files into fonts/

set -euo pipefail

VENV_DIR=".venv"
FONTS_DIR="fonts"


# ── 3. Download Noto Serif fonts ──────────────────────────────────
echo "[3/3] Downloading Noto Serif Regular + Bold fonts …"
mkdir -p "${FONTS_DIR}"

NOTO_REGULAR_URL="https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSerif/NotoSerif-Regular.ttf"
NOTO_BOLD_URL="https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSerif/NotoSerif-Bold.ttf"

download_font() {
    local url="$1"
    local dest="$2"
    if [ -f "${dest}" ]; then
        echo "  Already exists: ${dest}"
    else
        echo "  Downloading: ${dest}"
        if command -v wget &>/dev/null; then
            wget -q -O "${dest}" "${url}"
        elif command -v curl &>/dev/null; then
            curl -sL -o "${dest}" "${url}"
        else
            echo "  ERROR: Neither wget nor curl found. Install one and rerun."
            exit 1
        fi
        echo "  Done: ${dest}"
    fi
}

download_font "${NOTO_REGULAR_URL}" "${FONTS_DIR}/NotoSerif-Regular.ttf"
download_font "${NOTO_BOLD_URL}"    "${FONTS_DIR}/NotoSerif-Bold.ttf"
