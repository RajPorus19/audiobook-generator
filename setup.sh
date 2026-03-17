#!/usr/bin/env bash
# setup.sh — Bootstrap the audiobook TTS environment.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#
# What this script does:
#   1. Creates a Python 3.10 virtual environment in .venv/
#   2. Installs all dependencies from requirements.txt
#   3. Pre-downloads the Qwen3-TTS model weights from HuggingFace
#   4. Downloads the NLTK punkt tokenizer data

set -euo pipefail

PYTHON=${PYTHON:-python3.10}
VENV_DIR=".venv"

echo "============================================================"
echo "  Audiobook TTS — Environment Setup"
echo "============================================================"

# ── 1. Virtual environment ────────────────────────────────────────
if [ ! -d "${VENV_DIR}" ]; then
    echo "[1/4] Creating Python 3.10 virtual environment in ${VENV_DIR}/ …"
    "${PYTHON}" -m venv "${VENV_DIR}"
else
    echo "[1/4] Virtual environment already exists — skipping creation."
fi

# Activate the venv for the remainder of this script.
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "      Python : $(python --version)"
echo "      Pip    : $(pip --version)"

# ── 2. Install dependencies ───────────────────────────────────────
echo "[2/4] Installing dependencies from requirements.txt …"
pip install --upgrade pip wheel setuptools --quiet
pip install -r requirements.txt

# ── 3. Download Qwen3-TTS model ───────────────────────────────────
echo "[3/4] Pre-downloading Qwen3-TTS model weights from HuggingFace …"
echo "      (This is ~3.5 GB on first run; subsequent runs are instant.)"
python - <<'PYEOF'
import sys
try:
    from qwen_tts import Qwen3TTSModel
    import torch
    model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    print(f"  Downloading: {model_id}")
    Qwen3TTSModel.from_pretrained(model_id, device_map="cpu", dtype=torch.float32)
    print("  Qwen3-TTS ready.")
except Exception as exc:
    print(f"  WARNING: Could not download Qwen3-TTS model ({exc}).")
    print("  The model will be downloaded automatically on first synthesis run.")
    sys.exit(0)
PYEOF

# ── 4. NLTK tokenizer data ────────────────────────────────────────
echo "[4/4] Downloading NLTK punkt tokenizer data …"
python - <<'PYEOF'
import nltk
nltk.download("punkt",      quiet=False)
nltk.download("punkt_tab",  quiet=False)
print("  NLTK data ready.")
PYEOF

echo ""
echo "============================================================"
echo "  Setup complete!"
echo ""
echo "  Activate your environment:"
echo "    source ${VENV_DIR}/bin/activate"
echo ""
echo "  Run a quick synthesis test:"
echo "    python main.py --input 'Hello, world. This is a test.' \\"
echo "                   --output hello.mp3"
echo ""
echo "  Run the test suite:"
echo "    pytest test_tts.py -v"
echo "============================================================"
