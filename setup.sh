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
#   3. Pre-downloads the XTTS v2 model weights via Coqui TTS CLI
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
# Note: coqui-tts is the community-maintained fork of coqui-ai/TTS that
# supports Python 3.12+.  It is a drop-in replacement — all imports remain
# the same (from TTS.api import TTS).

# ── 3. Download XTTS v2 model ─────────────────────────────────────
echo "[3/4] Pre-downloading Coqui XTTS v2 model weights …"
echo "      (This is ~2 GB on first run; subsequent runs are instant.)"
python - <<'PYEOF'
from TTS.api import TTS
import os, sys

model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
print(f"  Downloading: {model_name}")
try:
    # Passing agree_to_terms=True avoids interactive prompts in CI.
    os.environ["COQUI_TOS_AGREED"] = "1"
    tts = TTS(model_name=model_name, progress_bar=True, gpu=False)
    print("  XTTS v2 ready.")
except Exception as exc:
    print(f"  WARNING: Could not download XTTS v2 ({exc}).")
    print("  The engine will fall back to tacotron2-DDC at runtime.")
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
