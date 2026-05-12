#!/usr/bin/env bash
# Run once on a fresh Lambda Labs (or any Ubuntu) instance.
# Usage:  bash environment/setup.sh
set -euo pipefail

# ── 1. Virtual environment ───────────────────────────────────────────────
if [ ! -d "$HOME/venv" ]; then
    echo "[setup] Creating venv at ~/venv"
    python3 -m venv "$HOME/venv"
else
    echo "[setup] ~/venv already exists, skipping creation"
fi

source "$HOME/venv/bin/activate"

# ── 2. Upgrade pip ───────────────────────────────────────────────────────
pip install --upgrade pip --quiet

# ── 3. Install requirements ──────────────────────────────────────────────
echo "[setup] Installing requirements..."
pip install -r environment/requirements.txt

# ── 4. HuggingFace login ─────────────────────────────────────────────────
echo ""
echo "[setup] ─────────────────────────────────────────────────────────────"
echo "[setup] You need to log in to HuggingFace to download gated models."
echo "[setup] Create a read-access token at https://huggingface.co/settings/tokens"
echo "[setup] Then run:   huggingface-cli login"
echo "[setup] ─────────────────────────────────────────────────────────────"
echo ""

# ── 5. Activate reminder ─────────────────────────────────────────────────
echo "[setup] Done. To activate the venv in every new shell, run:"
echo "    source ~/venv/bin/activate"
echo ""
echo "[setup] Add this to ~/.bashrc to activate automatically:"
echo "    echo 'source ~/venv/bin/activate' >> ~/.bashrc"
