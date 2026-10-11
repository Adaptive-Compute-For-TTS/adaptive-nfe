#!/usr/bin/env bash
# Creates .venv in this repo and installs torch + dots.tts + this repo's package.
# Run from the repo root:  bash scripts/setup_env.sh        (CUDA build)
#                          bash scripts/setup_env.sh cpu    (CPU build)
# Expects the dots.tts clone at ../dots.tts
set -euo pipefail

DEVICE="${1:-cuda}"
case "$DEVICE" in
  cuda) INDEX=https://download.pytorch.org/whl/cu128 ;;
  cpu)  INDEX=https://download.pytorch.org/whl/cpu ;;
  *) echo "usage: $0 [cuda|cpu]" >&2; exit 1 ;;
esac

[ -f ../dots.tts/pyproject.toml ] || { echo "dots.tts clone not found at ../dots.tts" >&2; exit 1; }
[ -x .venv/bin/python ] || python3 -m venv .venv
PY=.venv/bin/python

$PY -m pip install --upgrade pip

# 1. torch first, from the index matching the hardware, so pip can't pick a different build later.
$PY -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url "$INDEX"

# 2. dots.tts (editable), this repo (editable), test deps. Pinned by constraints.txt.
$PY -m pip install -r requirements.txt -c constraints.txt

# 3. Sanity check.
$PY -c "import torch, dots_tts, adaptive_nfe; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
