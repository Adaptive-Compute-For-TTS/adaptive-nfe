# Creates .venv in this repo and installs torch + dots.tts + test deps.
# Run from the repo root:  .\scripts\setup_env.ps1 -Device cuda   (or -Device cpu)
# Expects the dots.tts clone at ..\dots.tts

param(
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"
$constraints = "..\dots.tts\constraints\recommended.txt"

if (-not (Test-Path "..\dots.tts\pyproject.toml")) { throw "dots.tts clone not found at ..\dots.tts" }
if (-not (Test-Path $py)) { python -m venv .venv }

& $py -m pip install --upgrade pip

# 1. torch first, from the index matching the hardware. A plain `pip install` on Windows
#    would pull the CPU-only build, and dots.tts would silently run on CPU.
$index = if ($Device -eq "cuda") { "https://download.pytorch.org/whl/cu128" } else { "https://download.pytorch.org/whl/cpu" }
& $py -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url $index

# 2. dots.tts (editable) and its dependencies, pinned to the team's recommended versions.
& $py -m pip install -r requirements.txt -c $constraints

# 3. Sanity check.
& $py -c "import torch, dots_tts; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
