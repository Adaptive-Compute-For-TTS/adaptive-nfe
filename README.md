# dot-tts-baseline

Fixed-NFE baseline and scheduler integration work for dots.tts (WAT.ai GPU Systems).

dots.tts lives in a separate clone next to this repo (`..\dots.tts`) and is installed into this repo's venv in editable mode, so edits to its source take effect immediately.

## Setup (one time)

From this repo's root, with the NVIDIA GPU enabled:

```powershell
nvidia-smi                          # GPU visible? driver 570+ needed for cu128
.\scripts\setup_env.ps1 -Device cuda  # or -Device cpu
```

The last line of the setup output should say `cuda available: True`. The checkpoint (`dots-studio/dots.tts-base`) downloads from Hugging Face on the first `from_pretrained`.

In VS Code, select `.venv\Scripts\python.exe` as the interpreter.

## Scripts

| Script | What it does |
|---|---|
| `scripts/run_baseline.py` | Calls unmodified `generate()` at a fixed NFE. Saves `audio.wav`, `audio.pt` (exact tensor), and `config.json` under `runs/`. This is the reference for the equivalence test. |
| `scripts/trace_nfe.py` | Same call with counters wrapped around the patch loop and the ODE-step loop. Prints how many steps each patch got. |

```powershell
.\.venv\Scripts\python.exe scripts\trace_nfe.py --text "Hello there." --num-steps 10 --precision bfloat16
```

Always use `optimize=False` and a fixed seed when comparing outputs.
