# dot-tts-baseline

Internal repo for the WAT.ai GPU systems team. Setting up the baseline for the NFE scheduler and GPU execution calls on dots.tts. Testing with a fixed scheduler first (same NFE every patch) to make sure hooking in doesn't change anything.

## Folders

- `patches/`: our edit to dots.tts (the scheduler and executor hooks), saved as a git patch so anyone can apply it to their clone
- `src/adaptive_nfe/`: the scheduler (`scheduler.py`) and the DiT executor (`executor.py`, `ode.py`)
- `scripts/`: setup (`setup_env.sh`), the comparison test (`check_equivalence.py`), and `inspect_dit.py` for poking at the DiT
- `tests/`: fast checks that need no model weights (`pytest tests/`)
- `docs/design/`: executor design (`executor-baseline.md`) and next-phase notes (`batching-exploration.md`)
- `runs/`: results from running the scripts (reports, wavs). Not committed, it's in .gitignore

## What's going on

dots.tts makes audio one patch (~160 ms) at a time, and each patch gets `num_steps` denoising steps from the DiT. Right now every patch gets the same number. Plan is to let a scheduler pick it per patch.

Before doing anything smart, we checked that plugging a scheduler in gives the exact same output as stock dots.tts when it just returns a constant.

- `patches/dots_tts_scheduler_hook.patch`: small change to dots.tts `model.py`. Adds `model.nfe_scheduler` and `model.dit_executor` (both None by default). If a scheduler is set, it gets asked for `num_steps` before each patch. If an executor is set, it runs the DiT for the patch instead of dots.tts's own solver (eager flow matching only, errors otherwise).
- `src/adaptive_nfe/scheduler.py`: `Scheduler` base class (logs every choice) and `FixedScheduler(n)`.
- `scripts/check_equivalence.py`: the test. Same seed, for each text and NFE N it runs stock twice (A, A2) and once with `FixedScheduler(N)` (B). B is called with `num_steps=1` so it only passes if the hook actually works. Compares every patch and the final audio with `torch.equal`, and counts DiT calls per patch.
- `scripts/inspect_dit.py`: dumps velocity per step + EOS prob per patch. Might be useful later for deciding which patches are "hard".

All eager mode (`optimize=False`), bf16, seed 42, dots.tts commit `3cb6f94`.

## Results

First run (Oct 11, RTX 5060 Laptop 8 GB, torch 2.8.0+cu128): **all pass**. FixedScheduler output is bit for bit identical to stock at every NFE, and stock matches itself run to run, so exact comparison works on GPU.

| text | NFE | patches | stock vs stock | stock vs FixedScheduler | DiT calls per patch |
|---|---|---|---|---|---|
| "Hello, this is a quick speech synthesis test." | 4 | 23 | identical | identical | 4 |
| | 10 | 22 | identical | identical | 10 |
| | 32 | 21 | identical | identical | 32 |
| "The scheduler decides how many denoising steps each patch of audio receives." | 4 | 29 | identical | identical | 4 |
| | 10 | 27 | identical | identical | 10 |
| | 32 | 28 | identical | identical | 32 |

"Identical" means `torch.equal` on every patch and on the final waveform. Patch count changes a bit with NFE because the audio comes out slightly different, so the EOS head stops at a different point.

**Executor** (`--executor`, same setup, torchdiffeq 0.2.5): **all pass**. Running B through `adaptive_nfe.Executor` instead of dots.tts's solver gives the same table: every patch and the waveform identical to stock, same patch counts (23/22/21, 29/27/28), N DiT calls per patch, and every patch went through the executor. `pytest tests/`: 163 passed.

## Running it

WSL/Linux only (pynini won't build on Windows).

```bash
git clone https://github.com/tonyyytan/dot-tts-baseline.git
git clone https://github.com/studio-dots-ai/dots.tts.git
cd dots.tts && git switch -c scheduler-hook 3cb6f94
git apply ../dot-tts-baseline/patches/dots_tts_scheduler_hook.patch
cd ../dot-tts-baseline && bash scripts/setup_env.sh && source .venv/bin/activate
python scripts/check_equivalence.py --nfe 4 10 32
```

First run downloads the weights. Output ends up in `runs/equivalence/<timestamp>/`.

### Executor

`adaptive_nfe.Executor` runs the DiT for one patch at a given NFE (eager, Euler, one request) and must reproduce stock exactly. Design: `docs/design/executor-baseline.md`.

```bash
pytest tests/                                                  # no weights, CPU, seconds
python scripts/check_equivalence.py --nfe 4 10 32 --executor   # full model through the executor
```

## Todo

- try `optimize=True` (KV cache is sized per NFE so varying it will probably break)
- actual variable NFE scheduler
- sort out logging format + executor interface with everyone
