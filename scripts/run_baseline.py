"""Run unmodified dots.tts once at a fixed NFE and save everything needed to reproduce it.

This is the reference ("original") side of the equivalence test. It only uses the
public API: DotsTtsRuntime.from_pretrained(...).generate(...).

Example:
    python scripts/run_baseline.py --text "Hello there." --num-steps 10 --precision float32
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import soundfile as sf
import torch
from dots_tts.runtime import DotsTtsRuntime
from dots_tts.utils.util import seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="dots-studio/dots.tts-base")
    p.add_argument("--text", default="Hello, this is a quick speech synthesis test.")
    p.add_argument("--prompt-audio", default=None, help="Reference .wav for voice cloning (optional).")
    p.add_argument("--prompt-text", default=None, help="Exact transcript of --prompt-audio.")
    p.add_argument("--num-steps", type=int, default=10, help="NFE: denoising steps per patch.")
    p.add_argument("--guidance-scale", type=float, default=1.2)
    p.add_argument("--seed", type=int, default=42)
    # float32 on CPU (half precision is unreliable there); bfloat16 on GPU.
    p.add_argument("--precision", default="float32")
    p.add_argument("--out-dir", default="runs")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # optimize=False: torch.compile/CUDA graphs can change numerics, which breaks exact comparison.
    runtime = DotsTtsRuntime.from_pretrained(args.model, precision=args.precision, optimize=False)

    # Seed right before generate so the initial noise for every patch is reproducible.
    seed_everything(args.seed)
    result = runtime.generate(
        text=args.text,
        prompt_audio_path=args.prompt_audio,
        prompt_text=args.prompt_text,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
    )

    run_dir = Path(args.out_dir) / f"{time.strftime('%Y%m%d-%H%M%S')}_nfe{args.num_steps}"
    run_dir.mkdir(parents=True, exist_ok=True)

    audio = result["audio"].float().cpu()
    sf.write(run_dir / "audio.wav", audio.squeeze().numpy(), result["sample_rate"])
    # Raw tensor too: the .wav is lossy (float -> 16-bit), so compare tensors, not wavs.
    torch.save(audio, run_dir / "audio.pt")

    config = {
        **vars(args),
        "sample_rate": result["sample_rate"],
        "audio_seconds": audio.shape[-1] / result["sample_rate"],
        "time_used": result["time_used"],
        "rtf": result["rtf"],
        "device": str(runtime.device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"Saved {run_dir}  ({config['audio_seconds']:.2f}s audio, rtf={config['rtf']:.2f})")


if __name__ == "__main__":
    main()
