"""One prompt + one NFE -> one WAV, with config and measurements. The demo's GPU generation command.

Every patch gets the same NFE (FixedScheduler). Two paths:
  default     eager (optimize=False). Our Executor runs each patch's DiT loop via the dit_executor hook.
  --optimize  dots.tts's fast path (torch.compile + DiT KV cache, ~3 min warmup, CUDA only). The executor
              doesn't support it yet, so the DiT loop is stock dots.tts; the scheduler still sets NFE per patch.

--verify-stock also runs plain dots.tts (no hooks, same seed, same path) and checks the audio is bit-identical.

NFE here = dots.tts num_steps. CFG batches cond + uncond into the same DiT call, so NFE 10 is 10 DiT
calls per patch. Team convention (ml-evals too): NFE = num_steps.

Output, in <out-dir>/<run_id>/:
    audio.wav       float32
    config.json     settings + environment
    metrics.json    nfe per patch, latency, RTF, peak VRAM, audio hash, checks

Examples:
    python scripts/generate.py --prompt-id p004 --nfe 10
    python scripts/generate.py --text "Hello there." --nfe 4 --verify-stock
    python scripts/generate.py --prompt-id p004 --nfe 10 --optimize
    python scripts/generate.py --prompt-id p004 --nfe 10 --ref-audio ref.wav --ref-text "what ref.wav says"
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib.metadata as meta
import json
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
from adaptive_nfe import FixedScheduler
from adaptive_nfe.executor import Executor
from dots_tts.runtime import DotsTtsRuntime

PROMPTS_DEFAULT = Path(__file__).resolve().parents[2] / "ml-evals" / "benchmark_prompts_v1.json"


def audio_sha256(audio: torch.Tensor) -> str:
    # Hash of the samples, not the WAV file: libsndfile puts a timestamp in float WAVs.
    return hashlib.sha256(audio.float().cpu().contiguous().numpy().tobytes()).hexdigest()


def seed_all(s: int) -> None:
    # Same as ml-evals/run_baseline.py.
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def count_executor_calls(executor: Executor) -> list[int]:
    """Wrap executor.decode_patch to record the nfe of every patch it decodes."""
    calls: list[int] = []
    decode_patch = executor.decode_patch

    @functools.wraps(decode_patch)
    def wrapper(**kwargs):
        calls.append(kwargs["nfe"])
        return decode_patch(**kwargs)

    executor.decode_patch = wrapper
    return calls


def main() -> int:
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--text")
    src.add_argument("--prompt-id", help="id from the ml-evals benchmark prompts, e.g. p004")
    p.add_argument("--nfe", type=int, required=True, help="denoising steps per patch (dots.tts num_steps)")
    p.add_argument("--model", default="dots-studio/dots.tts-soar")
    p.add_argument("--prompts", default=str(PROMPTS_DEFAULT))
    p.add_argument("--ref-audio", help="reference audio for voice cloning (optional)")
    p.add_argument("--ref-text", help="transcript of --ref-audio (optional)")
    p.add_argument("--guidance-scale", type=float, default=1.2)
    p.add_argument("--speaker-scale", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--precision", default="bfloat16")
    p.add_argument("--optimize", action="store_true", help="dots.tts fast path; no executor (see top).")
    p.add_argument("--verify-stock", action="store_true", help="also run plain dots.tts and compare audio")
    p.add_argument("--no-warmup", action="store_true", help="skip the discarded warmup run (latency less accurate)")
    p.add_argument("--out-dir", default="runs/generate")
    p.add_argument("--run-id", help="default: <timestamp>_<prompt>_nfe<N>")
    a = p.parse_args()

    if a.prompt_id:
        prompts = {e["id"]: e for e in json.loads(Path(a.prompts).read_text(encoding="utf-8"))}
        if a.prompt_id not in prompts:
            p.error(f"--prompt-id {a.prompt_id} not in {a.prompts}")
        text = prompts[a.prompt_id]["text"]
    else:
        text = a.text
    run_id = a.run_id or f"{time.strftime('%Y%m%d-%H%M%S')}_{a.prompt_id or 'text'}_nfe{a.nfe}"
    out = Path(a.out_dir) / run_id
    out.mkdir(parents=True, exist_ok=True)

    rt = DotsTtsRuntime.from_pretrained(a.model, precision=a.precision, optimize=a.optimize)
    ode, steps, cfg = rt.resolve_sampling_options(ode_method=None, num_steps=a.nfe, guidance_scale=a.guidance_scale)
    gen_kwargs = dict(text=text, prompt_audio_path=a.ref_audio, prompt_text=a.ref_text, ode_method=ode,
                      num_steps=steps, guidance_scale=cfg, speaker_scale=a.speaker_scale)

    scheduler = FixedScheduler(steps)
    executor = None if a.optimize else Executor(rt.model)
    executor_calls = count_executor_calls(executor) if executor else []

    def generate(hooked: bool) -> dict:
        rt.model.nfe_scheduler = scheduler if hooked else None
        rt.model.dit_executor = executor if hooked else None
        scheduler.reset()
        executor_calls.clear()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        seed_all(a.seed)
        try:
            return rt.generate(**gen_kwargs)
        finally:
            rt.model.nfe_scheduler = None
            rt.model.dit_executor = None

    path = "optimize (stock DiT fast path)" if a.optimize else "eager + executor"
    print(f"run_id={run_id}  path={path}  nfe={steps}  cfg={cfg}  text={text!r}")
    if not a.no_warmup and not a.optimize:  # --optimize already warms up inside from_pretrained
        print("Warming up (discarded)...")
        generate(hooked=True)

    r = generate(hooked=True)
    peak_vram_mb = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None
    nfe_per_patch = [h["nfe"] for h in scheduler.history]
    sr = r["sample_rate"]
    sf.write(out / "audio.wav", r["audio"].float().cpu().squeeze().numpy(), sr, subtype="FLOAT")

    checks = {
        "nfe_fixed_all_patches": len(nfe_per_patch) > 0 and set(nfe_per_patch) == {steps},
        # eager: the executor decoded every patch at the chosen NFE
        "executor_used_all_patches": None if a.optimize else executor_calls == nfe_per_patch,
    }
    if a.verify_stock:
        stock = generate(hooked=False)
        checks["stock_identical"] = audio_sha256(stock["audio"]) == audio_sha256(r["audio"])

    metrics = dict(
        run_id=run_id, prompt_id=a.prompt_id, text=text, nfe=steps, path=path,
        num_patches=len(nfe_per_patch), nfe_per_patch=nfe_per_patch,
        audio_s=round(r["audio"].shape[-1] / sr, 4), time_s=round(r["time_used"], 4), rtf=round(r["rtf"], 6),
        peak_vram_mb=None if peak_vram_mb is None else round(peak_vram_mb, 1),
        audio_sha256=audio_sha256(r["audio"]), checks=checks,
    )
    config = dict(
        run_id=run_id, model=a.model, ode_method=ode, num_steps=steps, guidance_scale=cfg,
        speaker_scale=a.speaker_scale, seed=a.seed, precision=a.precision, optimize=a.optimize,
        ref_audio=a.ref_audio, ref_text=a.ref_text, sample_rate=sr, dots_tts_version=meta.version("dots.tts"),
        torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, argv=sys.argv,
    )
    (out / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    print(f"patches={metrics['num_patches']}  audio={metrics['audio_s']}s  time={metrics['time_s']}s  "
          f"RTF={metrics['rtf']}  peak_vram={metrics['peak_vram_mb']}MB")
    print("checks:", json.dumps(checks))
    print(f"-> {out}/")
    return 0 if all(v is not False for v in checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
