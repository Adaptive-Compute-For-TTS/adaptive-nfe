"""One prompt + one NFE -> one WAV, with config and measurements. The demo's GPU generation command.

--prompt-id and --nfe take several values; every combination runs in one process, so the model loads
(and with --optimize, compiles) once.

Every patch gets the same NFE (FixedScheduler). Two paths:
  default     eager (optimize=False). Our Executor runs each patch's DiT loop via the dit_executor hook.
  --optimize  dots.tts's fast path (torch.compile + DiT KV cache, ~3 min warmup, CUDA only). The executor
              doesn't support it yet, so the DiT loop is stock dots.tts; the scheduler still sets NFE per patch.

--verify-stock also runs plain dots.tts (no hooks, same seed, same path) and checks the audio is bit-identical.

NFE here = dots.tts num_steps. CFG batches cond + uncond into the same DiT call, so NFE 10 is 10 DiT
calls per patch. Team convention (ml-evals too): NFE = num_steps.

Output, one folder per prompt x NFE, in <out-dir>/<run_id>/:
    audio.wav       float32
    config.json     settings + environment
    metrics.json    nfe per patch, latency, RTF, peak VRAM, audio hash, checks

Examples:
    python scripts/generate.py --prompt-id p004 --nfe 10
    python scripts/generate.py --text "Hello there." --nfe 4 --verify-stock
    python scripts/generate.py --prompt-id p004 --nfe 10 --optimize
    python scripts/generate.py --prompt-id p001 p004 p007 --nfe 4 10 32 --optimize --verify-stock
    python scripts/generate.py --prompt-id p004 --nfe 10 --ref-audio ref.wav --ref-text "what ref.wav says"
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as meta
import json
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
from adaptive_nfe import generate_fixed_nfe
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


def main() -> int:
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--text")
    src.add_argument("--prompt-id", nargs="+", help="id(s) from the ml-evals benchmark prompts, e.g. p004")
    p.add_argument("--nfe", type=int, nargs="+", required=True, help="denoising steps per patch (dots.tts num_steps)")
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
    p.add_argument("--no-warmup", action="store_true", help="skip the discarded warmup run per NFE (latency less accurate)")
    p.add_argument("--out-dir", default="runs/generate")
    p.add_argument("--run-id", help="default: <timestamp>_<prompt>_nfe<N>; with several runs, used as a prefix")
    a = p.parse_args()

    if a.prompt_id:
        prompts = {e["id"]: e for e in json.loads(Path(a.prompts).read_text(encoding="utf-8"))}
        for pid in a.prompt_id:
            if pid not in prompts:
                p.error(f"--prompt-id {pid} not in {a.prompts}")
        texts = [(pid, prompts[pid]["text"]) for pid in a.prompt_id]
    else:
        texts = [(None, a.text)]

    rt = DotsTtsRuntime.from_pretrained(a.model, precision=a.precision, optimize=a.optimize)
    executor = None if a.optimize else Executor(rt.model)
    path = "optimize (stock DiT fast path)" if a.optimize else "eager + executor"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    single = len(texts) * len(a.nfe) == 1

    all_ok = True
    for nfe in a.nfe:
        ode, steps, cfg = rt.resolve_sampling_options(ode_method=None, num_steps=nfe, guidance_scale=a.guidance_scale)
        if not a.no_warmup:
            # Discarded run at this NFE. --optimize compiles in from_pretrained, but only at the default
            # num_steps; the first run at a new NFE still pays for new KV cache / CUDA graph setup.
            print(f"Warming up nfe={steps} (discarded)...")
            seed_all(a.seed)
            generate_fixed_nfe(rt, steps, use_executor=not a.optimize, executor=executor, text=texts[0][1],
                               prompt_audio_path=a.ref_audio, prompt_text=a.ref_text, ode_method=ode,
                               guidance_scale=cfg, speaker_scale=a.speaker_scale)
        for prompt_id, text in texts:
            name = f"{prompt_id or 'text'}_nfe{steps}"
            run_id = a.run_id if (a.run_id and single) else f"{a.run_id or stamp}_{name}"
            all_ok &= run_one(a, rt, executor, path, run_id, prompt_id, text, ode, steps, cfg)
    return 0 if all_ok else 1


def run_one(a, rt, executor, path, run_id, prompt_id, text, ode, steps, cfg) -> bool:
    out = Path(a.out_dir) / run_id
    out.mkdir(parents=True, exist_ok=True)
    gen_kwargs = dict(text=text, prompt_audio_path=a.ref_audio, prompt_text=a.ref_text, ode_method=ode,
                      guidance_scale=cfg, speaker_scale=a.speaker_scale)
    print(f"run_id={run_id}  path={path}  nfe={steps}  cfg={cfg}  text={text!r}")

    # generate_fixed_nfe raises if any patch missed the NFE or the executor.
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    seed_all(a.seed)
    result = generate_fixed_nfe(rt, steps, use_executor=not a.optimize, executor=executor, **gen_kwargs)
    r, nfe_per_patch = result.output, result.nfe_per_patch
    peak_vram_mb = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None
    sr = r["sample_rate"]
    sf.write(out / "audio.wav", r["audio"].float().cpu().squeeze().numpy(), sr, subtype="FLOAT")

    checks = {
        "nfe_fixed_all_patches": len(nfe_per_patch) > 0 and set(nfe_per_patch) == {steps},
        # eager: the executor decoded every patch at the chosen NFE
        "executor_used_all_patches": None if a.optimize else result.executor_nfe_per_patch == nfe_per_patch,
    }
    if a.verify_stock:
        seed_all(a.seed)
        stock = rt.generate(num_steps=steps, **gen_kwargs)  # no hooks: plain dots.tts
        checks["stock_identical"] = audio_sha256(stock["audio"]) == audio_sha256(r["audio"])

    metrics = dict(
        run_id=run_id, prompt_id=prompt_id, text=text, nfe=steps, path=path,
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
    return all(v is not False for v in checks.values())


if __name__ == "__main__":
    sys.exit(main())
