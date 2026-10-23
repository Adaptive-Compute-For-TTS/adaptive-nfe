"""Run the ml-evals benchmark prompts through stock dots.tts, our scheduler hook, or our executor.

Same seeding, generate() kwargs and output layout as ml-evals/run_baseline.py, so the output folder
can be dropped into ml-evals/baselines/ and scored by tts_eval_org like Nishant's baseline:

    <out-dir>/config.json      settings + environment
    <out-dir>/manifest.json    one row per prompt: wav path, sha256s, audio_s, time_s, rtf (+ nfe per patch)
    <out-dir>/wavs/<id>.wav    float32

--mode:
  stock      plain dots.tts, num_steps=N (should reproduce ml-evals run_baseline.py)
  scheduler  FixedScheduler(N) attached as model.nfe_scheduler
  executor   FixedScheduler(N) + adaptive_nfe.Executor attached as model.dit_executor

--compare-to <manifest.json> checks audio_sha256 (hash of the samples) per prompt id against another run. Only meaningful for a
run on the same GPU + software (Nishant's 3090 hashes won't match a 5060 even for stock).

Note: --num-steps is dots.tts num_steps = NFE (team convention). Nishant's first baseline config says
nfe_estimate=20 for 10 steps; that was the old steps x 2 CFG count, it's NFE 10.

Example:
    python scripts/run_benchmark.py --mode stock    --out-dir runs/benchmark/soar_steps10_stock
    python scripts/run_benchmark.py --mode executor --out-dir runs/benchmark/soar_steps10_executor \
        --compare-to runs/benchmark/soar_steps10_stock/manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as meta
import json
import sys
from pathlib import Path

import soundfile as sf
import torch
from adaptive_nfe import FixedScheduler
from adaptive_nfe.executor import Executor
from dots_tts.runtime import DotsTtsRuntime

PROMPTS_DEFAULT = Path(__file__).resolve().parents[2] / "ml-evals" / "benchmark_prompts_v1.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audio_sha256(audio) -> str:
    # Hash of the float32 samples. Use this to compare runs, not wav_sha256: libsndfile writes a
    # timestamped PEAK chunk into float WAVs, so the file hash changes every run even for identical audio.
    return hashlib.sha256(audio.float().cpu().contiguous().numpy().tobytes()).hexdigest()


def seed_all(s: int) -> None:
    # Same as ml-evals/run_baseline.py.
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["stock", "scheduler", "executor"], required=True)
    p.add_argument("--model", default="dots-studio/dots.tts-soar")
    p.add_argument("--prompts", default=str(PROMPTS_DEFAULT), help="ml-evals benchmark_prompts_v1.json")
    p.add_argument("--limit", type=int, help="Only the first N prompts (smoke test).")
    p.add_argument("--num-steps", type=int, default=10)
    p.add_argument("--guidance-scale", type=float, default=1.2)
    p.add_argument("--speaker-scale", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--precision", default="bfloat16")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--compare-to", help="Another run's manifest.json; checks audio_sha256 per prompt id.")
    a = p.parse_args()

    prompts = json.loads(Path(a.prompts).read_text(encoding="utf-8"))[: a.limit]
    out = Path(a.out_dir)
    wav_dir = out / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    # optimize=False: the executor only supports the eager path (see dit_executor hook).
    rt = DotsTtsRuntime.from_pretrained(a.model, precision=a.precision, optimize=False)
    ode, steps, cfg = rt.resolve_sampling_options(ode_method=None, num_steps=a.num_steps,
                                                  guidance_scale=a.guidance_scale)
    scheduler = FixedScheduler(steps) if a.mode != "stock" else None
    rt.model.nfe_scheduler = scheduler
    rt.model.dit_executor = Executor(rt.model) if a.mode == "executor" else None
    print(f"mode={a.mode}  ode={ode}  num_steps={steps}  guidance_scale={cfg}  prompts={len(prompts)}")

    gen_kwargs = dict(prompt_audio_path=None, prompt_text=None, ode_method=ode, num_steps=steps,
                      guidance_scale=cfg, speaker_scale=a.speaker_scale)

    print("Warming up (discarded)...")
    seed_all(a.seed)
    rt.generate(text=prompts[0]["text"], **gen_kwargs)

    rows = []
    for entry in prompts:
        if scheduler is not None:
            scheduler.reset()
        seed_all(a.seed)
        r = rt.generate(text=entry["text"], **gen_kwargs)

        sr = r["sample_rate"]
        wav_path = wav_dir / f"{entry['id']}.wav"
        sf.write(wav_path, r["audio"].float().cpu().squeeze().numpy(), sr, subtype="FLOAT")
        row = dict(
            id=entry["id"], text=entry["text"], length_bucket=entry["length_bucket"], category=entry["category"],
            wav=str(wav_path), wav_sha256=sha256(wav_path), audio_sha256=audio_sha256(r["audio"]),
            audio_s=round(r["audio"].shape[-1] / sr, 4), time_s=round(r["time_used"], 4), rtf=round(r["rtf"], 6),
        )
        if scheduler is not None:
            row["nfe_per_patch"] = [h["nfe"] for h in scheduler.history]
        rows.append(row)
        print(f"[{entry['id']}] {entry['length_bucket']:6}  {entry['category']:25}  RTF={r['rtf']:.3f}")

    config = dict(
        mode=a.mode, model=a.model, ode_method=ode, num_steps=steps, guidance_scale=cfg,
        speaker_scale=a.speaker_scale, seed=a.seed, precision=a.precision, optimize=False,
        sample_rate=sr, n_prompts=len(prompts), prompts_file=str(Path(a.prompts).resolve()),
        prompts_sha256=sha256(Path(a.prompts)), dots_tts_version=meta.version("dots.tts"),
        torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, argv=sys.argv,
    )
    (out / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))
    (out / "manifest.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"\nSaved {len(rows)} WAVs -> {wav_dir}/")

    if not a.compare_to:
        return 0
    ref = {row["id"]: row["audio_sha256"] for row in json.loads(Path(a.compare_to).read_text())}
    mismatched = [row["id"] for row in rows if ref.get(row["id"]) != row["audio_sha256"]]
    print(f"vs {a.compare_to}: {len(rows) - len(mismatched)}/{len(rows)} identical"
          + (f"  mismatched: {mismatched}" if mismatched else ""))
    return 1 if mismatched else 0


if __name__ == "__main__":
    sys.exit(main())
