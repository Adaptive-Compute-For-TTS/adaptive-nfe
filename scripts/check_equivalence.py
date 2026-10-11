"""Check that dots.tts driven by FixedScheduler(N) produces exactly the same audio as original dots.tts at NFE N.

For each text and each N, three runs with the same seed:
  A   original:      generate(num_steps=N), no scheduler
  A2  original again: same as A. If A != A2, dots.tts isn't deterministic on this machine
                      and an exact match against B is impossible.
  B   scheduler:     generate(num_steps=DECOY) with FixedScheduler(N) attached. DECOY != N on purpose:
                      if the hook were ignored, B would run at DECOY steps and fail loudly.

With --executor, B also runs every patch through adaptive_nfe.Executor instead of dots.tts's own
DiTSolver (eager path), so the comparison checks the executor reproduces stock exactly.

Compared per run: every patch latent the DiT produced (torch.equal), and the final waveform.
Also checked for B: the scheduler log says N for every patch, and the DiT really ran N times per patch
(and, with --executor, that the executor decoded every patch).

Needs the dots.tts clone on branch scheduler-hook. Exit code 1 if anything mismatches.

Example:
    python scripts/check_equivalence.py --nfe 4 10 32
    python scripts/check_equivalence.py --nfe 4 10 32 --executor
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import soundfile as sf
import torch
from adaptive_nfe import FixedScheduler
from adaptive_nfe.executor import Executor
from dots_tts.models.dots_tts.model import DotsTtsModel
from dots_tts.modules.backbone.dit_inference import DiTSolver, EagerDiTRunner
from dots_tts.runtime import DotsTtsRuntime
from dots_tts.utils.util import seed_everything

DEFAULT_TEXTS = [
    "Hello, this is a quick speech synthesis test.",
    "The scheduler decides how many denoising steps each patch of audio receives.",
]


@dataclass
class Capture:
    """Everything recorded during one generate() call."""

    patches: list[torch.Tensor] = field(default_factory=list)  # DiT output per patch, on CPU
    steps: list[int] = field(default_factory=list)  # DiT forward passes per patch
    audio: torch.Tensor | None = None
    executor_patches: int = 0  # patches decoded by the Executor instead of DiTSolver


current: Capture | None = None  # the run being recorded right now
current_executor: Executor | None = None  # set during runs that should go through the executor


def install_capture_hooks() -> None:
    """Wrap two dots.tts functions so every run records its patches and step counts."""
    decode_next_audio = DotsTtsModel._decode_next_audio
    velocity = EagerDiTRunner._flow_matching_velocity

    @functools.wraps(decode_next_audio)
    def decode_wrapper(self, state, **kwargs):
        current.steps.append(0)
        patch = decode_next_audio(self, state, **kwargs)
        current.patches.append(patch.detach().float().cpu().clone())
        return patch

    @functools.wraps(velocity)
    def velocity_wrapper(*args, **kwargs):
        current.steps[-1] += 1  # one DiT evaluation (cond + uncond batched together)
        return velocity(*args, **kwargs)

    DotsTtsModel._decode_next_audio = decode_wrapper
    EagerDiTRunner._flow_matching_velocity = velocity_wrapper


def install_executor_route() -> None:
    """Send the eager flow-matching DiT call through `current_executor` when one is set.

    Same effect as the hook in docs/design/executor-baseline.md (D1), without editing dots.tts.
    """
    decode_next = DiTSolver.decode_next

    @functools.wraps(decode_next)
    def decode_next_wrapper(self, state, **kwargs):
        if current_executor is None or self.optimize or self.meanflow:
            return decode_next(self, state, **kwargs)
        self._validate_decode_args(
            fm_seq_len=kwargs["fm_seq_len"],
            nfe=kwargs["nfe"],
            cfg_sequence=kwargs.get("cfg_sequence"),
            guidance_scale=kwargs.get("guidance_scale"),
        )
        sequence, null_g_cond, g_cond = kwargs["sequence"], kwargs["null_g_cond"], kwargs["g_cond"]
        g_cond = null_g_cond if g_cond is None else g_cond.to(device=null_g_cond.device, dtype=sequence.dtype)
        current.executor_patches += 1
        return current_executor.decode_patch(
            sequence=sequence,
            cfg_sequence=kwargs["cfg_sequence"],
            fm_seq_len=int(kwargs["fm_seq_len"]),
            g_cond=g_cond,
            nfe=int(kwargs["nfe"]),
            guidance_scale=kwargs["guidance_scale"],
            ode_method=kwargs.get("ode_method", "euler"),
        )

    DiTSolver.decode_next = decode_next_wrapper


def run(runtime: DotsTtsRuntime, text: str, *, num_steps: int, scheduler, seed: int,
        executor: Executor | None = None) -> Capture:
    global current, current_executor
    current = Capture()
    current_executor = executor
    runtime.model.nfe_scheduler = scheduler
    if scheduler is not None:
        scheduler.reset()
    seed_everything(seed)
    try:
        result = runtime.generate(text=text, num_steps=num_steps)
    finally:
        runtime.model.nfe_scheduler = None
        current_executor = None
    current.audio = result["audio"].detach().float().cpu()
    return current


def compare(a: Capture, b: Capture) -> dict:
    """Exact comparison of two runs. Returns a summary dict with 'equal' and where they diverge."""
    first_diff = None
    for i, (pa, pb) in enumerate(zip(a.patches, b.patches)):
        if not torch.equal(pa, pb):
            first_diff = i
            break
    same_shape = a.audio.shape == b.audio.shape
    audio_equal = same_shape and torch.equal(a.audio, b.audio)
    summary = {
        "equal": audio_equal and first_diff is None and len(a.patches) == len(b.patches),
        "patches": [len(a.patches), len(b.patches)],
        "first_differing_patch": first_diff,
        "audio_equal": audio_equal,
    }
    if first_diff is not None:
        summary["patch_max_abs_diff"] = (a.patches[first_diff] - b.patches[first_diff]).abs().max().item()
    if same_shape and not audio_equal:
        summary["audio_max_abs_diff"] = (a.audio - b.audio).abs().max().item()
    return summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="dots-studio/dots.tts-base")
    p.add_argument("--text", action="append", help="Repeatable. Defaults to two built-in sentences.")
    p.add_argument("--nfe", type=int, nargs="+", default=[4, 10, 32])
    p.add_argument("--precision", default="bfloat16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="runs/equivalence")
    p.add_argument("--executor", action="store_true", help="Run B's DiT through adaptive_nfe.Executor.")
    args = p.parse_args()
    texts = args.text or DEFAULT_TEXTS

    install_capture_hooks()
    install_executor_route()
    runtime = DotsTtsRuntime.from_pretrained(args.model, precision=args.precision, optimize=False)
    executor = Executor(runtime.model) if args.executor else None
    out_dir = Path(args.out_dir) / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    report = []
    for t_idx, text in enumerate(texts):
        for nfe in args.nfe:
            decoy = 1 if nfe != 1 else 2
            scheduler = FixedScheduler(nfe)
            a = run(runtime, text, num_steps=nfe, scheduler=None, seed=args.seed)
            a2 = run(runtime, text, num_steps=nfe, scheduler=None, seed=args.seed)
            b = run(runtime, text, num_steps=decoy, scheduler=scheduler, seed=args.seed, executor=executor)

            row = {
                "text": text,
                "nfe": nfe,
                "decoy_num_steps": decoy,
                "original_vs_itself": compare(a, a2),
                "original_vs_scheduler": compare(a, b),
                "scheduler_log_ok": [h["nfe"] for h in scheduler.history] == [nfe] * len(b.patches),
                "dit_steps_per_patch_ok": b.steps == [nfe] * len(b.patches) and a.steps == b.steps,
                "executor_used_ok": b.executor_patches == (len(b.patches) if executor else 0),
                "dit_steps_per_patch": b.steps,
                "scheduler_history": scheduler.history,
            }
            row["pass"] = (
                row["original_vs_scheduler"]["equal"] and row["scheduler_log_ok"] and row["dit_steps_per_patch_ok"]
                and row["executor_used_ok"]
            )
            report.append(row)

            b_name = "executor" if executor else "scheduler"
            for name, cap in (("original", a), (b_name, b)):
                sf.write(out_dir / f"t{t_idx}_nfe{nfe}_{name}.wav", cap.audio.squeeze().numpy(), runtime.sample_rate)

            status = "PASS" if row["pass"] else "FAIL"
            det = "deterministic" if row["original_vs_itself"]["equal"] else "NOT deterministic"
            print(f"[{status}] text {t_idx} nfe={nfe:<3} patches={len(b.patches):<3} original is {det}")
            if not row["pass"]:
                print("       ", json.dumps({k: row[k] for k in ("original_vs_scheduler", "scheduler_log_ok", "dit_steps_per_patch_ok", "executor_used_ok")}))

    config = {**vars(args), "texts": texts, "torch": torch.__version__, "cuda": torch.version.cuda,
              "device": str(runtime.device),
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    (out_dir / "report.json").write_text(json.dumps({"config": config, "results": report}, indent=2))
    all_pass = all(r["pass"] for r in report)
    print(f"\n{'ALL PASS' if all_pass else 'SOME FAILED'}  -> {out_dir / 'report.json'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
