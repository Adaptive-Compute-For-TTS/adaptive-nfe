"""Watch the two nested loops inside dots.tts without editing its source.

We wrap two internal functions with counters, then call generate() normally:
  - DotsTtsModel._decode_next_audio  -> called once per audio patch (outer, autoregressive loop)
  - ODESchedule.advance              -> called once per denoising step (inner loop, this is NFE)

Expected result: steps == patches * num_steps.
_decode_next_audio is also where the scheduler hook will go later.

Example:
    python scripts/trace_nfe.py --text "Hello there." --num-steps 10
"""

from __future__ import annotations

import argparse
import functools

from dots_tts.models.dots_tts.model import DotsTtsModel
from dots_tts.modules.backbone.dit_inference import ODESchedule
from dots_tts.runtime import DotsTtsRuntime
from dots_tts.utils.util import seed_everything

patch_log: list[dict] = []  # one entry per patch: {"patch": i, "nfe": n, "steps_run": k}
_steps_this_patch = 0


def wrap_decode_next_audio(original):
    @functools.wraps(original)
    def wrapper(self, state, *, num_steps, **kwargs):
        global _steps_this_patch
        _steps_this_patch = 0
        patch = original(self, state, num_steps=num_steps, **kwargs)
        patch_log.append({"patch": len(patch_log), "nfe": num_steps, "steps_run": _steps_this_patch})
        return patch

    return wrapper


def wrap_advance(original):
    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        global _steps_this_patch
        _steps_this_patch += 1
        return original(*args, **kwargs)

    return wrapper


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="dots-studio/dots.tts-base")
    p.add_argument("--text", default="Hello there.")
    p.add_argument("--num-steps", type=int, default=10)
    p.add_argument("--precision", default="float32")
    args = p.parse_args()

    DotsTtsModel._decode_next_audio = wrap_decode_next_audio(DotsTtsModel._decode_next_audio)
    ODESchedule.advance = wrap_advance(ODESchedule.advance)

    runtime = DotsTtsRuntime.from_pretrained(args.model, precision=args.precision, optimize=False)
    seed_everything(42)
    result = runtime.generate(text=args.text, num_steps=args.num_steps)

    for row in patch_log:
        print(row)
    total_steps = sum(r["steps_run"] for r in patch_log)
    seconds = result["audio"].shape[-1] / result["sample_rate"]
    print(f"\npatches={len(patch_log)}  nfe={args.num_steps}  total DiT steps={total_steps}  audio={seconds:.2f}s")
    print(f"expected steps = patches * nfe = {len(patch_log) * args.num_steps}")


if __name__ == "__main__":
    main()
