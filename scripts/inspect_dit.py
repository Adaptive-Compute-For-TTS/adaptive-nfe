"""Look inside one generation: the DiT's velocity at every ODE step, and the stop (EOS) probability.

Hooks (read-only, nothing in dots.tts is edited):
  - EagerDiTRunner._flow_matching_velocity(t, z) -> sees the noisy patch z and returns the CFG-combined velocity
  - DotsTtsModel._decode_next_audio  -> marks patch boundaries and reads the EOS probability

For each patch it prints:
  eos     probability the LLM thinks speech should end here (generation stops above 0.8)
  |v|     size of the velocity at each step
  cos     cosine similarity between consecutive velocities. Near 1.0 = straight path,
          so few steps would have been enough. Lower = curved path, needs more steps.

Saves everything to runs/inspect_nfe{N}.pt for plotting.

Example:
    python scripts/inspect_dit.py --text "Hello there." --num-steps 10 --precision bfloat16
"""

from __future__ import annotations

import argparse
import functools
from pathlib import Path

import torch
import torch.nn.functional as F
from dots_tts.models.dots_tts.model import DotsTtsModel
from dots_tts.modules.backbone.dit_inference import EagerDiTRunner
from dots_tts.runtime import DotsTtsRuntime
from dots_tts.utils.util import seed_everything

patches: list[dict] = []  # one entry per patch: {"eos": float, "z": [...], "v": [...]}


def wrap_decode_next_audio(original):
    @functools.wraps(original)
    def wrapper(self, state, **kwargs):
        # Same computation dots.tts uses in _should_stop_after_current_audio. Uses no randomness.
        eos = float("nan")
        if state.llm_hiddens is not None:
            eos = self.core.eos_proj(state.llm_hiddens).softmax(dim=-1)[0, -1, 1].item()
        patches.append({"eos": eos, "z": [], "v": []})
        return original(self, state, **kwargs)

    return wrapper


def wrap_velocity(original):
    @functools.wraps(original)
    def wrapper(self, t, z, **kwargs):
        vt = original(self, t, z, **kwargs)
        if patches:
            patches[-1]["z"].append(z.detach().float().cpu())
            patches[-1]["v"].append(vt.detach().float().cpu())
        return vt

    return wrapper


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="dots-studio/dots.tts-base")
    p.add_argument("--text", default="Hello there.")
    p.add_argument("--num-steps", type=int, default=10)
    p.add_argument("--precision", default="bfloat16")
    args = p.parse_args()

    DotsTtsModel._decode_next_audio = wrap_decode_next_audio(DotsTtsModel._decode_next_audio)
    EagerDiTRunner._flow_matching_velocity = wrap_velocity(EagerDiTRunner._flow_matching_velocity)

    # optimize=False matters here too: compiled code would bypass the Python hooks.
    runtime = DotsTtsRuntime.from_pretrained(args.model, precision=args.precision, optimize=False)
    seed_everything(42)
    runtime.generate(text=args.text, num_steps=args.num_steps)

    for i, patch in enumerate(patches):
        v = patch["v"]
        norms = [f"{x.norm().item():.1f}" for x in v]
        cos = [f"{F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item():.3f}" for a, b in zip(v, v[1:])]
        print(f"patch {i:3d}  eos={patch['eos']:.3f}")
        print(f"    |v|: {' '.join(norms)}")
        print(f"    cos: {' '.join(cos)}")

    out = Path("runs") / f"inspect_nfe{args.num_steps}.pt"
    out.parent.mkdir(exist_ok=True)
    torch.save({"args": vars(args), "patches": patches}, out)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
