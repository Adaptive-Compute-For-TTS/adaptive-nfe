"""Generate one utterance at one fixed NFE: the call a selector (or a script) hands its NFE to.

The NFE is accepted once and frozen for every patch (FixedScheduler). By default every patch's DiT
loop runs through our Euler Executor. Afterwards the per-patch log is checked: every patch got that
NFE, and the executor decoded every patch the scheduler saw. Any mismatch raises.

    rt = DotsTtsRuntime.from_pretrained("dots-studio/dots.tts-soar", precision="bfloat16", optimize=False)
    torch.manual_seed(42)  # seeding is the caller's job, as with rt.generate
    result = generate_fixed_nfe(rt, nfe=10, text="Hello there.", guidance_scale=1.2)
    result.output["audio"], result.num_patches, result.nfe_per_patch

The model's hooks are restored afterwards, so a plain rt.generate() call is stock dots.tts again.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from adaptive_nfe.scheduler import FixedScheduler

log = logging.getLogger(__name__)


@dataclass
class FixedNFEResult:
    output: dict  # what runtime.generate returned (audio, sample_rate, time_used, rtf, ...)
    nfe: int
    nfe_per_patch: list[int]  # from the scheduler, one entry per patch
    executor_nfe_per_patch: list[int] | None  # from the executor; None when it wasn't used

    @property
    def num_patches(self) -> int:
        return len(self.nfe_per_patch)


class _RecordingExecutor:
    """Passes decode_patch through to the real executor and records the nfe of every call."""

    def __init__(self, executor) -> None:
        self.executor = executor
        self.calls: list[int] = []

    def decode_patch(self, **kwargs):
        self.calls.append(kwargs["nfe"])
        return self.executor.decode_patch(**kwargs)


def generate_fixed_nfe(runtime, nfe: int, *, use_executor: bool = True, executor=None, **generate_kwargs) -> FixedNFEResult:
    """Run runtime.generate(**generate_kwargs) with every patch frozen at `nfe`, then check the patch log.

    `executor`: reuse an existing adaptive_nfe.Executor across calls; one is built when None.
    `use_executor=False` keeps dots.tts's own DiT solver (the scheduler still freezes the NFE), which is
    what the optimize=True runtime needs since the executor is eager only.
    """
    nfe = int(nfe)
    if nfe <= 0:
        raise ValueError(f"nfe must be positive, got {nfe}.")
    if generate_kwargs.setdefault("num_steps", nfe) != nfe:
        raise ValueError(f"num_steps={generate_kwargs['num_steps']} conflicts with nfe={nfe}; pass only nfe.")
    model = runtime.model
    if use_executor and getattr(model, "_optimize_enabled", False):
        raise NotImplementedError("The executor only supports optimize=False; pass use_executor=False.")

    scheduler = FixedScheduler(nfe)
    recorder = None
    if use_executor:
        if executor is None:
            from adaptive_nfe.executor import Executor  # imports dots.tts, so only when needed

            executor = Executor(model)
        recorder = _RecordingExecutor(executor)

    saved = (model.nfe_scheduler, model.dit_executor)
    model.nfe_scheduler, model.dit_executor = scheduler, recorder
    try:
        output = runtime.generate(**generate_kwargs)
    finally:
        model.nfe_scheduler, model.dit_executor = saved

    nfe_per_patch = [h["nfe"] for h in scheduler.history]
    executor_nfe_per_patch = recorder.calls if recorder else None
    log.info("fixed nfe=%d: %d patches, executor %s", nfe, len(nfe_per_patch),
             "off" if recorder is None else f"decoded {len(recorder.calls)}")

    if not nfe_per_patch:
        raise RuntimeError("No patches were decoded; is the dots.tts scheduler hook installed?")
    if set(nfe_per_patch) != {nfe}:
        raise RuntimeError(f"Expected nfe={nfe} on every patch, scheduler log has {nfe_per_patch}.")
    if recorder is not None and recorder.calls != nfe_per_patch:
        raise RuntimeError(f"Executor decoded {len(recorder.calls)} patches {recorder.calls}, "
                           f"scheduler saw {len(nfe_per_patch)} {nfe_per_patch}.")
    return FixedNFEResult(output, nfe, nfe_per_patch, executor_nfe_per_patch)
