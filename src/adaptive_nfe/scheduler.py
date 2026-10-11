"""NFE schedulers: decide how many denoising steps each audio patch gets.

dots.tts (branch scheduler-hook) calls `scheduler.next_nfe(...)` once per patch, right
before running the flow-matching solver, when one is attached:

    runtime.model.nfe_scheduler = FixedScheduler(10)

Set it back to None to get the original dots.tts behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PatchInfo:
    """What a scheduler knows when choosing the NFE for one patch."""

    patch_index: int  # 0-based position of the patch in the utterance
    default_nfe: int  # num_steps that was passed to generate()
    llm_hidden: torch.Tensor | None  # LLM hidden state conditioning this patch (future difficulty signal)


class Scheduler:
    """Base class. Subclasses implement choose_nfe; next_nfe also logs every choice."""

    def __init__(self) -> None:
        self.history: list[dict] = []  # one {"patch": i, "nfe": n} per decoded patch

    def next_nfe(self, *, patch_index: int, default_nfe: int, llm_hidden: torch.Tensor | None) -> int:
        nfe = int(self.choose_nfe(PatchInfo(patch_index, default_nfe, llm_hidden)))
        if nfe <= 0:
            raise ValueError(f"{type(self).__name__} returned nfe={nfe}; must be positive.")
        self.history.append({"patch": patch_index, "nfe": nfe})
        return nfe

    def choose_nfe(self, info: PatchInfo) -> int:
        raise NotImplementedError

    def reset(self) -> None:
        """Clear the log. Call before each generate()."""
        self.history.clear()


class FixedScheduler(Scheduler):
    """Baseline: every patch gets the same NFE, ignoring what generate() was given."""

    def __init__(self, nfe: int) -> None:
        super().__init__()
        self.nfe = int(nfe)

    def choose_nfe(self, info: PatchInfo) -> int:
        return self.nfe
