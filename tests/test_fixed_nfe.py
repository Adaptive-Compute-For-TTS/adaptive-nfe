"""generate_fixed_nfe: NFE frozen on every patch, executor used on every patch, hooks restored.

Uses a fake runtime that drives the two dots.tts hooks the way DotsTtsModel does, so it needs
neither dots.tts nor weights. The full-model check is `python scripts/generate.py --verify-stock`.

Run: pytest tests/
"""

from __future__ import annotations

import types

import pytest

pytest.importorskip("torch")

from adaptive_nfe.fixed_nfe import generate_fixed_nfe  # noqa: E402


class FakeExecutor:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def decode_patch(self, *, nfe, **kwargs):
        self.calls.append(nfe)
        return "latent"


class FakeRuntime:
    """Calls model.nfe_scheduler / model.dit_executor once per patch, like the dots.tts hooks."""

    def __init__(self, num_patches: int = 5, skip_executor_on: tuple[int, ...] = (), fail: bool = False) -> None:
        self.model = types.SimpleNamespace(nfe_scheduler=None, dit_executor=None)
        self.num_patches = num_patches
        self.skip_executor_on = skip_executor_on
        self.fail = fail
        self.generate_kwargs = None

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        if self.fail:
            raise RuntimeError("boom")
        for i in range(self.num_patches):
            nfe = kwargs["num_steps"]
            if self.model.nfe_scheduler is not None:
                nfe = self.model.nfe_scheduler.next_nfe(patch_index=i, default_nfe=nfe, llm_hidden=None)
            if self.model.dit_executor is not None and i not in self.skip_executor_on:
                self.model.dit_executor.decode_patch(nfe=nfe)
        return {"audio": "audio", "sample_rate": 48000}


def test_freezes_nfe_and_runs_every_patch_through_executor():
    rt, ex = FakeRuntime(num_patches=7), FakeExecutor()
    r = generate_fixed_nfe(rt, 10, executor=ex, text="hi")
    assert r.num_patches == 7
    assert r.nfe_per_patch == [10] * 7
    assert r.executor_nfe_per_patch == ex.calls == [10] * 7
    assert r.output == {"audio": "audio", "sample_rate": 48000}
    assert rt.generate_kwargs == {"text": "hi", "num_steps": 10}


def test_without_executor():
    r = generate_fixed_nfe(FakeRuntime(num_patches=3), 4, use_executor=False)
    assert r.nfe_per_patch == [4, 4, 4]
    assert r.executor_nfe_per_patch is None


def test_restores_previous_hooks_even_on_error():
    rt = FakeRuntime(fail=True)
    rt.model.nfe_scheduler, rt.model.dit_executor = "old_sched", "old_exec"
    with pytest.raises(RuntimeError, match="boom"):
        generate_fixed_nfe(rt, 10, executor=FakeExecutor())
    assert (rt.model.nfe_scheduler, rt.model.dit_executor) == ("old_sched", "old_exec")


def test_raises_when_executor_misses_a_patch():
    with pytest.raises(RuntimeError, match="Executor decoded 4 patches"):
        generate_fixed_nfe(FakeRuntime(num_patches=5, skip_executor_on=(2,)), 10, executor=FakeExecutor())


def test_raises_when_no_patches():
    with pytest.raises(RuntimeError, match="No patches"):
        generate_fixed_nfe(FakeRuntime(num_patches=0), 10, executor=FakeExecutor())


@pytest.mark.parametrize("nfe", [0, -3])
def test_rejects_non_positive_nfe(nfe):
    with pytest.raises(ValueError):
        generate_fixed_nfe(FakeRuntime(), nfe, executor=FakeExecutor())


def test_rejects_conflicting_num_steps():
    with pytest.raises(ValueError, match="conflicts"):
        generate_fixed_nfe(FakeRuntime(), 10, executor=FakeExecutor(), num_steps=32)


def test_executor_refused_on_optimized_runtime():
    rt = FakeRuntime()
    rt.model._optimize_enabled = True
    with pytest.raises(NotImplementedError):
        generate_fixed_nfe(rt, 10, executor=FakeExecutor())
    assert generate_fixed_nfe(rt, 10, use_executor=False).num_patches == 5
