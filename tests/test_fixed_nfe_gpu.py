"""generate_fixed_nfe on the real model: frozen NFE, executor on every patch, audio identical to stock.

Needs CUDA, dots.tts with the scheduler/executor hooks, and the weights (downloaded on first run).
Skipped everywhere else, so plain `pytest tests/` stays CPU-only. Model defaults to dots.tts-base;
set ADAPTIVE_NFE_MODEL=dots-studio/dots.tts-soar to test soar.

Run: pytest tests/test_fixed_nfe_gpu.py -v
"""

from __future__ import annotations

import os

import pytest
import torch

pytest.importorskip("dots_tts")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from dots_tts.runtime import DotsTtsRuntime  # noqa: E402

from adaptive_nfe import generate_fixed_nfe  # noqa: E402
from adaptive_nfe.executor import Executor  # noqa: E402

MODEL = os.environ.get("ADAPTIVE_NFE_MODEL", "dots-studio/dots.tts-base")
TEXT = "The quick brown fox jumps over the lazy dog."
SEED = 42


def seed_all(s: int) -> None:
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


@pytest.fixture(scope="module")
def rt():
    return DotsTtsRuntime.from_pretrained(MODEL, precision="bfloat16", optimize=False)


@pytest.fixture(scope="module")
def executor(rt):
    return Executor(rt.model)


def gen_kwargs(rt) -> dict:
    ode, _, cfg = rt.resolve_sampling_options(ode_method=None, num_steps=10, guidance_scale=1.2)
    return dict(text=TEXT, ode_method=ode, guidance_scale=cfg)


@pytest.mark.parametrize("nfe", [4, 10, 32])
def test_matches_stock_with_every_patch_frozen(rt, executor, nfe):
    seed_all(SEED)
    result = generate_fixed_nfe(rt, nfe, executor=executor, **gen_kwargs(rt))
    assert result.num_patches > 0
    assert result.nfe_per_patch == [nfe] * result.num_patches
    assert result.executor_nfe_per_patch == result.nfe_per_patch

    # Hooks are gone, so this is plain dots.tts.
    assert rt.model.nfe_scheduler is None and rt.model.dit_executor is None
    seed_all(SEED)
    stock = rt.generate(num_steps=nfe, **gen_kwargs(rt))
    assert torch.equal(result.output["audio"], stock["audio"])


def test_without_executor_matches_stock(rt):
    seed_all(SEED)
    result = generate_fixed_nfe(rt, 10, use_executor=False, **gen_kwargs(rt))
    assert result.executor_nfe_per_patch is None
    assert result.nfe_per_patch == [10] * result.num_patches
    seed_all(SEED)
    stock = rt.generate(num_steps=10, **gen_kwargs(rt))
    assert torch.equal(result.output["audio"], stock["audio"])


def test_back_to_back_calls_are_reproducible(rt, executor):
    # A selector will call this many times with one executor; nothing may leak between calls.
    seed_all(SEED)
    a = generate_fixed_nfe(rt, 4, executor=executor, **gen_kwargs(rt))
    generate_fixed_nfe(rt, 32, executor=executor, **gen_kwargs(rt))
    seed_all(SEED)
    b = generate_fixed_nfe(rt, 4, executor=executor, **gen_kwargs(rt))
    assert a.nfe_per_patch == b.nfe_per_patch
    assert torch.equal(a.output["audio"], b.output["audio"])
