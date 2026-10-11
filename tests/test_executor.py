"""Executor.decode_patch must match stock DiTSolver.decode_next (eager) exactly.

Uses a tiny randomly initialised DiT built with dots.tts's own classes, so no weights are
downloaded and it runs on CPU in seconds. The full-model check is
`python scripts/check_equivalence.py --executor`.

Run: pytest tests/
"""

from __future__ import annotations

import types

import pytest
import torch

pytest.importorskip("dots_tts")

from dots_tts.models.dots_tts.config import _DiTConfig  # noqa: E402
from dots_tts.modules.backbone.dit import DiT  # noqa: E402
from dots_tts.modules.backbone.dit_inference import DiTSolver, DiTSolverState  # noqa: E402

from adaptive_nfe.executor import Executor  # noqa: E402

HIDDEN = 64
LATENT_DIM = 16
LATENT_PATCH = 4  # config.patch_size
CAPACITY = 64  # FM buffer length


def tiny_model(dtype: torch.dtype):
    """Stand-in for DotsTtsModel with just the attributes the DiT solver reads from model.core."""
    torch.manual_seed(1234)
    dit_config = _DiTConfig(num_layers=2, num_heads=4, hidden_size=HIDDEN, ffn_hidden_size=128)
    core = torch.nn.Module()
    core.velocity_field_predictor = DiT(in_dim=HIDDEN, out_dim=LATENT_DIM, transformer_config=dit_config)
    core.coordinate_proj = torch.nn.Linear(LATENT_DIM, HIDDEN)
    core.hidden_patch_size = 1
    core.latent_patch_size = LATENT_PATCH
    core.latent_dim = LATENT_DIM
    core.fm_hidden_size = HIDDEN
    core.config = types.SimpleNamespace(fm_sigma=0.0)
    # DiT.initialize_weights zero-inits the adaLN/output layers, so an untrained DiT predicts zero
    # velocity and every solver gives the same answer. Random weights make the test sensitive.
    for param in core.parameters():
        param.data.normal_(0.0, 0.2)
    core.to(dtype).eval()
    return types.SimpleNamespace(core=core)


def make_inputs(dtype: torch.dtype, fm_seq_len: int, with_speaker: bool):
    gen = torch.Generator().manual_seed(fm_seq_len)
    sequence = torch.randn(1, CAPACITY, HIDDEN, generator=gen).to(dtype)
    cfg_sequence = torch.randn(1, CAPACITY, HIDDEN, generator=gen).to(dtype)
    null_g_cond = torch.zeros(1, HIDDEN, dtype=dtype)
    g_cond = torch.randn(1, HIDDEN, generator=gen).to(dtype) if with_speaker else None
    return sequence, cfg_sequence, null_g_cond, g_cond


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("nfe", [1, 3, 4, 10, 32])
@pytest.mark.parametrize("fm_seq_len", [1, 6, 26])  # first patch, second patch, later patch
@pytest.mark.parametrize("with_speaker", [True, False])
@torch.no_grad()
def test_executor_matches_stock(dtype, nfe, fm_seq_len, with_speaker):
    model = tiny_model(dtype)
    executor = Executor(model)  # fuses the DiT in place, as dots.tts does on first use
    solver = DiTSolver(executor.runner.context, optimize=False, bucket_resolver=lambda n: n)
    sequence, cfg_sequence, null_g_cond, g_cond = make_inputs(dtype, fm_seq_len, with_speaker)
    # dots.tts runs bf16 inside autocast (model.py _generate_latents_stream); CPU autocast stands in for CUDA's.
    amp = torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16)

    torch.manual_seed(42)
    with amp:
        expected = solver.decode_next(
            DiTSolverState(), sequence=sequence, cfg_sequence=cfg_sequence, fm_seq_len=fm_seq_len,
            null_g_cond=null_g_cond, g_cond=g_cond, nfe=nfe, ode_method="euler", guidance_scale=1.2,
        )
    rng_after_stock = torch.randn(8)

    # g_cond resolved the way DiTSolver.decode_next does it (the hook's job).
    resolved = null_g_cond if g_cond is None else g_cond.to(device=null_g_cond.device, dtype=sequence.dtype)
    torch.manual_seed(42)
    with amp:
        actual = executor.decode_patch(
            sequence=sequence, cfg_sequence=cfg_sequence, fm_seq_len=fm_seq_len,
            g_cond=resolved, nfe=nfe, guidance_scale=1.2,
        )
    rng_after_executor = torch.randn(8)

    assert actual.shape == (1, LATENT_PATCH, LATENT_DIM)
    assert actual.dtype == expected.dtype
    assert torch.equal(actual, expected)
    assert not torch.equal(actual, torch.zeros_like(actual))
    # Same number of random draws: the next patch's noise would be identical too.
    assert torch.equal(rng_after_executor, rng_after_stock)


def test_rejects_non_euler():
    executor = Executor(tiny_model(torch.float32))
    sequence, cfg_sequence, null_g_cond, _ = make_inputs(torch.float32, 6, False)
    with pytest.raises(NotImplementedError):
        executor.decode_patch(sequence=sequence, cfg_sequence=cfg_sequence, fm_seq_len=6,
                              g_cond=null_g_cond, nfe=4, guidance_scale=1.2, ode_method="rk4")
