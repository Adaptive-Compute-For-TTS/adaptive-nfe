"""Our Euler loop must match torchdiffeq.odeint(method="euler") exactly, as stock dots.tts calls it.

No model weights needed. Run: pytest tests/
"""

from __future__ import annotations

import pytest
import torch
from torchdiffeq import odeint

from adaptive_nfe.ode import euler_integrate

NFES = list(range(1, 33)) + [40, 64]


def make_velocity(out_dtype: torch.dtype, calls: list):
    """Deterministic, non-linear stand-in for the DiT that depends on both t and z."""
    weight = torch.linspace(-1.5, 1.5, 128)

    def velocity(t, z):
        calls.append(t.clone())
        v = torch.sin(3.0 * z.float() + t.float()) * weight + (1.0 - t.float()) * z.float()
        return v.to(out_dtype)

    return velocity


def stock(velocity, noise, nfe):
    """Exactly how EagerDiTRunner._decode_flow_matching calls odeint."""
    times = torch.tensor([0.0, 1.0], dtype=noise.dtype, device=noise.device)
    trajectory = odeint(
        func=velocity, y0=noise, t=times, atol=1e-5, rtol=1e-5, method="euler",
        options={"step_size": 1.0 / int(nfe)},
    )
    return trajectory[-1]


# out_dtype != state dtype covers the case where the velocity comes back in a wider dtype
# (e.g. under autocast), which changes the dtype of the intermediate Euler states.
@pytest.mark.parametrize("state_dtype,out_dtype", [
    (torch.bfloat16, torch.bfloat16),
    (torch.bfloat16, torch.float32),
    (torch.float32, torch.float32),
])
@pytest.mark.parametrize("nfe", NFES)
def test_matches_torchdiffeq(nfe, state_dtype, out_dtype):
    torch.manual_seed(0)
    noise = torch.randn(1, 4, 128).to(state_dtype)

    stock_calls, our_calls = [], []
    expected = stock(make_velocity(out_dtype, stock_calls), noise, nfe)
    actual = euler_integrate(make_velocity(out_dtype, our_calls), noise, nfe)

    assert actual.dtype == expected.dtype
    assert torch.equal(actual, expected)
    assert len(our_calls) == len(stock_calls)
    assert all(torch.equal(a, b) and a.dtype == b.dtype for a, b in zip(our_calls, stock_calls))
