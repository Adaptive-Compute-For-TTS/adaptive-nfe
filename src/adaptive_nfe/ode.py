"""Fixed-step Euler solver that reproduces torchdiffeq.odeint(method="euler") bit for bit.

Stock dots.tts (eager) integrates each patch with

    odeint(func, noise, t=tensor([0., 1.], dtype=bf16), method="euler", options={"step_size": 1 / nfe})

Owning the loop is what lets the executor change it later (per-row timesteps for batching),
but for the baseline it has to match torchdiffeq exactly. The details that matter
(torchdiffeq 0.2.5, _impl/solvers.py FixedGridODESolver):
  - the time grid is built in the dtype of t (bf16), so grid points are rounded
  - dt is recomputed per step as t1 - t0, so it isn't exactly 1/nfe
  - the last grid point is forced to t[-1] (exactly 1.0)
  - the update is dy = dt * f(t0, y0); y1 = y0 + dy, kept in whatever dtype that promotes to
  - f receives t cast to the state's dtype
  - the result is cast back to y0's dtype when stored in the solution tensor
"""

from __future__ import annotations

from typing import Callable

import torch

Velocity = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def euler_time_grid(nfe: int, *, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
    """Grid from 0 to 1 that torchdiffeq builds for step_size = 1/nfe."""
    if nfe <= 0:
        raise ValueError(f"nfe must be positive, got {nfe}.")
    t = torch.tensor([0.0, 1.0], dtype=dtype, device=device)
    step_size = 1.0 / int(nfe)
    niters = torch.ceil((t[-1] - t[0]) / step_size + 1).item()
    grid = torch.arange(0, niters, dtype=t.dtype, device=t.device) * step_size + t[0]
    grid[-1] = t[-1]
    return grid


def euler_integrate(velocity: Velocity, z0: torch.Tensor, nfe: int) -> torch.Tensor:
    """Integrate dz/dt = velocity(t, z) from t=0 to t=1. One velocity call per grid step."""
    grid = euler_time_grid(nfe, dtype=z0.dtype, device=z0.device)
    z = z0
    for t0, t1 in zip(grid[:-1], grid[1:]):
        dt = t1 - t0
        z = z + dt * velocity(t0.to(z.abs().dtype), z)
    return z.to(z0.dtype)
