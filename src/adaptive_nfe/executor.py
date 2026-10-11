"""DiT execution layer: runs the flow-matching ODE for one audio patch at a given NFE.

Baseline version: one request, eager mode (optimize=False), Euler only. It must produce exactly
what stock DiTSolver.decode_next produces, so it reuses dots.tts's own input preparation and DiT
call (EagerDiTRunner) and owns only the noise draw and the ODE loop. See
docs/design/executor-baseline.md.

    executor = Executor(runtime.model)
    patch = executor.decode_patch(sequence=..., cfg_sequence=..., fm_seq_len=..., g_cond=...,
                                  nfe=10, guidance_scale=1.2)
"""

from __future__ import annotations

import torch
from dots_tts.modules.backbone.dit_inference import DiTInferenceContext, EagerDiTRunner

from adaptive_nfe.ode import euler_integrate


class Executor:
    def __init__(self, model) -> None:
        # Same context dots.tts builds in DotsTtsModel._get_dit_solver (fusing the DiT is idempotent).
        self.runner = EagerDiTRunner(context=DiTInferenceContext.from_core(model.core))

    @torch.no_grad()
    def decode_patch(
        self,
        *,
        sequence: torch.Tensor,
        cfg_sequence: torch.Tensor,
        fm_seq_len: int,
        g_cond: torch.Tensor,
        nfe: int,
        guidance_scale: float,
        ode_method: str = "euler",
    ) -> torch.Tensor:
        """Run the DiT for one patch at `nfe` steps, return the patch latent [1, 4, 128].

        `g_cond` must already be resolved (null_g_cond when there is no speaker conditioning,
        cast to the sequence dtype), as DiTSolver.decode_next does before calling its runner.
        """
        if ode_method != "euler":
            raise NotImplementedError(f"Executor only supports ode_method='euler', got {ode_method!r}.")
        if nfe <= 0:
            raise ValueError(f"nfe must be positive, got {nfe}.")
        if fm_seq_len <= 0:
            raise RuntimeError("Cannot decode audio before any conditioning state has been prefetched.")

        runner = self.runner
        input_sequence, cfg_input_sequence, attn_mask, pos_ids = runner._prepare_inputs(
            sequence=sequence, cfg_sequence=cfg_sequence, fm_seq_len=fm_seq_len
        )
        guidance = sequence.new_tensor(float(guidance_scale))

        # Same draw, at the same point, as EagerDiTRunner._decode_flow_matching: the only random
        # call per patch, after input preparation (which uses no randomness).
        noise = torch.randn(
            (sequence.size(0), runner.latent_patch_size, runner.latent_dim),
            dtype=sequence.dtype,
            device=sequence.device,
        )

        def velocity(t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
            return runner._flow_matching_velocity(
                t,
                z,
                input_sequence=input_sequence,
                cfg_sequence=cfg_input_sequence,
                attn_mask=attn_mask,
                pos_ids=pos_ids,
                g_cond=g_cond,
                guidance_scale=guidance,
            )

        return euler_integrate(velocity, noise, nfe)
