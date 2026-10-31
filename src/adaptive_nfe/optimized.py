"""DiT execution for dots.tts's optimized path (optimize=True): KV cache + compiled step modules.

Same job as executor.py's decode_patch, for the fast path. The fast path does slightly different math
from eager, so this mirrors dots.tts's DiTSolver._decode_optimized (flow matching) rather than ode.py:
  - time grid t_i = i / nfe and a fixed dt = 1 / nfe (no bf16-rounded torchdiffeq grid)
  - the time only reaches the DiT through the precomputed adaLN table all_mods_by_ode[i]
    (the compiled step modules ignore t)
  - each step reads the earlier tokens' K/V from a cache that holds one copy per ODE step, and
    writes the previous patch's K/V back into it

dots.tts's solver still owns the compiled step modules (torch.compile + CUDA graphs), the cache
allocation and prefill, and the adaLN table. This owns the noise draw, the step loop, the Euler
update and the K/V write-back. Must match stock optimize=True bit for bit.

Called through the dit_executor hook as executor.decode_patch_optimized(...), with the model's
DiTSolver and this request's DiTSolverState (state.fm_dit_state).
"""

from __future__ import annotations

import torch


def decode_patch_optimized(
    *,
    solver,
    state,
    sequence: torch.Tensor,
    cfg_sequence: torch.Tensor,
    fm_seq_len: int,
    g_cond: torch.Tensor,
    nfe: int,
    guidance_scale: float,
) -> torch.Tensor:
    """Run the DiT for one patch at `nfe` steps on the optimized path, return the patch latent [1, 4, 128].

    `g_cond` must already be resolved (null_g_cond when there is no speaker conditioning), as for
    Executor.decode_patch.
    """
    if solver.meanflow or not solver.optimize:
        raise NotImplementedError("decode_patch_optimized only supports the flow-matching solver with optimize=True.")
    nfe, fm_seq_len = int(nfe), int(fm_seq_len)
    solver._validate_decode_args(fm_seq_len=fm_seq_len, nfe=nfe, cfg_sequence=cfg_sequence,
                                 guidance_scale=guidance_scale)

    ctx = solver.context
    hidden_patch_size, unit_len = int(ctx.hidden_patch_size), int(ctx.unit_len)
    prefix_len = fm_seq_len - hidden_patch_size  # tokens of earlier, finished patches
    if prefix_len < 0 or prefix_len % unit_len != 0:
        raise RuntimeError(f"Unexpected FM sequence layout: fm_seq_len={fm_seq_len} prefix_len={prefix_len} "
                           f"unit_len={unit_len}.")

    batch_size = int(sequence.size(0))
    current_hidden = sequence[:, prefix_len : prefix_len + hidden_patch_size]
    cfg_current_hidden = cfg_sequence[:, prefix_len : prefix_len + hidden_patch_size]
    # Same draw, at the same point, as stock: the only random call per patch.
    z = torch.randn((batch_size, int(ctx.latent_patch_size), ctx.noise_latent_dim),
                    dtype=sequence.dtype, device=sequence.device)
    # adaLN table + time grid, cached in `state` while nfe stays the same.
    all_mods_by_ode, schedule = solver._prepare_schedule(state, g_cond=g_cond, nfe=nfe)
    guidance = sequence.new_tensor(float(guidance_scale))
    flow_dt = sequence.new_tensor(1.0 / nfe)

    if prefix_len == 0:
        # First patch: no history, so no KV cache.
        runner = solver._get_cached_runner(capacity_patches=solver._resolve_bucket_patches(fm_seq_len),
                                           device=sequence.device, dtype=sequence.dtype)
        for i in range(nfe):
            vt = runner.current_step(z, current_hidden=current_hidden, all_mods=all_mods_by_ode[i],
                                     t=schedule.times[i], cfg_current_hidden=cfg_current_hidden,
                                     guidance_scale=guidance)
            z = z + flow_dt * vt
        return z

    # Later patches: the cache holds K/V for everything before the previous patch ("persistent");
    # the previous patch ("prev_unit") is fed in each step and its K/V written back below.
    persistent_len = prefix_len - unit_len
    prev_unit = sequence[:, persistent_len:prefix_len]
    cfg_prev_unit = cfg_sequence[:, persistent_len:prefix_len]
    runner, kv_cache = solver._ensure_cache(state, sequence=sequence[:, :fm_seq_len], nfe=nfe)
    if kv_cache.valid_tokens != persistent_len:
        # New or reallocated cache (first cached patch, a bigger length bucket, or a different nfe):
        # recompute K/V for the whole persistent prefix at every ODE step.
        runner.prefill(kv_cache=kv_cache, prefix_sequence=sequence[:, :persistent_len],
                       cfg_prefix_sequence=cfg_sequence[:, :persistent_len], all_mods_by_ode=all_mods_by_ode)
    block_mask, sdpa_mask = runner.masks_for(valid_persistent_tokens=persistent_len)
    rotary_cos, rotary_sin = runner.rotary_for(start_pos=persistent_len)
    dst = slice(persistent_len, prefix_len)

    for i in range(nfe):
        vt, new_k, new_v = runner.step(
            z, prev_unit=prev_unit, current_hidden=current_hidden, all_mods=all_mods_by_ode[i],
            cache_k=kv_cache.cache_k[i], cache_v=kv_cache.cache_v[i], block_mask=block_mask, sdpa_mask=sdpa_mask,
            rotary_cos=rotary_cos, rotary_sin=rotary_sin, t=schedule.times[i], cfg_prev_unit=cfg_prev_unit,
            cfg_current_hidden=cfg_current_hidden, guidance_scale=guidance,
        )
        z = z + flow_dt * vt
        kv_cache.cache_k[i, :, :, :, dst, :].copy_(new_k)
        kv_cache.cache_v[i, :, :, :, dst, :].copy_(new_v)
    kv_cache.valid_tokens = prefix_len
    return z
