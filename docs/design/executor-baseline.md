# Design: baseline DiT executor (fixed NFE)

**Author:** Yulchan · **Status:** Draft, for discussion · **Date:** 2026-10-11
**Task:** build the execution layer, run at fixed NFE for the baseline. Tony wires it into dots.tts and verifies the output is unchanged.
**Builds on:** Tony's executor notes (patch loop, CFG, equivalence rules), which aren't repeated here.
**dots.tts:** commit `3cb6f94`; line numbers refer to the unpatched source.
**Next phase (batching, variable NFE, `optimize=True`):** see [batching-exploration.md](batching-exploration.md).

## Answers to Tony's "Stuff to agree on"

| Tony's item | Proposed answer | Details |
|---|---|---|
| Interface: replace `DiTSolver.decode_next`, or something higher up? | **Replace `decode_next`** for the baseline, using your proposed signature unchanged. Batching later needs a loop above `generate()`, but that's additive | [D1](#d1-interface-and-hook) |
| Who owns noise generation and seeding | **The executor.** For the baseline, it draws from the global RNG exactly as stock does | [D3](#d3-noise) |
| Batching plans | **Across requests only.** Not part of the baseline | [Exploration doc](batching-exploration.md#e1-batching-policy) |
| Per-patch logging format | **Draft schema** below, for sign-off with Nishant and Afreed | [D4](#d4-per-patch-logging) |
| When we start on `optimize=True` | **After the baseline and eager batching.** Its KV cache is the main blocker | [Exploration doc](batching-exploration.md#e3-kv-cache-under-variable-nfe) |

## Scope

**In:** an `Executor` that runs the flow-matching ODE for one patch of one request at a given NFE. Eager mode (`optimize=False`), Euler only, bit-identical to stock.
**Out:** batching, NFE changing within a run, the optimized path. These are covered in the exploration doc.

---

## D1: Interface and hook

**Interface: Tony's proposal as-is.**

```python
class Executor:
    def __init__(self, model: DotsTtsModel): ...

    def decode_patch(self, *, sequence, cfg_sequence, fm_seq_len, g_cond,
                     nfe: int, guidance_scale: float, ode_method: str = "euler") -> torch.Tensor:
        """Run the DiT for one patch at `nfe` steps, return the patch latent [1, 4, 128]."""
```

- **`g_cond` must already be resolved.** `DiTSolver.decode_next` takes both `g_cond` and `null_g_cond`, and resolves them itself: it uses `null_g_cond` when `g_cond is None`, and otherwise casts `g_cond` to the sequence dtype/device (`dit_inference.py:1478`). The hook does that one line before calling the executor.
- **Only `ode_method="euler"` is supported.** Anything else raises `NotImplementedError`. Stock's default is Euler.
- **Later, a `decode_patches(jobs)` method is added for batching.** `decode_patch` stays as it is, so Tony's integration doesn't change.

**Hook in `_decode_next_audio`**, mirroring how `nfe_scheduler` is attached:

```python
executor = getattr(self, "dit_executor", None)
if executor is not None and solver_mode == "flow_matching" and not self._optimize_enabled:
    g = null_g_cond if g_cond is None else g_cond.to(device=null_g_cond.device, dtype=sequence.dtype)
    audio_patch = executor.decode_patch(
        sequence=sequence, cfg_sequence=cfg_sequence, fm_seq_len=state.fm_seq_len,
        g_cond=g, nfe=num_steps, guidance_scale=guidance_scale, ode_method=ode_method,
    )
else:
    audio_patch = self._get_dit_solver(solver_mode=solver_mode).decode_next(...)  # unchanged
```

`model.dit_executor = None` (the default) keeps stock behaviour.

**Where the code lives:** `src/adaptive_nfe/executor.py`, next to `scheduler.py`.

## D2: What `decode_patch` does

The executor reuses dots.tts's own input preparation and DiT call, and owns the noise and the ODE loop. Those are the two parts batching will change later.

| Step | Source | Owned by executor? |
|---|---|---|
| Pad sequences, build attention mask and `pos_ids` | `EagerDiTRunner._prepare_inputs` (`:1008`) | No, reused |
| Draw noise | `torch.randn([1, 4, 128])`, same dtype/device, before the loop | **Yes** ([D3](#d3-noise)) |
| Build time grid, Euler loop | Replaces `torchdiffeq.odeint` | **Yes** (below) |
| One DiT call per step (cond + uncond batched, CFG combine) | `EagerDiTRunner._flow_matching_velocity` (`:1041`) | No, reused |

Reusing `_flow_matching_velocity` has a side benefit: `check_equivalence.py` counts DiT calls by wrapping that method, so the call counting keeps working unchanged.

**The Euler loop has to reproduce `torchdiffeq`'s fixed-grid Euler exactly** (checked against torchdiffeq 0.2.5, `_impl/solvers.py`):

```python
# t = torch.tensor([0.0, 1.0], dtype=sequence.dtype)   # bf16, as stock
step = 1.0 / nfe                                       # Python float
niters = torch.ceil((t[-1] - t[0]) / step + 1).item()
grid = torch.arange(0, niters, dtype=t.dtype, device=t.device) * step + t[0]
grid[-1] = t[-1]                                       # last point forced to exactly 1.0
z = noise
for t0, t1 in zip(grid[:-1], grid[1:]):
    dt = t1 - t0                                       # recomputed per step, in bf16
    z = z + dt * velocity(t0, z)                       # dy = dt * f0;  y1 = y0 + dy
return z
```

**Why this matters:** the grid is built in bf16, so the timesteps are rounded and `dt` isn't exactly `1/nfe` at every step. A "clean" loop (`t = k/nfe`, `dt = 1/nfe`) would not be bit-identical to stock.

**Fallback:** if the hand-written loop can't be made bit-exact, call `odeint` exactly as stock does for the baseline, and move the custom loop to the batching phase.

## D3: Noise

**The executor draws the noise itself, but from the global RNG.** It makes exactly one `torch.randn(...)` call per patch, with the same shape, dtype and device as stock, before the ODE loop and after input preparation (which uses no randomness). Combined with `seed_everything(42)`, this gives the same noise stream as stock. Per-request generators are deferred to batching ([exploration doc](batching-exploration.md#e5-noise-in-batches)).

## D4: Per-patch logging

**Proposed format:** JSON Lines, one record per patch. Draft for sign-off with Nishant and Afreed:

```json
{"request_id": "r7", "patch": 12, "nfe": 10, "fm_seq_len": 840, "dit_ms": 9.3}
```

- `patch` and `nfe` overlap with `Scheduler.history`.
- `dit_ms` is executor-owned and measured with CUDA events (wall clock on CPU).
- Batching fields such as `batch_size` are added later.

## Testing

Implemented in `src/adaptive_nfe/executor.py` (executor) and `src/adaptive_nfe/ode.py` (Euler loop).

1. **`pytest tests/`** needs no weights and runs on CPU in seconds.
   - `test_ode.py`: our Euler loop vs `torchdiffeq.odeint`, bit for bit, NFE 1–32, 40, 64, in bf16 and fp32. The test catches a "clean" loop (`t = k/nfe`, `dt = 1/nfe`), which differs from stock at NFE 3, 5, 10, … (every non-power of two).
   - `test_executor.py`: `Executor.decode_patch` vs stock `DiTSolver.decode_next` on a tiny random DiT built from dots.tts's own classes. Covers fp32 and bf16 under autocast, with and without speaker conditioning, first and later patches, and that both consume the RNG identically.
2. **`python scripts/check_equivalence.py --nfe 4 10 32 --executor`** runs the full model. B's DiT calls go through the executor by wrapping `DiTSolver.decode_next`, the same logic as the hook in [D1](#d1-interface-and-hook), so dots.tts isn't edited. Needs ALL PASS, plus `executor_used_ok` (every patch decoded by the executor).

## Open questions

1. OK with the hook shape in [D1](#d1-interface-and-hook) (a `dit_executor` attribute next to `nfe_scheduler`, with `g_cond` resolved in the hook)?
2. Which `torchdiffeq` version is in your venv? It isn't pinned in `constraints.txt`. The loop in [D2](#d2-what-decode_patch-does) was checked against 0.2.5.
3. Logging schema sign-off ([D4](#d4-per-patch-logging)).
