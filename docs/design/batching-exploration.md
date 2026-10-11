# Exploration: batching and variable NFE

**Author:** Yulchan · **Status:** Exploration for the next phase, nothing measured yet · **Date:** 2026-10-11
**Context:** Tony's ask to explore *"batching, and how to handle the matrices that change when we change the flow matching step."* Builds on the [baseline executor](executor-baseline.md). dots.tts commit `3cb6f94`.

## Findings from the dots.tts code

**F1. The DiT accepts a different timestep per row.** `timesteps` is already a vector of length `2B` (`dit_inference.py:1071`), and AdaLN modulation is applied per row (`dit.py:9`). Rows at different ODE steps can share one forward pass with no model changes.

**F2. Eager and optimized paths differ.**
- **Eager:** every ODE step re-runs the full FM sequence, so cost grows with utterance length. Variable NFE works trivially.
- **Optimized:** each step runs only 2 units against a KV cache, compiled on CUDA. Variable NFE breaks the cache (F3).

**F3. The optimized KV cache has one copy per ODE step.** Its shape is `[nfe, layers, 2B, heads, capacity, head_dim]` (`:744`), because AdaLN depends on t, so history K/V differ per timestep. When NFE changes, `_ensure_cache` reallocates (`:1343`) and `prefill` recomputes the whole history at every new timestep (`:845`). Memory also scales with NFE. This is likely "the matrices that change". The K/V written for `prev_unit` don't depend on the noisy latent (`inference_utils.py:160`), so a cache slot can be filled without noise.

**F4. `generate()` is single-request** (`model.py:1390`), and patches within a request are sequential. Batching has to be across requests, through a new loop above `generate()`.

## Options

### E1: Batching policy

- **Pad to max NFE:** simple, but wastes compute and erases the NFE savings.
- **Bucket by NFE:** homogeneous batches, but smaller ones. Unnecessary given F1.
- **Mixed-NFE rows, each at its own t/dt** *(leaning)*: first a shrinking batch (rows leave when done), then continuous batching (rows join any time).
  - The Euler loop becomes per row, and the executor gains `decode_patches(jobs)`.
  - The LLM step may become the bottleneck.

### E2: Execution mode

Build in eager, where the bit-exact test exists and variable NFE is free. Measure final speed-ups against **stock optimized**, since that's what would be served.

### E3: KV cache under variable NFE

- **Status quo** (full re-prefill on NFE change): likely negates adaptive NFE.
- **One cache per NFE value:** memory × #NFEs, and the caches still go stale.
- **Master time grid** *(leaning)*: index the cache by `M` fixed timesteps and allow only NFEs that divide `M`. For example, M = 20 allows {1, 2, 4, 5, 10, 20}.
  - Switching NFE then never reallocates.
  - Slots a patch skips go stale. Fill them in one batched call per patch (eager back-fill), or only when needed (lazy catch-up).
- **Approximate K/V** (reuse the nearest slot): changes the output, so it's a research question.

### E4: Variable sequence length

Right-pad to the batch max, with a per-row attention mask `[B, L, L]` and `pos_ids [B, L]`.
- Padded positions must attend to at least themselves, or SDPA produces NaN.
- Unverified: whether the rotary embedding accepts batched `pos_ids`.

### E5: Noise in batches

Use a per-request seeded `torch.Generator`, so a request's noise doesn't depend on which other requests are in the batch.

## What to measure first (GPU)

1. Time per patch by stage (DiT vs LLM vs rest).
2. DiT step time vs batch size.
3. Re-prefill cost today vs back-fill cost.
4. Quality vs NFE (decides which NFEs, and M, are worth allowing).

## Open questions

1. Target the optimized path, with eager as the development vehicle?
2. Is restricting NFE to divisors of a master grid acceptable for the scheduler?
3. Is LLM batching in scope?
4. Does "the matrices that change" mean the per-step KV cache and AdaLN mods?
