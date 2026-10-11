# Exploration: batching and variable NFE

**Author:** Yulchan · **Status:** Exploration, next phase (not part of the baseline task) · **Date:** 2026-10-11
**Context:** answers Tony's ask to explore *"batching, and how to handle the matrices that change when we change the flow matching step, and making sure it works when we batch."* Builds on the [baseline executor](executor-baseline.md).
**dots.tts:** commit `3cb6f94`; line numbers refer to the unpatched source.

## Contents

1. [Findings from the dots.tts code](#1-findings-from-the-dotstts-code) (F1–F4)
2. [Design proposals](#2-design-proposals) (E1–E5): recommendation first, options collapsed
3. [Proposed roadmap](#3-proposed-roadmap)
4. [Open questions](#4-open-questions)

---

## 1. Findings from the dots.tts code

| ID | Finding | Why it matters |
|---|---|---|
| F1 | The DiT accepts a **different timestep per row** | Mixed-NFE batching needs no model changes |
| F2 | Eager and optimized paths behave very differently | Determines where to build first ([E2](#e2-execution-mode-eager-or-optimized)) |
| F3 | The optimized **KV cache has one copy per ODE step** | Changing NFE forces a full re-prefill ([E3](#e3-kv-cache-under-variable-nfe)) |
| F4 | `generate()` is **single-request only** | Batching needs a new loop above it ([E1](#e1-batching-policy)) |

### F1: The DiT accepts per-row timesteps

`_flow_matching_velocity` passes `timesteps` as a vector of length `2B` (`dit_inference.py:1071`); stock just broadcasts one scalar into it. `FusedAdaLNDiT.forward` turns per-row timesteps into per-row AdaLN modulation, and `modulate()` applies shift/scale per row (`dit.py:9`). Rows at different ODE times can share one forward pass.

### F2: Eager vs optimized paths

| | Eager (`optimize=False`) | Optimized (`optimize=True`) |
|---|---|---|
| Per ODE step | Full forward over the whole FM sequence | Forward over 2 units (`prev_unit` + `current_hidden` + noisy latent) against a KV cache |
| Step cost vs utterance length | Grows (linear layers O(L), attention O(L²)) | ~constant |
| ODE integration | `torchdiffeq.odeint`, bf16 grid, `dt = t1 − t0` per step | Hand-written loop, `t = k/nfe`, `dt = 1/nfe` |
| Variable NFE per patch | Works trivially | Triggers cache reallocation + full prefill ([F3](#f3-the-optimized-kv-cache-is-indexed-by-ode-step)) |
| Compilation | None | `torch.compile` on CUDA (`:1314`); flex attention by default |

### F3: The optimized KV cache is indexed by ODE step

```
cache_k, cache_v : [nfe, num_layers, 2B, num_heads, capacity_tokens, head_dim]   (allocate_cache, :744)
```

AdaLN modulation depends on t and applies to every token, history included, so history K/V differ per timestep. That means one full copy per ODE step. Consequences:

- **Changing NFE invalidates the cache.** `_ensure_cache` (`:1343`) reallocates and refuses to copy across NFE values (`:388`). `prefill` (`:845`) then recomputes K/V for the entire history at every new timestep. Frequent NFE changes could end up slower than fixed NFE.
- **Per-step AdaLN mods** (`all_mods_by_ode`, `:1436`) are also cached per NFE. They're cheap to recompute.
- **Cache memory scales linearly with NFE** per request, which bounds the batch size.

This is likely what "the matrices that change when we change the flow matching step" refers to.

**Useful property:** `prev_unit` queries attend only to the cache prefix and causally within `prev_unit` (`inference_utils.py:160`). So the K/V written back for `prev_unit` at timestep t (`:575`) don't depend on the noisy latent, and back-filling a cache slot is a noise-free forward over one unit.

### F4: Generation is single-request

`_generate_latents_stream` rejects batch size ≠ 1 (`model.py:1390`); LLM, patch encoder and FM buffers are per-request state. Patches within a request can't be batched either, because each depends on the previous one passing through the LLM. The DiT solver internals already handle a batch dimension (`apply_cfg` splits `[cond; uncond]` by `batch_size`).

---

## 2. Design proposals

### E1: Batching policy

**Recommendation:** batch **across requests**, with mixed-NFE rows in one DiT call, each at its own timestep ([F1](#f1-the-dit-accepts-per-row-timesteps)). Build it in two stages:
1. **Shrinking batch:** start a round with all pending patch jobs. Each DiT call advances every active row by one step; rows leave when they reach their NFE. No wasted compute, simple control flow.
2. **Continuous batching:** rows join as soon as a request needs a patch, as in Orca/vLLM. Highest GPU occupancy.

**What changes from the baseline executor:**
- **Interface:** add `decode_patches(jobs: list[PatchJob]) -> list[Tensor]`, where each job carries its own inputs, NFE and RNG generator. Continuous batching later needs a `submit(job)` / `step()` API. A new request loop above `generate()` gathers the jobs ([F4](#f4-generation-is-single-request)).
- **Euler loop:** becomes per row. Each row has its own time grid and `dt`, built with the same bf16 rules as the baseline loop, and the update is `z + dt[:, None, None] * v`.
- **CFG row layout** stays `[cond_0..cond_{B-1}; uncond_0..uncond_{B-1}]`, so `apply_cfg` works unchanged.
- **The optimized path** needs per-request solver state (cache, mods), kept inside the executor and keyed by `request_id`.

**LLM batching:** out of scope at first. Under continuous batching, the per-request LLM step serialises and may cap throughput, so revisit after profiling.

<details><summary>Options considered</summary>

| Option | Description | Pros | Cons |
|---|---|---|---|
| A. Pad to max NFE | All rows run max(NFE) steps | Simple | Wastes compute; erases NFE savings |
| B. Bucket by NFE | Separate batches per NFE | Homogeneous batches | Smaller batches, more queueing; unnecessary given F1 |
| **C1. Shrinking batch** ✅ first | Rows at own t/dt, leave when done, no mid-round joins | No wasted work; simple | Occupancy decays within a round |
| **C2. Continuous batching** ✅ later | Rows join/leave any time | Best occupancy and latency | Most complex; LLM becomes a bottleneck |

| LLM scope | Pros | Cons |
|---|---|---|
| **DiT only** ✅ | Contained | LLM may cap throughput under C2 |
| DiT + LLM + patch encoder | Full concurrency benefit | Much larger scope (LLM static cache, positions, encoder state) |

</details>

### E2: Execution mode: eager or optimized?

**Recommendation:** **target optimized, build in eager first.**
- Eager is where the bit-exact test lives and where variable NFE is free. Build and validate mixed-t batching, padding and per-request RNG there.
- Optimized is the real serving path. Final speed-ups must be measured against **stock optimized**, or they're misleading.
- Start optimized once eager batching is validated. Its blocker is [E3](#e3-kv-cache-under-variable-nfe).

**Open point:** the optimized path compiles on CUDA regardless of what we do. Whether stock-optimized vs ours-optimized can be bit-exact, or needs a tolerance, has to be found empirically.

<details><summary>Options considered</summary>

| Option | Pros | Cons |
|---|---|---|
| A. Eager only | Variable NFE free; test exists | Not what's served; misleading speed-ups |
| B. Optimized only | Real serving path | KV cache must be solved first; no test harness; compile complicates exactness |
| **C. Eager first, then optimized** ✅ | Correctness work where it's easy; cache work isolated | Some eager-only code (padding/masks) isn't reused |

</details>

### E3: KV cache under variable NFE

**Problem ([F3](#f3-the-optimized-kv-cache-is-indexed-by-ode-step)):** the cache holds one K/V copy per timestep. A patch at a new NFE needs timesteps the cache doesn't have, so today it re-prefills the whole history.

**Recommendation:** a **master time grid** with **eager back-fill**.

1. **Master grid.** Index the cache by a fixed set of `M` timesteps `{0, 1/M, …, (M−1)/M}` instead of by step number, and allow only NFE values that divide `M`. Every allowed NFE then uses a subset of the same slots, so switching NFE never reallocates.
2. **M = 20 to start.** It allows NFE {1, 2, 4, 5, 10, 20}, including stock's default of 10 (M = 16 would exclude it). Finalise after measuring quality vs NFE.
3. **Back-fill.** A patch only writes `prev_unit` K/V into the slots it visits. After each patch, update the skipped slots in **one batched forward**, so any NFE is ready for the next patch. Per [F3](#f3-the-optimized-kv-cache-is-indexed-by-ode-step), this needs no noise. If it proves expensive, switch to lazy catch-up (fill a slot only right before it's needed).

**Example (M = 20):**

| Patch | NFE | Slots used | Slots back-filled afterwards |
|---|---|---|---|
| 1 | 4 | 0, 5, 10, 15 | the other 16 |
| 2 | 10 | 0, 2, 4, …, 18 | the other 10 |

Without back-fill, patch 2 would find slots 2, 4, 6, … missing patch 1's K/V.

**Expected cost:** about `n` steps + 1 back-fill call per patch, instead of a full re-prefill whenever NFE changes.

**Notes:**
- `DiTKvCache.valid_tokens` becomes per slot.
- The cache always holds `M` copies per request, whatever NFE is chosen, so M bounds the batch size.
- The optimized path uses `t = k/nfe` (not the eager bf16 grid), so grid slot `s/M` equals stock's `k/n` whenever they're the same fraction. `all_mods_by_ode` is computed over a differently shaped batch, though, and may differ in the low bits. Test this.
- In a batch, rows read different slots from different requests' caches. This needs a per-row gather or a paged cache layout (as in vLLM).

<details><summary>Options considered</summary>

**Cache structure**

| Option | Description | Pros | Cons |
|---|---|---|---|
| 0. Status quo | Reallocate + full prefill on NFE change | No work | Likely negates adaptive NFE |
| 1. One cache per allowed NFE | Separate cache per NFE value | No realloc when switching back | Memory × #NFEs; caches still go stale |
| **2. Master time grid** ✅ | Cache indexed by grid slot; NFE divides M | One cache; free switching | Restricts NFE menu; M copies per request; stale slots |
| 3. Approximate K/V | Reuse nearest slot for off-grid t | No extra compute | Changes output; quality unknown |

**Filling stale slots (with option 2)**

| Strategy | Description | Pros | Cons |
|---|---|---|---|
| **(a) Eager back-fill** ✅ | All skipped slots updated in one batched call per patch | Cache always complete; predictable; simple | One extra call per patch, even if unused |
| (b) Lazy catch-up | Per-slot valid length; catch up before use | No wasted work | Bookkeeping; latency spikes after long low-NFE stretches |
| (c) Constrain scheduler | Limit how often/far NFE changes | Fewer catch-ups | Limits scheduler; not a fix alone |
| (d) Hybrid | (a) for common slots, (b) for rare ones | Balances waste vs spikes | Most complex; needs usage data |

</details>

### E4: Variable sequence length

**Problem:** requests in a batch have different FM sequence lengths (`fm_seq_len`), and each grows every patch. Tensors in one batch must have the same length.

**Recommendation:**
- **Eager:** right-pad every request to the longest in the batch, and give each row its own attention mask `[B, L, L]` and `pos_ids [B, L]` so it ignores the padding. For example, if request A has 300 tokens and B has 800, A is padded with 500 tokens that its mask hides. The SDPA path already accepts per-row masks (`dit_inference.py:146`). Add length bucketing later to cut padding waste.
- **Optimized:** the same idea on the cache. Each row gets its own valid cache length and RoPE start position; today `masks_for` and `rotary_for` take one scalar for the whole batch.

**Pitfall:** a padded position whose mask row is all `False` produces NaN in softmax, which can leak into real rows via `0 · NaN`. Let each padded position attend to itself.

**Unverified:** that the rotary embedding accepts batched `pos_ids [B, L]`.

<details><summary>Options considered</summary>

| Option | Pros | Cons |
|---|---|---|
| **A. Right-pad + per-row mask/pos_ids** ✅ | Straightforward | Everyone pays for the longest sequence |
| **B. A + length bucketing** ✅ later | Less padding waste | More queueing; needs a policy |
| C. Sequence packing, block-diagonal masks | No padding waste | More complex; not in the eager runner today |

</details>

### E5: Noise in batches

**Recommendation:** one seeded `torch.Generator` per request. Each request draws its own `[1, 4, 128]` noise from it, then the rows are stacked.

**Why:** with the global RNG used in the baseline, a request's noise would depend on which other requests happen to be in the batch and in what order. Per-request generators make results batch-invariant and reproducible.

<details><summary>Options considered</summary>

| Option | Pros | Cons |
|---|---|---|
| A. Global RNG, per-request draws in batch order | Closest to stock | Not batch-invariant |
| **B. Per-request `torch.Generator`** ✅ | Batch-invariant, reproducible | Not bit-identical to stock's global stream |
| C. One `[B, 4, 128]` draw from the global RNG | Simplest | Neither stock-exact nor batch-invariant |

</details>

---

## 3. Proposed roadmap

Starts after the baseline executor passes `check_equivalence.py`. Not yet agreed with Tony.

| Phase | Scope | Done when |
|---|---|---|
| 1 | Measurements (below) | Report under `runs/` |
| 2 | Eager batching: per-row t/dt, padding + masks, per-request RNG, shrinking-batch loop | Validation 1–3 |
| 3 | Optimized: master-grid cache, back-fill, per-row slot selection | Validation 4 |
| 4 | Continuous batching; revisit LLM batching | Throughput vs stock-optimized at matched quality |

**Validation**

| # | What | Pass criterion |
|---|---|---|
| 1 | One batched DiT call with mixed t vs per-row calls | Max abs diff under a tight tolerance |
| 2 | Full utterances, batched vs solo | Early-patch latents within tolerance; ASR transcript match (WER, `edit_playground_asr` extra); listening checks. Exact equality isn't possible: errors compound through the LLM and can shift the EOS patch |
| 3 | DiT call accounting | Forward passes per patch = scheduled NFE (back-fill counted separately) |
| 4 | Optimized path | Stock-optimized vs ours-optimized: bit-exact if achievable, else tolerance |

**Measurements** (NVIDIA GPU; dots.tts already times each stage via `measure_inference`)

1. **Time per patch by stage** (DiT vs LLM vs rest), eager and optimized.
2. **DiT step time vs batch size** (1 → 64).
3. **Re-prefill cost** today vs **back-fill cost** with the master grid.
4. **Quality vs NFE**, which decides the allowed NFE values and M.

## 4. Open questions

1. Confirm the target is the optimized path, with eager as the development vehicle ([E2](#e2-execution-mode-eager-or-optimized)).
2. Is restricting NFE to divisors of a master grid acceptable to the scheduler side? Is M = 20 a reasonable start ([E3](#e3-kv-cache-under-variable-nfe))?
3. Is LLM batching in scope, and if so, when ([E1](#e1-batching-policy))?
4. Do batched results need to be batch-invariant ([E5](#e5-noise-in-batches))?
5. Does "the matrices that change" refer to the per-step KV cache and AdaLN mods ([F3](#f3-the-optimized-kv-cache-is-indexed-by-ode-step)), or something else?
