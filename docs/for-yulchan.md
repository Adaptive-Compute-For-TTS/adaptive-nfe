# Notes for Yulchan: execution layer + dots.tts integration

Our task: build the GPU execution/scheduler layer, fixed NFE for the baseline phase. You build the execution layer, I wire it into dots.tts inference and check the speech still comes out the same. This is where my side is at and what I need from yours.

## Where I'm at

- The scheduler hook is in dots.tts and tested. With a fixed NFE, output is bit for bit identical to stock dots.tts at NFE 4, 10, 32 (2 sentences, every patch + final waveform, `torch.equal`). Results are in the README.
- Repo: https://github.com/tonyyytan/dot-tts-baseline. Setup steps are in the README (WSL/Linux only).
- Everything is eager mode (`optimize=False`), bf16, seed 42, dots.tts commit `3cb6f94`.

## How dots.tts runs a patch (the part you'd be replacing or wrapping)

One `generate()` call loops over patches (~160 ms each). Per patch:

```
model.py  _decode()                     outer loop, one iteration per patch, checks EOS
  └ _decode_next_audio()                <- our hook is here, picks num_steps
      └ DiTSolver.decode_next(nfe=...)  dit_inference.py ~L1459
          ├ optimize=False -> EagerDiTRunner.decode_next
          │    └ _decode_flow_matching   ~L1084
          │         noise = torch.randn(...)            one draw per patch, shape [1, 4, 128]
          │         odeint(euler, step 1/nfe, t 0 -> 1)
          │           └ _flow_matching_velocity          one DiT call per step,
          │                                              cond + uncond batched (2 rows), CFG g=1.2
          └ optimize=True  -> _decode_optimized          KV cache path, untested by us
  └ _consume_audio_patch()              patch -> patch encoder -> LLM -> next hidden state
```

So the cost per patch is NFE DiT forward passes, each with batch 2 (CFG). The LLM step runs once per patch no matter the NFE.

## The hook (what exists now)

```python
# in DotsTtsModel._decode_next_audio, before the solver runs
if self.nfe_scheduler is not None:
    num_steps = self.nfe_scheduler.next_nfe(
        patch_index=...,          # 0-based
        default_nfe=num_steps,    # whatever generate() got
        llm_hidden=state.llm_hiddens,
    )
```

Scheduler code is in `src/adaptive_nfe/scheduler.py`. `FixedScheduler(n)` always returns n. It only picks a number, the DiT still runs through dots.tts's own solver.

## What I need from you

Mainly the interface. My proposal, change whatever:

```python
class Executor:
    def decode_patch(self, *, sequence, cfg_sequence, fm_seq_len, g_cond,
                     nfe: int, guidance_scale: float, ode_method: str = "euler") -> torch.Tensor:
        """Run the DiT for one patch at `nfe` steps, return the patch latent [1, 4, 128]."""
```

Same inputs that `DiTSolver.decode_next` gets now, so I can swap the one call in `_decode_next_audio` for yours and nothing else in dots.tts changes. Scheduler picks NFE, executor runs it.

If you want something bigger (batching several patches or requests together, owning the noise, etc.) that's fine, just tell me the call so I know where to hook it.

## Rules if we want the baseline to stay bit for bit identical

The equivalence test has no tolerance, so for the fixed-NFE baseline your layer has to do exactly what stock does:

- Draw noise the same way: one `torch.randn` per patch, same shape/dtype/device, at the same point (before the ODE loop). Any extra or reordered random call shifts every patch after it.
- Same math in the same order: Euler, step `1/nfe`, cond and uncond in one batch of 2, CFG `v = cond + g*(cond - uncond)`.
- bf16, no `torch.compile` or CUDA graphs for now. Those change the numbers slightly.

If something has to differ (e.g. batching changes kernel choice) we can switch to a tolerance check, but let's see first.

## How we check it

Once your layer is plugged in I rerun:

```bash
python scripts/check_equivalence.py --nfe 4 10 32
```

It compares stock vs our version on every patch and the final audio, plus counts DiT calls per patch. Need ALL PASS. If it fails it tells you the first patch that differs and by how much. You can run it yourself too after setup.

## Heads up

- `optimize=True` (the fast path) keeps a KV cache across ODE steps that's sized for one NFE. Variable NFE per patch will probably break it. Fine for the baseline since NFE is fixed, but it'll matter when we go dynamic.
- Eager and optimized give slightly different numbers from each other, so the baseline is eager only for now.
- Compare `.pt` tensors not wavs, wav saving rounds the samples.

## Stuff to agree on

- [ ] Interface: does the executor replace `DiTSolver.decode_next`, or something higher up?
- [ ] Who owns noise generation and seeding
- [ ] Batching plans (across patches? across requests?) since that changes the interface
- [ ] Per-patch logging format (also with Nishant/Afreed)
- [ ] When we start on `optimize=True`
