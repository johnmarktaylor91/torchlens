# The deployment envelope: what is verified, at what size

TorchLens's target population loads large language models three ways --
`device_map="auto"` dispatch, CPU/disk offload, and 4/8-bit quantization --
usually with a PEFT/LoRA adapter on top. This page is the honest ledger of
what eager capture supports on those paths, what evidence backs each claim,
and exactly what remains unverified. **No scale claim on this page exceeds
its evidence.** Spellings referenced here are DOCUMENTED-UNSTABLE pending the
naming session.

## Verified today (CPU, small scale)

Every row below is pinned by the executable suites
`tests/test_deploy_env_*.py`, run on CPU against locally constructed real
architecture classes (a 2-layer GPT-2 of about one million parameters, and
small quantized MLPs). "Verified" means exactly what the named tests assert.

| Path | Evidence |
| --- | --- |
| `accelerate.dispatch_model` on a single execution device (the pure-CPU map) | capture runs, forward replay validation passes (`tl.validate(..., scope="forward") is True`) |
| `accelerate.disk_offload` / `accelerate.cpu_offload` (execution on CPU) | exact op-label parity with the plain twin (zero hook-infrastructure ops), materialized weights attribute to their prep-time `Param` records, captured logits equal the bare forward, forward replay validation passes |
| bitsandbytes `Linear8bitLt` / `Linear4bit` on CPU | captured outputs equal the bare forward; degradation DISCLOSED (see below), never silent |
| PEFT/LoRA (`peft.get_peft_model`) | output parity, adapter modules in the hierarchy, adapter params attributed at their addresses, forward replay validation passes |
| post-hoc `tl.debug.walk_grad_fn` / `sketch_grad_fn` | a real GPT-2 cross-entropy loss graph walks completely with named parameter leaves; structure-only legend on every render |

The compatibility report (`tl.compat.report(model, x)`) reflects the same
ledger: single-execution-device dispatch, offload, quantization, and PEFT
rows read `pass` with severity `info`/`warning` and evidence-stating details.

## How offloaded capture works

Accelerate's hooks materialize offloaded (meta) weights inside each module
call. TorchLens admits offload-backed meta parameters at the capture entry
gate with `weights_map` evidence (a meta parameter with NO offload hook still
refuses, fail-closed), runs the hook internals under `pause_logging` (weight
loading and device alignment are infrastructure, never model computation),
and re-stamps each materialized parameter with its prep-time identity so
weight consumptions attribute normally. Between forwards the parameters
return to meta, so:

- `Param.value` serves the LIVE handle -- meta between forwards. Value reads
  on an offloaded model's `Param` records are shape/dtype-only outside a
  forward.
- Weights tied at the storage level materialize as separate per-module
  objects under offload; tied-parameter metadata reflects the prep-time
  (meta) object topology.
- Default capture forces `requires_grad` for grad_fn metadata, and the
  resulting autograd graph holds every materialized shard alive until the
  trace is cleaned up. For large offloaded models prefer
  `tl.trace(..., inference_only=True)` (no autograd retention) or a
  selective `save=` predicate.

## Disclosed degradations (quantized capture)

bitsandbytes modules keep quantization state (`state.CB`/`state.SCB`, packed
uint8 payloads) outside the registered parameter/buffer contract:

- Reads of that state can appear as unattributed tensor arguments; the
  capture disclosure warning names the exact ops, and 8-bit captures may be
  ceilinged `capture_verified=False`.
- `tl.validate` keeps its tripwires armed: on 8-bit models the metadata
  invariant honestly flags those reads (a raise, not a pass); 4-bit replay
  reports `False`. A quantized model never validates as if it were dense.
- FLOPs counts and out-dtype metadata are best-effort (the standing
  quantized-module disclosure).

## NOT yet verified (gated on the GPU campaign)

The following publish NO claim until the C-DEPLOY cluster row runs them on
real hardware. The largest model verified by this repository's suite today
is the small-scale roster above; nothing bigger is claimed anywhere.

- Any real 7-8B (or larger) model on any path. C-DEPLOY runs ONE real 7-8B
  model through device_map / offload / 4-8-bit before any scale claim ships.
- Multi-GPU `device_map` sharding: cross-device activation moves are not
  capture-verified; the compat row keeps `known_broken` and multi-device
  maps stay refusal-adjacent until the campaign proves them.
- GPU bitsandbytes kernels (CPU kernels verified only).
- Buffer offload (`offload_buffers=True`): parameter capture works, but
  offloaded-buffer reads may stay unattributed (disclosed residual; the
  compat row reads severity `warning`).

A red C-DEPLOY keeps the claims and default flips off -- never the CPU-side
fixes documented here.
