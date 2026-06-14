# Backend Support

TorchLens resolves capture through `BackendSpec`. `backend=None` keeps the stable PyTorch eager
default and existing MLX module auto-routing; explicit `backend="jax"` enables the JAX functional
preview, and explicit `backend="tinygrad"` enables the tinygrad preview.

## Capability Matrix

| Backend | Capture | Validation | Payloads | Modules | Gradients |
|---|---|---|---|---|---|
| `torch` | Stable eager wrapper capture | Replay validation | Materialized `.tlspec` payloads | `torch_module` | True backward capture |
| `mlx` | Technical preview | Unsupported | Materialized forward array `.tlspec` payloads | `torch_module` compatibility mode | Unsupported |
| `jax` | Preview jaxpr-first functional capture | Live per-equation replay and parent perturbation | Materialized forward/derived array `.tlspec` payloads | `function_root`, Equinox/NNX `pytree_module` | `trace.derived_grads` only |
| `tinygrad` | Preview UOp-snapshot functional capture | Live UOp replay and parent perturbation on `DEV=PYTHON` payloads | Materialized forward/derived array `.tlspec` payloads | `function_root`, object `object_module` | `trace.derived_grads`; opt-in `trace.intermediate_derived_grads` |

## Public Option Spine

The public `tl.trace()` signature declares backend-growth options before their non-torch
implementations land. Torch accepts these options as inert cache-key inputs so torch behavior stays
unchanged. JAX now implements the control-flow policy knobs for `lax.scan`, `lax.cond`, and
`lax.while_loop`, and raw `function_root` JAX captures transparently inline pure nested `jax.jit`
calls before applying that control-flow policy. Nested JIT calls with closed constants, effects,
donated inputs, or explicit shardings remain rejected. tinygrad and
unimplemented option families reject explicit use with `BackendUnsupportedError` until the matching
backend phase implements the capability.

Declared future options:

| Option | Planned owner |
|---|---|
| `jax_control_flow` | JAX control-flow boundary or unroll policy; `lax.scan`/`cond`/`while_loop` support `unroll` or `reject` |
| `jax_max_control_flow_unroll` | JAX control-flow unroll safety limit |
| `module_identity_mode` | Backend module-mode selection; JAX supports raw `function_root` and Equinox/Flax NNX `pytree_module`; tinygrad supports raw `function_root` and callable object `object_module` |
| `payload_policy` | Backend payload codec/materialization policy; JAX/tinygrad/MLX use `array_payloads` |
| `save_preview` | Future non-torch save preview flag; JAX/tinygrad support static-label `save=`; MLX supports phase-A static labels (`tl.func`, `tl.label`, `tl.contains`) |

Capability epochs keep the public API, `CaptureOptions`, backend specs, per-backend capability
mirrors, cache keys, docs, and tests changing together.

## Predicate Control and Mutation

PyTorch eager capture can evaluate a predicate while a concrete tensor value is flowing through
the wrapped operation, then mutate that value or halt capture at the matching frontier. JAX and
tinygrad do not expose the same point in the current TorchLens preview backends. JAX first builds a
jaxpr over tracers and TorchLens labels are finalized after interpretation; a low-level primitive
interpreter can replace a primitive by position, but that is not the public static-label
`intervene=`/`halt=` contract. tinygrad builds a lazy UOp graph and concrete values appear only
after `Tensor.realize()`/`Tensor.item()`, with no stable public API for replacing one internal UOp
and rebuilding all descendants.

For that reason, non-torch backends support static-label `save=` only. Value-dependent `save=`,
`intervene=`, and `halt=` are rejected with typed backend errors instead of false partial traces or
validation passes. Live JAX/tinygrad selective-save traces still run real replay validation through
runtime-only hidden payloads; loaded traces report replay unavailable when those runtime captures
were stripped. MLX supports a narrower static-label `save=` phase for `tl.func`, `tl.label`,
`tl.contains`, and boolean composites of those; MLX validation is currently unsupported.

## MLX Preview

MLX is a technical-preview eager-dispatch backend. It supports forward capture and static-label
`save=` for `tl.func(...)`, `tl.label(...)`, `tl.contains(...)`, and safe boolean composites with
`&`, `|`, and `~`. Selective save is applied after full graph finalization: unsaved ops keep graph
metadata but drop public activation payloads, while saved payloads remain live MLX arrays in memory
and round-trip through portable `.tlspec` saves with `payload_policy="array_payloads"`.

MLX rejects `tl.module(...)` and `tl.in_module(...)` for `save=` because module hierarchy is
required and not yet available on MLX. It also rejects `tl.output(...)`, `tl.where(...)`,
`tl.followed_by(...)`, `tl.preceded_by(...)`, value-dependent predicates, `intervene=`, `halt=`,
streaming, `save_grads=`, `backward_ready=True`, and `tl.record(backend="mlx")`. MLX validation is
currently unsupported, so selective save does not fabricate a replay pass. Portable MLX saves
materialize forward array payloads and load them back as `mlx.core.array` values when the MLX runtime
is installed; loaded MLX traces still report replay validation as unavailable rather than as a pass.

JAX preview traces accept raw callables shaped like `fn(params, *inputs)`. Parameter records are
derived from the first pytree argument, so `Trace.param_source` is `"pytree-derived"` when tensor
leaves are present and `"none"` otherwise. Raw callables use `module_identity_mode="function_root"`
and expose only the root `"self"` module.

Equinox `eqx.Module` and Flax NNX `nnx.Module` roots default to
`module_identity_mode="pytree_module"`. TorchLens walks the module tree, attributes equations through
`jax.named_scope`, builds real `Trace.modules` entries with `training=False`, and derives parameter
owners from path-keyed pytree/state leaves. B2a strict mode rejects
`jax.jit`/`pjit`/`shard_map`, `lax.cond`, `lax.scan`, `lax.while_loop`, remat/custom-VJP, callback
effects, and NNX state rebinding inside attributed modules; move those transforms outside the module
or capture a raw `function_root` callable until widened attribution lands.

tinygrad raw callables use `module_identity_mode="function_root"`. Callable object graphs with
discoverable tinygrad module attributes default to `module_identity_mode="object_module"`; pass
`module_identity_mode="function_root"` to force the older root-only surface. Discovery walks object
attributes for callable children that own `Tensor` attributes or are known `tinygrad.nn.*` layer
objects, records shared children under the first address with later aliases, and attributes UOps from
the live module stack observed at construction time. If tinygrad reuses an existing UOp identity, the
UOp keeps its first observed module attribution rather than being duplicated for a later call.

## JAX Preview

Use explicit backend selection:

```python
import jax.numpy as jnp
import torchlens as tl

def fn(params, x):
    return jnp.tanh(x @ params["w"] + params["b"])

params = {"w": jnp.ones((3, 2)), "b": jnp.zeros((2,))}
trace = tl.trace(fn, (params, jnp.ones((4, 3))), backend="jax")
assert trace.validate_forward_pass([])
```

JAX leaf gradients are a derived-gradient preview, not backward capture:

```python
grad_options = tl.backends.jax.GradOptions(
    params=params,
    loss_fn=lambda output: jnp.sum(output * output),
    input_grad_argnums=(0,),
)
trace = tl.trace(fn, (params, jnp.ones((4, 3))), backend="jax", grad_options=grad_options)
trace.derived_grads["params.w"]
```

The JAX backend rejects transformed root callables (`jax.jit`, `jax.vmap`, `jax.grad`), root
capture from inside those transforms, unsupported nested-jaxpr primitives, callback effects,
closed-over array constants for raw function-root callables, value-dependent `save=` predicates,
`followed_by`/`preceded_by` window selectors, `intervene=`, `halt=`, streaming, `save_grads=`, and
`tl.record(backend="jax")`. Static-label `save=` selectors such as `tl.func(...)`,
`tl.label(...)`, `tl.contains(...)`, and module selectors are applied after full graph
finalization: unsaved ops keep metadata but drop public activation payloads. `lax.scan` is
unrolled by default with `jax_control_flow="unroll"` and guarded by
`jax_max_control_flow_unroll`; pass `jax_control_flow="reject"` to preserve the earlier scan
rejection behavior. Workarounds are to pass raw functions and explicit params/input leaves to
`tl.trace(..., backend="jax")`, or use the PyTorch backend for value-dependent predicate capture,
intervention, sparse fastlog, and true backward graphs.

JAX `.tlspec` support uses `payload_policy="array_payloads"`: default portable saves persist
forward and derived array payloads and load them back as `jax.Array` values.
Fully addressable single-host sharded arrays are saved as assembled host values; manifest
`codec_metadata` may include `jax_sharding_*` audit fields such as sharding kind, mesh axis
names, partition spec strings, and device counts, but load does not reconstruct that topology.
`trace.payload_load_status` records load-time materialization state. Runtime-only
`jax_equation_captures` are stripped from portable artifacts, so loaded JAX traces expose
`ValidationReplayStatus` at `trace.validation_replay_status.state == "unavailable"` with reason
`"loaded_trace_runtime_capture_stripped"` instead of reporting a false validation pass or failure.
Live JAX traces, including static-label selectively saved traces, still run real replay validation
using runtime-only replay payloads that are separate from the public saved-activation surface.

## tinygrad Preview

Install the optional runtime with `torchlens[tinygrad]`; the preview is pinned to the
`tinygrad>=0.13,<0.14` series and currently accepts `tinygrad==0.13.0` at runtime. For live
payload validation and derived gradients, run with `DEV=PYTHON`.

```python
from tinygrad import Tensor
import torchlens as tl

def fn(x):
    return ((x + 1).relu() * 2).sum()

trace = tl.trace(fn, Tensor([1.0, -2.0, 3.0]), backend="tinygrad")
assert trace.validate_forward_pass([fn(Tensor([1.0, -2.0, 3.0]))])
```

tinygrad callable object models expose object-module hierarchy and native parameter records when
TorchLens can discover them:

```python
from tinygrad import Tensor
import tinygrad.nn as nn
import torchlens as tl

class TinyMlp:
    def __init__(self):
        self.fc = nn.Linear(3, 4)

    def __call__(self, x):
        return self.fc(x).relu()

trace = tl.trace(TinyMlp(), Tensor.ones(1, 3), backend="tinygrad")
assert trace.module_identity_mode == "object_module"
assert trace.modules["fc"].params
```

tinygrad derived gradients are a bracketed leaf-gradient preview, not backward capture:

```python
grad_options = tl.backends.tinygrad.GradOptions(input_grad_argnums=(0,))
trace = tl.trace(fn, Tensor([1.0, -2.0, 3.0]), backend="tinygrad", grad_options=grad_options)
trace.derived_grads["inputs.0"]
```

Set `GradOptions(intermediate_grads=True)` to run tinygrad's separate no-realize
intermediate-gradient pass. It calls `loss.backward()` before realizing any copied outputs, then
attaches exact unambiguous records to `trace.intermediate_derived_grads`; each owning `Op` exposes
the payload through read-only `op.derived_grad`. Ambiguous signature matches are skipped instead of
attached, and `op.grads` / `trace.saved_grad_ops` remain true-backward-only.

The tinygrad backend rejects mid-capture `Tensor.realize()`, `Tensor.assign()`,
`Tensor.replace()`, setitem input mutation, TinyJit execution, value-dependent `save=`
predicates, `followed_by`/`preceded_by` window selectors, `intervene=`, `halt=`, streaming,
`save_grads=`, `backward_ready=True`, and `tl.record(backend="tinygrad")`. Static-label `save=`
selectors such as `tl.func(...)`, `tl.label(...)`, `tl.contains(...)`, and module selectors are
applied after full graph finalization: unsaved ops keep metadata but drop public activation
payloads. Workarounds are to return a pure lazy tinygrad expression from a raw callable and use
`tl.trace(..., backend="tinygrad")`, or use the PyTorch backend for value-dependent predicate
capture, intervention, sparse fastlog, and true backward graphs.

tinygrad `.tlspec` support uses `payload_policy="array_payloads"`: default portable saves persist
forward and derived array payloads and load them back as tinygrad `Tensor` values.
`trace.payload_load_status` records load-time materialization state. Runtime-only
`tinygrad_uop_captures` are stripped from portable artifacts, so loaded tinygrad traces expose
`ValidationReplayStatus` at `trace.validation_replay_status.state == "unavailable"` with reason
`"loaded_trace_runtime_capture_stripped"` instead of reporting a false validation pass or failure.
Live tinygrad traces, including static-label selectively saved traces, still run real replay
validation using runtime-only replay payloads that are separate from the public saved-activation
surface.
