# Backend Support

TorchLens resolves capture through `BackendSpec`. `backend=None` keeps the stable PyTorch eager
default and existing MLX module auto-routing; explicit `backend="jax"` enables the JAX functional
preview, and explicit `backend="tinygrad"` enables the tinygrad preview.

## Capability Matrix

| Backend | Capture | Validation | Payloads | Modules | Gradients |
|---|---|---|---|---|---|
| `torch` | Stable eager wrapper capture | Replay validation | Materialized `.tlspec` payloads | `torch_module` | True backward capture |
| `mlx` | Technical preview | Unsupported | Audit-only | `torch_module` compatibility mode | Unsupported |
| `jax` | Preview jaxpr-first functional capture | Live per-equation replay and parent perturbation | Materialized forward/derived array `.tlspec` payloads | `function_root`, Equinox `pytree_module` | `trace.derived_grads` only |
| `tinygrad` | Preview UOp-snapshot functional capture | Live UOp replay and parent perturbation on `DEV=PYTHON` payloads | Materialized forward/derived array `.tlspec` payloads | `function_root`, object `object_module` | `trace.derived_grads`; opt-in `trace.intermediate_derived_grads` |

## Public Option Spine

The public `tl.trace()` signature declares backend-growth options before their non-torch
implementations land. Torch accepts these options as inert cache-key inputs so torch behavior stays
unchanged. JAX now implements the control-flow policy knobs for `lax.scan`; tinygrad and
unimplemented option families reject explicit use with `BackendUnsupportedError` until the matching
backend phase implements the capability.

Declared future options:

| Option | Planned owner |
|---|---|
| `jax_control_flow` | JAX control-flow boundary or unroll policy; `lax.scan` supports `unroll` or `reject` |
| `jax_max_control_flow_unroll` | JAX `lax.scan` unroll safety limit |
| `module_identity_mode` | Backend module-mode selection; JAX supports raw `function_root` and Equinox `pytree_module`; tinygrad supports raw `function_root` and callable object `object_module` |
| `payload_policy` | Backend payload codec/materialization policy |
| `save_preview` | Future non-torch `save=` preview semantics |

Capability epochs keep the public API, `CaptureOptions`, backend specs, per-backend capability
mirrors, cache keys, docs, and tests changing together.

JAX preview traces accept raw callables shaped like `fn(params, *inputs)`. Parameter records are
derived from the first pytree argument, so `Trace.param_source` is `"pytree-derived"` when tensor
leaves are present and `"none"` otherwise. Raw callables use `module_identity_mode="function_root"`
and expose only the root `"self"` module.

Equinox `eqx.Module` roots default to `module_identity_mode="pytree_module"`. TorchLens walks the
module dataclass tree, attributes equations through `jax.named_scope`, builds real `Trace.modules`
entries with `training=False`, and derives parameter owners from pytree array paths. B2a strict mode
rejects `jax.jit`/`pjit`/`shard_map`, `lax.cond`, `lax.scan`, `lax.while_loop`, remat/custom-VJP,
and callback effects inside attributed modules; move those transforms outside the module or capture a
raw `function_root` callable until widened attribution lands.

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
closed-over array constants for raw function-root callables, selective save predicates,
intervention, halt, streaming, `save_grads=`, and `tl.record(backend="jax")`. `lax.scan` is
unrolled by default with `jax_control_flow="unroll"` and guarded by
`jax_max_control_flow_unroll`; pass `jax_control_flow="reject"` to preserve the earlier scan
rejection behavior. Workarounds are to pass raw functions and explicit params/input leaves to
full-save `tl.trace(..., backend="jax")`, or use the PyTorch backend for predicate capture,
intervention, sparse fastlog, and true backward graphs.

JAX `.tlspec` support uses `payload_policy="array_payloads"`: default portable saves persist
forward and derived array payloads and load them back as `jax.Array` values. Runtime-only
`jax_equation_captures` are stripped from portable artifacts, so loaded JAX traces expose
`trace.validation_replay_status.state == "unavailable"` with reason
`"loaded_trace_runtime_capture_stripped"` instead of reporting a false validation pass or failure.
Live JAX traces still run real replay validation.

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
`Tensor.replace()`, setitem input mutation, TinyJit execution, selective save predicates,
module predicates, intervention, halt, streaming, `save_grads=`, `backward_ready=True`, and
`tl.record(backend="tinygrad")`. Workarounds are to return a pure lazy tinygrad expression from a
raw callable and use full-save `tl.trace(..., backend="tinygrad")`, or use the PyTorch backend for
predicate capture, intervention, sparse fastlog, and true backward graphs.

tinygrad `.tlspec` support uses `payload_policy="array_payloads"`: default portable saves persist
forward and derived array payloads and load them back as tinygrad `Tensor` values. Runtime-only
`tinygrad_uop_captures` are stripped from portable artifacts, so loaded tinygrad traces expose
`trace.validation_replay_status.state == "unavailable"` with reason
`"loaded_trace_runtime_capture_stripped"` instead of reporting a false validation pass or failure.
Live tinygrad traces still run real replay validation.
