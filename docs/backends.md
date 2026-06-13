# Backend Support

TorchLens resolves capture through `BackendSpec`. `backend=None` keeps the stable PyTorch eager
default and existing MLX module auto-routing; explicit `backend="jax"` enables the JAX functional
preview, and explicit `backend="tinygrad"` enables the tinygrad preview.

## Capability Matrix

| Backend | Capture | Validation | Payloads | Modules | Gradients |
|---|---|---|---|---|---|
| `torch` | Stable eager wrapper capture | Replay validation | Materialized `.tlspec` payloads | `torch_module` | True backward capture |
| `mlx` | Technical preview | Unsupported | Audit-only | `torch_module` compatibility mode | Unsupported |
| `jax` | Preview jaxpr-first functional capture | Per-equation replay and parent perturbation | Audit-only `.tlspec` save/load | `function_root` | `trace.derived_grads` only |
| `tinygrad` | Preview UOp-snapshot functional capture | UOp replay and parent perturbation on live `DEV=PYTHON` payloads | Audit-only `.tlspec` save/load | `function_root` | `trace.derived_grads` only |

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
| `module_identity_mode` | Backend module-mode selection |
| `payload_policy` | Backend payload codec/materialization policy |
| `save_preview` | Future non-torch `save=` preview semantics |

Capability epochs keep the public API, `CaptureOptions`, backend specs, per-backend capability
mirrors, cache keys, docs, and tests changing together.

JAX preview traces accept raw callables shaped like `fn(params, *inputs)`. Parameter records are
derived from the first pytree argument, so `Trace.param_source` is `"pytree-derived"` when tensor
leaves are present and `"none"` otherwise. Function-root module accessors expose only the root
`"self"` module; module predicates and framework module traversal are not supported in JAX M1.

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
capture from inside those transforms, nested-jaxpr primitives such as `cond`, `while`, and
`custom_vjp`, callback effects, closed-over array constants, selective save predicates,
intervention, halt, streaming, `save_grads=`, and `tl.record(backend="jax")`. `lax.scan` is
unrolled by default with `jax_control_flow="unroll"` and guarded by
`jax_max_control_flow_unroll`; pass `jax_control_flow="reject"` to preserve the earlier scan
rejection behavior. Workarounds are to pass raw functions and explicit params/input leaves to
full-save `tl.trace(..., backend="jax")`, or use the PyTorch backend for predicate capture,
intervention, sparse fastlog, and true backward graphs.

JAX `.tlspec` support is audit-only: `trace.save(path, level="audit")` persists metadata. Default
materialized payload save fails with `BackendPayloadUnsupportedError` until a non-torch payload
codec exists.

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

tinygrad derived gradients are a bracketed leaf-gradient preview, not backward capture:

```python
grad_options = tl.backends.tinygrad.GradOptions(input_grad_argnums=(0,))
trace = tl.trace(fn, Tensor([1.0, -2.0, 3.0]), backend="tinygrad", grad_options=grad_options)
trace.derived_grads["inputs.0"]
```

The tinygrad backend rejects mid-capture `Tensor.realize()`, `Tensor.assign()`,
`Tensor.replace()`, setitem input mutation, TinyJit execution, selective save predicates,
module predicates, intervention, halt, streaming, `save_grads=`, `backward_ready=True`, and
`tl.record(backend="tinygrad")`. Workarounds are to return a pure lazy tinygrad expression from a
raw callable and use full-save `tl.trace(..., backend="tinygrad")`, or use the PyTorch backend for
predicate capture, intervention, sparse fastlog, and true backward graphs.

tinygrad `.tlspec` support is audit-only: `trace.save(path, level="audit")` persists metadata.
Loaded audit traces cannot replay-validate or materialize tinygrad payloads; default materialized
payload save fails with `BackendPayloadUnsupportedError` until a tinygrad payload codec exists.
