# Backend Support

TorchLens resolves capture through `BackendSpec`. `backend=None` keeps the stable PyTorch eager
default and existing MLX module auto-routing; explicit `backend="jax"` enables the JAX functional
preview, explicit `backend="tinygrad"` enables the tinygrad preview, and explicit
`backend="paddle"` enables the Paddle dygraph/eager preview. Explicit `backend="tf"` or
`backend="tensorflow"` enables the TensorFlow eager preview.

## Capability Matrix

| Backend | Capture | Validation | Payloads | Modules | Gradients |
|---|---|---|---|---|---|
| `torch` | Stable eager wrapper capture | Replay validation | Materialized `.tlspec` payloads | `torch_module` | True backward capture |
| `mlx` | Technical preview | Unsupported | Materialized forward/derived array `.tlspec` payloads | `function_root`, object `object_module` | `trace.derived_grads`; opt-in `trace.intermediate_derived_grads` |
| `jax` | Preview jaxpr-first functional capture | Live per-equation replay and parent perturbation | Materialized forward/derived array `.tlspec` payloads | `function_root`, Equinox/NNX `pytree_module` | `trace.derived_grads`; opt-in `trace.intermediate_derived_grads` |
| `tinygrad` | Preview UOp-snapshot functional capture | Live UOp replay and parent perturbation on `DEV=PYTHON` payloads | Materialized forward/derived array `.tlspec` payloads | `function_root`, object `object_module` | `trace.derived_grads`; opt-in `trace.intermediate_derived_grads` |
| `paddle` | Preview dygraph/eager capture | Live replay/perturbation plus static inventory guard | Materialized forward/derived array `.tlspec` payloads | `function_root`, object `object_module` | `trace.derived_grads`; opt-in `trace.intermediate_derived_grads` |
| `tf` | Preview eager op-callback capture; graph-only FuncGraph fallback planned | Callback self-consistency plus per-op replay/perturbation accounting | Materialized forward array `.tlspec` payloads | `function_root`, Keras/`tf.Module` object `object_module` | Deferred |

## Public Option Spine

The public `tl.trace()` signature declares backend-growth options before their non-torch
implementations land. Torch accepts these options as inert cache-key inputs so torch behavior stays
unchanged. JAX now implements the control-flow policy knobs for `lax.scan`, `lax.cond`, and
`lax.while_loop`, and raw `function_root` JAX captures transparently inline pure nested `jax.jit`
calls before applying that control-flow policy. Nested JIT calls with closed constants, effects,
donated inputs, or explicit shardings remain rejected. tinygrad, Paddle, TensorFlow, and
unimplemented option families reject explicit use with `BackendUnsupportedError` until the matching
backend phase implements the capability.

Declared future options:

| Option | Planned owner |
|---|---|
| `jax_control_flow` | JAX control-flow boundary or unroll policy; `lax.scan`/`cond`/`while_loop` support `unroll` or `reject`; `region` forces supported scan/while/custom-VJP-forward regions |
| `jax_max_control_flow_unroll` | JAX control-flow unroll safety limit |
| `module_identity_mode` | Backend module-mode selection; JAX supports raw `function_root` and Equinox/Flax NNX `pytree_module`; tinygrad, MLX, Paddle, and TensorFlow support raw `function_root` and callable object `object_module` |
| `payload_policy` | Backend payload codec/materialization policy; JAX/tinygrad/MLX/Paddle/TF use `array_payloads` |
| `save_preview` | Future non-torch save preview flag; JAX/tinygrad/MLX/Paddle/TF support static-label `save=` |

Capability epochs keep the public API, `CaptureOptions`, backend specs, per-backend capability
mirrors, cache keys, docs, and tests changing together.

## Predicate Control and Mutation

PyTorch eager capture can evaluate a predicate while a concrete tensor value is flowing through
the wrapped operation, then mutate that value or halt capture at the matching frontier. JAX,
tinygrad, Paddle, and TensorFlow do not expose the same point in the current TorchLens preview backends. JAX
first builds a jaxpr over tracers and TorchLens labels are finalized after interpretation; a
low-level primitive interpreter can replace a primitive by position, but that is not the public
static-label `intervene=`/`halt=` contract. tinygrad builds a lazy UOp graph and concrete values
appear only after `Tensor.realize()`/`Tensor.item()`, with no stable public API for replacing one
internal UOp and rebuilding all descendants. Paddle preview capture is eager but denies live
predicate-time mutation/halt while its op inventory and alias guards are still preview-scoped.
TensorFlow eager `op_callbacks` are read-only in the supported Keras-3 / TF>=2.16 runtime, so
interventions require a later writable monkeypatch layer.

For that reason, non-torch backends support static-label `save=` only. Value-dependent `save=`,
`intervene=`, and `halt=` are rejected with typed backend errors instead of false partial traces or
validation passes. Live JAX/tinygrad/Paddle/TF selective-save traces still run real replay validation
through runtime-only hidden payloads; loaded traces report replay unavailable when those runtime
captures were stripped. MLX supports static-label `save=` for `tl.func`, `tl.label`, `tl.module`,
`tl.in_module`, `tl.contains`, and boolean composites of those; MLX validation is currently
unsupported.

## TensorFlow Preview

Install the optional runtime with `torchlens[tf]` or `torchlens[tensorflow]`; the preview targets
Keras 3 on TensorFlow `>=2.16` and requires `keras.backend.backend() == "tensorflow"` for Keras
models. Use explicit backend selection:

```python
import tensorflow as tf
import torchlens as tl

model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(4, activation="relu"),
        tf.keras.layers.Dense(2),
    ]
)
x = tf.ones((1, 3), dtype=tf.float32)
trace = tl.trace(model, x, backend="tf")
print(trace.summary())
```

The TensorFlow backend is designed around two mechanisms. The shipped primary path is eager live
capture: TorchLens runs the real `model(*args, **kwargs)` forward under
`tensorflow.python.framework.op_callbacks`, records each eager op with real values and real
taken-branch control flow, and snapshots Keras/`tf.Module` call-stack frames for
`module_identity_mode="object_module"` when discovery succeeds. Raw callables can use the root-only
`function_root` mode. Graph-only entries such as compiled `Model.call`, `tf.function` roots,
SavedModel-style signatures, and `.predict()` are detected separately; the FuncGraph walk/prune
fallback is the planned static-mode path, and current P6 builds fail closed rather than producing a
partial graph-only trace.

Static-label `save=` selectors are applied after full graph finalization, with unsaved ops retaining
metadata and dropping public payloads. TensorFlow validation is non-vacuous but scoped to the
preview: callback self-consistency catches unresolved producers, pure allowlisted ops replay through
`tf.raw_ops` with parent perturbation, and the replay status reports replayed, pure-unverified, and
effect-region counts. A status of `unverified` is partial validation, not a pass.

TensorFlow `.tlspec` support uses `payload_policy="array_payloads"` for dense numeric/bool forward
payloads. `bfloat16` records logical dtype metadata because NumPy transports it as `uint16`; string,
resource, variant, and composite payloads fail closed.

TensorFlow interventions, `halt=`, true backward capture, fastlog/`tl.record()`, streaming, and
T1/intermediate derived gradients are deferred. These surfaces raise typed backend errors instead of
silently producing partial traces.

## MLX Preview

MLX is a technical-preview eager-dispatch backend. MLX `mlx.nn.Module` roots default to
`module_identity_mode="object_module"` using MLX public `named_modules()` traversal, with shared
module aliases represented by a deterministic primary address and `all_addresses`. Pass
`module_identity_mode="function_root"` to force a root-only module surface. Python list-contained
modules are attributed at addresses such as `layers.0`; TorchLens does not invent synthetic
container modules for non-module list objects.

MLX supports static-label `save=` for `tl.func(...)`, `tl.label(...)`, `tl.module(...)`,
`tl.in_module(...)`, `tl.contains(...)`, and safe boolean composites with `&`, `|`, and `~`.
Selective save is applied after full graph finalization: unsaved ops keep graph metadata but drop
public activation payloads, while saved payloads remain live MLX arrays in memory and round-trip
through portable `.tlspec` saves with `payload_policy="array_payloads"`.

MLX leaf gradients are a derived-gradient preview, not backward capture:

```python
grad_options = tl.backends.mlx.GradOptions(
    loss_fn=lambda output: mx.sum(output * output),
    input_grad_argnums=(0,),
)
trace = tl.trace(model, x, backend="mlx", grad_options=grad_options)
trace.derived_grads["inputs.0"]
```

For `mlx.nn.Module` roots, TorchLens passes `model.parameters()` as an explicit
`mx.value_and_grad` argument and rebinds it with `model.update(params)` during the AD rerun. The raw
AD-rerun output must match the captured forward output within dtype tolerance before records are
exposed.

Set `GradOptions(intermediate_grads=True, max_intermediate_grads=...)` to request MLX saved-op
intermediate derived gradients. The producer uses one extra `mx.value_and_grad` replay with
custom-VJP identity taps installed by the MLX wrappers before their logging early return; normal
capture logging remains off, so module call counts are not incremented by the AD replay. Attachment
uses grouped structural signatures rather than a capture-index map, and duplicate or ambiguous
groups are skipped. Each public record must pass an independent replacement-gradient and
perturbation oracle, and only records with `status == "exact"` reach
`trace.intermediate_derived_grads` and read-only `op.derived_grad`.

These derived gradients are not true backward capture, and `op.grads` /
`trace.saved_grad_ops` stay true-backward-only.

MLX rejects `tl.output(...)`, `tl.where(...)`,
`tl.followed_by(...)`, `tl.preceded_by(...)`, value-dependent predicates, `intervene=`, `halt=`,
streaming, `save_grads=`, `backward_ready=True`, and `tl.record(backend="mlx")`. MLX validation is
currently unsupported, so selective save does not fabricate a replay pass. Portable MLX saves
materialize forward and derived array payloads and load them back as `mlx.core.array` values when the
MLX runtime is installed; loaded MLX traces still report replay validation as unavailable rather than
as a pass.

## Paddle Preview

Install the optional runtime with `torchlens[paddle]`; the preview is pinned to
`paddlepaddle>=3.3,<3.4`. Paddle capture is dygraph/eager only:

```python
import paddle
import paddle.nn as nn
import torchlens as tl

model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
x = paddle.ones([1, 3], dtype="float32")
trace = tl.trace(model, x, backend="paddle")
```

Paddle module roots default to object module hierarchy when TorchLens can inspect the
`paddle.nn.Layer` tree, with `function_root` available for raw callables. Static-label `save=`
selectors are applied after full graph capture; value-dependent predicates, `intervene=`, `halt=`,
streaming, `save_grads=`, `backward_ready=True`, and `tl.record(backend="paddle")` are rejected.

Paddle validation is a live preview guard, not a completeness proof for arbitrary Paddle releases.
Each wrapped op emits an independent capture record, replay reconstructs argument templates and
perturbs parents, and a static inventory snapshot pins the wrapped/denied op surface. The preview
rejects in-place mutation, RNG, user-level tensor-derived Python scalar/control escapes, and active
stochastic/training composites. Deterministic eval-mode composites are allowed as coarse boundary
nodes. Same-object no-ops, such as an operation that returns the exact input tensor object, are
recorded as alias annotations rather than cloned value-producing ops.

Paddle leaf gradients are a derived-gradient preview, not true backward capture:

```python
grad_options = tl.backends.paddle.GradOptions(
    loss_fn=lambda output: paddle.sum(output * output),
    input_grad_argnums=(0,),
)
trace = tl.trace(model, x, backend="paddle", grad_options=grad_options)
trace.derived_grads["inputs.0"]
```

Set `GradOptions(intermediate_grads=True, max_intermediate_grads=...)` to request exact saved-op
intermediate derived gradients from a second guarded Paddle AD replay. Only unambiguous
`status == "exact"` records reach `trace.intermediate_derived_grads` and read-only
`op.derived_grad`; `has_backward_pass` remains `False`, and true-backward surfaces still raise.

Paddle `.tlspec` support uses `payload_policy="array_payloads"`: default portable saves persist
forward and derived array payloads and load them back as `paddle.Tensor` values when Paddle is
installed. Paddle `bfloat16` payloads record the logical dtype because NumPy transports them as
`uint16`; load restores the logical Paddle dtype.

## JAX Preview

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

Set `GradOptions(intermediate_grads=True, max_intermediate_grads=...)` to run JAX's separate
zero-tap AD replay for saved op boundaries. The producer uses one all-tap `jax.grad` pass, then an
O(k) oracle checks each attached boundary with an independent boundary-replacement VJP and finite
difference. The hard cap fails explicit requests above `max_intermediate_grads`. Only records whose
provenance has `status == "exact"` are exposed through `trace.intermediate_derived_grads` and
read-only `op.derived_grad`; oracle failures and non-float/control outputs are skipped. If the
producer cannot run on the installed JAX runtime, leaf `trace.derived_grads` remains intact and the
intermediate accessor is empty.

The JAX backend rejects transformed root callables (`jax.jit`, `jax.vmap`, `jax.grad`), root
capture from inside those transforms, unsupported nested-jaxpr primitives, callback effects,
closed-over array constants for raw function-root callables, value-dependent `save=` predicates,
`followed_by`/`preceded_by` window selectors, `intervene=`, `halt=`, streaming, `save_grads=`, and
`tl.record(backend="jax")`. Static-label `save=` selectors such as `tl.func(...)`,
`tl.label(...)`, `tl.contains(...)`, and module selectors are applied after full graph
finalization: unsaved ops keep metadata but drop public activation payloads. `lax.scan`,
`lax.cond`, and capped `lax.while_loop` are unrolled by default with
`jax_control_flow="unroll"` and guarded by `jax_max_control_flow_unroll`. Over-cap
`lax.scan` and `lax.while_loop` fall back to importer-owned graph regions with one boundary
node and per-output projection nodes; forward `custom_vjp_call` is also represented as a
region. These traces report `trace.validation_replay_status.state == "unverified"` after
replayable ops and region seam checks pass. Pass `jax_control_flow="reject"` to preserve the
earlier nested-primitive rejection behavior, or `jax_control_flow="region"` to force supported
scan/while/custom-VJP-forward boundaries to regions. Workarounds are to pass raw functions and explicit params/input leaves to
`tl.trace(..., backend="jax")`, or use the PyTorch backend for value-dependent predicate capture,
intervention, sparse fastlog, and true backward graphs.

JAX `.tlspec` support uses `payload_policy="array_payloads"`: default portable saves persist
forward and derived array payloads and load them back as `jax.Array` values.
Typed JAX PRNG keys round-trip as typed keys rather than legacy integer-key arrays.
Fully addressable single-host sharded arrays are saved as assembled host values; manifest
`codec_metadata` includes versioned `jax_sharding_*` fields plus a reconstructible
`jax_named_sharding` block for `NamedSharding`. The block stores mesh `axis_names`, mesh
`shape`, and `PartitionSpec` as JSON primitives (`null`, strings, and string lists for
multi-axis entries). Default load does not reconstruct that topology; it remains value-only.
To opt in, pass `PayloadLoadHints(jax=JaxPayloadLoadHint(sharding=...))` with a concrete
JAX sharding, or `JaxPayloadLoadHint(reconstruct_sharding=True, platform="cpu")` to rebuild
from metadata when the requested local mesh is available. `map_location` remains the
single-device `str | torch.device` load target and does not accept JAX shardings.
Multi-host or otherwise unaddressable sharded arrays fail closed during save.
`trace.payload_load_status` records load-time materialization state. Runtime-only
`jax_equation_captures` are stripped from portable artifacts, so loaded JAX traces expose
`ValidationReplayStatus` at `trace.validation_replay_status.state == "unavailable"` with reason
`"loaded_trace_runtime_capture_stripped"` instead of reporting a false validation pass or failure.
Live JAX traces, including static-label selectively saved traces, still run real replay validation
using runtime-only replay payloads that are separate from the public saved-activation surface.
Importer-owned JAX regions report `trace.validation_replay_status.state == "unverified"` after
all replayable ops and region seams pass; that status means replay was partial, is available for
inspection, and is not a validation pass.

## tinygrad Preview

Install the optional runtime with `torchlens[tinygrad]`; the preview is pinned to the
`tinygrad>=0.13,<0.14` series and currently accepts `tinygrad==0.13.0` at runtime. For live
payload validation and derived gradients, run with `DEV=PYTHON`.

tinygrad raw callables use `module_identity_mode="function_root"`. Callable object graphs with
discoverable tinygrad module attributes default to `module_identity_mode="object_module"`; pass
`module_identity_mode="function_root"` to force the older root-only surface. Discovery walks object
attributes for callable children that own `Tensor` attributes or are known `tinygrad.nn.*` layer
objects, records shared children under the first address with later aliases, and attributes UOps
from the live module stack observed at construction time. If tinygrad reuses an existing UOp
identity, the UOp keeps its first observed module attribution rather than being duplicated for a
later call.

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
If a future importer marks a region as outside per-op replay, successful replayable checks fold to
`trace.validation_replay_status.state == "unverified"` rather than `passed`; boolean coercion of
that status raises so it cannot be mistaken for a pass or fail.
