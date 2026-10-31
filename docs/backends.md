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
| `mlx` | Technical preview; static-label `intervene=`/`halt=` | Live per-op replay and parent perturbation over the captured (whitelisted) op set | Materialized forward/derived array `.tlspec` payloads | `function_root`, object `object_module` | `trace.derived_grads`; opt-in `trace.intermediate_derived_grads` |
| `jax` | Preview jaxpr-first functional capture | Live per-equation replay and parent perturbation | Materialized forward/derived array `.tlspec` payloads | `function_root`, Equinox/NNX `pytree_module` | `trace.derived_grads`; opt-in `trace.intermediate_derived_grads` |
| `tinygrad` | Preview UOp-snapshot functional capture | Live UOp replay and parent perturbation on `DEV=PYTHON` payloads | Materialized forward/derived array `.tlspec` payloads | `function_root`, object `object_module` | `trace.derived_grads`; opt-in `trace.intermediate_derived_grads` |
| `paddle` | Preview dygraph/eager capture with live forward `intervene=`/`halt=` | Live replay/perturbation plus static inventory guard; corroborated user-intervention carve-out | Materialized forward/derived array `.tlspec` payloads | `function_root`, object `object_module` | `trace.derived_grads`; opt-in `trace.intermediate_derived_grads` |
| `tf` | Preview eager op-callback capture; implemented graph-only FuncGraph static path for compiled/SavedModel entries | Callback self-consistency plus per-op replay/perturbation accounting | Materialized forward array `.tlspec` payloads | `function_root`, Keras/`tf.Module` object `object_module` | `trace.derived_grads`; opt-in `trace.intermediate_derived_grads` (eager entries only) |

Two cross-backend honesty notes apply to every preview row above:

- **`save=` on previews is a presentation filter, not a memory saver.** Preview backends capture
  the full forward first; static-label selectors then null the public payloads of unselected ops
  (retaining hidden copies for live replay validation). Unlike torch's single-pass selective
  capture, peak memory is unchanged. Value-dependent predicates, `followed_by`/`preceded_by`,
  lookback windows, and save shaping reject typed.
- **Preview `tl.validate(...)` may return a status object, not a plain bool.** Partial-coverage
  results return a `ValidationReplayStatus` whose `bool()` raises `TypeError` for non-final
  (`unverified`/`unavailable`) states instead of guessing; branch on `.state`/`.passed`.
- **Auto-routing applies to every preview, not just MLX.** Each registered preview declares a
  `can_handle` detector, so `backend=None` routes a genuine framework model (framework installed,
  no foreign tensors) to its preview automatically. Pass an explicit `backend=` to pin capture.
- **Recurrence and depth.** Every preview backend now groups recurrent calls into multi-pass
  layers through the same neutral grouper torch and JAX use: a reused submodule or repeated
  structurally identical block becomes one N-pass layer (`layer_label:pass` op labels,
  `pass_index`/`num_passes`, `recurrent_ops`), with grouped graph edges rewritten to the
  pass-qualified labels while raw capture labels stay resolvable. The stored
  `recurrence_detection` flag remains the EFFECTIVE value: `True` where the grouper ran (all
  eager preview captures by default; `recurrence_detection=False` opts back into the historical
  single-pass layout), `False` where finalize never grouped — the TF static FuncGraph importer
  keeps `False` because graph node names are one-shot sites. Grouping never weakens the
  validation tripwire: each backend's label-keyed replay sidecars stay keyed to raw capture
  identities, oracles compare in raw-label space, and per-backend tamper tests prove a
  stale-label sidecar FAILS validation rather than silently passing.
  `compute_input_output_distances` (input/output
  hop distances plus ancestor/descendant sets) defaults to `True` on every backend — the same
  torch `CaptureOptions` default — and `=False` disables the flood on every backend, torch and all
  five previews alike.
- **Preview replay validation scope.** Preview `validation_replay` replays each CAPTURED op from
  its declared parents' saved payloads and perturbation-checks the dependency; unlike torch's
  validator it does not independently re-run the model against a ground-truth output, so an op the
  preview failed to wrap is outside the replay denominator entirely. Calls with no perturbable
  tensor argument surface as `unverified`, never as a silent pass.

## Public Option Spine

The public `tl.trace()` signature declares backend-growth options before their non-torch
implementations land. Torch accepts these options as inert cache-key inputs so torch behavior stays
unchanged. JAX now implements the control-flow policy knobs for `lax.scan`, `lax.cond`, and
`lax.while_loop`, and raw `function_root` JAX captures transparently inline pure nested `jax.jit`
calls before applying that control-flow policy. Nested JIT calls with closed constants, effects,
donated inputs, or explicit shardings remain rejected. tinygrad, Paddle, TensorFlow, and
unimplemented option families reject explicit use with `BackendUnsupportedError` until the matching
backend phase implements the capability.

Capability-gated options are fail-closed in both directions: a `False` flag rejects with the
backend's policy message, and a `True` flag admits only through the implementing surface the spec
binds in `BackendSpec.capability_implementations`. A capability flag flipped to `True` on a backend
that registers no implementation raises `BackendCapabilityConformanceError` (at registration and at
the gate) instead of silently admitting and ignoring the option.

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
the wrapped operation, then mutate that value or halt capture at the matching frontier. Paddle
dygraph exposes the same point — the capture wrapper holds each concrete output before the caller
sees it — so the Paddle preview now supports live `intervene=` and `halt=` (see the Paddle Preview
section). The other previews do not expose that point. JAX
first builds a jaxpr over tracers and TorchLens labels are finalized after interpretation; a
low-level primitive interpreter can replace a primitive by position, but that is not the public
static-label `intervene=`/`halt=` contract. tinygrad builds a lazy UOp graph and concrete values
appear only after `Tensor.realize()`/`Tensor.item()`, with no stable public API for replacing one
internal UOp and rebuilding all descendants.
TensorFlow eager `op_callbacks` are read-only in the supported Keras-3 / TF>=2.16 runtime
(re-verified on TF 2.21: returned replacement outputs are ignored), so TensorFlow interventions run
through a writable wrap layer on top of the untouched callback capture spine — see the TensorFlow
section below.

For that reason, JAX and tinygrad support static-label `save=` only, and their `intervene=` and
`halt=` are rejected with typed backend errors instead of false partial traces or validation
passes. Value-dependent `save=` predicates stay rejected on every preview: preview `save=` is a
post-finalization payload filter, not single-pass selective capture. Live selective-save traces
still run real replay validation through runtime-only hidden payloads; loaded traces report
replay unavailable when those runtime captures were stripped.

Paddle supports static-label `trace(intervene=tl.when(...))` and `trace(halt=...)` on the
forward-only eager capture path (see the Paddle preview section). MLX is the other eager
exception: its wrapper sees each concrete lazy output before the caller, so MLX additionally
supports static-label `trace(intervene=tl.when(...))` and `trace(halt=...)` (see the MLX preview
section). TensorFlow supports static-label `intervene=` for eager entries through the writable
wrap layer with fail-closed site reachability; TF `halt=` keeps its typed refusal (capture-path
scope). Value-dependent predicates stay rejected on all three — predicate-time values would
require eager materialization that poisons their execution models. MLX supports static-label
`save=` for `tl.func`, `tl.label`, `tl.module`, `tl.in_module`, `tl.contains`, and boolean
composites of those.

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
SavedModel-style signatures, and `.predict()` are detected separately and route to the implemented
FuncGraph static importer (`torchlens/backends/tf/funcgraph.py`): a walk/prune pass with a tiered
concrete-value ladder, opaque regions recorded honestly as unverified rather than fabricated, and
zero module attribution in graph-only mode.

Static-label `save=` selectors are applied after full graph finalization, with unsaved ops retaining
metadata and dropping public payloads. TensorFlow validation is non-vacuous but scoped to the
preview: callback self-consistency catches unresolved producers, pure allowlisted ops replay through
`tf.raw_ops` with parent perturbation, and the replay status reports replayed, pure-unverified, and
effect-region counts. A status of `unverified` is partial validation, not a pass.

TensorFlow `.tlspec` support uses `payload_policy="array_payloads"` for dense numeric/bool forward
payloads. `bfloat16` records logical dtype metadata because NumPy transports it as `uint16`; string,
resource, variant, and composite payloads fail closed.

TensorFlow leaf gradients are a derived-gradient preview, not true backward capture:

```python
grad_options = tl.backends.tf.GradOptions(
    loss_fn=lambda output: tf.reduce_sum(output),
)
trace = tl.trace(model, x, backend="tf", grad_options=grad_options)
trace.derived_grads["inputs.0"]
```

The mechanism is one auxiliary forward replay under a persistent `tf.GradientTape` with the
op-callback session installed: input tensors and module variables are watched explicitly, the
replay raw output must match the captured raw output or the surface refuses with a typed
divergence error, and one `tape.gradient` call produces leaf gradients for floating inputs and
`trace.params` variables (mirrored onto `param.grad`). Set
`GradOptions(intermediate_grads=True, max_intermediate_grads=...)` to additionally
`tape.watch` every float op output live at callback time and attach exact op-level records. Only
unambiguous `status == "exact"` matches (keyed on stream position, op type, parents, and module
stack) reach `trace.intermediate_derived_grads` and read-only `op.derived_grad`;
`has_backward_pass` remains `False`, and true-backward surfaces still raise. `grad_options`
requires an eager live-capture entry: graph-only FuncGraph captures (compiled `Model.call`,
`tf.function` roots, SavedModel signatures) refuse it typed because the derived-gradient replay
cannot run inside a frozen graph.

TensorFlow supports static-label `intervene=` on eager entries through a two-level writable layer
on top of the read-only op-callback spine:

```python
ablated = tl.trace(model, x, backend="tf",
                   intervene=tl.when(tl.func("relu"), tl.zero_ablate()))
```

`tl.module(...)`/`tl.in_module(...)` conditions substitute Keras/`tf.Module` call outputs at the
module boundary (object-module attribution required); `tl.func`/`tl.label`/`tl.contains`
conditions substitute returns of a curated registry of python entry points (`tf.nn`/`tf.math`
core) that Keras-3 eager execution flows through. Replacement values are produced by real ops the
callback records, so the trace stays honest by construction: downstream records consume the
substituted values and the site op keeps its original payload plus a fire record. Site resolution
FAILS CLOSED: a selector matching captured ops whose calls never passed through the wrap layer
(gen_ops/C++ paths) raises a typed unreachable error instead of silently not intervening. Curated
helper actions are `zero_ablate`, `scale`, and `add`, plus plain callables; replacements must
preserve shape and dtype. Conditions must be static selectors; `direction="backward"`,
value-dependent callable conditions, shape-changing replacements, graph-only entries, and
combining `intervene=` with `grad_options=` refuse typed. `halt=` and `recipes=` keep typed
refusals inside the capture path.

TensorFlow true backward capture, `halt=`, fastlog/`tl.record()`, streaming, and value-dependent
predicates remain deferred. These surfaces raise typed backend errors instead of silently
producing partial traces.

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

MLX supports static-label live interventions and halt. `trace(intervene=tl.when(static_selector,
action))` substitutes matched output arrays at the wrapper boundary — the replacement genuinely
flows into downstream ops — and `trace(halt=static_selector)` finalizes an honest partial trace at
the matched frontier (`trace.halted`, `trace.halt_reason`; parameter accounting is restricted to
the captured ops). Static selectors are `tl.func`, `tl.label`, `tl.module`, `tl.in_module`,
`tl.contains`, and boolean composites. Supported actions are `tl.zero_ablate()`, `tl.scale(...)`,
`tl.add(scalar | mx.array)`, `tl.clamp(...)`, `tl.mean_ablate()` (self form), `tl.replace_with(...)`,
and callable transforms written against `mx.array`; other helpers build torch-only hooks and refuse
by name. Hooks must preserve shape and dtype; violations refuse typed at fire time. Replay
validation treats declared interventions as genuine user substitutions: the oracle recomputes the
raw producer output, re-applies the declared hook, and compares — stripping or forging the
declaration fails coverage against the emit-time inventory fingerprint. `mx.compile`/`mx.grad`/
`mx.vmap` traced-transform wrappers refuse at capture entry, and a compiled attribute on the model
ceilings the capture with `capture_verified=False` (`mlx_compiled_attribute_not_logged`).

MLX rejects `tl.output(...)`, `tl.where(...)`,
`tl.followed_by(...)`, `tl.preceded_by(...)`, value-dependent predicates, `recipes=`,
backward-direction interventions, streaming, `save_grads=`, `backward_ready=True`, and
`tl.record(backend="mlx")`. Portable MLX saves
materialize forward and derived array payloads and load them back as `mlx.core.array` values when the
MLX runtime is installed; loaded MLX traces report replay validation as unavailable rather than
as a pass when runtime replay captures were stripped.

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
selectors are applied after full graph capture; value-dependent `save=` predicates, streaming,
`save_grads=`, `backward_ready=True`, and `tl.record(backend="paddle")` are rejected.

Paddle supports live forward `intervene=` and `halt=` because dygraph capture is eager: the
wrapper holds each wrapped call's concrete output before the caller sees it, so predicates run
with real values and replacements flow into downstream computation.

```python
ablated = tl.trace(
    model,
    x,
    backend="paddle",
    intervene=tl.when(tl.func("functional.relu"), tl.zero_ablate()),
)
partial = tl.trace(model, x, backend="paddle", halt=tl.func("functional.relu"))
assert partial.halted
```

Intervention semantics on Paddle are honest by construction. The intervened op keeps its real
identity; its recorded payload is the replacement that actually flowed downstream, stamped with
`op.intervention_replaced` and hook-minted `FireRecord` entries, and replay validation checks the
original callable against the recorded pre-hook value through a narrow corroborated carve-out — a
replacement value presented as captured-native FAILS validation. Builtin helpers with Paddle
adapters are `tl.zero_ablate`, `tl.scale`, `tl.add`, and `tl.replace_with`; other helpers refuse
typed, and raw callables receiving/returning `paddle.Tensor` are supported. Hook internals run
outside capture logging (they do not appear as graph ops). Decisions are forward-only
(`direction="backward"` refuses), intervention predicates and `halt=` may be value-dependent
callables (the eager `RecordContext` carries real `tensor_requires_grad` / `is_scalar_bool` /
`bool_value`), `halt=` finalizes an honest partial trace at the matched frontier without executing
past it, and `recipes=` attaches per-trace facet recipes. `grad_options` cannot combine with
`intervene=`/`halt=` because the derived-gradient replay re-runs the forward without
interventions.

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
assert trace.validate_forward_pass([]) is True
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
assert trace.validate_forward_pass([fn(Tensor([1.0, -2.0, 3.0]))]) is True
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
