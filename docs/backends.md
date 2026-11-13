# Backend Support

TorchLens is built for PyTorch and also supports five other frameworks: JAX, TensorFlow, MLX,
PaddlePaddle and tinygrad. They have fewer features than PyTorch; the table below says which.
Every backend fills the same contract,
`BackendSpec`, so the same `tl.trace` call works on each. With `backend=None` a torch model
runs on the PyTorch backend, and a model from one of the other frameworks (framework installed,
no foreign tensors in the inputs) is routed to its backend automatically; pass `backend="jax"`,
`"tf"` (or `"tensorflow"`), `"mlx"`, `"paddle"` or `"tinygrad"` to pin one.

## What works on each framework

This table says which of TorchLens's main tasks work today on a small model from each framework. Every cell comes
from running the task, not from reading the code, and `tests/test_backend_feature_table.py`
reruns the Yes and Partial cells so the table cannot quietly go stale.

| Framework | Select | Extract | Inspect | Visualize | Intervene | Compare | Debug | Save and load | Gradients |
|---|---|---|---|---|---|---|---|---|---|
| PyTorch | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| JAX | Partial (1) | Partial (3) | Yes | Partial (4) | Partial (5) | Yes | Partial (7) | Yes | Yes |
| TensorFlow | Partial (1) | Partial (3) | Yes | Partial (4) | Partial (6) | Yes | Partial (7) | Yes | Yes |
| MLX | Partial (1) | Partial (3) | Yes | Partial (4) | Partial (6) | Yes | Partial (7) | Yes | Yes |
| PaddlePaddle | Partial (1) | Partial (3) | Yes | Partial (4) | Partial (8) | Yes | Partial (7) | Yes | Yes |
| tinygrad | Partial (1) | Partial (2) | Yes | Partial (4) | Partial (6) | Yes | Partial (7) | Yes | Yes |

1. You can select operations by function, by module, or both. Selecting by influence (which
   later units a given unit can reach) with `method="gradient"` works on every framework in both
   directions: TorchLens replays the captured operations between the two units and differentiates
   the replay with the framework's own autodiff, after checking the replay reproduces the
   capture. A trace without runtime replay records (a loaded trace) refuses with
   `gradient_replay_unavailable`. On PaddlePaddle, tinygrad, and MLX graphs with convolutions (no
   forward-mode rule in MLX 0.26), a projective probe runs one reverse probe per target element
   and refuses above 4096 target elements. Geometric influence selection works only on PyTorch;
   on the other frameworks it refuses with `influence_geometry_unavailable` or returns only a
   whole-input upper bound.
2. You can save only the operations you choose; values of unselected operations are never
   computed into a buffer. Stopping the forward pass early (`stop_after=`) and fast repeated
   recording (`tl.record`) raise an error.
3. You can save only the operations you choose, stop the forward pass early and record
   repeatedly with `tl.record`, which keeps values only for the selected operations. Streaming
   payloads to disk during `tl.trace` (`storage=`) works only on PyTorch and PaddlePaddle, and on
   PaddlePaddle it cannot be combined with `save=` or structure-only capture. On JAX `save=` is
   applied after the full capture.
4. The forward graph draws, including the collapsed overview. There is no backward pass to draw.
   On MLX this is a framework limit: MLX builds its backward pass inside C++, so TorchLens
   cannot list the backward operations. Gradient values for every recorded operation are
   available, and gradient edits work.
5. Edits work during a new run and on a finished trace (`trace.fork().do(...)`), on equation
   outputs chosen by function, by label or by `tl.in_module(...)`, with the six built-in helpers
   or a callable, and `stop_after=` stops the run at the first matching equation. With
   `grad_options=`, derived gradients follow the edited program and gradient edits at
   `tl.grad(...)` sites apply. Edits at a module's output (`tl.module(...)`), derived gradients
   of a stopped capture and patching in activations from another run raise an error.
6. Edits work during a new run and on a finished trace (`trace.fork().do(...)`), on operations
   chosen by function, by module or by a condition on their values (`tl.where` or a plain
   callable), with the six built-in helpers (ten on MLX) or a callable. Edits combine with
   `grad_options=`, and gradient edits at `tl.grad(...)` sites apply to derived gradients.
   Patching in activations from another run raises an error that names the framework. On
   tinygrad, `stop_after=`, edits at a module's output (`tl.module(...)`) and `trace.do` with a
   `tl.grad(...)` site raise an error; on MLX, a `tl.module(...)` edit with `grad_options=`
   raises an error.
7. The NaN finder and the dead-unit counter work. Guessing a model's input shape works on
   PyTorch, Paddle, tinygrad, TensorFlow, MLX and JAX; for a plain function it reports a working
   or ambiguous shape, never a determined one. On a convolution output the dead-unit counter
   needs the channel axis passed as `dim`.
8. Edits work during a new run and on a finished trace (`trace.fork().do(...)`), on operations
   chosen by function, by module or by a condition on their values, with the six built-in helpers
   or a callable. Gradient edits at `tl.grad(...)` sites apply to derived gradients, unless the
   gradient passes through an activation edit or another operation the replay cannot rerun, which
   raises an error naming it. Patching in activations from another run raises an error that names
   the framework.

Compare means comparing two runs on the same framework, for example two inputs through one
model. Comparing a run from one framework with a run from another (a PyTorch trace against an
MLX trace of the same model) refuses with a typed error: operation labels and order do not
correspond across frameworks, so pairing the runs by label would report differences that are
not there. Module-level comparison across frameworks is planned; see
[Comparing runs from different backends](reference/debug.md#comparing-runs-from-different-backends).

Checked with PyTorch 2.13, JAX 0.10.2 (Flax 0.12.8), TensorFlow 2.21 (Keras 3.15), MLX 0.26.3,
PaddlePaddle 3.3.1 and tinygrad 0.13.0, the versions each backend's test job pins.

Torch's `save=` is single-pass selective capture and lowers peak memory. On JAX `save=` chooses
which payloads a finished trace keeps and does not lower peak memory: JAX captures the full
forward first and then drops the public payloads of unselected ops (hidden copies stay for
replay validation). The other backends decide during capture. TensorFlow keeps a snapshot only
for selected ops and model inputs and frees intermediates as the model drops them. On MLX and
PaddlePaddle, static `save=` selectors move unselected payloads out of RAM during capture into a
private temporary directory that replay validation reads, removed when the trace is collected;
selectors that cannot be decided during capture, such as value-dependent ones, still apply after
it. On tinygrad the values of unselected ops are never computed into a buffer. Through
`tl.trace`, every non-PyTorch backend also takes value-dependent `save=` predicates (a plain callable or `tl.where`),
`tl.preceded_by`, `candidate & tl.followed_by(...)` with `lookback`, and save shaping
(`activation_transform` with or without `save_raw_activations`, `output_device`,
`detach_saved_activations`, `layer_visualizers`); op reprs show value statistics. `save_args`
still refuses, because these backends record no call arguments. Direct backend-object entries
keep the older refusals, and the TF and MLX direct entries take only static `save=`. On tinygrad,
selectors match tinygrad's lowered ops (a matmul lowers to reshape, expand, mul and sum), so a
structural selector can keep a different set of ops than on PyTorch.

## Backend protocol

Every backend, torch included, fills one contract, `tl.backends.BackendProtocol`: a framework
requirement, a capability table, capture, replay validation, a reference output computed with no
TorchLens wrapper installed, host copies of values, a realize-once hook for lazy frameworks and a
closed set of refusal codes. `BackendSpec.protocol()` returns a backend's protocol object. The
`CaptureBackend` adapter in `torchlens/backends/_protocol.py` is torch's internal capture seam,
not the backend contract.

| Backend | Materialization | Page |
|---|---|---|
| `torch` | eager | [PyTorch](backends/torch.md) |
| `tf` | eager | [TensorFlow](backends/tf.md) |
| `jax` | eager | [JAX](backends/jax.md) |
| `mlx` | lazy | [MLX](backends/mlx.md) |
| `paddle` | eager | [Paddle](backends/paddle.md) |
| `tinygrad` | lazy | [tinygrad](backends/tinygrad.md) |

No backend carries a maturity label: what each one supports is stated by the feature table above,
its page and its shipped scorecard. Each page has a runnable tutorial, a can and cannot table and
the backend's framework requirement (supported version range, the version the per-change CI leg pins, and the install
extra). The repository check `scripts/checks/backend_docs_lockstep.py` holds this table to the
registry and each page's can and cannot table to the backend's shipped capability summary.

Rules every backend follows:

- **Typed refusals.** An unsupported capability, option or input refuses with
  `BackendUnsupportedError` before any framework call with side effects; `exc.code` is a member of
  `tl.backends.REFUSAL_CODES`. A backend never returns a silent wrong result: where it cannot
  capture correctly it refuses, or marks the trace `provenance.verified=False` with the reason code
  and a warning.
- **Constants.** A tensor the forward reads from outside the traced call is recorded as a
  parentless `constant` op with a saved payload, listed as a parent by the op that read it. A
  literal producer (`ones`, `fill` inside the forward) keeps its own `func_name` and has no parents.
  Validation reads a constant's payload as a leaf (`constant_producer_leaf`,
  `constant_parent_payload`); a constant with no readable buffer refuses `constant_unrealizable`.
- **Realize-once.** On lazy frameworks (MLX, tinygrad) the forward runs once to its outputs and
  every recorded value is evaluated in one framework evaluation before any payload is read, so
  random-number consumption equals one eager forward. Random numbers drawn inside the traced
  forward refuse with `rng_state_in_graph`, because replay would draw them again.
- **Validation reasons.** `validate` returns a `ValidationStatus` whose per-op decisions name a
  reason from `tl.backends.VALIDATION_REASONS`; a failure names the first offending op. On the
  non-PyTorch backends `tl.validate(...)` may return that status object rather than a bool: `bool()`
  of a non-final state (`unverified`, `unavailable`) raises `TypeError`, so branch on `.state` or
  `.passed`. `unverified` means replay was partial; it is not a pass.

### Scorecards and `tl.report.doctor()`

The shared conformance suite (`tests/conformance/` in a checkout) runs the same case ids over
every backend and records each cell as `passed`, `failed`, `unsupported` (a typed refusal with a
registered code) or `not_run`; every cell stays in the denominator. Each backend ships the result
as `torchlens/backends/<name>/scorecard.json`, written by
`python -m tests.conformance.scorecard --backend <name> --write` on the pinned framework version
and checked by `--check` in its CI leg. The capability summary reads each `BackendCapabilities`
flag as `supported`, `unsupported` (the request refuses with the contracted code) or `untested`
(refusal not yet proven by a passing cell; the pages render it as "not yet proven").

`tl.report.doctor()` shows one row per registered backend, key `backend_scorecard:<name>`, read
from the shipped record only (no framework import). Its status is `pass` (durable tier reached, no
failed cell), `known_broken` (a cell failed), `scope` (framework not installed here) or
`not_tested` (no record ships); its details read
`<passed>/<denominator> cells passed, <unsupported> unsupported, <failed> failed; pinned <pin>,
installed <version or absent>, supported [<minimum>, <ceiling>)`, and the suggestion names the
install extra or the first failed cell. The severity is the same on every backend: `ok` for
`pass`, `warning` for `known_broken` (a failed cell in a shipped measurement is not a fault in
your environment, so it never turns the report red) and `info` otherwise.

### Graduation

`python -m tests.conformance.scorecard --backend <name> --graduation` checks six items. They are
evidence of how completely the shared suite covers a backend, embedded in its scorecard; they set
no label:

| Item | Criterion |
|---|---|
| G1 | passes the conformance suite torch passes; every unsupported capability is a typed refusal |
| G2 | replay validation works on it, including on a saved and reloaded trace |
| G3 | at least two pretrained families from its own ecosystem are captured and validated |
| G4 | a per-change CI leg installs the pinned framework version inside a stated supported range |
| G5 | its page has a runnable tutorial and a can and cannot table equal to its capability summary |
| G6 | its spellings are frozen: no unstable marker in its package, a glossary entry for every public name |

## Live edits and early stop

PyTorch capture can read a predicate while a concrete value flows through the wrapped operation,
then replace the value (`intervene=`) or stop the forward (`stop_after=`). Which backends
expose that point decides what they support:

- **Paddle and MLX** see each concrete output in their wrapper before the caller does, so both
  support live `intervene=tl.when(...)` and `stop_after=` on a new run, with static selectors or
  value-dependent conditions (`tl.where`, plain callables).
- **TensorFlow** records through read-only `op_callbacks` (a returned replacement is ignored, re-checked
  on TF 2.21), so `intervene=` and `stop_after=` run through a writable wrap layer on top of the
  callback capture, with static selectors or value-dependent conditions. Graph-only entries
  (`tf.function` bodies, SavedModels) refuse both.
- **JAX** evaluates its recorded jaxpr one equation at a time with concrete values, and the
  interpreter can replace an equation's outputs. That is the edit point: `intervene=` replaces
  equation outputs there, and `stop_after=` stops interpreting at the first matching equation.
- **tinygrad** builds a lazy graph whose values appear only after `realize()`. `intervene=`
  runs the forward twice: the first run matches the condition (a static selector, a `tl.where`
  leaf or a plain callable over the op context) against the finished ops, and the second applies
  the edit lazily to the Tensor calls that returned them, so the forward still realizes once.
  `stop_after=` refuses with a typed error, because a lazy forward builds its whole graph before
  anything runs.

Editing a finished trace (`trace.fork().do(...)`) works on every backend: the edit is applied at
the site and each downstream op is re-run once through the backend's replay adapter. Patching in
activations from another run (`tl.patch_from`) refuses on every non-PyTorch backend with an error
that names the framework.

## Backend options

`tl.options.CaptureOptions` carries the options the non-PyTorch backends read, declared before every
backend implements them (`jax_control_flow`, `jax_max_control_flow_unroll`, `module_identity_mode`,
`payload_policy`). Torch accepts them as inert cache-key inputs so torch behavior stays unchanged.
`facets=` is a `tl.trace` keyword on every backend. An option a backend does not implement refuses
with `BackendUnsupportedError`. The gate is fail-closed in both directions: a `False` capability
flag rejects with the backend's message, and a `True` flag admits only through the implementation
the spec binds in `BackendSpec.capability_implementations`; a flag flipped to `True` with no
implementation raises `BackendCapabilityConformanceError` at registration and at the gate.

| Option | What it controls |
|---|---|
| `jax_control_flow` | JAX control-flow boundary or unroll policy; `lax.scan`/`cond`/`while_loop` support `unroll` or `reject`; `region` forces supported scan/while/custom-VJP-forward regions |
| `jax_max_control_flow_unroll` | JAX control-flow unroll safety limit |
| `module_identity_mode` | Backend module-mode selection; JAX supports raw `function_root` and Equinox/Flax NNX `pytree_module`; tinygrad, MLX, Paddle, and TensorFlow support raw `function_root` and callable object `object_module` |
| `payload_policy` | Backend payload codec/materialization policy; JAX/tinygrad/MLX/Paddle/TF use `array_payloads` |
| `facets` | Per-trace facet recipes, backend-neutral: every backend (torch, JAX, tinygrad, MLX, Paddle, TF) snapshots them onto `trace.facet_functions` through one shared helper; a non-callable refuses `trace_option_unsupported` |

Two `CaptureOptions` fields behave the same on every backend: `recurrence_detection=True` (the
default) groups repeated calls of one module into passes (`layer_label:pass` op labels,
`pass_number`, `num_passes`, `recurrent_ops`), and `recurrence_detection=False` keeps one record per
call; the TensorFlow static importer keeps `False` because graph node names are one-shot sites.
`compute_input_output_distances=True` (the default) records input and output hop distances plus
ancestor and descendant sets; `False` disables that walk.

## TensorFlow

Install with `torchlens[tf]` or `torchlens[tensorflow]`; the backend targets Keras 3 on TensorFlow
`>=2.16` and requires `keras.backend.backend() == "tensorflow"` for Keras models.

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

What it does:

- Eager live capture is the primary path: the real `model(*args, **kwargs)` runs under
  `tensorflow.python.framework.op_callbacks`, each eager op is recorded with real values and the
  branch actually taken, and Keras or `tf.Module` call frames are attributed as modules
  (`module_identity_mode="object_module"`); raw callables use `function_root`.
- Graph-only entries (compiled `Model.call`, `tf.function` roots, SavedModel signatures,
  `.predict()`) route to the FuncGraph static importer (`torchlens/backends/tf/funcgraph.py`): a
  walk-and-prune pass with a tiered concrete-value ladder, opaque regions recorded as unverified,
  and no module attribution.
- Validation is scoped to the backend: callback self-consistency catches unresolved producers,
  allowlisted pure ops replay through `tf.raw_ops` with parent perturbation, and the status reports
  replayed, pure-unverified and effect-region counts. `unverified` is partial validation, not a pass.
- `save=` decides each op's payload during capture: only selected ops and model inputs keep a
  snapshot, op records drop their live tensors once their edges are recorded, and unsaved ops
  keep their metadata. `.tlspec` saves use `payload_policy="array_payloads"` for dense numeric
  and bool payloads; `bfloat16` records its logical dtype (NumPy transports it as `uint16`); string,
  resource, variant and composite payloads fail closed.
- `intervene=` on eager entries: `tl.module(...)` and `tl.in_module(...)` substitute Keras or
  `tf.Module` call outputs at the module boundary (object-module attribution required);
  `tl.func`, `tl.label` and `tl.label_contains` substitute the returns of a curated registry of
  `tf.nn` and `tf.math` entry points. A selector matching ops whose calls never passed through the
  wrap layer (gen_ops or C++ paths) raises a typed unreachable error rather than silently not
  intervening. Actions: the six built-in helpers (`tl.zero_ablate`, `tl.scale`, `tl.add`,
  `tl.clamp`, `tl.mean_ablate`, `tl.replace_with`) and plain callables; replacements keep shape
  and dtype. Value-dependent conditions (`tl.where`, plain callables) work as on PyTorch, and
  `stop_after=` stops an eager forward at the matched op. On the TensorFlow conformance roster the
  wrap layer reaches 2173 of 2174 recorded ops (variable reads and unary minus included);
  `tf.raw_ops.*` calls and ops inside a `tf.function` body stay read-only. `intervene=` and
  `stop_after=` combine with `grad_options=` (the derived gradients are those of the edited or
  halted program), and `tl.grad(...)` sites edit derived gradients at capture and through
  `trace.do(...)`. A finished trace takes edits through `trace.fork().do(...)`. Shape-changing
  replacements and graph-only entries refuse.
- Each op records the user source lines that ran it, as on PyTorch (`op.source.frames`;
  `CaptureOptions(save_source_context=True)` loads the source text).

```python
ablated = tl.trace(model, x, backend="tf", intervene=tl.when(tl.func("relu"), tl.zero_ablate()))
```

- `tl.record(..., backend="tf")` runs the eager forward with lean per-op events and keeps values
  only for the selected ops; graph-only callables refuse.
- Not available: true backward capture (`backward_ready=True`, `save_grads=`) and streaming
  (`storage=`); each refuses with a typed error.

Leaf gradients are derived gradients, not backward capture:

```python
from torchlens.options import CaptureOptions

grad_options = tl.backends.tf.GradOptions(
    loss_fn=lambda output: tf.reduce_sum(output),
)
trace = tl.trace(model, x, backend="tf", capture=CaptureOptions(grad_options=grad_options))
trace.autograd.derived_grads["inputs.0"]
```

One auxiliary forward replays under a persistent `tf.GradientTape` with the callback session
installed; inputs and module variables are watched explicitly, the replay output must match the
captured output or the call refuses with a typed divergence error, and one `tape.gradient` gives
gradients for floating inputs and `trace.params` variables (mirrored onto `param.grad`).
`GradOptions(intermediate_grads=True, max_intermediate_grads=...)` also watches every float op
output live and attaches exact op-level records: only unambiguous `status == "exact"` matches
(keyed on stream position, op type, parents and module stack) reach
`trace.autograd.intermediate_derived_grads` and read-only `op.autograd.derived_grad`. `has_backward_pass`
stays `False`. Graph-only captures refuse `grad_options`, because the replay cannot run inside a
frozen graph.

## MLX

Install with `torchlens[mlx]` (`mlx>=0.26,<0.27`). MLX is an eager-dispatch backend with lazy
arrays: the forward runs once and every recorded value is realized in one `mx.eval` before any
payload is read.

```python
import mlx.core as mx
import mlx.nn as nn
import torchlens as tl

model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
x = mx.ones((1, 3))
trace = tl.trace(model, x, backend="mlx")
print(trace.summary())
```

What it does:

- `mlx.nn.Module` roots default to `module_identity_mode="object_module"` through MLX's own
  `named_modules()`; shared module aliases get one primary address plus `all_addresses`. Pass
  `module_identity_mode="function_root"` for a root-only module surface. Modules kept in a Python
  list keep addresses such as `layers.0`; the list, tuple or dict holding them gets its own module
  record, as a torch `ModuleList` does.
- Array operators (`a + b` logs `add`, `v[0]` logs `getitem`), array methods and `v.T` are
  captured. An output no wrapped operation produced (an `mx.*` function outside the wrapped
  surface, an `mx.array(...)` built inside the forward) warns and ceilings the capture with
  `provenance.verified=False` (`unwrapped_op_in_capture`), or refuses with that code when the caller
  asks for a refusal. `mx.compile`, `mx.grad` and `mx.vmap` wrappers refuse at capture entry; a
  compiled attribute on the model ceilings the capture (`mlx_compiled_attribute_not_logged`).
- Static-label `save=`: `tl.func`, `tl.label`, `tl.module`, `tl.in_module`, `tl.label_contains`
  and boolean composites with `&`, `|` and `~`. A selector decided during the forward keeps the
  selected payloads in memory; unselected values are spilled to a temporary directory for
  validation and removed when the trace is collected, so peak memory falls. Saved payloads stay
  live MLX arrays and round-trip through `.tlspec` with `payload_policy="array_payloads"`; a
  loaded MLX trace reports replay validation as unavailable, not as a pass, when its runtime
  captures were stripped.
- `capture=CaptureOptions(structure_only=True)` runs the real forward and keeps the graph with no
  payloads; value validation refuses.
- `tl.record(..., backend="mlx")` emits lean per-op events and keeps values only for the selected
  ops; it takes `stop_after=`, and `storage=` writes the selected values to disk with bounded
  held bytes.
- Live `intervene=tl.when(selector, action)` substitutes matched outputs at the wrapper
  boundary, and `stop_after=selector` finalizes a partial trace at the matched frontier
  (`trace.is_stopped`, `trace.stop_reason`; parameter accounting covers the captured ops only).
  `tl.module(...)` fires once at the call's exit on its output arrays; a union of `tl.module` terms
  works, and mixing `tl.module` with another selector kind refuses before the forward runs.
  Actions: `tl.zero_ablate()`, `tl.scale(...)`, `tl.add(scalar | mx.array)`, `tl.clamp(...)`,
  `tl.mean_ablate(dim=None)`, `tl.replace_with(...)`, `tl.steer(...)`, `tl.project_onto(...)`,
  `tl.project_out(...)`, `tl.clip_norm(...)` and callables written against `mx.array`; other helpers
  refuse with `interventions_unsupported`. The same helper table applies on a finished trace
  (`trace.fork().do(site, helper)`) and in the gradient rerun. Hooks keep shape and dtype.
  Replay validation re-applies the declared hook to the recomputed
  producer output and compares; a stripped or forged declaration fails validation. The selector
  may be static or a value-dependent condition (`tl.where(...)`, a plain callable); a module
  selector combined with a non-module condition refuses, and `ctx.tensor_requires_grad` raises
  `MLXValueUnavailableError` because MLX arrays carry no gradient flag.
- `grad_options=` combines with `intervene=` and `stop_after=` on op-level sites: the derived
  gradients are those of the edited program (TorchLens reruns the model under
  `mx.value_and_grad` with the edits applied and refuses if the rerun does not reproduce the
  capture), and a stopped capture takes the loss at the stop frontier. `tl.grad(site)` edits the
  gradient flowing back through a forward op and needs `grad_options=`; a `tl.module(...)` edit
  with `grad_options=` refuses with `intervention_with_derived_grads_unsupported`.
- Each op records the user source lines that ran it, as on PyTorch (`op.source.frames`,
  `save_source_context`, `source_context_lines`).
- Refused: `tl.model_output(...)`, `intervene=` as a raw decision callable (not `tl.when`),
  `tl.grad` of a `tl.module(...)` site or of an autograd node, streaming through
  `tl.trace(storage=...)`, `save_grads=` and `backward_ready=True`. MLX builds its backward pass
  inside C++, so TorchLens cannot list the backward operations (a framework limit); gradient
  values for every recorded operation are available, and gradient edits work.

Leaf gradients are derived gradients, not backward capture:

```python
from torchlens.options import CaptureOptions

grad_options = tl.backends.mlx.GradOptions(
    loss_fn=lambda output: mx.sum(output * output),
    input_grad_argnums=(0,),
)
trace = tl.trace(model, x, backend="mlx", capture=CaptureOptions(grad_options=grad_options))
trace.autograd.derived_grads["inputs.0"]
```

For `mlx.nn.Module` roots TorchLens passes `model.parameters()` to `mx.value_and_grad` and rebinds
with `model.update(params)` during the rerun; the rerun output must match the captured forward
within dtype tolerance before records are exposed. `GradOptions(intermediate_grads=True,
max_intermediate_grads=...)` runs one more `mx.value_and_grad` replay with identity taps installed
by the MLX wrappers (capture logging off, so module call counts do not grow); records are matched
by structural signature, ambiguous groups are skipped, and each public record must pass an
independent replacement-gradient and perturbation check before it reaches
`trace.autograd.intermediate_derived_grads` and `op.autograd.derived_grad`. `op.autograd.grads` and
`trace.saved_grad_ops` stay true-backward-only. Full page: [MLX](backends/mlx.md).

## Paddle

Install with `torchlens[paddle]` (`paddlepaddle>=3.3,<3.4`). Paddle capture is dygraph (eager) only.

```python
import paddle
import paddle.nn as nn
import torchlens as tl

model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
x = paddle.ones([1, 3], dtype="float32")
trace = tl.trace(model, x, backend="paddle")
```

What it does:

- `paddle.nn.Layer` roots get the object module hierarchy when TorchLens can inspect the layer
  tree; raw callables use `function_root`.
- Static `save=` selectors decide during capture: unselected payloads leave RAM and replay
  validation reads them back from a private spool, so peak memory falls (value-dependent selectors
  still apply after capture). `capture=CaptureOptions(structure_only=True)` runs the real forward
  with no payloads; validation refuses typed. `storage=tl.to_disk(path)` streams payloads into a
  bundle during capture (not combined with `save=` or structure-only). `tl.record(backend="paddle")`
  works. Refused: `save_grads=` and `backward_ready=True`.
- Live `intervene=` and `stop_after=`: the wrapper holds each concrete output before the caller
  sees it, so predicates run with real values and may be callables (the eager `RecordContext`
  carries `tensor_requires_grad`, `is_scalar_bool` and `bool_value`), and replacements flow into
  downstream computation. The intervened op keeps its identity; its recorded payload is the
  replacement that flowed downstream, stamped with `op.is_edited` and `FireRecord` entries, and
  replay validation checks the original callable against the recorded pre-hook value through a
  narrow corroborated carve-out, so a replacement presented as captured-native fails validation.
  Actions with Paddle adapters: `tl.zero_ablate`, `tl.scale`, `tl.add`, `tl.clamp`,
  `tl.mean_ablate` (self form), `tl.replace_with` and raw callables on `paddle.Tensor`; other
  helpers refuse. Hook internals do not appear as graph ops. Gradient sites (`tl.grad(...)`) are
  armed on the finished trace and applied to derived gradients; `stop_after=` finalizes a partial
  trace without executing past the frontier; `facets=` works; with `grad_options`, derived
  gradients differentiate the edited or halted program.
  `tl.module(...)` fires once at the call's exit; a union of `tl.module` terms works, mixing kinds
  refuses.

```python
ablated = tl.trace(
    model,
    x,
    backend="paddle",
    intervene=tl.when(tl.func("relu"), tl.zero_ablate()),
)
partial = tl.trace(model, x, backend="paddle", stop_after=tl.func("relu"))
assert partial.is_stopped
```

- Validation is a live guard, not a completeness proof for arbitrary Paddle releases: each wrapped
  op emits an independent record, replay rebuilds argument templates and perturbs parents, and a
  static inventory snapshot pins the wrapped and denied op surface. The backend rejects in-place
  mutation, RNG, tensor-derived Python scalar and control escapes, and active stochastic or
  training composites; deterministic eval-mode composites become coarse boundary nodes, and an
  operation that returns its exact input object is recorded as an alias, not a new op.
- `.tlspec` saves use `payload_policy="array_payloads"` and load back as `paddle.Tensor` when
  Paddle is installed; `bfloat16` records its logical dtype and load restores it.

Leaf gradients are derived gradients, not backward capture:

```python
from torchlens.options import CaptureOptions

grad_options = tl.backends.paddle.GradOptions(
    loss_fn=lambda output: paddle.sum(output * output),
    input_grad_argnums=(0,),
)
trace = tl.trace(model, x, backend="paddle", capture=CaptureOptions(grad_options=grad_options))
trace.autograd.derived_grads["inputs.0"]
```

`GradOptions(intermediate_grads=True, max_intermediate_grads=...)` runs a second guarded Paddle AD
replay; only unambiguous `status == "exact"` records reach `trace.autograd.intermediate_derived_grads`
and `op.autograd.derived_grad`, and `has_backward_pass` stays `False`. Full page: [Paddle](backends/paddle.md).

## JAX

Install with `torchlens[jax]` (`jax[cpu]>=0.4.30`, Equinox, Flax). JAX traces accept raw callables
shaped like `fn(params, *inputs)`:

```python
import jax.numpy as jnp
import torchlens as tl

def fn(params, x):
    return jnp.tanh(x @ params["w"] + params["b"])

params = {"w": jnp.ones((3, 2)), "b": jnp.zeros((2,))}
trace = tl.trace(fn, (params, jnp.ones((4, 3))), backend="jax")
assert trace.validate_forward_pass([]) is True
```

What it does:

- Parameter records come from the first pytree argument, so `Trace.provenance.param_source` is
  `"pytree-derived"` when tensor leaves are present and `"none"` otherwise. Raw callables use
  `module_identity_mode="function_root"` and expose only the root `"self"` module.
- Equinox `eqx.Module` and Flax NNX `nnx.Module` roots default to `module_identity_mode="pytree_module"`:
  TorchLens walks the module tree (an Equinox list or tuple field of modules becomes a `field`
  container with `field.0`, `field.1` children, like torch's `ModuleList`), attributes equations
  through `jax.named_scope`, builds `Trace.modules` entries with `training=False` and derives
  parameter owners from path-keyed leaves. Inside attributed modules it rejects `jax.jit`,
  `pjit`, `shard_map`, `lax.cond`, `lax.scan`, `lax.while_loop`, remat and custom-VJP, callback
  effects and NNX state rebinding; move those outside the module or trace a raw `function_root`
  callable.
- Control flow in raw captures: pure nested `jax.jit` calls are inlined; `lax.scan`, `lax.cond`
  and capped `lax.while_loop` unroll by default (`jax_control_flow="unroll"`, guarded by
  `jax_max_control_flow_unroll`). Over-cap `scan` and `while_loop`, and forward `custom_vjp_call`,
  become importer-owned regions with one boundary node and per-output projection nodes, and the
  trace reports `trace.validation_replay_status.state == "unverified"` after the replayable ops
  and region seams pass. `jax_control_flow="reject"` restores the nested-primitive rejection and
  `"region"` forces regions. Nested JIT calls with closed constants, effects, donated inputs or
  explicit shardings are rejected.
- Readable closure arrays are recorded as constant sources; derived gradients hold them fixed.
- Inputs may be positional arrays, keyword arrays passed through `input_kwargs=`, or both.
- Each op records the user source lines that ran it, as on PyTorch (`op.source.frames`,
  `save_source_context`, `source_context_lines`).
- Refused: transformed root callables (`jax.jit`, `jax.vmap`, `jax.grad`) and capture from inside
  them, unsupported nested-jaxpr primitives, callback effects, streaming and `save_grads=`.
  `save=` on `tl.trace` applies after finalization.
- `intervene=` edits equation outputs at capture, with the six builtin helpers and callables;
  with `grad_options=`, derived gradients follow the edited program and `tl.grad(...)` sites
  edit the gradient at each matched output. Finished traces take edits through
  `trace.fork().do(...)`. `stop_after=` stops interpreting at the first matching equation and
  unwinds every nested frame before any later primitive binds.
- `tl.record(..., backend="jax")` records per-equation values with the same raw labels as
  `tl.trace`; only selected values are kept, and `stop_after=` stops at the frontier.
- Inside an Equinox or NNX module, submodules called in `jax.jit`, a taken `lax.cond` branch,
  `lax.scan` or `lax.while_loop` are attributed; each loop iteration is its own module call, and
  their returns are unrecorded (`CaptureGapWarning`). `checkpoint`/`remat`, `custom_vjp` and
  `shard_map` inside a module refuse typed.
- `.tlspec` saves use `payload_policy="array_payloads"` and load back as `jax.Array`; typed PRNG
  keys round-trip as typed keys. Fully addressable single-host sharded arrays are saved as
  assembled host values with versioned `jax_sharding_*` fields and a `jax_named_sharding` block
  (mesh `axis_names`, mesh `shape`, `PartitionSpec` as JSON primitives) in the manifest's
  `codec_metadata`; default load is value-only. To rebuild the topology pass
  `PayloadLoadHints(jax=JaxPayloadLoadHint(sharding=...))` or
  `JaxPayloadLoadHint(reconstruct_sharding=True, platform="cpu")`; `map_location` stays the
  single-device `str | torch.device` target. Multi-host or unaddressable arrays fail closed on save.
  `trace.payload_load_status` records what was materialized. Runtime-only `jax_equation_captures`
  are stripped from portable artifacts, so a loaded JAX trace reports
  `trace.validation_replay_status.state == "unavailable"` with reason
  `"loaded_trace_runtime_capture_stripped"`; live traces, selectively saved ones included, still
  run real replay validation from runtime-only payloads.

Leaf gradients are derived gradients, not backward capture:

```python
from torchlens.options import CaptureOptions

grad_options = tl.backends.jax.GradOptions(
    params=params,
    loss_fn=lambda output: jnp.sum(output * output),
    input_grad_argnums=(0,),
)
trace = tl.trace(fn, (params, jnp.ones((4, 3))), backend="jax", capture=CaptureOptions(grad_options=grad_options))
trace.autograd.derived_grads["params.w"]
```

`GradOptions(intermediate_grads=True, max_intermediate_grads=...)` runs one all-tap `jax.grad`
pass, then checks each attached boundary with an independent boundary-replacement VJP and a finite
difference; requests above the cap fail. Only records with provenance `status == "exact"` reach
`trace.autograd.intermediate_derived_grads` and `op.autograd.derived_grad`; oracle failures and
non-float or control outputs are skipped. If the producer cannot run on the installed JAX, leaf
`derived_grads` stay intact and the intermediate accessor is empty. Flax and Equinox module roots
(`module_identity_mode="pytree_module"`) also take derived gradients; their parameters are keyed
by module address in `trace.params`. Full page: [JAX](backends/jax.md).

## tinygrad

Install with `torchlens[tinygrad]` (`tinygrad>=0.13,<0.14`; `tinygrad==0.13.0` at runtime). For
live payload validation and derived gradients, run with `DEV=PYTHON`.

```python
from tinygrad import Tensor
import torchlens as tl

def fn(x):
    return ((x + 1).relu() * 2).sum()

trace = tl.trace(fn, Tensor([1.0, -2.0, 3.0]), backend="tinygrad")
assert trace.validate_forward_pass([fn(Tensor([1.0, -2.0, 3.0]))]) is True
```

- Inputs may be positional tensors, keyword tensors passed through `input_kwargs=`, or both.
- Each op records the user source lines that ran it, as on PyTorch (`op.source.frames`,
  `save_source_context`).
- `intervene=tl.when(condition, edit)` works with static selectors (`tl.func`, `tl.label`,
  `tl.label_contains`, `tl.in_module`) and with conditions on the op context (`tl.where`, a plain
  callable), evaluated on a first planning capture. The edit is applied lazily, so the forward
  still realizes once; the edited op keeps its label and parents and the trace validates. Edits
  take the six built-in helpers (`tl.zero_ablate`, `tl.scale`, `tl.add`, `tl.clamp`,
  `tl.mean_ablate`, `tl.replace_with`) or a callable, live and on a finished trace
  (`trace.fork().do(...)`).
- With `grad_options=`, the derived gradients are those of the edited program, and
  `tl.when(tl.grad(selector), edit)` edits the gradient at the selected ops (input and
  intermediate gradients). `trace.do` with a `tl.grad(...)` site refuses with
  `run_capability_unavailable`.

What it does:

- Raw callables use `module_identity_mode="function_root"`. Callable objects with discoverable
  tinygrad module attributes default to `object_module`: discovery walks object attributes for
  callable children that own `Tensor` attributes or are `tinygrad.nn.*` layers, records shared
  children under the first address with later aliases, and attributes UOps from the module stack
  observed at construction. A UOp tinygrad reuses keeps its first attribution. Pass
  `module_identity_mode="function_root"` for the root-only surface.

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
assert trace.options.module_identity_mode == "object_module"
assert trace.modules["fc"].params
```

- Refused: mid-capture `Tensor.realize()`, `Tensor.assign()`, `Tensor.replace()`, setitem input
  mutation, TinyJit execution, `tl.module(...)` edits, `stop_after=`, streaming,
  `save_grads=`, `backward_ready=True` and `tl.record(backend="tinygrad")`. `save=` is decided in
  the capture's single realization: values of unselected ops are never computed into a buffer.
  Return a pure lazy tinygrad expression from a raw callable, or use the PyTorch backend for the
  refused features.
- `.tlspec` saves use `payload_policy="array_payloads"` and load back as tinygrad `Tensor`;
  `trace.payload_load_status` records what was materialized. Runtime-only `tinygrad_uop_captures`
  are stripped from portable artifacts, so a loaded trace reports
  `trace.validation_replay_status.state == "unavailable"` with reason
  `"loaded_trace_runtime_capture_stripped"`; live traces still run real replay validation. If an
  importer marks a region as outside per-op replay, the status folds to `"unverified"` rather than
  `passed`, and `bool()` of it raises.

Derived gradients are bracketed leaf gradients, not backward capture:

```python
from torchlens.options import CaptureOptions

grad_options = tl.backends.tinygrad.GradOptions(input_grad_argnums=(0,))
trace = tl.trace(fn, Tensor([1.0, -2.0, 3.0]), backend="tinygrad", capture=CaptureOptions(grad_options=grad_options))
trace.autograd.derived_grads["inputs.0"]
```

`GradOptions(intermediate_grads=True)` runs a separate no-realize pass: it calls `loss.backward()`
before realizing any copied outputs and attaches exact unambiguous records to
`trace.autograd.intermediate_derived_grads` and `op.autograd.derived_grad`; ambiguous matches are skipped,
and `op.autograd.grads` and `trace.saved_grad_ops` stay true-backward-only. Full page:
[tinygrad](backends/tinygrad.md).

## Capability Matrix

The implementation view of the same facts, one row per backend:

| Backend | Capture | Validation | Payloads | Modules | Gradients |
|---|---|---|---|---|---|
| `torch` | Stable eager wrapper capture | Replay validation | Materialized `.tlspec` payloads | `torch_module` | True backward capture |
| `mlx` | Eager-dispatch capture; live `intervene=`/`stop_after=` (static or value-dependent), also with `grad_options=` and `tl.grad` sites | Live per-op replay and parent perturbation over the captured (whitelisted) op set | Materialized forward/derived array `.tlspec` payloads | `function_root`, object `object_module` | `trace.autograd.derived_grads`; opt-in `trace.autograd.intermediate_derived_grads` |
| `jax` | Jaxpr-first functional capture | Live per-equation replay and parent perturbation | Materialized forward/derived array `.tlspec` payloads | `function_root`, Equinox/NNX `pytree_module` | `trace.autograd.derived_grads`; opt-in `trace.autograd.intermediate_derived_grads` |
| `tinygrad` | UOp-snapshot functional capture; live `intervene=` (static or value-dependent), also with `grad_options=` and `tl.grad` sites; finished-trace edits | Live UOp replay and parent perturbation on `DEV=PYTHON` payloads | Materialized forward/derived array `.tlspec` payloads | `function_root`, object `object_module` | `trace.autograd.derived_grads`; opt-in `trace.autograd.intermediate_derived_grads` |
| `paddle` | Dygraph/eager capture with live forward `intervene=`/`stop_after=`, finished-trace edits and `tl.grad` sites on derived gradients | Live replay/perturbation plus static inventory guard; corroborated user-intervention carve-out | Materialized forward/derived array `.tlspec` payloads | `function_root`, object `object_module` | `trace.autograd.derived_grads`; opt-in `trace.autograd.intermediate_derived_grads` |
| `tf` | Eager op-callback capture with live `intervene=`/`stop_after=` (static or value-dependent); implemented graph-only FuncGraph static path for compiled/SavedModel entries | Callback self-consistency plus per-op replay/perturbation accounting | Materialized forward array `.tlspec` payloads | `function_root`, Keras/`tf.Module` object `object_module` | `trace.autograd.derived_grads`; opt-in `trace.autograd.intermediate_derived_grads` (eager entries only) |

Replay validation on the non-PyTorch backends (`has_validation_replay`) replays each captured op from
its parents' saved payloads and perturbation-checks the dependency. Unlike torch's validator it
does not re-run the whole model against a ground-truth output, so an op the backend failed to wrap
is outside the replay denominator, and a call with no perturbable tensor argument reports
`unverified`, never a silent pass. Grouping repeated calls into passes never weakens this check:
replay sidecars stay keyed to raw capture identities, oracles compare in raw-label space, and
per-backend tamper tests prove that a stale-label sidecar fails validation.
