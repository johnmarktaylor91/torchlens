# TensorFlow backend

The TensorFlow backend records the real eager forward pass through op callbacks,
including operation values and taken branches. Keras and `tf.Module` roots can
carry module attribution; raw callables use a function root. Graph-only entries
use a separate FuncGraph importer, with opaque regions reported as unverified.
It supports static-label payload selection, eager interventions with static or
value-dependent conditions, the six built-in edit helpers, edits on a finished trace
(`trace.fork().do(...)`), early stop (`stop_after=`), user source lines, array
payload persistence, and derived gradients, including gradients of an edited or
halted program and `tl.grad` edit sites.

## Tutorial

Install the runtime first:

```bash
pip install "torchlens[tf]"
```

```python
# tutorial: executed by tests/backends/test_tf_docs_page.py under the tf_backend marker
import keras
import tensorflow as tf
import torchlens as tl

keras.utils.set_random_seed(0)
model = keras.Sequential([
    keras.Input((3,)),
    keras.layers.Dense(4, activation="relu"),
    keras.layers.Dense(2),
])
x = tf.ones((1, 3), dtype=tf.float32)
trace = tl.trace(model, x, backend="tf")
print(trace.summary())
status = trace.validation_replay_status
print(status.state)
```

## Can and cannot

| Capability | Status | Note |
|---|---|---|
| backward_capture | unsupported | True backward graphs are deferred. |
| has_validation_replay | supported | Live replay reports its verification coverage. |
| fastlog | supported | `tl.record` runs the eager forward with lean per-op events; values only for selected ops; graph-only callables refuse. |
| can_intervene | supported | Eager replacements (static or value-dependent conditions) and `stop_after=` use the writable wrapper layer. |
| rng_replay | unsupported | RNG replay is not declared. |
| payload_materialization | supported | Dense numeric and boolean array payloads can be persisted. |
| has_streaming | unsupported | Streaming capture is deferred. |
| structure_only_capture | unsupported | Structure-only capture is not declared. |
| intermediate_derived_grads | supported | Exact matched intermediate gradients use auxiliary replay. |

### Early stop, edits and source lines

- `stop_after=` takes a selector or a callable returning bool, as on torch. The op
  callback that records the frontier op raises the halt, so no later user code runs;
  the captured prefix finalizes like a full capture and equals the same prefix of a
  full run op for op. A selector that never fires refuses with `stop_after_never_fired`;
  a callable that never fires warns with `stop_after_never_fired_callable`. When an
  edit and the halt match the same op, the edit lands first and the frontier carries
  the edited value.
- `intervene=` accepts `tl.when(selector, action)`, `tl.when(callable, action)`, a bare
  callable returning an action or `None`, and multi-rule specs. Conditions run once per
  recorded op output on its `RecordContext`, including the live `is_scalar_bool` and
  `bool_value` fields, so editing a branch flag changes the branch taken. A condition
  returning something that is not an action refuses with `predicate_return_invalid`. A
  multi-rule or value-dependent spec whose clause needs a module boundary refuses with
  `trace_option_unsupported`; pass module clauses as their own `tl.when` entry.
- Op callbacks are read-only, so edits land through a writable layer that wraps every
  generated python op wrapper (the `gen_*_ops` modules), the variable read
  (`ReadVariableOp`, so a Keras weight read is editable), unary minus and a curated
  set of public entry points. On the TensorFlow conformance roster this layer
  reaches 2173 of 2174 recorded ops (99.95%, variable reads included). The one
  op it misses is the `Erfc` inside exact `tf.nn.gelu`, which TensorFlow calls
  through an internal dispatch path. Ops that bypass the python wrappers, such as `tf.raw_ops.*` calls and ops inside a `tf.function` body,
  stay read-only: an edit whose condition matches one of them refuses with a typed
  unreachable error rather than silently not editing.
- Source lines record the user frames between the `tl.trace` call and each op,
  outermost first, as torch does (`trace[label].source.frames`). TensorFlow and Keras
  frames are filtered out, so a Keras functional model's ops carry only the
  `tl.trace` call frame. `CaptureOptions(save_source_context=True)` loads the
  surrounding code lines.

### Selective save and memory

- `save=` decides each op's payload at capture: only selected ops (and model inputs)
  keep a NumPy snapshot. Without a backward pass to prepare for, op records drop
  their live tensors once their edges are recorded, and the producer index holds
  tensors weakly, so an intermediate is freed as soon as the model drops it.
- On a 20-block chain of 4 MiB activations, `save=` with one late op peaks at about
  66 MiB (a bare forward peaks at about 48 MiB), and a full capture at about 260 MiB,
  which is the 244 MiB of saved payloads plus the forward.

### Edit helpers, finished-trace edits and gradients

- The six built-in helpers work on live captures and on finished traces:
  `tl.zero_ablate()`, `tl.scale(...)`, `tl.add(...)`, `tl.clamp(...)`,
  `tl.mean_ablate(...)` and `tl.replace_with(...)`. Their values are computed with
  TensorFlow ops, so an edited live capture passes replay validation. A value from
  another framework (a torch tensor, a JAX array), a `Population` source for
  `tl.mean_ablate` and any other helper refuse with `trace_option_unsupported`.
- `trace.fork().do(site, edit)` edits a finished TensorFlow trace: the edit is
  applied at the site and each downstream op is re-run once through the TensorFlow
  replay adapter from its updated parents. An edited fork equals a live
  `intervene=` capture of the same edit, passes `verify_backend_replay`, and reads
  `is_pristine` False. An op the adapter cannot replay refuses by its op name before
  anything is committed.
- `CaptureOptions(grad_options=...)` combines with `intervene=` and `stop_after=`.
  The derived gradients are the gradients of the program that actually ran: the
  replay applies the same edits, so an edit changes the gradients exactly as
  differentiating the edited code by hand would, and a halted capture
  differentiates the frontier op's value (the halted output) rather than the full
  model's output.
- `tl.when(tl.grad(selector), edit)` edits the gradient flowing back into the ops
  `selector` names. The forward is unchanged. Where TensorFlow computes gradients
  (the derived-gradient replay), the named op's output passes through an identity
  whose custom gradient applies the edit, and the edited gradient propagates
  upstream to the inputs and parameters. The edit takes a helper or a callable
  `edit(grad, *, ctx)`. A single site that mixes gradient and activation addresses,
  or names autograd nodes TensorFlow does not record, refuses with
  `backward_capture_unsupported`; pass gradient sites as their own `tl.when` clause.
- `trace.do(tl.grad(selector), edit)` on a finished TensorFlow trace arms the
  gradient edit on that trace and returns it; the derived-gradient replay then
  applies it. A `do()` call that also edits activations, or passes `loss=`, refuses
  with `run_capability_unavailable`: make the activation edit a separate `do()`
  call and compute edited gradients with `grad_options=` at capture.

### Known gaps

- A `save=` trace fails replay validation at ops whose own output or an input was
  not saved, where torch exempts those ops as `not_saved_by_user`.
- Replay validation reports `unverified`, rather than `passed`, on Keras models
  because parameter reads (`ReadVariableOp`) count as unverified regions.
- `tf.reduce_any` (`Any`) is not classified by replay validation yet, so a trace that
  uses it fails validation at that op, edited or not.
- An edit that replaces a value only Python control flow reads (a branch flag
  edited with `tl.zero_ablate()`) leaves its replacement op (`Fill`) with no
  consuming op, so replay validation fails at that op.
- Ops TensorFlow runs from C++ without returning through a python wrapper
  (`tf.raw_ops.*` calls, `tf.function` bodies, fused kernels' internals) cannot be
  edited live; see the writable-layer note above.
- Loaded traces cannot be replay-validated because runtime captures are not persisted.
- Array-valued closure constants (closure weights, `tf.constant([...])`) are recorded as
  `constant` source ops with a saved payload; rank-zero literals stay replayable values on
  the op that consumed them. A `tf.ones`/`tf.fill` inside the forward is a literal
  producer: zero parents, payload saved, replayed from its arguments.

## Framework requirement

- Supported range: `tensorflow>=2.16`, with Keras 3 using the TensorFlow backend.
- Tested through: TensorFlow `2.21.0`.
- Pinned CI version: TensorFlow `2.21.0`, Keras `3.15.1`.
