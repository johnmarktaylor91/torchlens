# TensorFlow backend

The TensorFlow backend records the real eager forward pass through op callbacks,
including operation values and taken branches. Keras and `tf.Module` roots can
carry module attribution; raw callables use a function root. Graph-only entries
use a separate FuncGraph importer, with opaque regions reported as unverified.
It supports static-label payload selection, eager interventions with static or
value-dependent conditions, early stop (`stop_after=`), user source lines, array
payload persistence, and derived gradients; true backward capture remains deferred.

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
| fastlog | unsupported | Sparse recording is deferred. |
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
  generated python op wrapper (the `gen_*_ops` modules) plus a curated set of public
  entry points. On the TensorFlow conformance roster this layer reaches about
  99.8% of recorded compute ops (1535 of 1538; the curated entry points alone reached 17.4%). Ops that bypass the
  python wrappers, such as `tf.raw_ops.*` calls and ops inside a `tf.function` body,
  stay read-only: an edit whose condition matches one of them refuses with a typed
  unreachable error rather than silently not editing.
- Source lines record the user frames between the `tl.trace` call and each op,
  outermost first, as torch does (`trace[label].source.frames`). TensorFlow and Keras
  frames are filtered out, so a Keras functional model's ops carry only the
  `tl.trace` call frame. `CaptureOptions(save_source_context=True)` loads the
  surrounding code lines.

### Known gaps

- Replay validation reports `unverified`, rather than `passed`, on Keras models
  because parameter reads (`ReadVariableOp`) count as unverified regions.
- Loaded traces cannot be replay-validated because runtime captures are not persisted.
- Array-valued closure constants (closure weights, `tf.constant([...])`) are recorded as
  `constant` source ops with a saved payload; rank-zero literals stay replayable values on
  the op that consumed them. A `tf.ones`/`tf.fill` inside the forward is a literal
  producer: zero parents, payload saved, replayed from its arguments.

## Framework requirement

- Supported range: `tensorflow>=2.16`, with Keras 3 using the TensorFlow backend.
- Tested through: TensorFlow `2.21.0`.
- Pinned CI version: TensorFlow `2.21.0`, Keras `3.15.1`.
