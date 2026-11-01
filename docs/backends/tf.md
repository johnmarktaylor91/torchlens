# TensorFlow backend (preview)

The TensorFlow preview records the real eager forward pass through op callbacks,
including operation values and taken branches. Keras and `tf.Module` roots can
carry module attribution; raw callables use a function root. Graph-only entries
use a separate FuncGraph importer, with opaque regions reported as unverified.
The preview supports static-label payload selection, eager interventions, array
payload persistence, and derived gradients; true backward capture remains deferred.

## Tutorial

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
| can_intervene | supported | Static-label eager replacements use the writable wrapper layer. |
| rng_replay | unsupported | RNG replay is not declared. |
| payload_materialization | supported | Dense numeric and boolean array payloads can be persisted. |
| has_streaming | unsupported | Streaming capture is deferred. |
| structure_only_capture | unsupported | Structure-only capture is not declared. |
| intermediate_derived_grads | supported | Exact matched intermediate gradients use auxiliary replay. |

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
