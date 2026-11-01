# Paddle backend (preview)

`backend="paddle"` captures PaddlePaddle dygraph (eager) forwards. The backend is a
**preview**: its capabilities are measured by the shared conformance suite and shown in
`tl.compat.report()` as the `backend_scorecard:paddle` row. Install it with
`pip install "torchlens[paddle]"` (PaddlePaddle `>=3.3,<3.4`; the per-change test leg
runs the pinned 3.3 release).

## Tutorial

Capture a small convolutional network, read its ops, and replay-validate the capture.

```python
import numpy as np
import paddle
import torchlens as tl


class SmallCNN(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv = paddle.nn.Conv2D(1, 2, 3, padding=1)
        self.pool = paddle.nn.MaxPool2D(2)
        self.fc = paddle.nn.Linear(18, 3)

    def forward(self, x):
        h = self.pool(paddle.nn.functional.relu(self.conv(x)))
        return self.fc(paddle.flatten(h, 1))


paddle.seed(0)
model = SmallCNN()
image = paddle.to_tensor(np.linspace(-1, 1, 36, dtype="float32").reshape(1, 1, 6, 6))

trace = tl.trace(model, image, backend="paddle")
print([op.layer_type for op in trace.layer_list])
# ['input', 'conv2d', 'functional.relu', 'functional.max_pool2d', 'flatten', 'functional.linear']

assert tl.validate(model, image, scope="forward", backend="paddle") is True
```

`Conv1D`, `Conv2D` and `Conv3D` layers appear as one op each (`conv1d`, `conv2d`,
`conv3d`) whose parent is the layer input; the ops inside the layer's `forward` are not
recorded separately.

Tensors a function or layer reads from outside the traced call (a closure variable, a
module-level global, a plain tensor attribute of a layer) are recorded as **constants**:
parentless ops with `layer_type == "constant"` and a saved value, listed as parents of the
op that read them. Ops that build a tensor from literals (`paddle.ones`, `paddle.full`,
...) stay parentless ops and are replayed by validation.

```python
weight = paddle.to_tensor(np.eye(4, dtype="float32"))


def project(v):
    return paddle.nn.functional.relu(paddle.matmul(v, weight))


trace = tl.trace(project, paddle.ones([2, 4], dtype="float32"), backend="paddle")
print([op.layer_type for op in trace.layer_list])
# ['input', 'constant', 'matmul', 'functional.relu']
```

A tensor created *during* the forward by an operation TorchLens does not wrap is not a
constant: it stays an untraced input, and validation fails naming the op that read it
(reason `parent_payload_missing`).

## Can and cannot

The rows follow the backend's capability summary in its scorecard: **supported** means
declared and exercised by the suite, **unsupported** means refused with a typed error
(`exc.code` below), and **not yet proven** means the refusal has not passed its
conformance cell yet.

| Capability | Status | Refusal code |
|---|---|---|
| Forward capture (eager dygraph) | supported | |
| Replay validation (`has_validation_replay`) | supported | |
| Payload materialization after load (`payload_materialization`) | supported | |
| Live forward interventions and halt (`interventions`) | supported | |
| Derived gradients, including intermediates (`intermediate_derived_grads`) | supported | |
| True backward capture (`backward_capture`) | not yet proven | `backward_capture_unsupported` |
| Fast logging (`fastlog`) | not yet proven | `fastlog_unsupported` |
| RNG replay (`rng_replay`) | not yet proven | `rng_replay_unsupported` |
| Streaming (`has_streaming`) | not yet proven | `streaming_unsupported` |
| Structure-only capture (`structure_only_capture`) | not yet proven | `structure_only_capture_unsupported` |

Other refusals the backend raises: `explicit_seed_unsupported` (`random_seed=`),
`trace_option_unsupported` (an option outside the backend's list, or `grad_options=`
combined with `intervene=`/`halt=`), `module_identity_mode_unsupported`,
`value_dependent_branch_unsupported` (a tensor turned into a Python scalar inside the
forward), `rng_replay_unsupported` (a stochastic Paddle op inside the forward),
`foreign_tensor_input` and `predicate_return_invalid`.

Known limits: static-graph and PIR modes are refused (code `runtime_mode_unsupported`);
tensors registered with `register_buffer` are not recorded as constants and stay untraced
inputs; transposed convolution layers are not captured at the layer level.
