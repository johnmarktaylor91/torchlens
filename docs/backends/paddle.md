# Paddle backend

`backend="paddle"` captures PaddlePaddle dygraph (eager) forwards. Its capabilities are
measured by the shared conformance suite and shown in `tl.report.doctor()` as the
`backend_scorecard:paddle` row. Install it with
`pip install "torchlens[paddle]"` (PaddlePaddle `>=3.3,<3.4`; the per-change test leg
runs the pinned 3.3.1 release).

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
print([op.type for op in trace])
# ['input', 'conv2d', 'relu', 'max_pool2d', 'flatten', 'linear']

assert tl.validate(model, image, scope="forward", backend="paddle") is True
```

`Conv1D`, `Conv2D` and `Conv3D` layers appear as one op each (`conv1d`, `conv2d`,
`conv3d`) whose parent is the layer input; the ops inside the layer's `forward` are not
recorded separately. `Conv1DTranspose`, `Conv2DTranspose` and `Conv3DTranspose` layers
likewise appear as one op each (`conv1d_transpose`, `conv2d_transpose`,
`conv3d_transpose`) inside the layer's module, as do direct calls to those functionals.

Keyword arguments to the forward are passed with `input_kwargs=`; a keyword tensor
becomes an input op like a positional one. Tensors registered on a layer with
`register_buffer` are recorded as torch records buffers: one parentless op of type
`buffer` per buffer, listed in `trace.buffers` under its `named_buffers` address, with
an edge to every op that reads it.

```python
class Scaled(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.fc = paddle.nn.Linear(4, 4)
        self.register_buffer("scale", paddle.full([4], 2.0))

    def forward(self, x, bias=None):
        return self.fc(x) * self.scale + bias


scaled = Scaled()
x = paddle.ones([2, 4], dtype="float32")
bias = {"bias": paddle.zeros([4], dtype="float32")}
trace = tl.trace(scaled, x, input_kwargs=bias, backend="paddle")
print([op.type for op in trace])
# ['input', 'input', 'linear', 'buffer', 'mul', 'add']
print([buffer.address for buffer in trace.buffers])
# ['scale']
assert tl.validate(scaled, x, input_kwargs=bias, backend="paddle") is True
```

A buffer the forward overwrites through a path TorchLens does not see (a kernel that
updates it in place) is not credited: its reads stay untraced, the trace records a
`parameter written during forward: buffer <address>` marker, and validation fails.
Paddle's own `BatchNorm` running statistics are parameters, not buffers, so they appear
in `trace.params`.

Each op records the user source lines that ran it, as in torch: `op.source.frames`
lists the call site, the root's frame and every frame down to the line that ran the op
(last), skipping TorchLens and Paddle's layer call machinery. Input and buffer ops have
no frames. `save_source_context=True` keeps the source text with the trace.

Tensors a function or layer reads from outside the traced call (a closure variable, a
module-level global, a plain tensor attribute of a layer) are recorded as **constants**:
parentless ops with `op.type == "constant"` and a saved value, listed as parents of the
op that read them. Ops that build a tensor from literals (`paddle.ones`, `paddle.full`,
...) stay parentless ops and are replayed by validation.

```python
weight = paddle.to_tensor(np.eye(4, dtype="float32"))


def project(v):
    return paddle.nn.functional.relu(paddle.matmul(v, weight))


trace = tl.trace(project, paddle.ones([2, 4], dtype="float32"), backend="paddle")
print([op.type for op in trace])
# ['input', 'constant', 'matmul', 'relu']
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
| Live forward interventions and `stop_after=` (`can_intervene`) | supported | |
| Derived gradients, including intermediates (`intermediate_derived_grads`) | supported | |
| True backward capture (`backward_capture`) | unsupported | `backward_capture_unsupported` |
| Fast logging (`fastlog`) | unsupported | `fastlog_unsupported` |
| RNG replay (`rng_replay`) | unsupported | `rng_replay_unsupported` |
| Streaming (`has_streaming`) | unsupported | `streaming_unsupported` |
| Structure-only capture (`structure_only_capture`) | unsupported | `structure_only_capture_unsupported` |

Other refusals the backend raises: `explicit_seed_unsupported` (`random_seed=`),
`trace_option_unsupported` (an option outside the backend's list, or `grad_options=`
combined with `intervene=`/`stop_after=`), `module_identity_mode_unsupported`,
`value_dependent_branch_unsupported` (a tensor turned into a Python scalar inside the
forward), `rng_replay_unsupported` (a stochastic Paddle op inside the forward),
`inplace_op_unsupported` (an in-place or global-state Paddle op inside the forward),
`foreign_tensor_input` and `predicate_return_invalid`.

Known limits: static-graph and PIR modes are refused (code `runtime_mode_unsupported`).
They build a program instead of running ops eagerly, so there is no forward for the
dygraph wrappers to observe.
