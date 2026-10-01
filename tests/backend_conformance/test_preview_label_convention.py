"""Preview backends must match torch's exact relation-label convention.

Torch (``postprocess/labeling.py``) relabels ``parents``/``children`` (and the
lineage sets ``input_ancestors``/``output_descendants``/``root_ancestors``)
through a CONDITIONAL mapping: a referenced op's bare ``layer_label`` when its
layer has a single pass, and its pass-qualified ``label`` only when the
referenced layer is multi-pass. ``trace.input_layers``/``trace.output_layers``
resolve through the identical conditional mapping (torch's own rename always
emits the bare label there too, but only because torch always wraps real
computation in single-pass input/output pseudo-ops; a preview's output can be
the real last-pass op of a multi-pass layer directly, where a bare label would
be ambiguous). Previews must reproduce this exactly so backend-neutral code
sees one contract regardless of backend.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.backend_parity


def _assert_conditional_label(trace, label: str, *, context: str) -> None:
    """A label is bare iff its own layer is single-pass; checked for lineage."""

    layer_num_calls = trace.layer_num_calls
    if ":" in label:
        layer_label, _, pass_suffix = label.rpartition(":")
        assert pass_suffix.isdigit(), f"{label!r} ({context}) has a non-numeric suffix after ':'"
    else:
        layer_label = label
    num_calls = layer_num_calls[layer_label]
    if num_calls == 1:
        assert label == layer_label, (
            f"single-pass layer {layer_label!r} must be referenced bare ({context}), got {label!r}"
        )
    else:
        assert label != layer_label, (
            f"multi-pass layer {layer_label!r} must be referenced "
            f"pass-qualified ({context}), got the bare label"
        )


def _assert_torch_label_convention(trace) -> None:
    """Every parent/child/input/output reference follows the conditional rule."""

    for op in trace.layer_list:
        for neighbor_label in (*op.parents, *op.children):
            _assert_conditional_label(
                trace, neighbor_label, context=f"referenced from {op.label!r}"
            )
    for label in trace.input_layers:
        _assert_conditional_label(trace, label, context="trace.input_layers")
    for label in trace.output_layers:
        _assert_conditional_label(trace, label, context="trace.output_layers")


@pytest.mark.backend_tinygrad
def test_tinygrad_label_convention_matches_torch() -> None:
    pytest.importorskip("tinygrad")
    from tinygrad import Tensor

    import torchlens as tl

    def model(x, w):
        for _ in range(2):
            x = (x @ w).relu()
        return x

    w = Tensor.arange(16, dtype="float").reshape(4, 4) / 20.0
    trace = tl.trace(model, (Tensor.ones(2, 4), w), backend="tinygrad")
    assert trace.recurrence_detection is True
    _assert_torch_label_convention(trace)


@pytest.mark.backend_jax
def test_jax_label_convention_matches_torch() -> None:
    jnp = pytest.importorskip("jax.numpy")
    import torchlens as tl

    def model(x):
        w = jnp.ones((4, 4))
        for _ in range(3):
            x = jnp.tanh(x @ w)
        return x

    trace = tl.trace(model, jnp.ones((1, 4)), backend="jax")
    assert trace.recurrence_detection is True
    _assert_torch_label_convention(trace)


@pytest.mark.backend_mlx
def test_mlx_label_convention_matches_torch() -> None:
    mx = pytest.importorskip("mlx.core")
    mnn = pytest.importorskip("mlx.nn")
    import torchlens as tl

    class M(mnn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = mnn.Linear(4, 4)

        def __call__(self, x):
            for _ in range(3):
                x = mnn.tanh(self.lin(x))
            return x

    trace = tl.trace(M(), mx.ones((1, 4)), backend="mlx")
    assert trace.recurrence_detection is True
    _assert_torch_label_convention(trace)


@pytest.mark.backend_paddle
def test_paddle_label_convention_matches_torch() -> None:
    paddle = pytest.importorskip("paddle")
    import torchlens as tl

    class M(paddle.nn.Layer):
        def __init__(self) -> None:
            super().__init__()
            self.lin = paddle.nn.Linear(4, 4)

        def forward(self, x):
            for _ in range(3):
                x = paddle.nn.functional.tanh(self.lin(x))
            return x

    trace = tl.trace(M(), paddle.ones([1, 4]), backend="paddle")
    assert trace.recurrence_detection is True
    _assert_torch_label_convention(trace)


@pytest.mark.tf_backend
def test_tf_label_convention_matches_torch() -> None:
    tf = pytest.importorskip("tensorflow")
    keras = pytest.importorskip("keras")
    import torchlens as tl

    class M(keras.Model):
        def __init__(self) -> None:
            super().__init__()
            self.dense = keras.layers.Dense(4)

        def call(self, x):
            for _ in range(3):
                x = tf.math.tanh(self.dense(x))
            return x

    model = M()
    inputs = tf.ones((1, 4))
    model(inputs)
    trace = tl.trace(model, inputs, backend="tf")
    assert trace.recurrence_detection is True
    _assert_torch_label_convention(trace)
