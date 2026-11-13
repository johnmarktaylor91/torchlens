"""Paddle registered buffers are recorded the way torch records buffers.

A ``register_buffer`` tensor read by the forward becomes one parentless
``buffer`` source layer with its ``named_buffers`` address, listed in
``Trace.buffers``; every read of the same buffer is an edge from that one
layer, and replay validation passes. A buffer the forward writes through a
path the capture cannot see is never credited: the trace carries a write
marker and validation fails closed.
"""

from __future__ import annotations

from typing import Any

import pytest

paddle = pytest.importorskip("paddle")

import torchlens as tl  # noqa: E402
from torchlens.backends.paddle._param_writes import PARAMETER_WRITE_MARKER  # noqa: E402
from torchlens.backends.paddle.backend import PaddleBackend  # noqa: E402

pytestmark = [pytest.mark.backend_paddle, pytest.mark.rare]


class _ScaleBuffer(paddle.nn.Layer):
    """Linear layer whose output is multiplied by a registered buffer."""

    def __init__(self) -> None:
        """Build the layer and register the ``scale`` buffer."""

        super().__init__()
        self.fc = paddle.nn.Linear(4, 4)
        self.register_buffer("scale", paddle.full([4], 2.0))

    def forward(self, x: Any) -> Any:
        """Scale the linear output by the buffer."""

        return self.fc(x) * self.scale


class _Offset(paddle.nn.Layer):
    """Child layer that reads its buffer twice."""

    def __init__(self) -> None:
        """Register the ``offset`` buffer."""

        super().__init__()
        self.register_buffer("offset", paddle.to_tensor([0.5, -1.0, 2.0, 0.25]))

    def forward(self, x: Any) -> Any:
        """Add the buffer, then subtract half of it."""

        return (x + self.offset) - self.offset * 0.5


class _NestedOffset(paddle.nn.Layer):
    """Root layer holding a buffer-reading child."""

    def __init__(self) -> None:
        """Build the child."""

        super().__init__()
        self.block = _Offset()

    def forward(self, x: Any) -> Any:
        """Run the child and a ReLU."""

        return paddle.nn.functional.relu(self.block(x))


class _HiddenBufferWrite(paddle.nn.Layer):
    """Training batch norm over registered-buffer running statistics.

    ``batch_norm`` in training mode overwrites ``running_mean`` and
    ``running_var`` inside its kernel, a write no wrapper sees; the forward
    then reads the written mean.
    """

    def __init__(self) -> None:
        """Register the running-statistic buffers."""

        super().__init__()
        self.register_buffer("running_mean", paddle.zeros([3]))
        self.register_buffer("running_var", paddle.ones([3]))

    def forward(self, x: Any) -> Any:
        """Normalize in training mode, then add the written running mean."""

        normed = paddle.nn.functional.batch_norm(
            x, self.running_mean, self.running_var, training=True
        )
        return normed + self.running_mean


def _x() -> Any:
    """Return a deterministic ``[2, 4]`` input."""

    return paddle.arange(8, dtype="float32").reshape([2, 4]) / 8.0


def _buffer_layers(trace: Any) -> list[Any]:
    """Return the trace's buffer layers."""

    return [layer for layer in trace.layers if layer.layer_type == "buffer"]


def test_paddle_buffer_is_a_parentless_buffer_layer() -> None:
    """The buffer read is one ``buffer_1`` layer that the multiply consumes."""

    paddle.seed(0)
    model = _ScaleBuffer()
    trace = tl.trace(model, _x(), backend="paddle")

    buffers = _buffer_layers(trace)
    assert [layer.label for layer in buffers] == ["buffer_1:1"]
    assert list(buffers[0].parents) == []
    assert [buffer.address for buffer in trace.buffers] == ["scale"]
    mul = next(layer for layer in trace.layers if layer.layer_type == "mul")
    assert "buffer_1" in list(mul.parents)
    assert PaddleBackend().validate_trace(trace) is True
    assert tl.validate(model, _x(), backend="paddle") is True


def test_paddle_buffer_reads_share_one_layer_with_nested_address() -> None:
    """Two reads of a child's buffer are edges from one ``block.offset`` layer."""

    model = _NestedOffset()
    trace = tl.trace(model, _x(), backend="paddle")

    buffers = _buffer_layers(trace)
    assert len(buffers) == 1
    assert len(list(buffers[0].children)) == 2
    assert [buffer.address for buffer in trace.buffers] == ["block.offset"]
    assert PaddleBackend().validate_trace(trace) is True
    assert tl.validate(model, _x(), backend="paddle") is True


def test_paddle_buffer_written_by_hidden_kernel_fails_closed() -> None:
    """A buffer changed during the forward is marked and never credited."""

    model = _HiddenBufferWrite()
    x = paddle.arange(12, dtype="float32").reshape([4, 3]) / 4.0 + 1.0
    trace = tl.trace(model, x, backend="paddle")

    written = sorted(
        marker
        for marker in trace._paddle_capture_gap_markers
        if marker.startswith(f"{PARAMETER_WRITE_MARKER}: buffer ")
    )
    assert written == [
        f"{PARAMETER_WRITE_MARKER}: buffer running_mean",
        f"{PARAMETER_WRITE_MARKER}: buffer running_var",
    ]
    assert _buffer_layers(trace) == []
    assert PaddleBackend().validate_trace(trace) is False


def test_paddle_buffer_capture_leaves_buffer_untouched() -> None:
    """Recording a buffer does not change the model's buffer value."""

    model = _ScaleBuffer()
    before = model.scale.numpy().copy()
    tl.trace(model, _x(), backend="paddle")

    assert (model.scale.numpy() == before).all()
