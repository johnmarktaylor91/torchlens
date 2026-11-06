"""The drawing's sizes on preview backends: real bytes, never ``None`` or a made-up ``0 B``.

The launch pictures showed ``170 params (0 B, all trainable)`` on JAX, MLX and
Paddle (the backends counted parameters but never stamped their byte total;
TensorFlow also sized Keras 3 variables at 0, their dtype being a string)
and ``(1, 4, 4, 4), None`` on every Paddle op (its byte probe called the wrapped
``Tensor.numel`` mid-capture, whose ``int()`` the scalar-escape guard refuses).
The framework tests capture the same tiny CNN in each backend and read the
strings the drawing prints; the formatter tests pin that an unknown size is left
out of a row instead of printed.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import torchlens as tl
from torchlens._visualization._header_block import network_count_line, params_caption_detail
from torchlens._visualization._label_format import format_memory, shape_rows, sized_row

#: Conv2d(1 -> 4, 3x3) plus Linear(64 -> 2), float32: 40 + 130 parameters.
_NUM_PARAMS = 170
_PARAM_BYTES = 4 * _NUM_PARAMS


def _assert_param_bytes(trace: Any) -> None:
    """Assert the trace stamps its parameter bytes and the header prints them.

    Only the size is pinned: the trainable split is each framework's own flag
    (tinygrad leaves ``requires_grad`` unset until an optimizer claims a tensor).
    """

    assert trace.num_params == _NUM_PARAMS
    recorded = sum(int(param.param_memory) for param in trace.params.values())
    assert recorded == _PARAM_BYTES
    assert int(trace.cost.total_param_memory) == _PARAM_BYTES
    header = params_caption_detail(trace)
    assert header.startswith(f"{_NUM_PARAMS} params ({format_memory(_PARAM_BYTES)}, "), header


def _assert_op_bytes(trace: Any) -> None:
    """Assert every op records its output bytes and no drawn row prints ``None``."""

    total = 0
    for op in trace:
        memory = op.cost.activation_memory
        assert isinstance(memory, int) and memory > 0, (op.label, memory)
        total += int(memory)
        row = shape_rows(op, with_memory=True)[0]
        assert row.endswith(format_memory(memory)), (op.label, row)
        assert "None" not in row
    assert network_count_line(trace, None, "unrolled") == (
        f"{len(trace)} ops ({format_memory(total)})"
    )


@pytest.mark.backend_jax
@pytest.mark.rare
def test_jax_nnx_trace_reports_param_bytes() -> None:
    """A Flax NNX trace stamps its parameter bytes and every op's output bytes."""

    jnp = pytest.importorskip("jax.numpy")
    nnx = pytest.importorskip("flax.nnx")

    class Net(nnx.Module):
        """The launch CNN in Flax NNX."""

        def __init__(self, rngs: Any) -> None:
            """Build the conv and the linear head."""

            self.conv = nnx.Conv(1, 4, (3, 3), padding="VALID", rngs=rngs)
            self.head = nnx.Linear(64, 2, rngs=rngs)

        def __call__(self, x: Any) -> Any:
            """Run conv, relu, flatten and the head."""

            return self.head(nnx.relu(self.conv(x)).reshape(1, -1))

    trace = tl.trace(Net(nnx.Rngs(0)), jnp.ones((1, 6, 6, 1)), backend="jax")
    _assert_param_bytes(trace)
    _assert_op_bytes(trace)


@pytest.mark.backend_mlx
@pytest.mark.rare
def test_mlx_trace_reports_param_bytes() -> None:
    """An MLX trace stamps its parameter bytes and every op's output bytes."""

    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")

    class Net(nn.Module):
        """The launch CNN in mlx.nn."""

        def __init__(self) -> None:
            """Build the conv and the linear head."""

            super().__init__()
            self.conv = nn.Conv2d(1, 4, 3)
            self.head = nn.Linear(64, 2)

        def __call__(self, x: Any) -> Any:
            """Run conv, relu, flatten and the head."""

            return self.head(nn.relu(self.conv(x)).reshape(1, -1))

    trace = tl.trace(Net(), mx.ones((1, 6, 6, 1)), backend="mlx")
    _assert_param_bytes(trace)
    _assert_op_bytes(trace)


@pytest.mark.backend_paddle
@pytest.mark.rare
def test_paddle_trace_reports_param_and_op_bytes() -> None:
    """A Paddle trace stamps its parameter bytes, and every op (not only the input) its size."""

    paddle = pytest.importorskip("paddle")
    nn = paddle.nn
    model = nn.Sequential(nn.Conv2D(1, 4, 3), nn.ReLU(), nn.Flatten(), nn.Linear(64, 2))
    trace = tl.trace(model, paddle.ones([1, 1, 6, 6]), backend="paddle")
    _assert_param_bytes(trace)
    _assert_op_bytes(trace)


@pytest.mark.tf_backend
@pytest.mark.rare
def test_tf_keras_trace_reports_param_bytes() -> None:
    """A Keras trace sizes its variables (Keras 3 spells their dtype as a string)."""

    tf = pytest.importorskip("tensorflow")
    keras = tf.keras
    model = keras.Sequential(
        [
            keras.Input((6, 6, 1)),
            keras.layers.Conv2D(4, 3),
            keras.layers.ReLU(),
            keras.layers.Flatten(),
            keras.layers.Dense(2),
        ]
    )
    trace = tl.trace(model, tf.ones((1, 6, 6, 1)), backend="tf")
    _assert_param_bytes(trace)


@pytest.mark.backend_tinygrad
@pytest.mark.rare
def test_tinygrad_trace_reports_param_bytes() -> None:
    """A tinygrad trace stamps its parameter bytes (the same shared totals helper)."""

    tinygrad = pytest.importorskip("tinygrad")

    class Net:
        """The launch CNN in tinygrad."""

        def __init__(self) -> None:
            """Build the conv and the linear head."""

            self.conv = tinygrad.nn.Conv2d(1, 4, 3)
            self.head = tinygrad.nn.Linear(64, 2)

        def __call__(self, x: Any) -> Any:
            """Run conv, relu, flatten and the head."""

            return self.head(self.conv(x).relu().flatten(1))

    trace = tl.trace(Net(), tinygrad.Tensor.ones(1, 1, 6, 6), backend="tinygrad")
    _assert_param_bytes(trace)


def test_unknown_size_leaves_the_row_bare() -> None:
    """A row with no recorded size prints the shape alone, never ``None``."""

    assert sized_row("(1, 2)", None) == "(1, 2)"
    assert sized_row("(1, 2)", 8) == "(1, 2), 8 B"
    op = SimpleNamespace(shape=(1, 2), cost=SimpleNamespace(activation_memory=None))
    assert shape_rows(op, with_memory=True) == ["(1, 2)"]


def _stub_trace(memories: list[int | None], param_bytes: int) -> Any:
    """Return a duck-typed trace with one input op and ops of the given sizes."""

    ops = {
        f"op_{index}": SimpleNamespace(activation_memory=memory, modules=())
        for index, memory in enumerate(memories)
    }
    return SimpleNamespace(
        ops=ops,
        input_layers=["op_0"],
        output_layers=[],
        num_params=_NUM_PARAMS,
        num_params_frozen=0,
        num_params_trainable=_NUM_PARAMS,
        _cost_total_param_memory=param_bytes,
    )


def test_header_leaves_out_unknown_sizes() -> None:
    """The header omits a size it does not know instead of printing a partial sum or ``0 B``."""

    partial = _stub_trace([144, None, None], param_bytes=0)
    assert network_count_line(partial, None, "unrolled") == "3 ops"
    assert network_count_line(partial, None, "rolled") == "3 layers (3 ops)"
    assert params_caption_detail(partial) == f"{_NUM_PARAMS} params (all trainable)"
    known = _stub_trace([144, 256], param_bytes=_PARAM_BYTES)
    assert network_count_line(known, None, "unrolled") == "2 ops (400 B)"
    assert params_caption_detail(known) == f"{_NUM_PARAMS} params (680 B, all trainable)"
