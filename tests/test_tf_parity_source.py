"""TensorFlow source lines point at user code, like torch's ``code_context``.

Each recorded op carries the user frames between the ``tl.trace`` call and the op,
outermost first, with TensorFlow, Keras and torchlens frames filtered out. Keras
functional models run no user code inside the forward, so their ops carry only the
``tl.trace`` call frame. ``save_source_context=True`` loads the surrounding lines.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest
from conftest import tensorflow_backend_modules

import torchlens as tl

tf, keras, _TF_BACKEND_SKIP_REASON = tensorflow_backend_modules()

pytestmark = [
    pytest.mark.tf_backend,
    pytest.mark.skipif(
        _TF_BACKEND_SKIP_REASON is not None,
        reason=_TF_BACKEND_SKIP_REASON or "TensorFlow backend stack is supported",
    ),
    pytest.mark.rare,
]

_THIS_FILE = __file__
_LINES: dict[str, int] = {}


def _mark(name: str) -> None:
    """Record the line number of the caller's next line under ``name``."""

    frame = inspect.currentframe()
    assert frame is not None and frame.f_back is not None
    _LINES[name] = frame.f_back.f_lineno + 1


def _user_forward(x: Any) -> Any:
    """Raw eager forward with two user-visible op lines."""

    _mark("relu")
    hidden = tf.nn.relu(x)
    _mark("sum")
    return tf.reduce_sum(hidden * 2.0)


class _Subclassed(keras.Model if keras is not None else object):  # type: ignore[misc]
    """Subclassed Keras model whose ``call`` holds a user op line."""

    def __init__(self) -> None:
        super().__init__()
        self.dense = keras.layers.Dense(3)

    def call(self, x: Any) -> Any:
        _mark("subclassed_tanh")
        return tf.math.tanh(self.dense(x))


def _frames(trace: Any, func_name: str) -> list[Any]:
    """Return the source frames of the first layer recording ``func_name``."""

    layer = next(layer for layer in trace._layer_list if str(layer.func_name) == func_name)
    return list(layer.source.frames or ())


def test_tf_raw_eager_source_lines_point_at_user_code() -> None:
    """Raw eager ops carry the trace call and the user line, outermost first."""

    _mark("trace_call")
    trace = tl.trace(_user_forward, tf.constant([1.0, -2.0, 3.0]), backend="tf")
    relu_frames = _frames(trace, "Relu")
    assert [frame.file for frame in relu_frames] == [_THIS_FILE, _THIS_FILE]
    assert relu_frames[0].line_number == _LINES["trace_call"]
    assert relu_frames[0].func_name == "test_tf_raw_eager_source_lines_point_at_user_code"
    assert relu_frames[-1].line_number == _LINES["relu"]
    assert relu_frames[-1].func_name == "_user_forward"
    sum_frames = _frames(trace, "Sum")
    assert sum_frames[-1].line_number == _LINES["sum"]


def test_tf_subclassed_model_source_lines_point_at_call() -> None:
    """A subclassed model's op carries its ``call`` line; Keras frames are filtered."""

    model = _Subclassed()
    _mark("trace_call")
    trace = tl.trace(model, tf.ones((2, 4)), backend="tf")
    frames = _frames(trace, "Tanh")
    assert all(frame.file == _THIS_FILE for frame in frames)
    assert frames[0].line_number == _LINES["trace_call"]
    assert frames[-1].line_number == _LINES["subclassed_tanh"]
    assert frames[-1].func_name == "call"


def test_tf_functional_model_source_lines_stop_at_trace_call() -> None:
    """A functional model runs no user code in its forward: only the trace call frame."""

    inputs = keras.Input(shape=(4,))
    model = keras.Model(inputs, keras.layers.Dense(2, activation="relu")(inputs))
    _mark("trace_call")
    trace = tl.trace(model, tf.ones((2, 4)), backend="tf")
    frames = _frames(trace, "Relu")
    assert [(frame.file, frame.line_number) for frame in frames] == [
        (_THIS_FILE, _LINES["trace_call"])
    ]


def test_tf_save_source_context_loads_code_lines() -> None:
    """``save_source_context=True`` loads the user's code lines around each frame."""

    trace = tl.trace(
        _user_forward,
        tf.constant([1.0, -2.0, 3.0]),
        backend="tf",
        capture=tl.options.CaptureOptions(save_source_context=True),
    )
    innermost = _frames(trace, "Relu")[-1]
    context = innermost.code_context or []
    assert any("tf.nn.relu(x)" in line for line in context)
