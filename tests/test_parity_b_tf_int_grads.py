"""TensorFlow derived gradients of the edited program and of a halted boundary.

A capture with ``intervene=`` edits ran an edited program; its derived
gradients must be TensorFlow's gradient of that edited program, worked out
here by writing the edited function by hand and differentiating it with
``tf.GradientTape``. With ``stop_after=``, ``loss_fn`` receives the halted
boundary and the gradients are those of the loss at that boundary.

The engine tests call ``attach_tf_derived_grads`` on a live edited capture.
The entry tests go through ``tl.trace(..., grad_options=)`` and run once the
capture entry passes ``intervene=`` and ``stop_after=`` through to the
derived-gradient replay.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
keras = pytest.importorskip("keras")

import torchlens as tl  # noqa: E402
from torchlens.backends.tf import GradOptions  # noqa: E402
from torchlens.backends.tf.derived_grads import attach_tf_derived_grads  # noqa: E402
from torchlens.errors import BackendUnsupportedError  # noqa: E402

pytestmark = [pytest.mark.tf_backend, pytest.mark.rare]

_KERNEL = np.asarray([[0.2, -0.4], [0.7, 0.3], [-0.5, 0.1]], dtype=np.float32)
_BIAS = np.asarray([0.05, -0.1], dtype=np.float32)
_X = np.asarray([[1.0, -2.0, 0.5], [0.3, 0.1, -0.8]], dtype=np.float32)


class _DenseRelu(keras.Model):
    """``relu(h) + 0.5 * h`` with ``h = dense(x)``: the rectifier is the edit site."""

    def __init__(self) -> None:
        super().__init__(name="m")
        self.dense = keras.layers.Dense(2, name="dense")
        self.dense.build((None, 3))
        self.dense.kernel.assign(_KERNEL)
        self.dense.bias.assign(_BIAS)

    def call(self, x: Any) -> Any:
        h = self.dense(x)
        return tf.nn.relu(h) + 0.5 * h


def _loss(output: Any) -> Any:
    """A scalar loss that weights every output element differently."""

    weights = tf.reshape(tf.range(1.0, 1.0 + tf.cast(tf.size(output), tf.float32)), output.shape)
    return tf.reduce_sum(output * weights)


def _variable(value: Any) -> Any:
    """The TensorFlow variable behind a Keras variable."""

    inner = getattr(value, "value", None)
    return inner if inner is not None and not callable(inner) else value


def _by_hand(
    model: _DenseRelu, x: Any, edit: Callable[[Any], Any], *, halt: bool = False
) -> tuple[Any, list[Any]]:
    """The edited program's output and its tape gradients for x, kernel, bias."""

    kernel, bias = _variable(model.dense.kernel), _variable(model.dense.bias)
    with tf.GradientTape() as tape:
        tape.watch(x)
        h = tf.matmul(x, kernel) + bias
        site = edit(tf.nn.relu(h))
        output = site if halt else site + 0.5 * h
        loss = _loss(output)
    return output, tape.gradient(loss, [x, kernel, bias])


_EDITS: dict[str, tuple[Callable[[], Any], Callable[[Any], Any]]] = {
    "scale": (lambda: tl.scale(3.0), lambda r: r * 3.0),
    "zero_ablate": (tl.zero_ablate, tf.zeros_like),
    "clamp": (lambda: tl.clamp(low=0.1, high=0.3), lambda r: tf.clip_by_value(r, 0.1, 0.3)),
    "add": (lambda: tl.add(0.25), lambda r: r + 0.25),
    "replace_with": (
        lambda: tl.replace_with(tf.fill((2, 2), 0.5)),
        lambda r: tf.fill((2, 2), 0.5),
    ),
}


def _grads(trace: Any) -> list[Any]:
    """The trace's derived gradients for the input, the kernel and the bias."""

    records = trace.autograd.derived_grads
    kernel = next(key for key in trace.params.keys() if key.endswith("kernel"))
    bias = next(key for key in trace.params.keys() if key.endswith("bias"))
    return [records[key].grad for key in ("inputs.0", f"params.{kernel}", f"params.{bias}")]


def _assert_grads(actual: list[Any], expected: list[Any]) -> None:
    for got, want in zip(actual, expected, strict=True):
        want_array = np.zeros_like(np.asarray(got)) if want is None else np.asarray(want)
        got_array = np.zeros_like(want_array) if got is None else np.asarray(got)
        np.testing.assert_allclose(got_array, want_array, rtol=1e-5, atol=1e-6)


def _engine(model: Any, x: Any, *, intervene: Any = None, halt: Any = None, output: Any) -> Any:
    """A live edited capture with derived gradients attached by the replay engine."""

    trace = tl.trace(model, x, backend="tf", intervene=intervene, stop_after=halt)
    attach_tf_derived_grads(
        tf=tf,
        trace=trace,
        callable_obj=model,
        args=(x,),
        kwargs={},
        captured_output=output,
        grad_options=GradOptions(loss_fn=_loss),
        module_tree=None,
        intervene=intervene,
        halt=halt,
    )
    return trace


@pytest.mark.parametrize("name", sorted(_EDITS))
def test_engine_grads_are_the_edited_programs_grads(name: str) -> None:
    helper, edit = _EDITS[name]
    model, x = _DenseRelu(), tf.constant(_X)
    output, expected = _by_hand(model, x, edit)
    clause = tl.when(tl.func("relu"), helper())
    trace = _engine(model, x, intervene=clause, output=output)
    _assert_grads(_grads(trace), expected)


def test_engine_edit_changes_the_gradient() -> None:
    model, x = _DenseRelu(), tf.constant(_X)
    _clean_output, clean = _by_hand(model, x, lambda r: r)
    output, edited = _by_hand(model, x, lambda r: r * 3.0)
    trace = _engine(model, x, intervene=tl.when(tl.func("relu"), tl.scale(3.0)), output=output)
    assert not np.allclose(np.asarray(clean[0]), np.asarray(edited[0]))
    _assert_grads(_grads(trace), edited)


def test_engine_refuses_an_unedited_output_for_an_edited_capture() -> None:
    model, x = _DenseRelu(), tf.constant(_X)
    clean_output, _clean = _by_hand(model, x, lambda r: r)
    with pytest.raises(BackendUnsupportedError):
        _engine(model, x, intervene=tl.when(tl.func("relu"), tl.scale(3.0)), output=clean_output)


def test_engine_halt_differentiates_the_halted_boundary() -> None:
    model, x = _DenseRelu(), tf.constant(_X)
    output, expected = _by_hand(model, x, lambda r: r, halt=True)
    trace = _engine(model, x, halt=tl.func("relu"), output=output)
    _assert_grads(_grads(trace), expected)
    assert trace.is_stopped is True


def test_engine_halt_after_an_edit_differentiates_the_edited_boundary() -> None:
    model, x = _DenseRelu(), tf.constant(_X)
    output, expected = _by_hand(model, x, lambda r: r * 3.0, halt=True)
    trace = _engine(
        model,
        x,
        intervene=tl.when(tl.func("relu"), tl.scale(3.0)),
        halt=tl.func("relu"),
        output=output,
    )
    _assert_grads(_grads(trace), expected)


def test_engine_replay_leaves_the_halt_record_as_captured() -> None:
    model, x = _DenseRelu(), tf.constant(_X)
    output, _expected = _by_hand(model, x, lambda r: r, halt=True)
    trace = tl.trace(model, x, backend="tf", stop_after=tl.func("relu"))
    latch, reason = trace.__dict__.get("_stop_requested"), trace.stop_reason
    attach_tf_derived_grads(
        tf=tf,
        trace=trace,
        callable_obj=model,
        args=(x,),
        kwargs={},
        captured_output=output,
        grad_options=GradOptions(loss_fn=_loss),
        module_tree=None,
        halt=tl.func("relu"),
    )
    assert trace.__dict__.get("_stop_requested") is latch
    assert trace.stop_reason is reason


def _entry(model: Any, x: Any, **kwargs: Any) -> Any:
    """``tl.trace`` with grad_options; skipped while the entry still refuses the pair."""

    try:
        return tl.trace(
            model,
            x,
            backend="tf",
            capture=tl.options.CaptureOptions(grad_options=GradOptions(loss_fn=_loss)),
            **kwargs,
        )
    except BackendUnsupportedError as refusal:
        if "does not combine" in str(refusal):
            pytest.skip("the tf capture entry does not yet pass edits to the gradient replay")
        raise


@pytest.mark.parametrize("name", sorted(_EDITS))
def test_entry_grads_are_the_edited_programs_grads(name: str) -> None:
    helper, edit = _EDITS[name]
    model, x = _DenseRelu(), tf.constant(_X)
    _output, expected = _by_hand(model, x, edit)
    trace = _entry(model, x, intervene=tl.when(tl.func("relu"), helper()))
    _assert_grads(_grads(trace), expected)


def test_entry_halt_differentiates_the_halted_boundary() -> None:
    model, x = _DenseRelu(), tf.constant(_X)
    _output, expected = _by_hand(model, x, lambda r: r * 3.0, halt=True)
    trace = _entry(
        model,
        x,
        intervene=tl.when(tl.func("relu"), tl.scale(3.0)),
        stop_after=tl.func("relu"),
    )
    _assert_grads(_grads(trace), expected)
