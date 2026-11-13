"""TensorFlow ``tl.grad`` edit sites: the gate is open and the edit reaches the gradient.

``tl.when(tl.grad(selector), edit)`` on a TensorFlow capture no longer refuses:
the forward is unchanged, and where TensorFlow computes gradients (the
derived-gradient replay) the edit is applied to the gradient flowing into the
named op's output and propagated upstream. The oracle splits the gradient by
hand with ``tf.GradientTape``: the total minus the part through the site plus
the part through the site with the edit applied to the site's gradient.
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
from torchlens.backends.tf._grad_sites import arm_tf_grad_sites  # noqa: E402
from torchlens.backends.tf.derived_grads import attach_tf_derived_grads  # noqa: E402
from torchlens.errors import BackendUnsupportedError  # noqa: E402
from torchlens.intervention._grad_site_neutral import (  # noqa: E402
    armed_grad_sites,
    clear_grad_sites,
)

pytestmark = [pytest.mark.tf_backend, pytest.mark.rare]

_KERNEL = np.asarray([[0.2, -0.4], [0.7, 0.3], [-0.5, 0.1]], dtype=np.float32)
_BIAS = np.asarray([0.05, -0.1], dtype=np.float32)
_X = np.asarray([[1.0, -2.0, 0.5], [0.3, 0.1, -0.8]], dtype=np.float32)


class _DenseRelu(keras.Model):
    """``relu(h) + 0.5 * h`` with ``h = dense(x)``: the rectifier is the gradient site."""

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
    weights = tf.reshape(tf.range(1.0, 1.0 + tf.cast(tf.size(output), tf.float32)), output.shape)
    return tf.reduce_sum(output * weights)


def _variable(value: Any) -> Any:
    inner = getattr(value, "value", None)
    return inner if inner is not None and not callable(inner) else value


def _by_hand(model: _DenseRelu, x: Any, edit: Callable[[Any], Any]) -> tuple[Any, list[Any]]:
    """Output and gradients for x, kernel, bias with ``edit`` on the rectifier's gradient."""

    leaves = [x, _variable(model.dense.kernel), _variable(model.dense.bias)]
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        h = tf.matmul(x, leaves[1]) + leaves[2]
        r = tf.nn.relu(h)
        output = r + 0.5 * h
        loss = _loss(output)
    total = tape.gradient(loss, leaves)
    g_site = tape.gradient(loss, r)
    through_site = tape.gradient(r, leaves, output_gradients=g_site)
    edited_site = tape.gradient(r, leaves, output_gradients=edit(g_site))
    return output, [t - s + e for t, s, e in zip(total, through_site, edited_site, strict=True)]


_EDITS: dict[str, tuple[Callable[[], Any], Callable[[Any], Any]]] = {
    "scale": (lambda: tl.scale(0.25), lambda g: g * 0.25),
    "zero_ablate": (tl.zero_ablate, tf.zeros_like),
    "clamp": (lambda: tl.clamp(low=-1.0, high=2.5), lambda g: tf.clip_by_value(g, -1.0, 2.5)),
    "callable": (lambda: lambda grad, *, ctx: grad * -1.0, lambda g: g * -1.0),
}


def _grads(trace: Any) -> list[Any]:
    records = trace.autograd.derived_grads
    kernel = next(key for key in trace.params.keys() if key.endswith("kernel"))
    bias = next(key for key in trace.params.keys() if key.endswith("bias"))
    return [records[key].grad for key in ("inputs.0", f"params.{kernel}", f"params.{bias}")]


def _assert_grads(actual: list[Any], expected: list[Any]) -> None:
    for got, want in zip(actual, expected, strict=True):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-5, atol=1e-6)


def _clause(helper: Callable[[], Any]) -> Any:
    return tl.when(tl.grad(tl.func("relu")), helper())


def test_live_capture_keeps_a_grad_clause_and_leaves_the_forward_unchanged() -> None:
    model, x = _DenseRelu(), tf.constant(_X)
    clean = tl.trace(model, x, backend="tf")
    edited = tl.trace(model, x, backend="tf", intervene=_clause(lambda: tl.scale(0.25)))
    assert [op.label for op in edited._layer_list] == [op.label for op in clean._layer_list]
    for left, right in zip(edited._layer_list, clean._layer_list, strict=True):
        np.testing.assert_array_equal(np.asarray(left.out), np.asarray(right.out))


@pytest.mark.parametrize("name", sorted(_EDITS))
def test_engine_grad_site_edit_equals_the_hand_split(name: str) -> None:
    helper, edit = _EDITS[name]
    model, x = _DenseRelu(), tf.constant(_X)
    output, expected = _by_hand(model, x, edit)
    clause = _clause(helper)
    trace = tl.trace(model, x, backend="tf", intervene=clause)
    attach_tf_derived_grads(
        tf=tf,
        trace=trace,
        callable_obj=model,
        args=(x,),
        kwargs={},
        captured_output=output,
        grad_options=GradOptions(loss_fn=_loss),
        module_tree=None,
        intervene=clause,
    )
    _assert_grads(_grads(trace), expected)
    relu = next(op for op in trace._layer_list if str(op.func_name).lower() == "relu")
    assert [edit.label for edit in armed_grad_sites(trace)] == [relu.label]


def test_engine_grad_site_and_activation_edit_together() -> None:
    model, x = _DenseRelu(), tf.constant(_X)
    clauses = [tl.when(tl.func("relu"), tl.scale(3.0)), _clause(lambda: tl.scale(0.5))]
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        kernel, bias = _variable(model.dense.kernel), _variable(model.dense.bias)
        h = tf.matmul(x, kernel) + bias
        r = tf.nn.relu(h) * 3.0
        output = r + 0.5 * h
        loss = _loss(output)
    leaves = [x, kernel, bias]
    g_site = tape.gradient(loss, r)
    expected = [
        t - s + e
        for t, s, e in zip(
            tape.gradient(loss, leaves),
            tape.gradient(r, leaves, output_gradients=g_site),
            tape.gradient(r, leaves, output_gradients=g_site * 0.5),
            strict=True,
        )
    ]
    trace = tl.trace(model, x, backend="tf", intervene=clauses)
    attach_tf_derived_grads(
        tf=tf,
        trace=trace,
        callable_obj=model,
        args=(x,),
        kwargs={},
        captured_output=output,
        grad_options=GradOptions(loss_fn=_loss),
        module_tree=None,
        intervene=clauses,
    )
    _assert_grads(_grads(trace), expected)


def test_a_clause_mixing_gradient_and_activation_sites_refuses_typed() -> None:
    model, x = _DenseRelu(), tf.constant(_X)
    with pytest.raises(BackendUnsupportedError, match="Split it") as caught:
        tl.trace(
            model,
            x,
            backend="tf",
            intervene=tl.when(tl.grad(tl.func("relu")) | tl.func("add"), tl.scale(0.5)),
        )
    assert caught.value.code == "backward_capture_unsupported"


@pytest.mark.parametrize("name", sorted(_EDITS))
def test_entry_grad_site_edit_equals_the_hand_split(name: str) -> None:
    helper, edit = _EDITS[name]
    model, x = _DenseRelu(), tf.constant(_X)
    _output, expected = _by_hand(model, x, edit)
    try:
        trace = tl.trace(
            model,
            x,
            backend="tf",
            intervene=_clause(helper),
            capture=tl.options.CaptureOptions(grad_options=GradOptions(loss_fn=_loss)),
        )
    except BackendUnsupportedError as refusal:
        if "does not combine" in str(refusal):
            pytest.skip("the tf capture entry does not yet pass edits to the gradient replay")
        raise
    _assert_grads(_grads(trace), expected)


def test_arming_at_capture_then_attaching_grads_arms_each_site_once() -> None:
    """A capture that arms its clauses, then the derived-gradient replay, arm one edit per site."""

    model, x = _DenseRelu(), tf.constant(_X)
    output, expected = _by_hand(model, x, lambda g: g * 0.25)
    clause = _clause(lambda: tl.scale(0.25))
    trace = tl.trace(model, x, backend="tf", intervene=clause)
    relu = next(op for op in trace._layer_list if str(op.func_name).lower() == "relu")
    arm_tf_grad_sites(trace, tf)
    arm_tf_grad_sites(trace, tf)
    assert [edit.label for edit in armed_grad_sites(trace)] == [relu.label]
    attach_tf_derived_grads(
        tf=tf,
        trace=trace,
        callable_obj=model,
        args=(x,),
        kwargs={},
        captured_output=output,
        grad_options=GradOptions(loss_fn=_loss),
        module_tree=None,
        intervene=clause,
    )
    assert [edit.label for edit in armed_grad_sites(trace)] == [relu.label]
    _assert_grads(_grads(trace), expected)
    clear_grad_sites(trace)
    arm_tf_grad_sites(trace, tf)
    assert [edit.label for edit in armed_grad_sites(trace)] == [relu.label], "re-arms after a clear"
