"""tinygrad ``tl.grad`` edit sites in derived gradients.

``trace(intervene=tl.when(tl.grad(site), edit), grad_options=...)`` edits the
gradient of ``site``'s output and lets the edited gradient flow upstream. The
oracle is tinygrad's own ``backward`` of the two halves of the program with the
same edit applied by hand to the site gradient in between.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import torchlens as tl
from torchlens.backends.tinygrad import GradOptions
from torchlens.errors import BackendUnsupportedError

tinygrad = pytest.importorskip("tinygrad")
Tensor = tinygrad.Tensor

pytestmark = [pytest.mark.backend_tinygrad, pytest.mark.rare]

_X = np.asarray([[1.0, -0.5, 0.25, 2.0], [-1.5, 0.75, 1.0, -0.25]], dtype=np.float32)
_W1 = np.asarray(
    [[0.5, -0.25, 0.75], [0.25, 0.5, -0.5], [-0.75, 0.25, 0.5], [0.5, 0.75, 0.25]],
    dtype=np.float32,
)
_B1 = np.asarray([0.1, -0.2, 0.3], dtype=np.float32)
_W2 = np.asarray([[1.0, -0.5], [0.5, 0.25], [-0.25, 0.75]], dtype=np.float32)


class _TinyMLP:
    """Two-layer tinygrad MLP with fixed weights; the relu output is the gradient site."""

    def __init__(self) -> None:
        """Hold the fixed weights as realized tinygrad tensors."""

        self.w1 = Tensor(_W1).realize()
        self.b1 = Tensor(_B1).realize()
        self.w2 = Tensor(_W2).realize()

    def __call__(self, x: Any) -> Any:
        """Return ``relu(x @ w1 + b1) @ w2``."""

        return self.hidden(x).matmul(self.w2)

    def hidden(self, x: Any) -> Any:
        """Return ``relu(x @ w1 + b1)``."""

        return (x.matmul(self.w1) + self.b1).relu()


def _np(value: Any) -> np.ndarray:
    """A tinygrad value as a numpy array."""

    return np.asarray(value.numpy())


def _relu_op(trace: Any) -> Any:
    """The one relu op (tinygrad records relu as ``where``)."""

    ops = [op for op in trace._layer_list if op.func_name in {"relu", "where"}]
    assert len(ops) == 1, [op.func_name for op in trace._layer_list]
    return ops[0]


def _oracle(edit: Callable[[Any], Any]) -> tuple[np.ndarray, np.ndarray]:
    """``(input gradient, edited site gradient)`` by tinygrad ``backward`` of each half."""

    model = _TinyMLP()
    hidden = Tensor(_np(model.hidden(Tensor(_X))))
    hidden.matmul(model.w2).sum().backward()
    site_grad = Tensor(_np(edit(hidden.grad)))
    x = Tensor(_X)
    (model.hidden(x) * site_grad).sum().backward()
    return _np(x.grad), _np(site_grad)


def _options(**overrides: Any) -> Any:
    """Capture options with derived gradients of ``out.sum()`` for input 0."""

    fields: dict[str, Any] = {
        "loss_fn": lambda out: out.sum(),
        "input_grad_argnums": (0,),
        "intermediate_grads": True,
    }
    fields.update(overrides)
    return tl.options.CaptureOptions(grad_options=GradOptions(**fields))


_EDITS: list[tuple[str, Any, Callable[[Any], Any]]] = [
    ("scale", tl.scale(0.25), lambda g: g * 0.25),
    ("clamp", tl.clamp(low=-0.5, high=0.5), lambda g: g.clip(-0.5, 0.5)),
    ("callable", lambda grad, *, ctx: grad * 2.0, lambda g: g * 2.0),
]


@pytest.mark.parametrize(
    ("edit", "by_hand"),
    [pytest.param(edit, by_hand, id=name) for name, edit, by_hand in _EDITS],
)
def test_grad_site_edit_equals_tinygrad_by_hand(edit: Any, by_hand: Callable[[Any], Any]) -> None:
    """The input gradient and the site's gradient equal the edited backward by hand."""

    trace = tl.trace(
        _TinyMLP(),
        Tensor(_X),
        backend="tinygrad",
        intervene=tl.when(tl.grad(tl.func("where")), edit),
        capture=_options(),
    )
    input_grad, site_grad = _oracle(by_hand)
    np.testing.assert_allclose(
        _np(trace.autograd.derived_grads["inputs.0"].grad), input_grad, rtol=1e-5, atol=1e-6
    )
    relu = _relu_op(trace)
    np.testing.assert_allclose(
        _np(trace.autograd.intermediate_derived_grads[relu.label].grad),
        site_grad,
        rtol=1e-5,
        atol=1e-6,
    )
    # The forward is not edited: only the gradient is.
    assert relu.is_edited is not True


def test_identity_grad_site_edit_equals_plain_derived_gradients() -> None:
    """An identity edit gives the plain capture's derived gradients (engine == tinygrad)."""

    plain = tl.trace(_TinyMLP(), Tensor(_X), backend="tinygrad", capture=_options())
    armed = tl.trace(
        _TinyMLP(),
        Tensor(_X),
        backend="tinygrad",
        intervene=tl.when(tl.grad(tl.func("where")), lambda grad, *, ctx: grad),
        capture=_options(),
    )
    np.testing.assert_allclose(
        _np(armed.autograd.derived_grads["inputs.0"].grad),
        _np(plain.autograd.derived_grads["inputs.0"].grad),
        rtol=1e-5,
        atol=1e-6,
    )
    relu = _relu_op(armed).label
    np.testing.assert_allclose(
        _np(armed.autograd.intermediate_derived_grads[relu].grad),
        _np(plain.autograd.intermediate_derived_grads[relu].grad),
        rtol=1e-5,
        atol=1e-6,
    )


def test_grad_site_without_grad_options_refuses_typed() -> None:
    """A gradient edit needs a gradient computation: tinygrad has no later backward."""

    with pytest.raises(BackendUnsupportedError) as caught:
        tl.trace(
            _TinyMLP(),
            Tensor(_X),
            backend="tinygrad",
            intervene=tl.when(tl.grad(tl.func("where")), tl.scale(0.5)),
        )
    assert caught.value.code == "backward_capture_unsupported"


def test_grad_site_naming_no_op_refuses_typed() -> None:
    """A ``tl.grad`` site whose selector names no op refuses instead of editing nothing."""

    with pytest.raises(Exception, match="names no recorded op") as caught:
        tl.trace(
            _TinyMLP(),
            Tensor(_X),
            backend="tinygrad",
            intervene=tl.when(tl.grad(tl.func("conv2d")), tl.scale(0.5)),
            capture=_options(),
        )
    assert caught.value.fields["code"] == "run_capability_unavailable"
