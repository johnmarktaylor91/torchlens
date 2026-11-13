"""Paddle derived gradients of the program the capture executed.

With ``GradOptions``, an edited capture (``intervene=``) must report the
gradients of the edited program, a halted capture (``stop_after=``) the
gradients of the loss of the halted frontier, and a ``tl.grad(site)`` edit the
gradients with the edit applied to the gradient arriving at the site. Each is
compared with ``paddle.grad`` of the same program written out by hand.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

paddle = pytest.importorskip("paddle")

import torchlens as tl  # noqa: E402
from torchlens.backends.paddle import GradOptions  # noqa: E402
from torchlens.errors import BackendUnsupportedError  # noqa: E402

pytestmark = [pytest.mark.backend_paddle, pytest.mark.rare]


class _Net(paddle.nn.Layer):
    """``linear -> relu -> linear`` with fixed parameters; ``edit`` acts on the relu output."""

    def __init__(self, edit: Callable[[Any], Any] | None = None) -> None:
        super().__init__()
        self.first = paddle.nn.Linear(3, 4)
        self.second = paddle.nn.Linear(4, 2)
        self.first.weight.set_value(
            paddle.to_tensor(
                [[0.2, -0.4, 0.1, 0.5], [0.7, 0.3, -0.2, 0.4], [-0.5, 0.1, 0.6, -0.3]],
                dtype="float32",
            )
        )
        self.first.bias.set_value(paddle.to_tensor([0.05, -0.1, 0.2, 0.0], dtype="float32"))
        self.second.weight.set_value(
            paddle.to_tensor([[0.3, -0.2], [0.1, 0.4], [-0.6, 0.2], [0.5, 0.5]], dtype="float32")
        )
        self.second.bias.set_value(paddle.to_tensor([0.1, -0.05], dtype="float32"))
        self._edit = edit

    def forward(self, x: Any) -> Any:
        hidden = paddle.nn.functional.relu(self.first(x))
        if self._edit is not None:
            hidden = self._edit(hidden)
        return self.second(hidden)


def _x() -> Any:
    return paddle.to_tensor([[1.0, -2.0, 0.5], [0.3, 0.1, -0.8]], dtype="float32")


def _loss(output: Any) -> Any:
    return paddle.sum(output * output)


def _options(**kwargs: Any) -> Any:
    return tl.options.CaptureOptions(grad_options=GradOptions(loss_fn=_loss, **kwargs))


def _close(actual: Any, expected: Any) -> None:
    """Equal values; a missing gradient (``None``) equals zeros, and two missing ones agree."""

    if actual is None and expected is None:
        return  # no gradient reaches this leaf in either program
    actual = paddle.zeros_like(expected) if actual is None else actual
    expected = paddle.zeros_like(actual) if expected is None else expected
    assert paddle.allclose(actual, expected, rtol=1e-5, atol=1e-6), (actual, expected)


def _by_hand(net: Any, loss_of_output: Callable[[Any], Any], x: Any) -> tuple[Any, ...]:
    """``paddle.grad`` of ``loss_of_output(net(x))`` for the input and every parameter."""

    x = x.detach()
    x.stop_gradient = False
    params = [net.first.weight, net.first.bias, net.second.weight, net.second.bias]
    for param in params:
        param.stop_gradient = False
    return tuple(paddle.grad(loss_of_output(net(x)), [x, *params], allow_unused=True))


_PATHS = (
    "inputs.0",
    "params.first.weight",
    "params.first.bias",
    "params.second.weight",
    "params.second.bias",
)


def _derived(trace: Any, path: str) -> Any:
    """The derived gradient at ``path``, or ``None`` when the trace has no record for it."""

    grads = trace.autograd.derived_grads
    return grads[path].grad if path in grads else None


def _assert_derived(trace: Any, expected: tuple[Any, ...]) -> None:
    for path, grad in zip(_PATHS, expected, strict=True):
        _close(_derived(trace, path), grad)


_EDITS: dict[str, tuple[Any, Callable[[Any], Any]]] = {
    "scale": (tl.scale(3.0), lambda h: h * 3.0),
    "zero_ablate": (tl.zero_ablate(), paddle.zeros_like),
    "clamp": (tl.clamp(low=0.1, high=0.4), lambda h: paddle.clip(h, 0.1, 0.4)),
    "callable": (lambda h: h * h + 1.0, lambda h: h * h + 1.0),
}


@pytest.mark.parametrize("name", sorted(_EDITS))
def test_edited_capture_derived_grads_equal_the_edited_program(name: str) -> None:
    edit, by_hand = _EDITS[name]
    trace = tl.trace(
        _Net(),
        _x(),
        backend="paddle",
        intervene=tl.when(tl.func("functional.relu"), edit),
        capture=_options(),
    )
    expected = _by_hand(_Net(by_hand), _loss, _x())
    _assert_derived(trace, expected)
    unedited = _by_hand(_Net(), _loss, _x())
    edited_weight = paddle.zeros_like(unedited[1]) if expected[1] is None else expected[1]
    assert not paddle.allclose(edited_weight, unedited[1])


def test_module_exit_edit_derived_grads_equal_the_edited_program() -> None:
    trace = tl.trace(
        _Net(),
        _x(),
        backend="paddle",
        intervene=tl.when(tl.module("first"), tl.scale(2.0)),
        capture=_options(),
    )

    class _Scaled(_Net):
        def forward(self, x: Any) -> Any:
            return self.second(paddle.nn.functional.relu(self.first(x) * 2.0))

    _assert_derived(trace, _by_hand(_Scaled(), _loss, _x()))


def test_halted_capture_derived_grads_differentiate_the_frontier() -> None:
    trace = tl.trace(
        _Net(),
        _x(),
        backend="paddle",
        stop_after=tl.func("functional.relu"),
        capture=_options(),
    )

    class _Frontier(_Net):
        def forward(self, x: Any) -> Any:
            return paddle.nn.functional.relu(self.first(x))

    expected = _by_hand(_Frontier(), _loss, _x())
    assert expected[3] is None and expected[4] is None
    for path, grad in zip(_PATHS[:3], expected[:3], strict=True):
        _close(_derived(trace, path), grad)
    for path in _PATHS[3:]:
        grad = _derived(trace, path)
        assert grad is None or not bool(paddle.any(grad != 0))


def _site_edited_by_hand(scale: float) -> tuple[Any, Any, Any]:
    """Input and first-weight gradients with the relu-output gradient scaled, and that gradient."""

    net = _Net()
    x = _x().detach()
    x.stop_gradient = False
    for param in net.parameters():
        param.stop_gradient = False
    hidden = paddle.nn.functional.relu(net.first(x))
    hidden_leaf = hidden.detach()
    hidden_leaf.stop_gradient = False
    (grad_hidden,) = paddle.grad(_loss(net.second(hidden_leaf)), [hidden_leaf])
    edited = grad_hidden * scale
    grad_x, grad_w = paddle.grad([hidden], [x, net.first.weight], grad_outputs=[edited])
    return grad_x, grad_w, edited


def test_grad_site_edit_derived_grads_equal_the_edit_by_hand() -> None:
    trace = tl.trace(
        _Net(),
        _x(),
        backend="paddle",
        intervene=tl.when(tl.grad(tl.func("functional.relu")), tl.scale(0.5)),
        capture=_options(intermediate_grads=True),
    )
    grad_x, grad_w, edited = _site_edited_by_hand(0.5)
    _close(_derived(trace, "inputs.0"), grad_x)
    _close(_derived(trace, "params.first.weight"), grad_w)
    relu = next(op for op in trace._layer_list if "relu" in str(op.func_name))
    _close(trace.autograd.intermediate_derived_grads[relu.label].grad, edited)


def test_grad_site_with_an_activation_edit_on_the_gradient_path_refuses_by_name() -> None:
    """An activation edit the gradient passes through has no replay recipe: refuse, never drop."""

    from torchlens.errors import RunCapabilityUnavailableError

    spec = tl.when(tl.grad(tl.func("functional.relu")), tl.scale(0.5)) & tl.when(
        tl.module("second"), tl.scale(2.0)
    )
    with pytest.raises(RunCapabilityUnavailableError, match="cannot be replayed"):
        tl.trace(_Net(), _x(), backend="paddle", intervene=spec, capture=_options())


def test_autograd_node_selector_refuses_typed() -> None:
    with pytest.raises(BackendUnsupportedError, match="autograd-node"):
        tl.trace(
            _Net(),
            _x(),
            backend="paddle",
            intervene=tl.when(tl.grad_fn("relu"), tl.scale(0.5)),
        )


class _WithDropout(_Net):
    """A training-mode dropout the replay adapter refuses, whose output is still reproducible.

    The dropout acts on zeros, so the derived-gradient re-run reproduces the captured
    output whatever mask it draws; the gradient pass still reaches the dropout op.
    """

    def forward(self, x: Any) -> Any:
        hidden = paddle.nn.functional.relu(self.first(x))
        dropped = paddle.nn.functional.dropout(hidden * 0.0, p=0.5, training=True)
        return self.second(hidden + dropped)


def test_grad_site_reaching_an_unreplayable_op_refuses_by_name() -> None:
    from torchlens.errors import RunCapabilityUnavailableError

    paddle.seed(0)
    with pytest.raises(RunCapabilityUnavailableError, match="cannot be replayed") as caught:
        tl.trace(
            _WithDropout(),
            _x(),
            backend="paddle",
            intervene=tl.when(tl.grad(tl.func("functional.relu")), tl.scale(0.5)),
            capture=_options(),
        )
    assert "dropout" in str(caught.value)
