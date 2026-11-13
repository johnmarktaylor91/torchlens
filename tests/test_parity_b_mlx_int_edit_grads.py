"""MLX derived gradients of the edited or halted program, and ``tl.grad`` sites.

Each capture combines ``grad_options=`` with ``intervene=`` or ``stop_after=``.
The expected gradients are MLX's own ``mx.grad`` / ``mx.vjp`` of the edited (or
truncated) program written out by hand, never the capture's replay.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

pytestmark = [pytest.mark.backend_mlx, pytest.mark.rare]

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

import torchlens as tl  # noqa: E402
from torchlens.backends.mlx import GradOptions, MLXBackend  # noqa: E402
from torchlens.errors import BackendUnsupportedError  # noqa: E402

_RTOL, _ATOL = 1e-5, 1e-6


class _MLP(nn.Module):
    """``l2(relu(l1(x)))`` with fixed weights (mixed signs, so relu is not all-on)."""

    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.Linear(4, 3)
        self.l2 = nn.Linear(3, 2)
        self.l1.weight = mx.array(
            [[0.5, -0.25, 0.1, 0.3], [-0.4, 0.2, 0.6, -0.1], [0.3, 0.3, -0.2, 0.5]]
        )
        self.l1.bias = mx.array([0.1, -0.05, 0.2])
        self.l2.weight = mx.array([[0.25, -0.5, 0.4], [0.6, 0.1, -0.3]])
        self.l2.bias = mx.array([0.05, -0.2])

    def __call__(self, x: Any) -> Any:
        return self.l2(nn.relu(self.l1(x)))


def _x() -> Any:
    return mx.array([[1.0, 0.5, -0.5, 2.0], [0.25, -1.0, 1.5, 0.75]])


def _loss(out: Any) -> Any:
    return mx.sum(out * out)


def _capture(model: _MLP, **kwargs: Any) -> Any:
    """Capture ``model`` with parameter and input gradients and the given edit or halt."""

    return tl.trace(
        model,
        _x(),
        backend="mlx",
        capture=tl.options.CaptureOptions(
            grad_options=GradOptions(loss_fn=_loss, input_grad_argnums=(0,))
        ),
        **kwargs,
    )


def _reference(model: _MLP, program: Callable[[_MLP, Any], Any]) -> dict[str, Any]:
    """``mx.grad`` of ``program`` over the parameters and the input, keyed like derived grads."""

    params = model.parameters()

    def value(p: Any, x: Any) -> Any:
        model.update(p)
        return program(model, x)

    try:
        grad_params, grad_x = mx.grad(value, argnums=(0, 1))(params, _x())
    finally:
        model.update(params)
    found = {"inputs.0": grad_x}
    for layer in ("l1", "l2"):
        for name in ("weight", "bias"):
            found[f"params.{layer}.{name}"] = grad_params[layer][name]
    mx.eval(found)
    return found


def _assert_grads(trace: Any, expected: dict[str, Any], keys: tuple[str, ...] = ()) -> None:
    """Every expected record (or the named ones) equals the capture's derived gradient."""

    derived = trace.autograd.derived_grads
    for key in keys or tuple(expected):
        got = derived[key].grad
        mx.eval(got)
        np.testing.assert_allclose(
            np.asarray(got), np.asarray(expected[key]), rtol=_RTOL, atol=_ATOL, err_msg=key
        )


_FORWARD_EDITS: list[tuple[str, Callable[[], Any], Callable[[Any], Any]]] = [
    ("scale", lambda: tl.scale(2.0), lambda h: h * 2.0),
    ("zero_ablate", tl.zero_ablate, mx.zeros_like),
    ("add", lambda: tl.add(0.5), lambda h: h + 0.5),
    ("clamp", lambda: tl.clamp(high=0.3), lambda h: mx.clip(h, None, 0.3)),
    ("mean_ablate", lambda: tl.mean_ablate(dim=None), lambda h: mx.zeros_like(h) + mx.mean(h)),
    ("callable", lambda: lambda out, *, ctx: out * out, lambda h: h * h),
    (
        "steer",
        lambda: tl.steer(mx.array([1.0, -1.0, 0.5]), strength=0.5, dim=1),
        lambda h: h + mx.array([[1.0, -1.0, 0.5]]) * 0.5,
    ),
    (
        "clip_norm",
        lambda: tl.clip_norm(0.2),
        lambda h: h * mx.minimum(1.0, 0.2 / mx.sqrt(mx.sum(h * h))),
    ),
]


@pytest.mark.parametrize(
    ("name", "edit", "hand"), _FORWARD_EDITS, ids=[case[0] for case in _FORWARD_EDITS]
)
def test_forward_edit_derived_grads_equal_mx_grad_of_edited_program(
    name: str, edit: Callable[[], Any], hand: Callable[[Any], Any]
) -> None:
    """Parameter and input gradients are those of ``l2(edit(relu(l1(x))))``."""

    model = _MLP()
    trace = _capture(model, intervene=tl.when(tl.func("relu"), edit()))
    expected = _reference(model, lambda m, x: _loss(m.l2(hand(nn.relu(m.l1(x))))))
    clean = _reference(model, lambda m, x: _loss(m(x)))
    _assert_grads(trace, expected)
    assert set(trace.autograd.derived_grads.keys()) == set(expected), name
    assert not np.allclose(np.asarray(expected["inputs.0"]), np.asarray(clean["inputs.0"])), name
    assert trace.params["l1.weight"].grad is trace.autograd.derived_grads["params.l1.weight"].grad


def test_halted_capture_derived_grads_equal_mx_grad_of_the_prefix() -> None:
    """``stop_after=`` with grad_options: gradients of ``loss(relu(l1(x)))``."""

    model = _MLP()
    trace = _capture(model, stop_after=tl.func("relu"))
    assert trace.is_stopped
    expected = _reference(model, lambda m, x: _loss(nn.relu(m.l1(x))))
    _assert_grads(trace, expected, keys=("inputs.0", "params.l1.weight", "params.l1.bias"))
    for key in ("params.l2.weight", "params.l2.bias"):
        if key in trace.autograd.derived_grads:
            grad = trace.autograd.derived_grads[key].grad
            mx.eval(grad)
            assert not np.any(np.asarray(grad)), key


def test_edit_then_halt_derived_grads_equal_mx_grad_of_the_edited_prefix() -> None:
    """An edit before the halt frontier is in the replayed program too."""

    model = _MLP()
    trace = _capture(
        model,
        intervene=tl.when(tl.func("relu"), tl.scale(3.0)),
        stop_after=tl.func("relu"),
    )
    expected = _reference(model, lambda m, x: _loss(nn.relu(m.l1(x)) * 3.0))
    _assert_grads(trace, expected, keys=("inputs.0", "params.l1.weight", "params.l1.bias"))


def _final(trace: Any) -> Any:
    """The trace's model output value."""

    outputs = [op for op in trace._layer_list if getattr(op, "is_output", False)]
    return (outputs or list(trace._layer_list))[-1].out


def _suffix_cotangent(model: _MLP) -> tuple[Any, Any]:
    """The relu output and the gradient of the loss arriving at it, by hand."""

    hidden = nn.relu(model.l1(_x()))
    arriving = mx.grad(lambda h: _loss(model.l2(h)))(hidden)
    mx.eval(hidden, arriving)
    return hidden, arriving


def _expected_with_cotangent(model: _MLP, edited: Any) -> dict[str, Any]:
    """Gradients when ``edited`` replaces the cotangent at the relu output (``mx.vjp``)."""

    def prefix(weight: Any, bias: Any, x: Any) -> Any:
        return nn.relu(x @ weight.T + bias)

    primals = [model.l1.weight, model.l1.bias, _x()]
    _, (grad_weight, grad_bias, grad_x) = mx.vjp(prefix, primals, [edited])
    suffix = _reference(model, lambda m, x: _loss(m(x)))
    found = {
        "inputs.0": grad_x,
        "params.l1.weight": grad_weight,
        "params.l1.bias": grad_bias,
        "params.l2.weight": suffix["params.l2.weight"],
        "params.l2.bias": suffix["params.l2.bias"],
    }
    mx.eval(found)
    return found


_GRAD_EDITS: list[tuple[str, Callable[[], Any], Callable[[Any], Any]]] = [
    ("scale", lambda: tl.scale(0.5), lambda g: g * 0.5),
    ("zero_ablate", tl.zero_ablate, mx.zeros_like),
    ("callable", lambda: lambda grad, *, ctx: -grad, lambda g: -g),
    (
        "clip_norm",
        lambda: tl.clip_norm(0.05),
        lambda g: g * min(1.0, 0.05 / float(np.linalg.norm(np.asarray(g).reshape(-1)))),
    ),
]


@pytest.mark.parametrize(
    ("name", "edit", "hand"), _GRAD_EDITS, ids=[case[0] for case in _GRAD_EDITS]
)
def test_grad_site_edits_the_gradient_arriving_at_the_op(
    name: str, edit: Callable[[], Any], hand: Callable[[Any], Any]
) -> None:
    """``tl.grad(site)``: the forward is unchanged; upstream gradients use the edited cotangent."""

    model = _MLP()
    clean = tl.trace(model, _x(), backend="mlx")
    trace = _capture(model, intervene=tl.when(tl.grad(tl.func("relu")), edit()))
    clean_out, edited_out = _final(clean), _final(trace)
    mx.eval(clean_out, edited_out)
    np.testing.assert_array_equal(np.asarray(edited_out), np.asarray(clean_out))
    _hidden, arriving = _suffix_cotangent(model)
    _assert_grads(trace, _expected_with_cotangent(model, hand(arriving)))
    assert name


def _chain(v: Any) -> Any:
    """``sum((relu(v*v - 1) + 3)^2)`` with a nonlinear suffix so every op's gradient varies."""

    a = mx.multiply(v, v)
    h = mx.maximum(mx.subtract(a, 1.0), 0.0)
    shifted = mx.add(h, 3.0)
    return mx.sum(mx.multiply(shifted, shifted))


def _chain_trace(**kwargs: Any) -> Any:
    return tl.trace(
        _chain,
        mx.array([1.5, -2.0]),
        backend="mlx",
        capture=tl.options.CaptureOptions(
            grad_options=GradOptions(input_grad_argnums=(0,), intermediate_grads=True)
        ),
        **kwargs,
    )


def _op(trace: Any, func_name: str) -> Any:
    return next(op for op in trace._layer_list if op.func_name == func_name)


def test_intermediate_grads_with_a_forward_edit() -> None:
    """Op gradients are those of the edited program, taken at the edited values."""

    trace = _chain_trace(intervene=tl.when(tl.func("maximum"), tl.scale(2.0)))
    square, site = _op(trace, "multiply"), _op(trace, "maximum")
    a = mx.array([1.5, -2.0]) ** 2
    expected_site = mx.grad(lambda b: mx.sum((b + 3.0) ** 2))(site.out)
    expected_square = mx.grad(lambda s: mx.sum((2.0 * mx.maximum(s - 1.0, 0.0) + 3.0) ** 2))(a)
    for op, expected in ((site, expected_site), (square, expected_square)):
        assert op.label in trace.autograd.intermediate_derived_grads, op.label
        got = op.autograd.derived_grad
        mx.eval(got, expected)
        np.testing.assert_allclose(np.asarray(got), np.asarray(expected), rtol=_RTOL, atol=_ATOL)


def test_intermediate_grads_at_a_grad_site_hold_the_edited_gradient() -> None:
    """At a ``tl.grad`` site the op's gradient is the edited one (as torch saves it)."""

    trace = _chain_trace(intervene=tl.when(tl.grad(tl.func("maximum")), tl.scale(0.5)))
    square, site = _op(trace, "multiply"), _op(trace, "maximum")
    a = mx.array([1.5, -2.0]) ** 2
    h = mx.maximum(a - 1.0, 0.0)
    expected_site = 0.5 * 2.0 * (h + 3.0)
    expected_square = expected_site * (a > 1.0)
    for op, expected in ((site, expected_site), (square, expected_square)):
        assert op.label in trace.autograd.intermediate_derived_grads, op.label
        got = op.autograd.derived_grad
        mx.eval(got, expected)
        np.testing.assert_allclose(np.asarray(got), np.asarray(expected), rtol=_RTOL, atol=_ATOL)
    leaf = trace.autograd.derived_grads["inputs.0"].grad
    v = mx.array([1.5, -2.0])
    mx.eval(leaf)
    np.testing.assert_allclose(
        np.asarray(leaf), np.asarray(expected_square * 2.0 * v), rtol=_RTOL, atol=_ATOL
    )


def test_grad_site_without_grad_options_refuses() -> None:
    """MLX edits gradients only in the derived-gradient run."""

    with pytest.raises(BackendUnsupportedError, match="grad_options") as info:
        tl.trace(
            _MLP(),
            _x(),
            backend="mlx",
            intervene=tl.when(tl.grad(tl.func("relu")), tl.scale(0.5)),
        )
    assert info.value.code == "intervention_with_derived_grads_unsupported"


def test_module_edit_with_grad_options_refuses() -> None:
    """A tl.module edit applies outside any wrapped call; the replay cannot reproduce it."""

    with pytest.raises(BackendUnsupportedError, match="tl.module") as info:
        _capture(_MLP(), intervene=tl.when(tl.module("l1"), tl.scale(2.0)))
    assert info.value.code == "intervention_with_derived_grads_unsupported"


@pytest.mark.parametrize(
    "site",
    [
        pytest.param(lambda: tl.grad(tl.func("relu")) | tl.func("relu"), id="mixed-universe"),
        pytest.param(lambda: tl.grad(tl.module("l1")), id="grad-of-module"),
    ],
)
def test_unsupported_grad_site_shapes_refuse(site: Callable[[], Any]) -> None:
    """Mixed activation/gradient sites and gradient-of-module sites refuse typed."""

    with pytest.raises(BackendUnsupportedError) as info:
        _capture(_MLP(), intervene=tl.when(site(), tl.scale(0.5)))
    assert info.value.code == "backward_capture_unsupported"


def test_replay_that_misses_an_edit_site_refuses() -> None:
    """A model that takes another path in the gradient run refuses instead of answering."""

    class _Flaky(_MLP):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def __call__(self, x: Any) -> Any:
            self.calls += 1
            hidden = self.l1(x)
            return self.l2(nn.relu(hidden) if self.calls == 1 else nn.gelu(hidden))

    with pytest.raises(BackendUnsupportedError) as info:
        _capture(_Flaky(), intervene=tl.when(tl.func("relu"), tl.scale(2.0)))
    assert info.value.code == "intervention_with_derived_grads_unsupported"


def test_plain_grad_options_capture_is_unchanged() -> None:
    """Without an edit or halt the derived-gradient run calls the model itself."""

    model = _MLP()
    trace = MLXBackend().capture_trace(
        model, _x(), grad_options=GradOptions(loss_fn=_loss, input_grad_argnums=(0,))
    )
    _assert_grads(trace, _reference(model, lambda m, x: _loss(m(x))))
