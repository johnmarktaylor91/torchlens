"""JAX parity: ``tl.grad`` sites edit the gradient in derived gradients.

``tl.trace(..., backend="jax", intervene=tl.when(tl.grad(site), edit))`` with
``grad_options`` applies ``edit`` to the cotangent flowing into each matched
equation output. Each test compares the derived gradients with ``jax.vjp`` of
the two program halves and the edit applied in between, written by hand.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

import torchlens as tl
from torchlens.backends.jax import GradOptions
from torchlens.errors import BackendUnsupportedError

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

pytestmark = [pytest.mark.backend_jax, pytest.mark.rare]

_SITE = tl.grad(tl.func("tanh"))


def _loss(output: Any) -> Any:
    """Return a nonlinear scalar loss."""

    return jnp.sum(output * output)


def _params() -> dict[str, Any]:
    """Return deterministic tiny parameters."""

    return {
        "w": jnp.asarray([[0.2, -0.4], [0.7, 0.3], [-0.5, 0.1]], dtype=jnp.float32),
        "v": jnp.asarray([1.5, -0.5], dtype=jnp.float32),
    }


def _x() -> Any:
    """Return a deterministic input batch."""

    return jnp.asarray([[1.0, -2.0, 0.5], [0.3, 0.1, -0.8]], dtype=jnp.float32)


def _model(params: dict[str, Any], x: Any) -> Any:
    """Return ``tanh(x @ w) * v`` (one ``tanh`` equation)."""

    return jnp.tanh(x @ params["w"]) * params["v"]


def _trace(model: Any, args: Any, intervene: Any, **grad_kwargs: Any) -> Any:
    """Trace ``model`` on JAX with ``intervene`` and derived gradients."""

    return tl.trace(
        cast(Any, model),
        args,
        backend="jax",
        intervene=intervene,
        capture=tl.options.CaptureOptions(grad_options=GradOptions(loss_fn=_loss, **grad_kwargs)),
    )


def _code(error: Any) -> str | None:
    """Return a typed refusal's code."""

    return getattr(error, "code", None) or error.fields.get("code")


def _oracle(params: dict[str, Any], x: Any, edit: Any) -> tuple[Any, Any]:
    """``jax.vjp`` of each half, ``edit`` applied to the tanh output's cotangent."""

    hidden, pull_hidden = jax.vjp(lambda p, xx: jnp.tanh(xx @ p["w"]), params, x)
    loss, pull_out = jax.vjp(lambda h, p: _loss(h * p["v"]), hidden, params)
    grad_hidden, grad_p_direct = pull_out(jnp.ones((), loss.dtype))
    grad_p_hidden, grad_x = pull_hidden(edit(grad_hidden))
    return jax.tree.map(jnp.add, grad_p_direct, grad_p_hidden), grad_x


@pytest.mark.parametrize(
    ("action", "edit"),
    [
        (tl.scale(0.5), lambda g: g * 0.5),
        (tl.zero_ablate(), jnp.zeros_like),
        (tl.clamp(low=-0.2, high=0.2), lambda g: jnp.clip(g, -0.2, 0.2)),
        (tl.add(np.asarray([0.25, -1.0], np.float32)), lambda g: g + jnp.asarray([0.25, -1.0])),
        (lambda grad, *, ctx: grad * 3.0, lambda g: g * 3.0),
    ],
)
def test_grad_site_edit_matches_vjp_by_hand(action: Any, edit: Any) -> None:
    """Param and input grads equal ``jax.vjp`` with the edit on the site's cotangent."""

    params, x = _params(), _x()
    trace = _trace(
        _model, (params, x), tl.when(_SITE, action), params=params, input_grad_argnums=(0,)
    )
    expected_p, expected_x = _oracle(params, x, edit)
    clean_p, _ = _oracle(params, x, lambda g: g)
    grads = trace.autograd.derived_grads
    assert jnp.allclose(grads["params.w"].grad, expected_p["w"], atol=1e-6)
    assert jnp.allclose(grads["params.v"].grad, expected_p["v"], atol=1e-6)
    assert jnp.allclose(grads["inputs.0"].grad, expected_x, atol=1e-6)
    assert not jnp.allclose(clean_p["w"], expected_p["w"])
    assert jnp.allclose(clean_p["v"], expected_p["v"])
    assert not [op for op in trace.ops if op.is_edited]
    np.testing.assert_allclose(
        np.asarray(trace[trace.output_layers[0]].out), np.asarray(_model(params, x)), rtol=1e-6
    )
    assert bool(trace.validate_forward_pass([]))


def test_grad_site_union_and_first_pass_qualifier() -> None:
    """``tl.grad(a) | tl.grad(b)`` edits both sites; ``in_backward_pass(1)`` is the pass."""

    def model(params: dict[str, Any], x: Any) -> Any:
        """Return ``tanh(sin(x @ w)) * v``."""

        return jnp.tanh(jnp.sin(x @ params["w"])) * params["v"]

    def by_hand(p: dict[str, Any], xx: Any) -> Any:
        """Return the loss with both site cotangents halved (custom VJP identities)."""

        @jax.custom_vjp
        def half(value: Any) -> Any:
            return value

        half.defvjp(lambda value: (value, None), lambda _r, g: (g * 0.5,))
        return _loss(half(jnp.tanh(half(jnp.sin(xx @ p["w"])))) * p["v"])

    params, x = _params(), _x()
    site = (tl.grad(tl.func("tanh")) | tl.grad(tl.func("sin"))) & tl.in_backward_pass(1)
    trace = _trace(model, (params, x), tl.when(site, tl.scale(0.5)), params=params)
    expected = jax.grad(by_hand)(params, x)
    grads = trace.autograd.derived_grads
    assert jnp.allclose(grads["params.w"].grad, expected["w"], atol=1e-6)
    assert jnp.allclose(grads["params.v"].grad, expected["v"], atol=1e-6)


@pytest.mark.parametrize(
    ("site", "with_grads"),
    [
        (_SITE, False),
        (_SITE | tl.func("tanh"), True),
        (_SITE & tl.in_backward_pass(2), True),
    ],
    ids=["no-grad-options", "mixed-universes", "second-pass"],
)
def test_grad_site_refuses_typed(site: Any, with_grads: bool) -> None:
    """No derived gradients, a mixed clause or a later pass refuse with a typed code."""

    params, x = _params(), _x()
    with pytest.raises(BackendUnsupportedError) as caught:
        if with_grads:
            _trace(_model, (params, x), tl.when(site, tl.scale(0.5)), params=params)
        else:
            tl.trace(
                cast(Any, _model),
                (params, x),
                backend="jax",
                intervene=tl.when(site, tl.scale(0.5)),
            )
    assert _code(caught.value) == "backward_capture_unsupported"


def test_grad_site_in_loop_body_refuses_typed() -> None:
    """A site bound once per loop iteration has no single gradient position."""

    def looped(params: dict[str, Any], x: Any) -> Any:
        """Return three tanh iterations under fori_loop, then a product with v."""

        h = jax.lax.fori_loop(0, 3, lambda _i, h: jnp.tanh(h @ params["w"][:2]), x[:, :2])
        return h * params["v"]

    params = _params()
    with pytest.raises(BackendUnsupportedError) as caught:
        _trace(looped, (params, _x()), tl.when(_SITE, tl.scale(0.5)), params=params)
    assert _code(caught.value) == "intervention_with_derived_grads_unsupported"
