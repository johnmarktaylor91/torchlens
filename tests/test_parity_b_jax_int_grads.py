"""JAX parity: derived gradients differentiate the edited program of ``intervene=``.

Each test traces with a capture-time edit and ``grad_options`` and compares the
trace's derived gradients with ``jax.grad`` of the edited program written out
by hand. The unedited gradient is checked to differ, so a path that silently
reran the unedited model would fail.
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

_CODE = "intervention_with_derived_grads_unsupported"


def _loss(output: Any) -> Any:
    """Return a nonlinear scalar loss, so an additive edit changes the gradient."""

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


def _grads_trace(model: Any, args: Any, intervene: Any, **grad_kwargs: Any) -> Any:
    """Trace ``model`` on JAX with ``intervene`` and derived gradients."""

    options = GradOptions(loss_fn=_loss, **grad_kwargs)
    return tl.trace(
        cast(Any, model),
        args,
        backend="jax",
        intervene=intervene,
        capture=tl.options.CaptureOptions(grad_options=options),
    )


def _refusal_code(model: Any, args: Any, intervene: Any, **grad_kwargs: Any) -> str | None:
    """Return the typed refusal code of an edited derived-gradient trace."""

    with pytest.raises(BackendUnsupportedError) as caught:
        _grads_trace(model, args, intervene, **grad_kwargs)
    error = cast(Any, caught.value)
    return getattr(error, "code", None) or error.fields.get("code")


def _hand_grads(edited: Any, params: Any, x: Any) -> tuple[Any, Any]:
    """Return ``jax.grad`` of the hand-written edited program for params and input."""

    return jax.grad(lambda p, xx: _loss(edited(p, xx)), argnums=(0, 1))(params, x)


@pytest.mark.parametrize(
    ("action", "edit"),
    [
        (tl.scale(0.5), lambda h: h * 0.5),
        (tl.zero_ablate(), jnp.zeros_like),
        (tl.add(np.asarray([0.25, -1.0], np.float32)), lambda h: h + jnp.asarray([0.25, -1.0])),
        (tl.clamp(low=-0.3, high=0.3), lambda h: jnp.clip(h, -0.3, 0.3)),
        (tl.mean_ablate(dim=None), lambda h: jnp.full(h.shape, jnp.mean(h), h.dtype)),
        (
            tl.replace_with(jnp.full((2, 2), 0.4, jnp.float32)),
            lambda h: jnp.full((2, 2), 0.4, jnp.float32),
        ),
        (lambda out, *, ctx: out * out, lambda h: h * h),
    ],
)
def test_edited_function_root_grads_match_hand_edited_program(action: Any, edit: Any) -> None:
    """Param and input grads equal ``jax.grad`` of the edited function, not the clean one."""

    params, x = _params(), _x()
    trace = _grads_trace(
        _model,
        (params, x),
        tl.when(tl.func("tanh"), action),
        params=params,
        input_grad_argnums=(0,),
    )

    def edited(p: dict[str, Any], xx: Any) -> Any:
        """Return the hand-edited forward."""

        return edit(jnp.tanh(xx @ p["w"])) * p["v"]

    expected_p, expected_x = _hand_grads(edited, params, x)
    clean_p, _ = _hand_grads(_model, params, x)
    grads = trace.autograd.derived_grads
    assert jnp.allclose(grads["params.w"].grad, expected_p["w"], atol=1e-6)
    assert jnp.allclose(grads["params.v"].grad, expected_p["v"], atol=1e-6)
    assert jnp.allclose(grads["inputs.0"].grad, expected_x, atol=1e-6)
    assert not jnp.allclose(clean_p["v"], expected_p["v"])
    assert trace.params["w"].grad is grads["params.w"].grad
    assert len([op for op in trace.ops if op.is_edited]) == 1
    assert bool(trace.validate_forward_pass([]))


def test_edited_grads_through_closure_constants_and_jit() -> None:
    """An edit inside a jitted helper, with a closed-over array after it."""

    shift = jnp.asarray([0.3, -0.7], dtype=jnp.float32)

    @jax.jit
    def block(w: Any, x: Any) -> Any:
        """Return ``sin(x @ w)`` under ``jit``."""

        return jnp.sin(x @ w)

    def model(params: dict[str, Any], x: Any) -> Any:
        """Return a jitted block, a closed-over shift and a tanh."""

        return jnp.tanh(block(params["w"], x) + shift) * params["v"]

    def edited(p: dict[str, Any], xx: Any) -> Any:
        """Return the hand-edited forward (sin output scaled by 3)."""

        return jnp.tanh(jnp.sin(xx @ p["w"]) * 3.0 + shift) * p["v"]

    params, x = _params(), _x()
    trace = _grads_trace(model, (params, x), tl.when(tl.func("sin"), tl.scale(3.0)), params=params)
    expected_p, _ = _hand_grads(edited, params, x)
    grads = trace.autograd.derived_grads
    assert jnp.allclose(grads["params.w"].grad, expected_p["w"], atol=1e-6)
    assert jnp.allclose(grads["params.v"].grad, expected_p["v"], atol=1e-6)


def test_edited_grads_on_equinox_module_root() -> None:
    """Module-root grads differentiate the edited module forward."""

    eqx = pytest.importorskip("equinox")

    class Mlp(eqx.Module):
        """Two-layer Equinox MLP with a tanh hidden layer."""

        fc1: Any
        fc2: Any

        def __init__(self) -> None:
            """Initialize deterministic layers."""

            key1, key2 = jax.random.split(jax.random.PRNGKey(10))
            self.fc1 = eqx.nn.Linear(3, 4, key=key1)
            self.fc2 = eqx.nn.Linear(4, 2, key=key2)

        def __call__(self, x: Any) -> Any:
            """Return the MLP output."""

            return self.fc2(jnp.tanh(self.fc1(x)))

    model = Mlp()
    x = jnp.asarray([0.5, -1.0, 2.0], dtype=jnp.float32)
    trace = tl.trace(
        model,
        x,
        backend="jax",
        intervene=tl.when(tl.func("tanh"), tl.scale(0.5)),
        capture=tl.options.CaptureOptions(grad_options=GradOptions(params=model, loss_fn=_loss)),
    )
    expected = eqx.filter_grad(lambda m: _loss(m.fc2(0.5 * jnp.tanh(m.fc1(x)))))(model)
    clean = eqx.filter_grad(lambda m: _loss(m(x)))(model)
    for address, ref in (
        ("fc1.weight", expected.fc1.weight),
        ("fc1.bias", expected.fc1.bias),
        ("fc2.weight", expected.fc2.weight),
        ("fc2.bias", expected.fc2.bias),
    ):
        assert jnp.allclose(trace.params[address].grad, ref, atol=1e-6), address
    assert not jnp.allclose(clean.fc2.weight, expected.fc2.weight)
    assert bool(trace.validate_forward_pass([]))


def test_edit_without_grads_still_edits_everywhere() -> None:
    """Edits inside control flow keep working when no gradients are asked for."""

    def looped(x: Any) -> Any:
        """Return three tanh iterations under fori_loop."""

        return jax.lax.fori_loop(0, 3, lambda _i, h: jnp.tanh(h), x)

    trace = tl.trace(
        cast(Any, looped),
        _x(),
        backend="jax",
        intervene=tl.when(tl.func("tanh"), tl.zero_ablate()),
    )
    assert np.allclose(np.asarray(trace[trace.output_layers[0]].out), 0.0)


def test_edit_in_loop_body_with_grads_refuses_typed() -> None:
    """An edit on an equation bound once per iteration has no single gradient position."""

    def looped(params: dict[str, Any], x: Any) -> Any:
        """Return three tanh iterations under fori_loop, then a product with v."""

        h = jax.lax.fori_loop(0, 3, lambda _i, h: jnp.tanh(h @ params["w"][:2]), x[:, :2])
        return h * params["v"]

    params = _params()
    code = _refusal_code(
        looped, (params, _x()), tl.when(tl.func("tanh"), tl.scale(0.5)), params=params
    )
    assert code == _CODE


def test_edit_in_cond_branch_with_grads_refuses_typed() -> None:
    """An edit inside a cond branch cannot be re-applied when the branch binds whole."""

    def branchy(params: dict[str, Any], x: Any) -> Any:
        """Return a cond over a value-dependent predicate."""

        h = x @ params["w"]
        h = jax.lax.cond(jnp.sum(h) > -100.0, jnp.tanh, jnp.sin, h)
        return h * params["v"]

    params = _params()
    code = _refusal_code(
        branchy, (params, _x()), tl.when(tl.func("tanh"), tl.scale(0.5)), params=params
    )
    assert code == _CODE


def test_edit_with_intermediate_grads_refuses_typed() -> None:
    """Intermediate derived gradients replay the jaxpr per boundary and refuse edits."""

    params = _params()
    code = _refusal_code(
        _model,
        (params, _x()),
        tl.when(tl.func("tanh"), tl.scale(0.5)),
        params=params,
        intermediate_grads=True,
    )
    assert code == _CODE


def test_unmatched_edit_keeps_clean_grads() -> None:
    """A predicate that never fires leaves the clean derived gradients."""

    params, x = _params(), _x()
    with pytest.warns(UserWarning, match="matched zero sites"):
        trace = _grads_trace(
            _model, (params, x), tl.when(tl.func("cosh"), tl.scale(0.5)), params=params
        )
    clean_p, _ = _hand_grads(_model, params, x)
    assert jnp.allclose(trace.autograd.derived_grads["params.w"].grad, clean_p["w"], atol=1e-6)


def test_halted_capture_refuses_derived_grads_typed() -> None:
    """A trace halted by ``stop_after=`` never differentiates the whole jaxpr."""

    from types import SimpleNamespace

    from torchlens.backends.jax._converted_grads import attach_jax_derived_grads

    with pytest.raises(BackendUnsupportedError) as caught:
        attach_jax_derived_grads(
            object(),
            trace=SimpleNamespace(is_stopped=True),
            capture=cast(Any, None),
            derive=lambda *_args: None,
        )
    error = cast(Any, caught.value)
    assert (getattr(error, "code", None) or error.fields.get("code")) == _CODE
