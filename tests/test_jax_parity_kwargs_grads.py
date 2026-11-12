"""JAX parity: keyword inputs and derived gradients on module roots and closures."""

from __future__ import annotations

from typing import Any, cast

import pytest

import torchlens as tl
from torchlens.backends.jax import GradOptions

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

pytestmark = [pytest.mark.backend_jax, pytest.mark.rare]


def _loss(output: Any) -> Any:
    """Return a scalar loss for a model output."""

    return jnp.sum(output * output)


def _params() -> dict[str, Any]:
    """Return deterministic tiny parameters."""

    return {
        "w": jnp.asarray([[0.2, -0.4], [0.7, 0.3], [-0.5, 0.1]], dtype=jnp.float32),
        "b": jnp.asarray([0.05, -0.1], dtype=jnp.float32),
    }


def _x() -> Any:
    """Return a deterministic input batch."""

    return jnp.asarray([[1.0, -2.0, 0.5], [0.3, 0.1, -0.8]], dtype=jnp.float32)


def _kw_model(params: dict[str, Any], x: Any, *, bias: Any, double: bool = False) -> Any:
    """Return a model output using an array keyword and a static flag keyword."""

    out = jnp.tanh(x @ params["w"] + params["b"]) + bias
    return out * 2.0 if double else out


def _validates(trace: Any) -> bool:
    """Return the forward-pass replay validation verdict."""

    return bool(trace.validate_forward_pass([]))


def test_jax_input_kwargs_array_leaves_and_static_flags() -> None:
    """Array keyword values trace as inputs; Python flags bind statically."""

    bias = jnp.asarray([0.5, -0.25], dtype=jnp.float32)
    trace = tl.trace(
        cast(Any, _kw_model),
        (_params(), _x()),
        input_kwargs={"bias": bias, "double": True},
        backend="jax",
    )
    expected = _kw_model(_params(), _x(), bias=bias, double=True)
    output = trace[trace.output_layers[0]].out
    assert jnp.allclose(jnp.asarray(output), expected)
    # params (2 leaves) + x + the bias keyword leaf.
    assert len(list(trace.input_ops)) == 4
    assert _validates(trace)


def test_jax_input_kwargs_static_only() -> None:
    """A keyword input with no array leaves changes the traced graph statically."""

    bias = jnp.zeros(2, dtype=jnp.float32)
    plain = tl.trace(
        cast(Any, lambda p, x, b, double=False: _kw_model(p, x, bias=b, double=double)),
        (_params(), _x(), bias),
        input_kwargs={"double": False},
        backend="jax",
    )
    doubled = tl.trace(
        cast(Any, lambda p, x, b, double=False: _kw_model(p, x, bias=b, double=double)),
        (_params(), _x(), bias),
        input_kwargs={"double": True},
        backend="jax",
    )
    assert len(doubled.ops) > len(plain.ops)
    assert _validates(plain) and _validates(doubled)


def test_jax_input_kwargs_with_derived_grads() -> None:
    """Derived gradients still differentiate params when keyword inputs are present."""

    bias = jnp.asarray([0.5, -0.25], dtype=jnp.float32)
    params = _params()
    trace = tl.trace(
        cast(Any, _kw_model),
        (params, _x()),
        input_kwargs={"bias": bias},
        backend="jax",
        capture=tl.options.CaptureOptions(grad_options=GradOptions(params=params, loss_fn=_loss)),
    )
    expected = jax.grad(lambda p: _loss(_kw_model(p, _x(), bias=bias)))(params)
    assert jnp.allclose(trace.params["w"].grad, expected["w"])
    assert jnp.allclose(trace.params["b"].grad, expected["b"])


def test_jax_closure_constant_derived_grads() -> None:
    """A function closing over an array gets param grads; the closure is a fixed leaf."""

    shift = jnp.asarray([0.3, -0.7], dtype=jnp.float32)

    def model(params: dict[str, Any], x: Any) -> Any:
        """Return an output that reads a closed-over array."""

        return jnp.tanh(x @ params["w"] + params["b"]) * shift

    params = _params()
    trace = tl.trace(
        cast(Any, model),
        (params, _x()),
        backend="jax",
        capture=tl.options.CaptureOptions(
            grad_options=GradOptions(params=params, loss_fn=_loss, input_grad_argnums=(0,))
        ),
    )
    expected_p, expected_x = jax.grad(lambda p, x: _loss(model(p, x)), argnums=(0, 1))(params, _x())
    grads = trace.autograd.derived_grads
    assert jnp.allclose(grads["params.w"].grad, expected_p["w"])
    assert jnp.allclose(grads["params.b"].grad, expected_p["b"])
    assert jnp.allclose(grads["inputs.0"].grad, expected_x)
    assert trace.params["w"].has_grad


def _equinox_model() -> Any:
    """Return a tiny deterministic Equinox MLP."""

    eqx = pytest.importorskip("equinox")

    class Mlp(eqx.Module):
        """Two-layer Equinox MLP."""

        fc1: Any
        fc2: Any

        def __init__(self) -> None:
            """Initialize deterministic layers."""

            key1, key2 = jax.random.split(jax.random.PRNGKey(10))
            self.fc1 = eqx.nn.Linear(3, 4, key=key1)
            self.fc2 = eqx.nn.Linear(4, 2, key=key2)

        def __call__(self, x: Any) -> Any:
            """Return the MLP output."""

            return self.fc2(jax.nn.relu(self.fc1(x)))

    return Mlp()


def test_jax_equinox_module_root_derived_grads() -> None:
    """Equinox module-root grads match eqx.filter_grad and mirror onto trace.params."""

    eqx = pytest.importorskip("equinox")
    model = _equinox_model()
    x = jnp.asarray([0.5, -1.0, 2.0], dtype=jnp.float32)
    trace = tl.trace(
        model,
        x,
        backend="jax",
        capture=tl.options.CaptureOptions(grad_options=GradOptions(params=model, loss_fn=_loss)),
    )
    expected = eqx.filter_grad(lambda m: _loss(m(x)))(model)
    for address, ref in (
        ("fc1.weight", expected.fc1.weight),
        ("fc1.bias", expected.fc1.bias),
        ("fc2.weight", expected.fc2.weight),
        ("fc2.bias", expected.fc2.bias),
    ):
        assert trace.params[address].has_grad, address
        assert jnp.allclose(trace.params[address].grad, ref), address
    assert _validates(trace)


def test_jax_nnx_module_root_derived_grads() -> None:
    """Flax NNX module-root grads match jax.grad over the split state."""

    nnx = pytest.importorskip("flax.nnx")

    class Mlp(nnx.Module):
        """Two-layer NNX MLP."""

        def __init__(self) -> None:
            """Initialize deterministic layers."""

            rngs = nnx.Rngs(20)
            self.fc1 = nnx.Linear(3, 4, rngs=rngs)
            self.fc2 = nnx.Linear(4, 2, rngs=rngs)

        def __call__(self, x: Any) -> Any:
            """Return the MLP output."""

            return self.fc2(jax.nn.relu(self.fc1(x)))

    model = Mlp()
    x = jnp.asarray([0.5, -1.0, 2.0], dtype=jnp.float32)
    trace = tl.trace(
        model,
        x,
        backend="jax",
        capture=tl.options.CaptureOptions(grad_options=GradOptions(params=model, loss_fn=_loss)),
    )
    graphdef, state = nnx.split(model)
    expected = jax.grad(lambda s: _loss(nnx.merge(graphdef, s)(x)))(state)
    for layer in ("fc1", "fc2"):
        for leaf in ("kernel", "bias"):
            ref = expected[layer][leaf]
            ref = getattr(ref, "value", ref)
            address = f"{layer}.{leaf}"
            assert trace.params[address].has_grad, address
            assert jnp.allclose(trace.params[address].grad, ref), address


def test_jax_module_root_rejects_foreign_params() -> None:
    """A module root refuses GradOptions.params that are not the module itself."""

    model = _equinox_model()
    x = jnp.asarray([0.5, -1.0, 2.0], dtype=jnp.float32)
    with pytest.raises(tl.errors.BackendUnsupportedError) as excinfo:
        tl.trace(
            model,
            x,
            backend="jax",
            capture=tl.options.CaptureOptions(
                grad_options=GradOptions(params={"w": jnp.ones(2)}, loss_fn=_loss)
            ),
        )
    assert excinfo.value.code == "grad_options_unsupported"
