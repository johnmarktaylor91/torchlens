"""JAX backend derived-gradient preview tests."""

from __future__ import annotations

from typing import Any, cast

import pytest

import torchlens as tl
from torchlens.backends import BackendUnsupportedError
from torchlens.backends.jax import GradOptions
from torchlens.backends.jax.backend import _experimental_per_op_boundary_vjp_oracle

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


pytestmark = pytest.mark.backend_jax


def _params() -> dict[str, Any]:
    """Return deterministic tiny MLP parameters.

    Returns
    -------
    dict[str, Any]
        Parameter pytree.
    """

    return {
        "w": jnp.asarray([[0.2, -0.4], [0.7, 0.3], [-0.5, 0.1]], dtype=jnp.float32),
        "b": jnp.asarray([0.05, -0.1], dtype=jnp.float32),
    }


def _model(params: dict[str, Any], x: Any, scale: Any = 1.0) -> Any:
    """Return a small JAX model output.

    Parameters
    ----------
    params
        Parameter pytree.
    x
        Input array.
    scale
        Output scale.

    Returns
    -------
    Any
        Model output.
    """

    return jnp.tanh(x @ params["w"] + params["b"]) * scale


def _loss(output: Any) -> Any:
    """Return scalar loss for a model output.

    Parameters
    ----------
    output
        Raw model output.

    Returns
    -------
    Any
        Scalar loss.
    """

    return jnp.sum(output * output)


def test_jax_derived_grads_match_value_and_grad_oracle() -> None:
    """Derived param and input gradients should match direct JAX AD."""

    params = _params()
    x = jnp.asarray([[1.0, -2.0, 0.5], [0.3, 0.1, -0.8]], dtype=jnp.float32)
    trace = tl.trace(
        cast(Any, _model),
        (params, x),
        backend="jax",
        grad_options=GradOptions(params=params, loss_fn=_loss, input_grad_argnums=(0,)),
    )

    def value_fn(test_params: dict[str, Any], test_x: Any) -> Any:
        """Return direct oracle loss.

        Parameters
        ----------
        test_params
            Parameter pytree.
        test_x
            Input array.

        Returns
        -------
        Any
            Scalar oracle loss.
        """

        return _loss(_model(test_params, test_x))

    _value, (expected_param_grads, expected_x_grad) = jax.value_and_grad(value_fn, argnums=(0, 1))(
        params, x
    )

    assert set(trace.derived_grads.keys()) == {"params.b", "params.w", "inputs.0"}
    assert jnp.allclose(trace.derived_grads["params.w"].grad, expected_param_grads["w"])
    assert jnp.allclose(trace.derived_grads["params.b"].grad, expected_param_grads["b"])
    assert jnp.allclose(trace.derived_grads["inputs.0"].grad, expected_x_grad)
    assert trace.params["w"].grad is trace.derived_grads["params.w"].grad
    assert trace.params["b"].grad is trace.derived_grads["params.b"].grad
    assert trace.params["w"].has_grad


def test_jax_derived_grads_allow_scalar_output_without_loss_fn() -> None:
    """Scalar raw outputs can be differentiated without an explicit loss_fn."""

    params = {"w": jnp.asarray([0.25, -0.5], dtype=jnp.float32)}
    x = jnp.asarray([2.0, -3.0], dtype=jnp.float32)

    def scalar_model(local_params: dict[str, Any], local_x: Any) -> Any:
        """Return scalar dot-product output.

        Parameters
        ----------
        local_params
            Parameter pytree.
        local_x
            Input array.

        Returns
        -------
        Any
            Scalar output.
        """

        return jnp.sum(local_params["w"] * local_x)

    trace = tl.trace(
        cast(Any, scalar_model),
        (params, x),
        backend="jax",
        grad_options=GradOptions(params=params),
    )

    assert jnp.allclose(trace.derived_grads["params.w"].grad, x)


def test_jax_derived_grads_reject_non_scalar_output_without_loss_fn() -> None:
    """Non-scalar raw outputs require a loss function."""

    with pytest.raises(ValueError, match="loss_fn"):
        tl.trace(
            cast(Any, _model),
            (_params(), jnp.ones((2, 3), dtype=jnp.float32)),
            backend="jax",
            grad_options=GradOptions(params=_params()),
        )


def test_jax_derived_grads_are_not_backward_capture() -> None:
    """JAX traces should reject true backward surfaces with derived-grad guidance."""

    params = _params()
    trace = tl.trace(
        cast(Any, _model),
        (params, jnp.ones((2, 3), dtype=jnp.float32)),
        backend="jax",
        grad_options=GradOptions(params=params, loss_fn=_loss),
    )

    with pytest.raises(BackendUnsupportedError, match="trace\\.derived_grads"):
        trace.log_backward(cast(Any, trace[trace.output_layers[0]].out))
    with pytest.raises(ValueError, match="trace\\.derived_grads"):
        _ = trace.backward_passes
    with pytest.raises(ValueError, match="trace\\.derived_grads"):
        _ = trace.saved_grad_ops
    with pytest.raises(ValueError, match="trace\\.derived_grads"):
        _ = trace[0].grads


def test_jax_derived_grads_reject_closed_over_scalar_host_state() -> None:
    """Closed-over host scalar state is refused before derived grads are exposed."""

    hidden_scale = 2.0

    def uses_global(params: dict[str, Any], x: Any) -> Any:
        """Return output using a closed-over host scalar."""

        return _model(params, x) * hidden_scale

    with pytest.raises(ValueError, match="closed-over host-state"):
        tl.trace(
            cast(Any, uses_global),
            (_params(), jnp.ones((2, 3), dtype=jnp.float32)),
            backend="jax",
            grad_options=GradOptions(params=_params(), loss_fn=_loss),
        )


def test_jax_derived_grads_reject_aux_output_divergence() -> None:
    """A changed second-run raw output should refuse derived grads."""

    calls = {"count": 0}

    def impure_model(params: dict[str, Any], x: Any) -> Any:
        """Return different outputs across Python calls."""

        calls["count"] += 1
        return _model(params, x) + jnp.asarray(float(calls["count"]), dtype=x.dtype)

    with pytest.raises(ValueError, match="raw output diverged"):
        tl.trace(
            cast(Any, impure_model),
            (_params(), jnp.ones((2, 3), dtype=jnp.float32)),
            backend="jax",
            grad_options=GradOptions(params=_params(), loss_fn=_loss),
        )


def test_jax_private_boundary_vjp_oracle_is_capped_and_not_public() -> None:
    """Private JAX T1 oracle computes capped boundary VJPs without trace surface."""

    x = jnp.asarray([1.0, -2.0, 0.5], dtype=jnp.float32)
    hidden = jnp.tanh(x * 2.0)

    def suffix(replacement: Any) -> Any:
        """Return output reconstructed from a replacement boundary.

        Parameters
        ----------
        replacement
            Replacement hidden activation.

        Returns
        -------
        Any
            Model suffix output.
        """

        return replacement * replacement

    result = _experimental_per_op_boundary_vjp_oracle(
        boundaries={"hidden": (hidden, suffix)},
        loss_fn=lambda output: jnp.sum(output),
        max_save_set=1,
    )
    _loss_value, expected_grad = jax.value_and_grad(
        lambda replacement: jnp.sum(suffix(replacement))
    )(hidden)
    trace = tl.trace(
        cast(Any, lambda params, value: suffix(jnp.tanh(value * 2.0))), ({}, x), backend="jax"
    )

    assert result.skipped == {}
    assert set(result.records.keys()) == {"hidden"}
    assert jnp.allclose(result.records["hidden"].grad, expected_grad)
    assert not hasattr(trace, "jax_intermediate_derived_grads")
    assert len(trace.intermediate_derived_grads) == 0

    with pytest.raises(ValueError, match="capped"):
        _experimental_per_op_boundary_vjp_oracle(
            boundaries={"a": (hidden, suffix), "b": (hidden, suffix)},
            loss_fn=lambda output: jnp.sum(output),
            max_save_set=1,
        )
