"""JAX backend derived-gradient preview tests."""

from __future__ import annotations

from typing import Any, cast

import pytest

import torchlens as tl
from torchlens.backends import BackendUnsupportedError
from torchlens.backends.jax import GradOptions
from torchlens.backends.jax.backend import (
    _JaxIntermediateTapSpec,
    _experimental_per_op_boundary_vjp_oracle,
    _jax_intermediate_oracle_passes,
)
from torchlens.backends.jax.jaxpr import _jax_zero_tap
from torchlens.data_classes.derived_grad import (
    IntermediateDerivedGradAccessor,
    IntermediateDerivedGradRecord,
)
from torchlens.ir.refs import DtypeRef

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


def _jax_capture_graph_signature(trace: Any) -> tuple[Any, ...]:
    """Return graph-only JAX capture metadata for byte-identity checks.

    Parameters
    ----------
    trace
        JAX trace to summarize.

    Returns
    -------
    tuple[Any, ...]
        Stable finalized graph signature excluding derived-gradient accessors.
    """

    return (
        tuple(
            (
                op.label,
                op.layer_label,
                op.func_name,
                tuple(op.parents),
                op.has_saved_activation,
                op.shape,
                str(op.dtype),
                tuple(op.annotations.get("jax_outvars", ())),
            )
            for op in trace.layer_list
        ),
        dict(trace.jax_outvar_key_to_capture_index),
        dict(trace.jax_capture_index_to_final_op_label),
    )


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


def test_jax_intermediate_derived_grads_remain_deferred_with_leaf_grads() -> None:
    """JAX should expose leaf derived grads without public intermediate T1 records."""

    params = _params()
    x = jnp.asarray([[1.0, -2.0, 0.5], [0.3, 0.1, -0.8]], dtype=jnp.float32)
    trace = tl.trace(
        cast(Any, _model),
        (params, x),
        backend="jax",
        grad_options=GradOptions(params=params, loss_fn=_loss, input_grad_argnums=(0,)),
    )

    assert set(trace.derived_grads.keys()) == {"params.b", "params.w", "inputs.0"}
    assert len(trace.intermediate_derived_grads) == 0
    assert not hasattr(trace, "jax_intermediate_derived_grads")


def test_jax_intermediate_derived_grads_zero_tap_public_surface() -> None:
    """Opt-in JAX T1 should expose exact op-level records only."""

    params = _params()
    x = jnp.asarray([[1.0, -2.0, 0.5], [0.3, 0.1, -0.8]], dtype=jnp.float32)
    trace = tl.trace(
        cast(Any, _model),
        (params, x),
        backend="jax",
        grad_options=GradOptions(
            params=params,
            loss_fn=_loss,
            input_grad_argnums=(0,),
            intermediate_grads=True,
            max_intermediate_grads=16,
        ),
    )
    tanh_op = next(op for op in trace.layer_list if op.func_name == "tanh")
    expected_tanh_grad = 2.0 * tanh_op.out

    assert set(trace.derived_grads.keys()) == {"params.b", "params.w", "inputs.0"}
    assert len(trace.intermediate_derived_grads) > 0
    assert tanh_op.label in trace.intermediate_derived_grads
    assert jnp.allclose(tanh_op.derived_grad, expected_tanh_grad)
    assert all(
        record.provenance["mechanism"] == "jax_zero_tap" and record.provenance["status"] == "exact"
        for record in trace.intermediate_derived_grads.values()
    )


def test_jax_intermediate_derived_grads_do_not_mutate_capture_graph() -> None:
    """T1 off/requested modes should keep the finalized JAX capture graph identical."""

    params = _params()
    x = jnp.ones((2, 3), dtype=jnp.float32)
    base = tl.trace(cast(Any, _model), (params, x), backend="jax")
    leaf_only = tl.trace(
        cast(Any, _model),
        (params, x),
        backend="jax",
        grad_options=GradOptions(params=params, loss_fn=_loss),
    )
    t1 = tl.trace(
        cast(Any, _model),
        (params, x),
        backend="jax",
        grad_options=GradOptions(
            params=params,
            loss_fn=_loss,
            intermediate_grads=True,
            max_intermediate_grads=16,
        ),
    )

    assert _jax_capture_graph_signature(base) == _jax_capture_graph_signature(leaf_only)
    assert _jax_capture_graph_signature(base) == _jax_capture_graph_signature(t1)


def test_jax_zero_tap_preserves_signed_zero_and_nan() -> None:
    """Zero-tap replay primitive should preserve exact primal edge cases."""

    value = jnp.asarray([-0.0, jnp.nan, 2.0], dtype=jnp.float32)
    delta = jnp.zeros_like(value)
    tapped = _jax_zero_tap(value, delta)
    grad = jax.grad(lambda tap: jnp.nansum(_jax_zero_tap(value, tap)))(delta)

    assert bool(jnp.signbit(tapped[0]))
    assert bool(jnp.isnan(tapped[1]))
    assert jnp.allclose(grad, jnp.asarray([1.0, 0.0, 1.0], dtype=jnp.float32))


def test_jax_intermediate_derived_grads_cap_hard_fails() -> None:
    """Explicit JAX T1 should hard-fail when saved boundaries exceed the cap."""

    params = _params()
    x = jnp.ones((2, 3), dtype=jnp.float32)
    with pytest.raises(BackendUnsupportedError, match="capped"):
        tl.trace(
            cast(Any, _model),
            (params, x),
            backend="jax",
            grad_options=GradOptions(
                params=params,
                loss_fn=_loss,
                intermediate_grads=True,
                max_intermediate_grads=1,
            ),
        )


def test_jax_intermediate_derived_grads_follow_selective_save() -> None:
    """JAX T1 should attach only to public saved ops after static save filtering."""

    params = _params()
    x = jnp.ones((2, 3), dtype=jnp.float32)
    trace = tl.trace(
        cast(Any, _model),
        (params, x),
        backend="jax",
        save=tl.func("dot_general"),
        grad_options=GradOptions(
            params=params,
            loss_fn=_loss,
            intermediate_grads=True,
            max_intermediate_grads=4,
        ),
    )

    assert set(trace.intermediate_derived_grads.keys()) == {
        op.label for op in trace.saved_ops if op.func_name == "dot_general"
    }
    assert all(op.derived_grad is None for op in trace.layer_list if not op.has_saved_activation)


def test_jax_intermediate_derived_grads_skip_non_float_outputs() -> None:
    """JAX T1 should skip non-float/control outputs cleanly."""

    params: dict[str, Any] = {}
    x = jnp.asarray([-1.0, 2.0, -3.0], dtype=jnp.float32)

    def masked_model(_params: dict[str, Any], value: Any) -> Any:
        """Return an output that includes a boolean-producing equation."""

        mask = value > 0.0
        return jnp.where(mask, value, -value)

    trace = tl.trace(
        cast(Any, masked_model),
        (params, x),
        backend="jax",
        grad_options=GradOptions(
            params=params,
            loss_fn=lambda output: jnp.sum(output),
            intermediate_grads=True,
            max_intermediate_grads=8,
        ),
    )
    bool_ops = [op for op in trace.layer_list if str(op.dtype) == "bool"]

    assert bool_ops
    assert all(op.label not in trace.intermediate_derived_grads for op in bool_ops)


def test_jax_intermediate_derived_grads_degrade_to_leaf_grads(monkeypatch: Any) -> None:
    """Producer errors should leave leaf derived gradients intact and T1 empty."""

    import torchlens.backends.jax.backend as jax_backend

    params = _params()
    x = jnp.ones((2, 3), dtype=jnp.float32)
    original_interpreter = jax_backend.interpret_closed_jaxpr_with_inlining

    def failing_interpreter(*args: Any, **kwargs: Any) -> Any:
        """Raise only for the zero-tap intermediate producer path."""

        if kwargs.get("tap_values_by_capture_output"):
            raise AttributeError("simulated jax zero-tap API drift")
        return original_interpreter(*args, **kwargs)

    monkeypatch.setattr(jax_backend, "interpret_closed_jaxpr_with_inlining", failing_interpreter)
    trace = tl.trace(
        cast(Any, _model),
        (params, x),
        backend="jax",
        grad_options=GradOptions(
            params=params,
            loss_fn=_loss,
            intermediate_grads=True,
            max_intermediate_grads=8,
        ),
    )

    assert set(trace.derived_grads.keys()) == {"params.b", "params.w"}
    assert len(trace.intermediate_derived_grads) == 0
    assert trace.jax_intermediate_derived_grad_status["status"] == "degraded"


def test_intermediate_derived_grad_accessor_and_op_filter_unverified() -> None:
    """Default public T1 access should filter oracle-unconfirmed records."""

    params = _params()
    x = jnp.ones((2, 3), dtype=jnp.float32)
    trace = tl.trace(cast(Any, _model), (params, x), backend="jax")
    op = next(op for op in trace.layer_list if op.func_name == "tanh")
    record = IntermediateDerivedGradRecord(
        op_label=op.label,
        layer_label=op.layer_label,
        aval="ShapedArray((), float32)",
        dtype_ref=DtypeRef.from_value(jnp.float32),
        grad=jnp.asarray(1.0, dtype=jnp.float32),
        provenance={"status": "unverified"},
    )

    trace.intermediate_derived_grads = IntermediateDerivedGradAccessor({op.label: record})

    assert len(trace.intermediate_derived_grads) == 0
    assert op.derived_grad is None


def test_jax_intermediate_oracle_catches_off_by_one_tap() -> None:
    """The oracle should reject a producer grad from the wrong boundary."""

    value = jnp.asarray([1.5, -0.5], dtype=jnp.float32)
    wrong_grad = 3.0 * value
    spec = _JaxIntermediateTapSpec(
        op_label="relu_1:1",
        layer_label="relu_1",
        capture_index=0,
        output_index=0,
        value=value,
        aval="ShapedArray((2,), float32)",
        dtype_ref=DtypeRef.from_value(value.dtype),
    )

    assert not _jax_intermediate_oracle_passes(
        spec=spec,
        tap_index=0,
        producer_grad=wrong_grad,
        zero_taps=(jnp.zeros_like(value),),
        loss_from_replacement=lambda _spec, replacement: jnp.sum(replacement * replacement),
    )


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
