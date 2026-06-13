"""JAX backend hardening and public-surface matrix tests."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest

import torchlens as tl
from torchlens.backends import (
    BackendPayloadUnsupportedError,
    BackendUnsupportedError,
    get_backend_spec,
)
from torchlens.backends.jax import capabilities

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
lax = pytest.importorskip("jax.lax")


pytestmark = pytest.mark.backend_jax


def _params() -> dict[str, Any]:
    """Return deterministic JAX parameter leaves.

    Returns
    -------
    dict[str, Any]
        Parameter pytree.
    """

    return {"w": jnp.ones((3, 2), dtype=jnp.float32), "b": jnp.zeros((2,), dtype=jnp.float32)}


def _model(params: dict[str, Any], x: Any) -> Any:
    """Return a tiny JAX model output.

    Parameters
    ----------
    params
        Parameter pytree.
    x
        Input array.

    Returns
    -------
    Any
        Model output.
    """

    return jnp.tanh(x @ params["w"] + params["b"])


def _trace(**kwargs: Any) -> Any:
    """Trace the shared tiny JAX model.

    Parameters
    ----------
    **kwargs
        Public trace keyword overrides.

    Returns
    -------
    Any
        Captured JAX trace.
    """

    return tl.trace(
        cast(Any, _model),
        (_params(), jnp.ones((2, 3), dtype=jnp.float32)),
        backend="jax",
        **kwargs,
    )


@pytest.mark.parametrize(
    ("builder", "pattern"),
    (
        (
            lambda: jax.jit(_model),
            "transformed callable.*root model",
        ),
        (
            lambda: jax.vmap(_model, in_axes=({"w": None, "b": None}, 0)),
            "transformed callable.*jax\\.vmap",
        ),
        (
            lambda: jax.grad(lambda params, x: jnp.sum(_model(params, x))),
            "transformed callable.*jax\\.grad",
        ),
    ),
)
def test_jax_rejects_transformed_callable_as_model(
    builder: Callable[[], Callable[..., Any]],
    pattern: str,
) -> None:
    """Transformed root callables should fail with raw-function guidance."""

    with pytest.raises(BackendUnsupportedError, match=pattern):
        tl.trace(
            cast(Any, builder()),
            (_params(), jnp.ones((2, 3), dtype=jnp.float32)),
            backend="jax",
        )


def test_jax_rejects_root_capture_inside_jit() -> None:
    """Calling TorchLens capture under a root JAX transform should fail loudly."""

    def traced_under_jit(x: Any) -> Any:
        """Attempt a nested TorchLens capture under ``jax.jit``."""

        tl.trace(cast(Any, _model), (_params(), x), backend="jax")
        return x

    with pytest.raises(BackendUnsupportedError, match="inside jax\\.jit.*concrete"):
        jax.jit(traced_under_jit)(jnp.ones((2, 3), dtype=jnp.float32))


@pytest.mark.parametrize(
    ("fn", "kwargs", "pattern"),
    (
        (
            lambda params, x: lax.scan(lambda carry, item: (carry + item, carry), x[0], x)[1],
            {"jax_control_flow": "reject"},
            "unsupported nested primitive: scan",
        ),
        (
            lambda params, x: lax.cond(
                x.sum() > 0,
                lambda y: y @ params["w"],
                lambda y: (y @ params["w"]) + params["b"],
                x,
            ),
            {"jax_control_flow": "reject"},
            "unsupported nested primitive: cond",
        ),
        (
            lambda params, x: lax.while_loop(
                lambda state: state[0] < 2,
                lambda state: (state[0] + 1, state[1] + 1),
                (0, x),
            )[1],
            {"jax_control_flow": "reject"},
            "unsupported nested primitive: while",
        ),
    ),
)
def test_jax_rejects_nested_jaxpr_primitives(
    fn: Callable[[dict[str, Any], Any], Any],
    kwargs: dict[str, Any],
    pattern: str,
) -> None:
    """Nested jaxpr control-flow primitives should name the unsupported primitive."""

    with pytest.raises(ValueError, match=pattern):
        tl.trace(
            cast(Any, fn),
            (_params(), jnp.ones((2, 3), dtype=jnp.float32)),
            backend="jax",
            **kwargs,
        )


def test_jax_accepts_default_unrolled_cond_and_while() -> None:
    """Default JAX control-flow policy should unroll supported cond and while primitives."""

    def uses_cond(params: dict[str, Any], x: Any) -> Any:
        """Return a conditional model output."""

        return lax.cond(
            x.sum() > 0,
            lambda y: y @ params["w"],
            lambda y: (y @ params["w"]) + params["b"],
            x,
        )

    def uses_while(params: dict[str, Any], x: Any) -> Any:
        """Return a while-loop model output."""

        return (
            lax.while_loop(
                lambda state: state[0] < 2,
                lambda state: (state[0] + 1, state[1] + 1),
                (0, x),
            )[1]
            @ params["w"]
        )

    args = (_params(), jnp.ones((2, 3), dtype=jnp.float32))
    cond_trace = tl.trace(cast(Any, uses_cond), args, backend="jax")
    while_trace = tl.trace(cast(Any, uses_while), args, backend="jax")

    assert any(
        op.annotations.get("jax_capture_kind") == "cond_decision" for op in cond_trace.layer_list
    )
    assert any(
        op.annotations.get("jax_capture_kind") == "while_decision" for op in while_trace.layer_list
    )
    assert cond_trace.validate_forward_pass([]) is True
    assert while_trace.validate_forward_pass([]) is True


def test_jax_rejects_custom_vjp_nested_primitive() -> None:
    """User custom VJP call primitives should be rejected for M1."""

    @jax.custom_vjp
    def custom_square(x: Any) -> Any:
        """Return a custom-VJP square."""

        return x * x

    def fwd(x: Any) -> tuple[Any, Any]:
        """Return custom VJP forward output and residual."""

        return custom_square(x), x

    def bwd(residual: Any, grad: Any) -> tuple[Any]:
        """Return custom VJP backward output."""

        return (2 * residual * grad,)

    custom_square.defvjp(fwd, bwd)

    def uses_custom_vjp(params: dict[str, Any], x: Any) -> Any:
        """Return output through a custom-VJP function."""

        del params
        return custom_square(x)

    with pytest.raises(ValueError, match="unsupported nested primitive: custom_vjp_call"):
        tl.trace(
            cast(Any, uses_custom_vjp),
            (_params(), jnp.ones((2, 3), dtype=jnp.float32)),
            backend="jax",
        )


def test_jax_rejects_callback_effects() -> None:
    """JAX callback effects should be rejected with effect wording."""

    def callback(value: Any) -> Any:
        """Return callback value unchanged."""

        return value

    def uses_callback(params: dict[str, Any], x: Any) -> Any:
        """Return output through ``jax.pure_callback``."""

        del params
        return jax.pure_callback(callback, jax.ShapeDtypeStruct(x.shape, x.dtype), x)

    with pytest.raises(ValueError, match="unsupported.*effect"):
        tl.trace(cast(Any, uses_callback), (_params(), jnp.ones((2, 3))), backend="jax")


@pytest.mark.parametrize(
    ("kwargs", "pattern"),
    (
        ({"save": tl.func("tanh")}, "full-save only.*save-shaping"),
        ({"layers_to_save": ["tanh"]}, "full-save only"),
        ({"lookback": 1}, "full-save only"),
        ({"intervene": tl.when(tl.func("tanh"), tl.zero_ablate())}, "full-save only"),
        ({"halt": tl.func("tanh")}, "full-save only"),
        ({"save_grads": True}, "GradOptions"),
    ),
)
def test_jax_rejects_save_shaping_kwargs(kwargs: dict[str, Any], pattern: str) -> None:
    """Save-shaping and runtime mutation kwargs should fail before capture."""

    with pytest.raises(BackendUnsupportedError, match=pattern):
        _trace(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "pattern"),
    (
        ({"module_identity_mode": "pytree_module"}, "module_identity_mode selection"),
        ({"payload_policy": "full"}, "payload_policy.*not implemented"),
        ({"save_preview": True}, "save_preview.*not implemented"),
    ),
)
def test_jax_rejects_declared_future_public_options(
    kwargs: dict[str, Any],
    pattern: str,
) -> None:
    """Declared public-option spine knobs should reject until JAX phases implement them."""

    with pytest.raises(BackendUnsupportedError, match=pattern):
        _trace(**kwargs)


def test_jax_record_backend_jax_rejected() -> None:
    """Sparse ``tl.record`` should stay torch-only for backend v1."""

    with pytest.raises(BackendUnsupportedError, match="torch-only.*tl\\.trace"):
        tl.record(
            cast(Any, _model),
            (_params(), jnp.ones((2, 3), dtype=jnp.float32)),
            backend="jax",
        )


def test_jax_module_predicate_surfaces_rejected_for_capture() -> None:
    """Module predicate capture should fail because JAX M1 has only function_root."""

    with pytest.raises(BackendUnsupportedError, match="full-save only.*save-shaping"):
        _trace(save=tl.in_module("self"))


def test_jax_rejects_closed_over_array_params() -> None:
    """Closed-over array parameters should be passed as explicit pytree leaves."""

    hidden = jnp.ones((3, 2), dtype=jnp.float32)

    def uses_hidden(params: dict[str, Any], x: Any) -> Any:
        """Return output from a hidden array."""

        del params
        return x @ hidden

    with pytest.raises(ValueError, match="closed-jaxpr constants.*explicit"):
        tl.trace(cast(Any, uses_hidden), ({}, jnp.ones((2, 3))), backend="jax")


def test_jax_public_surface_matrix(tmp_path: Path) -> None:
    """Assert supported and unsupported public surfaces on a real JAX trace."""

    trace = _trace()

    assert trace.backend == "jax"
    assert trace.validate_forward_pass([]) is True
    assert isinstance(trace.summary(), str)
    assert len(trace.to_pandas()) == len(trace.layer_list)
    assert trace.modules["self"].address == "self"
    assert trace.module_identity_mode == "function_root"
    assert trace.param_source == "pytree-derived"
    assert set(trace.params.keys()) == {"w", "b"}
    assert trace.has_backward_pass is False
    assert trace.num_backward_passes == 0
    assert trace.num_backward_edges is None

    outpath = tmp_path / "jax_graph"
    dot = trace.draw(vis_outpath=str(outpath), vis_save_only=True, vis_fileformat="dot")
    assert isinstance(dot, str)

    audit_path = tmp_path / "jax_audit.tlspec"
    trace.save(audit_path, level="audit")
    loaded = tl.load(audit_path)
    assert loaded.backend == "jax"
    assert loaded.param_source == "pytree-derived"
    assert all(op.out is None for op in loaded.layer_list)

    with pytest.raises(BackendPayloadUnsupportedError, match="audit-only|materialize") as exc_info:
        trace.save(tmp_path / "jax_portable.tlspec")
    assert "expected a tensor for portable blobification" not in str(exc_info.value)
    with pytest.raises(BackendUnsupportedError, match="backward capture.*derived_grads"):
        trace.log_backward(cast(Any, trace[trace.output_layers[0]].out))
    with pytest.raises(ValueError, match="derived_grads"):
        _ = trace.backward_passes
    with pytest.raises(ValueError, match="derived_grads"):
        _ = trace[0].grads


def test_jax_capability_flags_match_preview_contract() -> None:
    """JAX capability exports should match the M1 preview surface."""

    spec = get_backend_spec("jax")

    assert spec.capabilities.backward_capture is False
    assert spec.capabilities.validation_replay is True
    assert spec.capabilities.fastlog is False
    assert spec.capabilities.interventions is False
    assert spec.capabilities.payload_materialization is False
    assert spec.capabilities.module_identity_modes == ("function_root",)
    assert spec.serialization_policy.payload_policy == "audit_only"
    assert capabilities.supports_backward_capture is False
    assert capabilities.supports_validation_replay is True
    assert capabilities.supports_fastlog is False
    assert capabilities.supports_intervention is False
    assert capabilities.supports_payload_materialization is False
    assert capabilities.module_identity_modes == ("function_root",)
    assert capabilities.payload_policy == "audit_only"
    assert capabilities.trace_options == (
        "jax_static_argnums",
        "grad_options",
        "jax_control_flow",
        "jax_max_control_flow_unroll",
    )
