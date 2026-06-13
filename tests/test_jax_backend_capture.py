"""JAX backend capture tests."""

from __future__ import annotations

from typing import Any

import pytest

import torchlens as tl
from torchlens.backends import BackendUnsupportedError

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


pytestmark = pytest.mark.backend_jax


def _mlp(params: dict[str, Any], x: Any) -> Any:
    """Return a tiny JAX MLP output.

    Parameters
    ----------
    params
        Parameter pytree.
    x
        Input array.

    Returns
    -------
    Any
        Output array.
    """

    hidden = jnp.tanh(x @ params["w1"] + params["b1"])
    return hidden @ params["w2"]


def _relu_block(params: dict[str, Any], x: Any) -> Any:
    """Return an output that lowers through JAX's library ReLU wrapper.

    Parameters
    ----------
    params
        Parameter pytree.
    x
        Input array.

    Returns
    -------
    Any
        Output array.
    """

    return jax.nn.relu(x @ params["w"])


def _params() -> dict[str, Any]:
    """Build a nested parameter pytree.

    Returns
    -------
    dict[str, Any]
        Parameter pytree.
    """

    return {
        "w1": jnp.ones((3, 4)),
        "b1": jnp.zeros((4,)),
        "w2": jnp.ones((4, 2)),
    }


def _nested_params() -> dict[str, Any]:
    """Build a nested parameter pytree.

    Returns
    -------
    dict[str, Any]
        Nested parameter pytree.
    """

    return {
        "encoder": {"w": jnp.ones((3, 4)), "b": jnp.zeros((4,))},
        "head": {"w": jnp.ones((4, 2))},
    }


def test_jax_trace_captures_equation_ops_and_params() -> None:
    """JAX raw functions should produce equation-backed TorchLens traces."""

    params = _params()
    x = jnp.ones((2, 3))

    trace = tl.trace(_mlp, (params, x), backend="jax")

    assert trace.backend == "jax"
    assert trace.module_identity_mode == "function_root"
    assert trace.param_source == "pytree-derived"
    assert trace.modules["self"].address == "self"
    assert set(trace.params.keys()) == {"w1", "b1", "w2"}
    assert trace.params["w1"].backend_address == "pytree:w1"
    assert trace.params["w1"].resolver_status == "metadata_only"
    primitive_names = [op.func_name for op in trace.layer_list]
    assert "dot_general" in primitive_names
    assert "tanh" in primitive_names
    assert all(op.has_saved_activation for op in trace.layer_list)
    assert trace.validate_forward_pass([])


def test_jax_trace_preserves_nested_param_paths() -> None:
    """Nested pytree parameter leaves should be addressable without raising."""

    def nested_mlp(params: dict[str, Any], x: Any) -> Any:
        """Return an MLP result from nested parameters."""

        hidden = jnp.tanh(x @ params["encoder"]["w"] + params["encoder"]["b"])
        return hidden @ params["head"]["w"]

    trace = tl.trace(nested_mlp, (_nested_params(), jnp.ones((2, 3))), backend="jax")

    assert set(trace.params.keys()) == {"encoder.w", "encoder.b", "head.w"}
    assert trace.params["encoder.w"].backend_address == "pytree:encoder.w"
    assert trace.param_source == "pytree-derived"


def test_jax_trace_accepts_declared_static_argnums() -> None:
    """Declared static positional args should be excluded from dynamic jaxpr inputs."""

    def scaled(params: dict[str, Any], x: Any, scale: int) -> Any:
        """Return a statically scaled JAX result."""

        return (x @ params["w1"]) * scale

    trace = tl.trace(
        scaled,
        (_params(), jnp.ones((2, 3)), 3),
        backend="jax",
        jax_static_argnums=2,
    )

    assert trace.jax_static_argnums == (2,)
    assert "mul" in {op.func_name for op in trace.layer_list}
    assert trace.validate_forward_pass([])


def test_jax_trace_inlines_allowlisted_pure_library_calls() -> None:
    """JAX library pure-call wrappers should inline into primitive equation ops."""

    params = {"w": jnp.ones((3, 3))}
    x = jnp.ones((2, 3))

    trace = tl.trace(_relu_block, (params, x), backend="jax")

    assert "custom_jvp_call" in trace.jax_inlined_call_primitives
    assert any(capture.inlined for capture in trace.jax_equation_captures)
    assert "max" in {op.func_name for op in trace.layer_list}


def test_jax_trace_rejects_save_shaping_kwargs() -> None:
    """JAX preview should reject selective-save shaping."""

    with pytest.raises(BackendUnsupportedError, match="full-save only"):
        tl.trace(_mlp, (_params(), jnp.ones((2, 3))), backend="jax", layers_to_save=["tanh"])


def test_jax_trace_rejects_nested_control_flow() -> None:
    """Unsupported nested jaxprs should raise actionable errors."""

    def uses_cond(params: dict[str, Any], x: Any) -> Any:
        """Return a conditional JAX result."""

        return jax.lax.cond(
            x.sum() > 0,
            lambda y: y @ params["w1"],
            lambda y: (y @ params["w1"]) - params["b1"],
            x,
        )

    with pytest.raises(ValueError, match="unsupported nested primitive: cond"):
        tl.trace(uses_cond, (_params(), jnp.ones((2, 3))), backend="jax")


def test_jax_trace_rejects_hidden_consts() -> None:
    """Hidden closure constants should be rejected as undeclared leaves."""

    hidden = jnp.ones((3, 3))

    def uses_hidden(params: dict[str, Any], x: Any) -> Any:
        """Return a result using a closure constant."""

        del params
        return x @ hidden

    with pytest.raises(ValueError, match="closed-jaxpr constants"):
        tl.trace(uses_hidden, ({}, jnp.ones((2, 3))), backend="jax")
