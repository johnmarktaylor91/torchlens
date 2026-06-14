"""JAX Equinox module hierarchy tests."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import torchlens as tl
from torchlens.backends import BackendUnsupportedError
from torchlens.validation.invariants import check_metadata_invariants

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
eqx = pytest.importorskip("equinox")

pytestmark = pytest.mark.backend_jax


class SimpleEquinoxMlp(eqx.Module):
    """Tiny Equinox MLP with two direct Linear children."""

    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear

    def __init__(self) -> None:
        """Initialize deterministic linear layers."""

        key1, key2 = jax.random.split(jax.random.PRNGKey(10))
        self.fc1 = eqx.nn.Linear(3, 4, key=key1)
        self.fc2 = eqx.nn.Linear(4, 2, key=key2)

    def __call__(self, x: Any) -> Any:
        """Return the MLP output for ``x``."""

        return self.fc2(jax.nn.relu(self.fc1(x)))


class NestedBlock(eqx.Module):
    """Nested Equinox block with a Linear child and local activation."""

    proj: eqx.nn.Linear

    def __init__(self) -> None:
        """Initialize the projection layer."""

        self.proj = eqx.nn.Linear(3, 4, key=jax.random.PRNGKey(11))

    def __call__(self, x: Any) -> Any:
        """Return projected and activated ``x``."""

        return jnp.tanh(self.proj(x))


class NestedEquinoxMlp(eqx.Module):
    """Equinox model with a user block nested under the root."""

    encoder: NestedBlock
    head: eqx.nn.Linear

    def __init__(self) -> None:
        """Initialize nested block and head layers."""

        self.encoder = NestedBlock()
        self.head = eqx.nn.Linear(4, 2, key=jax.random.PRNGKey(12))

    def __call__(self, x: Any) -> Any:
        """Return nested-model output for ``x``."""

        return self.head(self.encoder(x))


class SharedProjection(eqx.Module):
    """Shared Equinox submodule used through two root addresses."""

    weight: Any
    bias: Any

    def __init__(self) -> None:
        """Initialize deterministic shared parameters."""

        self.weight = jnp.asarray(
            [[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]],
            dtype=jnp.float32,
        )
        self.bias = jnp.asarray([0.25, -0.5], dtype=jnp.float32)

    def __call__(self, x: Any) -> Any:
        """Return a projected vector."""

        return x @ self.weight + self.bias


class SharedEquinoxMlp(eqx.Module):
    """Equinox model with the same submodule instance at two paths."""

    left: SharedProjection
    right: SharedProjection

    def __init__(self) -> None:
        """Initialize both paths with one shared module instance."""

        shared = SharedProjection()
        self.left = shared
        self.right = shared

    def __call__(self, x: Any) -> Any:
        """Call the shared submodule through both aliases."""

        return self.left(x) + self.right(x + 1.0)


class JitInsideEquinox(eqx.Module):
    """Equinox module that calls ``jax.jit`` inside ``__call__``."""

    core: eqx.nn.Linear

    def __init__(self) -> None:
        """Initialize the core layer."""

        self.core = eqx.nn.Linear(3, 2, key=jax.random.PRNGKey(13))

    def __call__(self, x: Any) -> Any:
        """Return output through an unsupported inner ``jax.jit``."""

        def inner(y: Any) -> Any:
            """Return the core layer output for ``y``."""

            return self.core(y)

        return jax.jit(inner)(x)


class ScanInsideEquinox(eqx.Module):
    """Equinox module that calls ``lax.scan`` inside ``__call__``."""

    scale: Any

    def __init__(self) -> None:
        """Initialize a trainable scalar."""

        self.scale = jnp.asarray(2.0, dtype=jnp.float32)

    def __call__(self, x: Any) -> Any:
        """Return output through an unsupported inner ``lax.scan``."""

        def body(carry: Any, item: Any) -> tuple[Any, Any]:
            """Return one scan step."""

            updated = carry + item * self.scale
            return updated, updated

        final, _ys = jax.lax.scan(body, jnp.asarray(0.0, dtype=x.dtype), x)
        return final


class CondInsideEquinox(eqx.Module):
    """Equinox module that calls ``lax.cond`` inside ``__call__``."""

    scale: Any

    def __init__(self) -> None:
        """Initialize a trainable scalar."""

        self.scale = jnp.asarray(2.0, dtype=jnp.float32)

    def __call__(self, x: Any) -> Any:
        """Return output through an unsupported inner ``lax.cond``."""

        return jax.lax.cond(
            jnp.sum(x) > 0,
            lambda y: y * self.scale,
            lambda y: y - self.scale,
            x,
        )


class WhileInsideEquinox(eqx.Module):
    """Equinox module that calls ``lax.while_loop`` inside ``__call__``."""

    bias: Any

    def __init__(self) -> None:
        """Initialize a trainable scalar."""

        self.bias = jnp.asarray(1.0, dtype=jnp.float32)

    def __call__(self, x: Any) -> Any:
        """Return output through an unsupported inner ``lax.while_loop``."""

        def cond_fn(state: tuple[Any, Any]) -> Any:
            """Return whether iteration should continue."""

            index, _value = state
            return index < 2

        def body_fn(state: tuple[Any, Any]) -> tuple[Any, Any]:
            """Return the next loop state."""

            index, value = state
            return index + 1, value + self.bias

        _index, value = jax.lax.while_loop(cond_fn, body_fn, (0, x))
        return value


def test_jax_equinox_simple_mlp_uses_pytree_module_hierarchy() -> None:
    """Simple Equinox modules should produce real pytree-module logs."""

    model = SimpleEquinoxMlp()
    trace = tl.trace(model, jnp.ones(3, dtype=jnp.float32), backend="jax")

    assert trace.module_identity_mode == "pytree_module"
    assert [module.address for module in trace.modules] == ["self", "fc1", "fc2"]
    assert set(trace.modules["self"].address_children) == {"fc1", "fc2"}
    assert trace.modules["fc1"].address_parent == "self"
    assert trace.modules["fc2"].address_parent == "self"
    assert trace.modules["fc1"].training is False
    assert trace.modules["fc1"].params
    assert {param.module_address for param in trace.modules["fc1"].params} == {"fc1"}
    assert trace.params["fc1.weight"].is_trainable is True
    fc1_labels = trace.resolve_sites(tl.in_module("fc1"), max_fanout=8).labels()
    assert fc1_labels
    assert all("fc1:1" in trace[label].modules for label in fc1_labels)
    check_metadata_invariants(trace)
    assert trace.validate_forward_pass([]) is True


def test_jax_equinox_nested_modules_preserve_address_tree_and_selectors() -> None:
    """Nested Equinox modules should preserve parent/child addresses."""

    model = NestedEquinoxMlp()
    trace = tl.trace(model, jnp.ones(3, dtype=jnp.float32), backend="jax")

    assert trace.module_identity_mode == "pytree_module"
    assert set(module.address for module in trace.modules) == {
        "self",
        "encoder",
        "encoder.proj",
        "head",
    }
    assert set(trace.modules["self"].address_children) == {"encoder", "head"}
    assert trace.modules["encoder"].address_parent == "self"
    assert trace.modules["encoder"].address_children == ["encoder.proj"]
    assert trace.modules["encoder.proj"].address_parent == "encoder"
    assert trace.modules["head"].params
    assert {param.module_address for param in trace.modules["encoder.proj"].params} == {
        "encoder.proj"
    }
    proj_labels = trace.resolve_sites(tl.in_module("encoder.proj"), max_fanout=8).labels()
    encoder_labels = trace.resolve_sites(tl.in_module("encoder"), max_fanout=8).labels()
    assert set(proj_labels) < set(encoder_labels)
    assert all("encoder.proj:1" in trace[label].modules for label in proj_labels)
    check_metadata_invariants(trace)
    assert trace.validate_forward_pass([]) is True


def test_jax_equinox_shared_submodule_aliases_and_multicall() -> None:
    """Shared Equinox module instances should mirror torch alias semantics."""

    model = SharedEquinoxMlp()
    trace = tl.trace(model, jnp.ones(3, dtype=jnp.float32), backend="jax")

    shared = trace.modules["left"]
    assert trace.modules["right"] is shared
    assert shared.address == "left"
    assert shared.all_addresses == ["left", "right"]
    assert shared.num_calls == 2
    assert set(shared.ops.keys()) == {1, 2}
    assert shared.call_labels == ["left:1", "left:2"]
    first_call = shared.ops.get(1)
    second_call = shared.ops.get(2)
    assert first_call is not None
    assert second_call is not None
    assert first_call.ops
    assert second_call.ops
    assert all(f"left:{call_index}" in trace.modules._pass_dict for call_index in (1, 2))
    assert all("left" in trace[op_label].modules[-1] for op_label in first_call.ops)

    assert trace.modules["self"].address_children == ["left"]
    assert {param.module_address for param in shared.params} == {"left"}
    assert {tuple(param.all_module_addresses) for param in shared.params} == {("left", "right")}
    assert {tuple(param.all_addresses) for param in shared.params} == {
        ("left.weight", "right.weight"),
        ("left.bias", "right.bias"),
    }
    check_metadata_invariants(trace)
    assert trace.validate_forward_pass([]) is True


def test_jax_equinox_module_call_forward_args_are_populated() -> None:
    """Pytree-module calls should retain public per-call positional args."""

    model = SimpleEquinoxMlp()
    x = jnp.ones(3, dtype=jnp.float32)
    trace = tl.trace(model, x, backend="jax")

    call = trace.module_calls["fc1:1"]
    assert call.forward_args is not None
    assert len(call.forward_args) == 1
    assert jnp.array_equal(call.forward_args[0], x)
    assert call.num_forward_pos_args == 1
    assert call.num_forward_args_total == 1
    assert "Array" in call.forward_args_summary


def test_jax_raw_function_traces_remain_function_root() -> None:
    """Raw JAX callables should keep the existing function-root module mode."""

    def raw_fn(params: dict[str, Any], x: Any) -> Any:
        """Return a tiny raw-function output."""

        return x @ params["w"] + params["b"]

    params = {
        "w": jnp.ones((3, 2), dtype=jnp.float32),
        "b": jnp.zeros((2,), dtype=jnp.float32),
    }
    trace = tl.trace(raw_fn, (params, jnp.ones((4, 3), dtype=jnp.float32)), backend="jax")

    assert trace.module_identity_mode == "function_root"
    assert [module.address for module in trace.modules] == ["self"]
    check_metadata_invariants(trace)
    assert trace.validate_forward_pass([]) is True


@pytest.mark.parametrize(
    ("model", "primitive_name"),
    (
        (JitInsideEquinox(), "jit"),
        (CondInsideEquinox(), "cond"),
        (ScanInsideEquinox(), "scan"),
        (WhileInsideEquinox(), "while"),
    ),
)
def test_jax_equinox_pytree_module_strict_rejects_inner_transforms(
    model: Any,
    primitive_name: str,
) -> None:
    """B2a strict mode should reject transforms/control flow inside modules."""

    match = f"pytree_module strict mode.*{primitive_name!r}.*attributed module 'self'"
    with pytest.raises(ValueError, match=match):
        tl.trace(model, jnp.ones(3, dtype=jnp.float32), backend="jax")


def _pytree_surface_trace() -> Any:
    """Return a real Equinox pytree-module trace for surface tests.

    Returns
    -------
    Any
        Captured TorchLens trace.
    """

    return tl.trace(SimpleEquinoxMlp(), jnp.ones(3, dtype=jnp.float32), backend="jax")


def _assert_pytree_modules_surface(trace: Any, tmp_path: Path) -> None:
    """Assert module accessors and predicates on a pytree-module trace."""

    del tmp_path
    assert trace.module_identity_mode == "pytree_module"
    assert len(trace.modules) > 1
    assert trace.modules["fc1"].address == "fc1"
    assert trace.resolve_sites(tl.in_module("fc1"), max_fanout=8).labels()


def _assert_pytree_module_children_surface(trace: Any, tmp_path: Path) -> None:
    """Assert static and dynamic child surfaces on a pytree-module trace."""

    del tmp_path
    root = trace.modules["self"]
    assert set(root.address_children) == {"fc1", "fc2"}
    assert set(root.call_children) == {"fc1", "fc2"}
    assert trace.modules["fc1"].address_children == []
    assert trace.modules["fc1"].call_children == []


def _assert_pytree_module_params_surface(trace: Any, tmp_path: Path) -> None:
    """Assert direct and recursive parameter surfaces."""

    del tmp_path
    assert trace.modules["fc1"].params
    assert trace.modules["self"].params
    assert trace.modules["self"].recursive_params
    assert {param.module_address for param in trace.modules["fc2"].params} == {"fc2"}


def _assert_pytree_forward_args_surface(trace: Any, tmp_path: Path) -> None:
    """Assert module call forward-arg surfaces."""

    del tmp_path
    call = trace.module_calls["fc1:1"]
    assert call.forward_args is not None
    assert call.forward_kwargs == {}
    assert call.forward_args_summary
    assert trace.modules["fc1"].forward_args is not None


def _assert_pytree_draw_surface(trace: Any, tmp_path: Path) -> None:
    """Assert graph drawing remains available."""

    dot = trace.draw(
        vis_outpath=str(tmp_path / "pytree_graph"),
        vis_save_only=True,
        vis_fileformat="dot",
    )
    assert isinstance(dot, str)


def _assert_pytree_tabular_and_summary_surface(trace: Any, tmp_path: Path) -> None:
    """Assert tabular exports and summaries."""

    del tmp_path
    assert isinstance(trace.summary(), str)
    assert len(trace.to_pandas()) == len(trace.layer_list)
    assert len(trace.modules.to_pandas()) == len(trace.modules)
    assert len(trace.module_calls["fc1:1"].to_pandas()) == 1


def _assert_pytree_save_load_surface(trace: Any, tmp_path: Path) -> None:
    """Assert portable save/load materializes arrays with explicit replay status."""

    audit_path = tmp_path / "jax_pytree_audit.tlspec"
    trace.save(audit_path, level="audit")
    loaded = tl.load(audit_path)
    assert loaded.backend == "jax"
    assert loaded.module_identity_mode == "pytree_module"
    assert all(op.out is None for op in loaded.layer_list)

    portable_path = tmp_path / "jax_pytree_portable.tlspec"
    expected = np.asarray(trace[trace.output_layers[0]].out)
    trace.save(portable_path)
    loaded_portable = tl.load(portable_path)
    loaded_out = loaded_portable[loaded_portable.output_layers[0]].out
    assert loaded_portable.backend == "jax"
    assert loaded_portable.module_identity_mode == "pytree_module"
    assert getattr(loaded_portable, "payload_load_status") == "loaded_device_best_effort"
    assert isinstance(loaded_out, jax.Array)
    assert loaded_out.shape == expected.shape
    assert str(loaded_out.dtype) == str(expected.dtype)
    np.testing.assert_allclose(np.asarray(loaded_out), expected)
    loaded_status = loaded_portable.validate_forward_pass([])
    assert loaded_status is loaded_portable.validation_replay_status
    assert loaded_status.state == "unavailable"
    assert loaded_status.reason == "loaded_trace_runtime_capture_stripped"


def _assert_pytree_validate_surface(trace: Any, tmp_path: Path) -> None:
    """Assert replay validation remains available."""

    del tmp_path
    assert trace.validate_forward_pass([]) is True


def _assert_pytree_grad_surface(trace: Any, tmp_path: Path) -> None:
    """Assert current pytree-module gradient surface contract."""

    del tmp_path
    assert trace.has_backward_pass is False
    with pytest.raises(ValueError, match="derived_grads"):
        _ = trace[0].grads


def _assert_pytree_log_backward_surface(trace: Any, tmp_path: Path) -> None:
    """Assert backward capture rejects with derived-gradient guidance."""

    del tmp_path
    with pytest.raises(BackendUnsupportedError, match="backward capture.*derived_grads"):
        trace.log_backward(trace[trace.output_layers[0]].out)


@pytest.mark.parametrize(
    "surface_assertion",
    (
        _assert_pytree_modules_surface,
        _assert_pytree_module_children_surface,
        _assert_pytree_module_params_surface,
        _assert_pytree_forward_args_surface,
        _assert_pytree_draw_surface,
        _assert_pytree_tabular_and_summary_surface,
        _assert_pytree_save_load_surface,
        _assert_pytree_validate_surface,
        _assert_pytree_grad_surface,
        _assert_pytree_log_backward_surface,
    ),
)
def test_jax_equinox_pytree_public_surface_matrix(
    surface_assertion: Callable[[Any, Path], None],
    tmp_path: Path,
) -> None:
    """Assert executable public surfaces on a real Equinox pytree trace."""

    surface_assertion(_pytree_surface_trace(), tmp_path)
