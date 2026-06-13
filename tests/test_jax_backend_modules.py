"""JAX Equinox module hierarchy tests."""

from __future__ import annotations

from typing import Any

import pytest

import torchlens as tl
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
