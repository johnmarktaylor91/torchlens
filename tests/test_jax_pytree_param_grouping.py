"""Recurrence grouping on JAX pytree modules keys on param identity, as in torch.

Two distinct modules of the same shape (each with its own weights) must stay separate
layers; one module instance called twice still folds into one multi-pass layer. Before the
param barcodes were stamped ahead of grouping, the grouper saw no params on pytree ops and
folded distinct linears together, and ``check_metadata_invariants`` failed ``param_xrefs``.
"""

from __future__ import annotations

from typing import Any

import pytest

import torchlens as tl
from torchlens.validation import check_metadata_invariants

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
eqx = pytest.importorskip("equinox")
nnx = pytest.importorskip("flax.nnx")

pytestmark = [pytest.mark.backend_jax, pytest.mark.rare]


class _EqxPair(eqx.Module):
    """Two same-shape Equinox linears, used in sequence or one of them twice."""

    first: eqx.nn.Linear
    second: eqx.nn.Linear
    reuse: bool = eqx.field(static=True)

    def __call__(self, value: Any) -> Any:
        """Apply ``first`` then ``second`` (or ``first`` twice)."""

        hidden = jnp.maximum(self.first(value), 0.0)
        return (self.first if self.reuse else self.second)(hidden)


class _NnxPair(nnx.Module):
    """Two same-shape NNX linears applied in sequence."""

    def __init__(self) -> None:
        """Build the linears."""

        self.first = nnx.Linear(4, 4, rngs=nnx.Rngs(1))
        self.second = nnx.Linear(4, 4, rngs=nnx.Rngs(2))

    def __call__(self, value: Any) -> Any:
        """Apply both linears."""

        return self.second(jnp.maximum(self.first(value), 0.0))


def _eqx_pair(reuse: bool) -> _EqxPair:
    """Return an Equinox pair with distinct weights."""

    first_key, second_key = jax.random.split(jax.random.PRNGKey(0))
    return _EqxPair(eqx.nn.Linear(4, 4, key=first_key), eqx.nn.Linear(4, 4, key=second_key), reuse)


def _multi_pass_layers(trace: Any) -> set[str]:
    """Return the layer labels that hold more than one pass."""

    return {op.layer_label for op in trace._layer_list if op.num_passes > 1}


@pytest.mark.parametrize(
    ("model", "value"),
    [
        pytest.param(_eqx_pair(reuse=False), jnp.linspace(-1.0, 1.0, 4), id="equinox"),
        pytest.param(_NnxPair(), jnp.linspace(-1.0, 1.0, 8).reshape(2, 4), id="nnx"),
    ],
)
def test_distinct_same_shape_modules_stay_separate_layers(model: Any, value: Any) -> None:
    """Distinct params keep same-shape ops in separate layers; invariants hold."""

    trace = tl.trace(model, value, backend="jax")
    assert _multi_pass_layers(trace) == set()
    assert check_metadata_invariants(trace) is True
    assert trace.validate_forward_pass([]) is True


def test_one_module_called_twice_still_groups() -> None:
    """The same module instance called twice still folds into multi-pass layers."""

    trace = tl.trace(_eqx_pair(reuse=True), jnp.linspace(-1.0, 1.0, 4), backend="jax")
    assert _multi_pass_layers(trace)
    assert check_metadata_invariants(trace) is True
    assert trace.validate_forward_pass([]) is True
