"""Gradient influence probes on JAX traces: framework gradient, finite differences, torch twin.

The subject is the feature-table JAX model (``tests/test_backend_feature_table.py``);
the checks live in ``tests/test_parity_b_influence_support.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import torchlens as tl
from tests import (
    test_backend_feature_table as feature_table,
    test_parity_b_influence_support as support,
)

pytestmark = pytest.mark.backend_jax


def _kit() -> support.Kit:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    subject = feature_table._jax_subject()
    model = subject.model
    x = np.asarray(subject.x)

    def unit_gradient(k: int) -> np.ndarray:
        return np.asarray(jax.grad(lambda value: model(value)[0, k])(jnp.asarray(x)))

    def head_gradient(hidden: np.ndarray, k: int) -> np.ndarray:
        head = lambda value: model.head(value.reshape((1, -1)))[0, k]  # noqa: E731
        return np.asarray(jax.grad(head)(jnp.asarray(hidden)))

    kernel = np.asarray(model.conv.kernel.value)
    twin = support.Twin(
        (np.transpose(kernel, (3, 2, 0, 1)), np.asarray(model.conv.bias.value)),
        (np.asarray(model.head.kernel.value).T, np.asarray(model.head.bias.value)),
        channels_last=True,
    )
    return support.Kit(
        trace=subject.trace(),
        x=x,
        center=(0, 3, 3, 0),
        activation=subject.act_func,
        unit_gradient=unit_gradient,
        head_gradient=head_gradient,
        forward=lambda value: np.asarray(model(jnp.asarray(value))),
        twin=twin,
        rng_state=lambda: (),
    )


def _two_input_trace(a: np.ndarray, b: np.ndarray) -> Any:
    jnp = pytest.importorskip("jax.numpy")
    return tl.trace(
        lambda left, right: (left * left, right + 1.0),
        (jnp.asarray(a), jnp.asarray(b)),
        backend="jax",
    )


@pytest.fixture
def kit() -> support.Kit:
    # No trace.cleanup(): it raises on TF, MLX and tinygrad traces that hold parameters.
    return _kit()


def test_receptive_probe_equals_the_framework_gradient(kit: support.Kit) -> None:
    support.check_receptive_matches_framework(kit)


def test_receptive_probe_matches_finite_differences(kit: support.Kit) -> None:
    support.check_receptive_matches_finite_differences(kit)


def test_probe_from_the_activation_equals_the_head_gradient(kit: support.Kit) -> None:
    support.check_intermediate_source(kit)


def test_projective_column_equals_the_framework_jacobian(kit: support.Kit) -> None:
    support.check_projective_matches_framework(kit)


def test_both_directions_match_the_torch_twin(kit: support.Kit) -> None:
    support.check_twin_matches_torch(kit)


def test_gradient_selection_answers_in_both_directions(kit: support.Kit) -> None:
    support.check_selection_methods(kit)


def test_probes_leave_backward_state_and_rng_alone(kit: support.Kit) -> None:
    support.check_probe_hygiene(kit)


def test_disconnected_input_gets_a_zero_gradient() -> None:
    rng = np.random.default_rng(2)
    a = rng.standard_normal((2, 3)).astype(np.float32)
    b = rng.standard_normal((2, 3)).astype(np.float32)
    support.check_two_inputs_disconnected(_two_input_trace(a, b), a)
