"""Gradient influence probes on MLX traces: framework gradient, finite differences, torch twin.

The subject is the feature-table MLX model (``tests/test_backend_feature_table.py``);
the checks live in ``tests/test_parity_b_influence_support.py``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

import torchlens as tl
from tests import (
    test_backend_feature_table as feature_table,
    test_parity_b_influence_support as support,
)

pytestmark = pytest.mark.backend_mlx


def _kit() -> support.Kit:
    subject = feature_table._mlx_subject()
    mx = pytest.importorskip("mlx.core")
    model = subject.model
    x = np.asarray(subject.x)

    def unit_gradient(k: int) -> np.ndarray:
        return np.array(mx.grad(lambda value: model(value)[0, k])(mx.array(x)))

    def head_gradient(hidden: np.ndarray, k: int) -> np.ndarray:
        head = lambda value: model.head(value.reshape(1, -1))[0, k]  # noqa: E731
        return np.array(mx.grad(head)(mx.array(hidden)))

    weight = np.array(model.conv.weight)
    twin = support.Twin(
        (np.transpose(weight, (0, 3, 1, 2)), np.array(model.conv.bias)),
        (np.array(model.head.weight), np.array(model.head.bias)),
        channels_last=True,
    )
    return support.Kit(
        trace=subject.trace(),
        x=x,
        center=(0, 3, 3, 0),
        activation=subject.act_func,
        unit_gradient=unit_gradient,
        head_gradient=head_gradient,
        forward=lambda value: np.array(model(mx.array(value))),
        twin=twin,
        rng_state=lambda: tuple(np.array(item) for item in mx.random.state),
    )


def _two_input_trace(a: np.ndarray, b: np.ndarray) -> Any:
    feature_table._mlx_subject()
    mx = pytest.importorskip("mlx.core")
    return tl.trace(
        lambda left, right: (left * left, right + 1.0),
        (mx.array(a), mx.array(b)),
        backend="mlx",
    )


@pytest.fixture
def kit() -> Iterator[support.Kit]:
    built = _kit()
    yield built
    built.trace.cleanup()


def test_cleanup_releases_the_parameters(kit: support.Kit) -> None:
    support.check_cleanup_releases_parameters(kit.trace)


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
