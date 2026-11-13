"""Gradient influence probes on tinygrad traces: framework gradient, finite differences, torch twin.

The subject is the feature-table tinygrad model (``tests/test_backend_feature_table.py``);
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

pytestmark = pytest.mark.backend_tinygrad


def _kit() -> support.Kit:
    tinygrad = pytest.importorskip("tinygrad")
    tensor = tinygrad.Tensor
    subject = feature_table._tinygrad_subject()
    model = subject.model
    x = np.asarray(subject.x.numpy())

    def _grad(fn: Any, at: np.ndarray) -> np.ndarray:
        value = tensor(at)
        (grad,) = fn(value).gradient(value)
        return grad.numpy()

    def unit_gradient(k: int) -> np.ndarray:
        return _grad(lambda value: model(value)[0, k], x)

    def head_gradient(hidden: np.ndarray, k: int) -> np.ndarray:
        return _grad(lambda value: model.head(value.flatten(1))[0, k], hidden)

    twin = support.Twin(
        (model.conv.weight.numpy(), model.conv.bias.numpy()),
        (model.head.weight.numpy(), model.head.bias.numpy()),
        channels_last=False,
    )
    return support.Kit(
        trace=subject.trace(),
        x=x,
        center=(0, 0, 3, 3),
        activation=subject.act_func,
        unit_gradient=unit_gradient,
        head_gradient=head_gradient,
        forward=lambda value: model(tensor(value)).numpy(),
        twin=twin,
        rng_state=lambda: (int(tensor._seed),),
    )


def _two_input_trace(a: np.ndarray, b: np.ndarray) -> Any:
    tinygrad = pytest.importorskip("tinygrad")
    return tl.trace(
        lambda left, right: (left * left, right + 1.0),
        (tinygrad.Tensor(a).realize(), tinygrad.Tensor(b).realize()),
        backend="tinygrad",
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
