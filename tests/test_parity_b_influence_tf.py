"""Gradient influence probes on TensorFlow traces: framework gradient, finite differences, torch twin.

The subject is the feature-table TensorFlow model (``tests/test_backend_feature_table.py``);
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

pytestmark = pytest.mark.tf_backend


def _kit() -> support.Kit:
    tf = pytest.importorskip("tensorflow")
    subject = feature_table._tf_subject()
    model = subject.model
    x = np.asarray(subject.x)

    def unit_gradient(k: int) -> np.ndarray:
        value = tf.constant(x)
        with tf.GradientTape() as tape:
            tape.watch(value)
            output = model(value)[0, k]
        return tape.gradient(output, value).numpy()

    def head_gradient(hidden: np.ndarray, k: int) -> np.ndarray:
        value = tf.constant(hidden)
        with tf.GradientTape() as tape:
            tape.watch(value)
            output = model.get_layer("head")(tf.reshape(value, (1, -1)))[0, k]
        return tape.gradient(output, value).numpy()

    conv, head = model.get_layer("conv"), model.get_layer("head")
    twin = support.Twin(
        (np.transpose(conv.kernel.numpy(), (3, 2, 0, 1)), conv.bias.numpy()),
        (head.kernel.numpy().T, head.bias.numpy()),
        channels_last=True,
    )
    return support.Kit(
        trace=subject.trace(),
        x=x,
        center=(0, 3, 3, 0),
        activation=subject.act_func,
        unit_gradient=unit_gradient,
        head_gradient=head_gradient,
        forward=lambda value: model(tf.constant(value)).numpy(),
        twin=twin,
        rng_state=lambda: (),
    )


def _two_input_trace(a: np.ndarray, b: np.ndarray) -> Any:
    tf = pytest.importorskip("tensorflow")
    return tl.trace(
        lambda left, right: (left * left, right + 1.0),
        (tf.constant(a), tf.constant(b)),
        backend="tf",
    )


@pytest.fixture
def kit() -> Iterator[support.Kit]:
    built = _kit()
    try:
        yield built
    finally:
        built.trace.cleanup()


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
    trace = _two_input_trace(a, b)
    try:
        support.check_two_inputs_disconnected(trace, a, b)
    finally:
        trace.cleanup()
