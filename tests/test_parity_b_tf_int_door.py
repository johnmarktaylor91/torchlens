"""TensorFlow ``Trace.do`` with a ``tl.grad`` site: the door arms the edit on the trace.

The armed edit is then replayed by the shared reverse engine over the
TensorFlow replay adapter, and the input gradient equals ``tf.GradientTape``
split by hand at the site with the edit applied to the site's gradient.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

import torchlens as tl  # noqa: E402
from tests.test_parity_b_gradsites_support import _replay_values  # noqa: E402
from torchlens.backends._replay_adapters import get_adapter  # noqa: E402
from torchlens.errors import RunCapabilityUnavailableError  # noqa: E402
from torchlens.intervention._grad_site_neutral import (  # noqa: E402
    armed_grad_sites,
    clear_grad_sites,
    edited_gradients_for_backend,
)

pytestmark = [pytest.mark.tf_backend, pytest.mark.rare]

_X = np.asarray([1.0, -0.2, 0.9, 1.4], dtype=np.float32)


def _program(x: Any) -> Any:
    return tf.reduce_sum(tf.nn.relu(x * x - 0.5) * x)


def _oracle(x: Any, edit: Any) -> np.ndarray:
    """``d loss / d x`` with ``edit`` applied to the gradient entering the rectifier."""

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        r = tf.nn.relu(x * x - 0.5)
        loss = tf.reduce_sum(r * x)
    total = tape.gradient(loss, x)
    g_site = tape.gradient(loss, r)
    through = tape.gradient(r, x, output_gradients=g_site)
    edited = tape.gradient(r, x, output_gradients=edit(g_site))
    return np.asarray(total - through + edited, dtype=np.float64)


def _replayed_input_grad(trace: Any, x: Any) -> np.ndarray:
    """Replay the armed edits through the reverse engine; return the input's gradient."""

    adapter = get_adapter("tf")
    values, graph_ops, source = _replay_values(trace, adapter, x)
    loss = adapter.recipe(trace, graph_ops[-1]).output_slots[0]
    result = edited_gradients_for_backend(
        trace,
        "tf",
        graph_ops,
        values.__getitem__,
        {loss: tf.ones_like(values[loss])},
        armed_grad_sites(trace),
    )
    label = next(name for name in source if name in result.grads)
    return np.asarray(adapter.to_host(result.grads[label]), dtype=np.float64)


@pytest.mark.parametrize(
    ("helper", "edit"),
    [
        (lambda: tl.scale(0.25), lambda g: g * 0.25),
        (tl.zero_ablate, tf.zeros_like),
        (lambda: lambda grad, *, ctx: grad * -2.0, lambda g: g * -2.0),
    ],
    ids=["scale", "zero_ablate", "callable"],
)
def test_do_arms_a_grad_site_and_the_engine_matches_tf_by_hand(helper: Any, edit: Any) -> None:
    x = tf.constant(_X)
    trace = tl.trace(_program, x, backend="tf")
    relu = next(op for op in trace._layer_list if str(op.func_name).lower() == "relu")
    try:
        result = trace.do(tl.grad(tl.func("relu")), helper())
        assert result is trace
        assert [armed.label for armed in armed_grad_sites(trace)] == [relu.label]
        np.testing.assert_allclose(
            _replayed_input_grad(trace, x), _oracle(x, edit), rtol=1e-5, atol=1e-6
        )
    finally:
        clear_grad_sites(trace)


def test_do_with_an_activation_half_refuses_before_arming() -> None:
    x = tf.constant(_X)
    trace = tl.trace(_program, x, backend="tf")
    with pytest.raises(RunCapabilityUnavailableError, match="separate") as caught:
        trace.do(tl.grad(tl.func("relu")) | tl.func("sum"), tl.scale(0.5))
    assert caught.value.fields["code"] == "run_capability_unavailable"
    assert armed_grad_sites(trace) == ()
