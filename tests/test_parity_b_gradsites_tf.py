"""TensorFlow: an edited gradient at a ``tl.grad`` site equals ``tape.gradient`` by hand."""

from __future__ import annotations

from typing import Any

import pytest

tf = pytest.importorskip("tensorflow")

import torchlens as tl  # noqa: E402
from tests import test_parity_b_gradsites_support as support  # noqa: E402

pytestmark = [pytest.mark.tf_backend, pytest.mark.rare]


def _f1(x: Any) -> Any:
    return x * x + x


def _f2(s: Any, x: Any) -> Any:
    return tf.reduce_sum(s * s + s * x)


def _program(x: Any) -> Any:
    return _f2(_f1(x), x)


def _edits() -> list[tuple[str, Any]]:
    return [
        ("scale", lambda g: g * support.SCALE),
        ("clip", lambda g: tf.clip_by_value(g, -support.CLIP, support.CLIP)),
    ]


def _oracle(x: Any, edits: list[tuple[str, Any]]) -> Any:
    """``tape.gradient`` of each half, the edits applied to the site gradient in between."""

    with tf.GradientTape() as tape_s:
        tape_s.watch(x)
        s = _f1(x)
    with tf.GradientTape() as tape_loss:
        tape_loss.watch([s, x])
        loss = _f2(s, x)
    grad_s, grad_x_direct = tape_loss.gradient(loss, [s, x])
    for _name, fn in edits:
        grad_s = fn(grad_s)
    return tape_s.gradient(s, x, output_gradients=grad_s) + grad_x_direct


@pytest.mark.parametrize("edited", [False, True])
def test_edited_site_gradient_equals_tf_by_hand(edited: bool) -> None:
    x = tf.constant(support.x_np())
    trace = tl.trace(_program, x, backend="tf")
    edits = _edits() if edited else []
    run = support.run_neutral(trace, "tf", x, edits, tf.ones_like)
    support.assert_matches_oracle(run, _oracle(x, edits).numpy(), edited=edited)
