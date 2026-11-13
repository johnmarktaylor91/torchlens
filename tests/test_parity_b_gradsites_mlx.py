"""MLX: an edited gradient at a ``tl.grad`` site equals ``mx.vjp`` with the edit by hand.

MLX has no observable backward pass; the neutral module needs only gradient
values, so ``tl.grad`` sites work here too.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytest.importorskip("mlx")
import mlx.core as mx  # noqa: E402

import torchlens as tl  # noqa: E402
from tests import test_parity_b_gradsites_support as support  # noqa: E402

pytestmark = [pytest.mark.backend_mlx, pytest.mark.rare]


def _f1(x: Any) -> Any:
    return x * x + x


def _f2(s: Any, x: Any) -> Any:
    return mx.sum(s * s + s * x)


def _program(x: Any) -> Any:
    return _f2(_f1(x), x)


def _edits() -> list[tuple[str, Any]]:
    return [
        ("scale", lambda g: g * support.SCALE),
        ("clip", lambda g: mx.clip(g, -support.CLIP, support.CLIP)),
    ]


def _oracle(x: Any, edits: list[tuple[str, Any]]) -> Any:
    """``mx.vjp`` of each half, the edits applied to the site gradient in between."""

    (s,), _ = mx.vjp(lambda v: [_f1(v)], [x], [mx.ones_like(x)])
    _out, (grad_s, grad_x_direct) = mx.vjp(lambda a, b: [_f2(a, b)], [s, x], [mx.array(1.0)])
    for _name, fn in edits:
        grad_s = fn(grad_s)
    _out, (grad_x,) = mx.vjp(lambda v: [_f1(v)], [x], [grad_s])
    total = grad_x + grad_x_direct
    mx.eval(total)
    return np.array(total)


@pytest.mark.parametrize("edited", [False, True])
def test_edited_site_gradient_equals_mlx_by_hand(edited: bool) -> None:
    x = mx.array(support.x_np())
    trace = tl.trace(_program, x, backend="mlx")
    edits = _edits() if edited else []
    run = support.run_neutral(trace, "mlx", x, edits, mx.ones_like)
    support.assert_matches_oracle(run, _oracle(x, edits), edited=edited)
