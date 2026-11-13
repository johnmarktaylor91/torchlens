"""Paddle: an edited gradient at a ``tl.grad`` site equals ``paddle.grad`` with the edit by hand."""

from __future__ import annotations

from typing import Any

import pytest

paddle = pytest.importorskip("paddle")

import torchlens as tl  # noqa: E402
from tests import test_parity_b_gradsites_support as support  # noqa: E402

pytestmark = [pytest.mark.backend_paddle, pytest.mark.rare]


def _f1(x: Any) -> Any:
    return x * x + x


def _f2(s: Any, x: Any) -> Any:
    return paddle.sum(s * s + s * x)


def _program(x: Any) -> Any:
    return _f2(_f1(x), x)


def _edits() -> list[tuple[str, Any]]:
    return [
        ("scale", lambda g: g * support.SCALE),
        ("clip", lambda g: paddle.clip(g, -support.CLIP, support.CLIP)),
    ]


def _leaf() -> Any:
    return paddle.to_tensor(support.x_np(), stop_gradient=False)


def _oracle(edits: list[tuple[str, Any]]) -> Any:
    """``paddle.grad`` of each half, the edits applied to the site gradient in between."""

    x_s = _leaf()
    s = _f1(x_s)
    s_leaf = s.detach()
    s_leaf.stop_gradient = False
    x_loss = _leaf()
    grad_s, grad_x_direct = paddle.grad([_f2(s_leaf, x_loss)], [s_leaf, x_loss])
    for _name, fn in edits:
        grad_s = fn(grad_s)
    (grad_x,) = paddle.grad([s], [x_s], grad_outputs=[grad_s])
    return (grad_x + grad_x_direct).numpy()


@pytest.mark.parametrize("edited", [False, True])
def test_edited_site_gradient_equals_paddle_by_hand(edited: bool) -> None:
    x = paddle.to_tensor(support.x_np())
    trace = tl.trace(_program, (x,), backend="paddle")
    edits = _edits() if edited else []
    run = support.run_neutral(trace, "paddle", x, edits, paddle.ones_like)
    support.assert_matches_oracle(run, _oracle(edits), edited=edited)
