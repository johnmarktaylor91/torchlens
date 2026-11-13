"""tinygrad: an edited gradient at a ``tl.grad`` site equals ``backward`` with the edit by hand."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("tinygrad")
from tinygrad import Tensor  # noqa: E402

import torchlens as tl  # noqa: E402
from tests import test_parity_b_gradsites_support as support  # noqa: E402

pytestmark = [pytest.mark.backend_tinygrad, pytest.mark.rare]


def _f1(x: Any) -> Any:
    return x * x + x


def _f2(s: Any, x: Any) -> Any:
    return (s * s + s * x).sum()


def _program(x: Any) -> Any:
    return _f2(_f1(x), x)


def _edits() -> list[tuple[str, Any]]:
    return [
        ("scale", lambda g: g * support.SCALE),
        ("clip", lambda g: g.clip(-support.CLIP, support.CLIP)),
    ]


def _oracle(edits: list[tuple[str, Any]]) -> Any:
    """``backward`` of each half, the edits applied to the site gradient in between."""

    s_leaf = Tensor(_f1(Tensor(support.x_np())).numpy(), requires_grad=True)
    x_loss = Tensor(support.x_np(), requires_grad=True)
    _f2(s_leaf, x_loss).backward()
    grad_s = s_leaf.grad
    for _name, fn in edits:
        grad_s = fn(grad_s)
    grad_s = Tensor(grad_s.numpy())
    x_s = Tensor(support.x_np(), requires_grad=True)
    (_f1(x_s) * grad_s).sum().backward()
    return x_s.grad.numpy() + x_loss.grad.numpy()


@pytest.mark.parametrize("edited", [False, True])
def test_edited_site_gradient_equals_tinygrad_by_hand(edited: bool) -> None:
    x = Tensor(support.x_np())
    trace = tl.trace(_program, x, backend="tinygrad")
    edits = _edits() if edited else []
    run = support.run_neutral(trace, "tinygrad", x, edits, lambda v: v.ones_like())
    support.assert_matches_oracle(run, _oracle(edits), edited=edited)
