"""JAX: an edited gradient at a ``tl.grad`` site equals ``jax.vjp`` with the edit by hand."""

from __future__ import annotations

from typing import Any

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import torchlens as tl  # noqa: E402
from tests import test_parity_b_gradsites_support as support  # noqa: E402

pytestmark = [pytest.mark.backend_jax, pytest.mark.rare]


def _f1(x: Any) -> Any:
    return x * x + x


def _f2(s: Any, x: Any) -> Any:
    return jnp.sum(s * s + s * x)


def _program(x: Any) -> Any:
    return _f2(_f1(x), x)


def _edits() -> list[tuple[str, Any]]:
    return [
        ("scale", lambda g: g * support.SCALE),
        ("clip", lambda g: jnp.clip(g, -support.CLIP, support.CLIP)),
    ]


def _oracle(x: Any, edits: list[tuple[str, Any]]) -> Any:
    """``jax.vjp`` of each half, the edits applied to the site gradient in between."""

    s, pull_s = jax.vjp(_f1, x)
    _loss, pull_loss = jax.vjp(_f2, s, x)
    grad_s, grad_x_direct = pull_loss(jnp.ones((), dtype=x.dtype))
    for _name, fn in edits:
        grad_s = fn(grad_s)
    return pull_s(grad_s)[0] + grad_x_direct


@pytest.mark.parametrize("edited", [False, True])
def test_edited_site_gradient_equals_jax_by_hand(edited: bool) -> None:
    x = jnp.asarray(support.x_np())
    trace = tl.trace(_program, x, backend="jax")
    edits = _edits() if edited else []
    run = support.run_neutral(trace, "jax", x, edits, jnp.ones_like)
    support.assert_matches_oracle(run, _oracle(x, edits), edited=edited)
