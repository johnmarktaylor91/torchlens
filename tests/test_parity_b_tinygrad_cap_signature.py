"""The tinygrad structural UOp signature stays linear in a graph that reuses values.

The capture keys output matching, T1 matching and lineage on a structural
signature of each UOp. Expanding every source recursively doubles the string
at each reused value (``relu`` reads its input twice), so a deep forward
spent most of its capture time and memory building signatures. Each source
now enters as a digest of its own signature, computed once per call.
"""

from __future__ import annotations

from typing import Any

import pytest

from torchlens.backends.tinygrad._uop_graph import _uop_signature

tinygrad = pytest.importorskip("tinygrad")
Tensor = tinygrad.Tensor

pytestmark = [pytest.mark.backend_tinygrad, pytest.mark.rare]


def _reuse_chain(depth: int) -> Any:
    """A chain where every step reads the previous value twice."""

    x = Tensor([1.0, -2.0, 3.0]).realize()
    for _ in range(depth):
        x = (x * x + 1.0).relu()
    return x


def test_signature_length_is_bounded_on_a_deep_reuse_chain() -> None:
    """Forty reused steps give a short signature (the expanded form has 2**40 leaves)."""

    signature = _uop_signature(_reuse_chain(40).uop)

    assert len(signature) < 4096


def test_signature_separates_structures_and_matches_equal_ones() -> None:
    """Equal structure gives an equal signature; a changed constant deep inside does not."""

    base = Tensor([1.0, 2.0]).realize()

    def build(constant: float) -> Any:
        y = base * constant
        for _ in range(6):
            y = (y * y).relu()
        return y

    assert _uop_signature(build(3.0).uop) == _uop_signature(build(3.0).uop)
    assert _uop_signature(build(3.0).uop) != _uop_signature(build(4.0).uop)
