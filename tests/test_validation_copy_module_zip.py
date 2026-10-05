"""grind-p3 T11.10: validation-copy attr restoration refuses misaligned trees.

``_restore_simple_plain_attrs_on_copy`` zipped ``source.modules()`` against
``copied.modules()`` positionally with no arity check, so a deepcopy that
adds or drops a submodule (a ``__deepcopy__`` hook, lazy child
materialization) silently shifted every later pair and aligned plain
attributes onto the WRONG modules. The zip is now ``strict=True``; the
caller's existing fallback then validates against the live model with a
disclosure instead of proceeding misaligned.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from torchlens._plain_attr_fidelity import (
    restore_simple_plain_attrs_on_copy as _restore_simple_plain_attrs_on_copy,
)


class _TwoBlocks(nn.Module):
    """Parent with two named child blocks carrying plain attributes."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(2, 2)
        self.b = nn.Linear(2, 2)
        self.a.mode = "a-mode"
        self.b.mode = "b-mode"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run both blocks."""

        return self.b(self.a(x))


def test_module_count_mismatch_refuses_instead_of_shifting():
    """A dropped submodule in the copy raises, never misaligns attributes."""

    source = _TwoBlocks()
    shifted = _TwoBlocks()
    del shifted.a  # copy tree now enumerates [root, b] vs source [root, a, b]
    shifted.b.mode = "stale"
    with pytest.raises(ValueError):
        _restore_simple_plain_attrs_on_copy(source, shifted)
    # And the misalignment the strict zip prevents: positional pairing would
    # have compared source ``a`` against copied ``b``.
    assert shifted.b.mode == "stale"


def test_matched_trees_still_align_attributes():
    """Equal-arity trees keep the historical attribute restoration."""

    source = _TwoBlocks()
    copied = _TwoBlocks()
    copied.a.mode = "drifted"
    _restore_simple_plain_attrs_on_copy(source, copied)
    assert copied.a.mode == "a-mode"
    assert copied.b.mode == "b-mode"
