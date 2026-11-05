"""grind-p3 T11.7: _fields presence is not proof of an *args constructor.

The capture-entry normalizers (``_coerce_input_args`` and
``copy_arg_tree``) inferred a namedtuple ``*args`` constructor from the mere
presence of ``_fields``, so a tuple subclass exposing ``_fields`` as a list
or property -- with an ordinary single-iterable constructor -- crashed plain
``tl.trace`` with an untyped ``TypeError`` blamed on the user's container.
The ``_move_tensors_to_device`` rebuild ladder (try ``*args``, then the
single-iterable shape, verify the exact type) now applies to both siblings,
strengthened with an exact item-identity check so a single-iterable
constructor reached via ``*items`` can never silently EXPAND a lone iterable
element.
"""

from __future__ import annotations

import collections

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._input_coerce import _coerce_input_args
from torchlens._utils.arg_handling import copy_arg_tree, rebuild_tuple_like


class _MalformedFields(tuple):
    """Tuple subclass with a NON-namedtuple ``_fields`` and iterable ctor."""

    _fields = ["a", "b"]


class _PropertyFields(tuple):
    """Tuple subclass whose ``_fields`` is a property, iterable ctor."""

    @property
    def _fields(self) -> tuple[str, ...]:
        """Return a fake schema."""

        return ("a", "b")


class _NeedsExtraArgs(tuple):
    """Tuple subclass whose constructor cannot be called generically."""

    def __new__(cls, values: object, meta: object) -> _NeedsExtraArgs:
        """Build from values plus a required metadata argument."""

        instance = super().__new__(cls, values)  # type: ignore[arg-type]
        instance.meta = meta  # type: ignore[attr-defined]
        return instance


_Point = collections.namedtuple("_Point", ["x", "y"])


class _TakesModel(nn.Module):
    """Model consuming the first element of a tuple-like container."""

    def forward(self, box: tuple) -> torch.Tensor:
        """Add one to the container's first element."""

        return box[0] + 1


def test_coerce_input_args_handles_malformed_fields_subclass():
    """The coercer no longer crashes on a lying ``_fields``."""

    import numpy as np

    box = _MalformedFields((np.ones(2), torch.zeros(2)))
    coerced = _coerce_input_args(nn.Identity(), box)
    assert type(coerced) is _MalformedFields
    assert isinstance(coerced[0], torch.Tensor)


def test_property_fields_subclass_refuses_typed_end_to_end():
    """Plain tl.trace REFUSES a property-``_fields`` tuple subclass input, typed.

    Contract progression (reviewed, fw4): originally this shape crashed with an
    untyped ``TypeError`` from the *args constructor inference (T11.7 red);
    T11.7's fix made it trace -- but with every enclosed tensor leaf silently
    UNATTRIBUTED (this test formerly tolerated that gap with a warning filter).
    B3R4-R12-1 settled the honest contract: a tuple subclass DECLARING a
    ``_fields`` schema that cannot account for its physical elements refuses
    capture entry with a typed ``InvalidArgumentError`` naming the remedy --
    never an untyped crash, and never a trace with silently dropped input
    provenance.
    """

    from torchlens._errors import InvalidArgumentError

    box = _PropertyFields((torch.ones(2), torch.zeros(2)))
    with pytest.raises(InvalidArgumentError, match="_fields"):
        tl.trace(_TakesModel(), box)


def test_unreconstructable_subclass_passes_by_reference():
    """A constructor needing extra args falls back instead of crashing."""

    box = _NeedsExtraArgs((torch.ones(2),), meta="m")
    assert copy_arg_tree(box) is box
    assert _coerce_input_args(nn.Identity(), box) is box


def test_ladder_never_expands_a_lone_iterable_element():
    """``*items`` reaching an iterable ctor cannot silently expand the item."""

    rebuilt = rebuild_tuple_like(_MalformedFields, [[1, 2, 3]])
    assert rebuilt is not None
    assert tuple.__len__(rebuilt) == 1
    assert tuple.__getitem__(rebuilt, 0) == [1, 2, 3]
