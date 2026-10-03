"""grind-p3 T11.1: property-shadowed namedtuple fields fail closed.

A namedtuple subclass shadowing a DECLARED field with a ``property`` bypassed
the whole declared-schema proof: physical arity matched, no instance state
existed, no ``__getattribute__``/``__getattr__`` was overridden -- so every
walker's ``getattr`` read the property's DECOY while the hidden physical slot
(``tuple.__getitem__``) steered forward control flow with no witness, and
replay read the decoy as VERIFIED. The declared-schema gate now requires the
stock positional ``_tuplegetter`` per declared field (mirroring the slots
rule), refusing ``instance_state_uninspectable`` without ever running the
shadowing descriptor. TRIPWIRE STRENGTHENING -- never weaken.
"""

from __future__ import annotations

import collections
import typing

import pytest
import torch

from torchlens._input_walk import (
    _namedtuple_field_descriptors_shadowed,
    snapshot_input_boundary,
    undeclared_instance_state,
)

_Base = collections.namedtuple("_Base", ["x", "flag"])


class _PropertyShadow(_Base):
    """Declared field ``flag`` shadowed by a decoy-returning property."""

    __slots__ = ()

    @property
    def flag(self) -> str:  # type: ignore[override]
        """Return the decoy value replay would read."""

        return "decoy"


class _TypedTuple(typing.NamedTuple):
    """Stock typing.NamedTuple control."""

    x: torch.Tensor
    n: int


class _ExtraProperty(_Base):
    """Subclass adding a NON-field convenience property (must stay admitted)."""

    __slots__ = ()

    @property
    def doubled(self) -> object:
        """Return a derived convenience value."""

        return self.n if hasattr(self, "n") else None


def _refusal_reasons(value: object) -> list[str]:
    """Snapshot one value and return its refusal reasons."""

    return [refusal["reason"] for refusal in snapshot_input_boundary(value)["refusals"]]


@pytest.mark.smoke
def test_property_shadowed_field_refuses_uninspectable():
    """The decoy container refuses instead of witnessing the decoy."""

    hidden = _PropertyShadow(torch.ones(2), "steer-me")
    assert hidden.flag == "decoy"  # the decoy is live
    assert tuple.__getitem__(hidden, 1) == "steer-me"  # the hidden state is live
    assert "instance_state_uninspectable" in _refusal_reasons(hidden)


def test_property_shadowed_field_reads_as_undeclared_state():
    """The W-family fail-closed judgment also fires."""

    hidden = _PropertyShadow(torch.ones(2), "steer-me")
    assert undeclared_instance_state(hidden, "namedtuple") is True


def test_transposed_stock_getter_refuses():
    """A stock getter bound to the WRONG index is not the declared schema."""

    class _Transposed(_Base):
        __slots__ = ()

    _Transposed.x, _Transposed.flag = _Base.flag, _Base.x
    swapped = _Transposed(torch.ones(2), "steer-me")
    assert _namedtuple_field_descriptors_shadowed(swapped) is True
    assert "instance_state_uninspectable" in _refusal_reasons(swapped)


def test_stock_namedtuples_stay_admitted():
    """collections and typing namedtuples keep zero refusals."""

    assert _refusal_reasons(_Base(torch.ones(2), 3)) == []
    assert _refusal_reasons(_TypedTuple(torch.ones(2), 3)) == []


@pytest.mark.smoke
def test_extra_nonfield_property_stays_admitted():
    """A convenience property OUTSIDE the declared schema is not a shadow."""

    assert _namedtuple_field_descriptors_shadowed(_ExtraProperty(torch.ones(2), 3)) is False
    assert _refusal_reasons(_ExtraProperty(torch.ones(2), 3)) == []


def test_shadowing_property_is_never_executed_by_the_gate():
    """Fail-closed judgment must be inert: the hostile descriptor never runs."""

    calls: list[str] = []

    class _Booby(_Base):
        __slots__ = ()

        @property
        def flag(self) -> str:  # type: ignore[override]
            calls.append("ran")
            return "decoy"

    booby = _Booby(torch.ones(2), "steer-me")
    assert _namedtuple_field_descriptors_shadowed(booby) is True
    assert undeclared_instance_state(booby, "namedtuple") is True
    assert calls == [], "the shadowing property was executed during the inert proof"
