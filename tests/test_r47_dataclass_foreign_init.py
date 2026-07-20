"""Round-47 secB_1 immunizer -- dataclass foreign-``__init__`` forged-flag recompute.

Sparse runnable reconstruction sets only captured fields and NEVER invokes ``__init__`` (that would
be the SEC1 construction-gadget RCE surface). So a dataclass output whose WINNING ``__init__`` is a
user-authored / foreign constructor can compute a dropped tensor-derived non-field extra -- exactly
like ``__post_init__``. Before r47 only ``__post_init__`` was type-recompute-flagged, so a forged
``lossy_reconstruction=False`` on a custom-``__init__`` dataclass forced a false ``VERIFIED``.

r47 adds ``_dataclass_has_foreign_init``: a dataclass is faithful ONLY when its winning ``__init__``
is the dataclasses-GENERATED one (``__dataclass_params__.init`` true AND its ``co_filename`` equals
the feature-detected generated marker). A user init, ``init=False`` custom init, or undetectable
shape fails CLOSED to lossy. A dataclass that GENERATES its own init over an evil base stays
lossless (the generated init wins).

The fixtures below are defined in THIS real source file (not ``exec``/``<string>``) so a user
``__init__`` reads a real ``co_filename`` -- the exec/``<string>``-compiled custom-init case is the
documented narrow residual (Fork D-sub).
"""

from __future__ import annotations

import dataclasses

import pytest

from torchlens.ir.container import (
    CONTAINER_KIND_CAPABILITIES,
    ContainerSpec,
    _GENERATED_DC_INIT_MARKER,
    _dataclass_has_foreign_init,
    reconstruction_is_lossy_by_type,
)
from torchlens._runnable_execution import _spec_node_reconstruction_lossy

_FIELDS = ("a", "b")


@dataclasses.dataclass
class Plain:
    a: int
    b: int


@dataclasses.dataclass
class WithPost:
    a: int
    b: int

    def __post_init__(self) -> None:
        self.derived = self.a + self.b


@dataclasses.dataclass
class UserInit:
    a: int
    b: int

    def __init__(self, a: int, b: int) -> None:
        self.a = a
        self.b = b
        self.derived = a + b  # dropped, tensor-derived in a real forward


@dataclasses.dataclass(init=False)
class InitFalse:
    a: int
    b: int

    def __init__(self) -> None:
        self.a = 7
        self.b = 8
        self.extra = 1


class _Evil:
    def __init__(self) -> None:
        self.secret = 1


@dataclasses.dataclass
class InheritEvilInit(_Evil):
    a: int
    b: int


def test_generated_marker_detected() -> None:
    """The feature-detected generated-init marker is present on this interpreter (non-vacuous)."""

    assert _GENERATED_DC_INIT_MARKER is not None


@pytest.mark.parametrize(
    ("cls", "want_lossy"),
    [
        (Plain, False),
        (InheritEvilInit, False),  # generated init shadows the evil base -> stays lossless
        (WithPost, True),  # __post_init__ regression pin
        (UserInit, True),  # user-authored init -> foreign
        (InitFalse, True),  # init=False custom init -> foreign
    ],
)
def test_reconstruction_lossy_by_type(cls: type, want_lossy: bool) -> None:
    """The LOAD-time type recompute flags exactly the foreign-init / post-init dataclasses; a
    generated field-mirroring init (incl. one shadowing an evil base) stays VERIFIED-eligible."""

    assert reconstruction_is_lossy_by_type(cls, _FIELDS, "dataclass") is want_lossy


@pytest.mark.parametrize(
    ("cls", "want_foreign"),
    [
        (Plain, False),
        (InheritEvilInit, False),
        (WithPost, False),  # __post_init__ is NOT a foreign __init__ (caught separately)
        (UserInit, True),
        (InitFalse, True),
    ],
)
def test_dataclass_has_foreign_init(cls: type, want_foreign: bool) -> None:
    assert _dataclass_has_foreign_init(cls) is want_foreign


def _dataclass_spec(qualname: str, lossy: bool) -> ContainerSpec:
    return ContainerSpec(
        kind="dataclass",
        fields=_FIELDS,
        type_module=__name__,
        type_qualname=qualname,
        lossy_reconstruction=lossy,
    )


def test_forged_lossy_flag_cannot_bless_user_init_dataclass() -> None:
    """Behavioral: a forged ``lossy_reconstruction=False`` naming a user-``__init__`` dataclass is
    independently recomputed lossy at load -> UNVERIFIABLE, never a false VERIFIED."""

    assert _spec_node_reconstruction_lossy(_dataclass_spec("UserInit", lossy=False)) is True


def test_generated_init_dataclass_stays_verified_eligible() -> None:
    """Over-trigger pin: a genuinely generated-init dataclass with a false persisted flag is NOT
    recomputed lossy (stays VERIFIED-eligible)."""

    assert _spec_node_reconstruction_lossy(_dataclass_spec("Plain", lossy=False)) is False
    assert _spec_node_reconstruction_lossy(_dataclass_spec("InheritEvilInit", lossy=False)) is False


def test_type_recompute_kinds_are_the_expected_set() -> None:
    """Meta pin: the forged-flag type-recompute defense covers exactly ``dataclass`` /
    ``hf_model_output`` / ``namedtuple`` (so a new stateful kind cannot silently escape)."""

    type_recompute_kinds = {
        kind
        for kind, caps in CONTAINER_KIND_CAPABILITIES.items()
        if caps.get("instance_state_rule") == "type_recompute"
    }
    assert type_recompute_kinds == {"dataclass", "hf_model_output", "namedtuple"}


def test_forged_flag_cannot_bless_stateful_namedtuple() -> None:
    """Meta-cell (namedtuple type-recompute): a namedtuple SUBCLASS that can carry per-instance
    state (no ``__slots__ = ()``) with a forged false flag is recomputed lossy."""

    spec = ContainerSpec(
        kind="namedtuple",
        fields=(),
        keys=(),
        type_module=__name__,
        type_qualname="StatefulNamedTuple",
        lossy_reconstruction=False,
    )
    assert _spec_node_reconstruction_lossy(spec) is True


import collections  # noqa: E402  (kept adjacent to the namedtuple fixture it constructs)

_BaseNT = collections.namedtuple("_BaseNT", ["a", "b"])


class StatefulNamedTuple(_BaseNT):  # type: ignore[misc,valid-type]
    """A namedtuple subclass WITHOUT ``__slots__ = ()`` -- it can carry per-instance state, so the
    forged-flag recompute must treat it as lossy."""
