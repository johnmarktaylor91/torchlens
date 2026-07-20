"""Round-49 secB_1 immunizer -- gate/reconstruction coupled through ONE substitution predicate.

The load-time forged-flag lossy gate and the sparse reconstruction path used DIFFERENT faithfulness
criteria: reconstruction substitutes a PLAIN ``dict`` / ``tuple`` (dropping the recorded type + any
``__new__``-computed state) for a type whose ``__new__`` is not an inert allocator (dataclass / hf)
or whose namedtuple ``__new__`` is not a Python ``FunctionType``, in cases the gate never inspected
(it checked instance-dict / descriptors / init but never ``__new__`` / generated-namedtuple). A
forged ``lossy_reconstruction=False`` naming such a type -> false ``VERIFIED`` on a type-substituted
output (the r48 secB_1 finding).

r49 introduces ONE shared predicate ``_reconstruction_would_substitute_plain`` consulted by BOTH
``_rebuild_container_from_spec`` (reconstruction) AND ``reconstruction_is_lossy_by_type`` /
``_spec_node_reconstruction_lossy`` (gate). Coupling is enforced by shared code, so a future
gate/reconstruction divergence is a failing coverage meta-test rather than a silent false
``VERIFIED``. This immunizer pins the coupling over every reconstruction kind x {custom ``__init__``,
``__post_init__``, custom ``__new__``, generated, C-``__new__`` tuple subclass} and the over-trigger
controls (plain dataclass / real namedtuple / inert ModelOutput stay VERIFIED-eligible).
"""

from __future__ import annotations

import collections
import dataclasses

import pytest

from torchlens.ir.container import (
    ContainerSpec,
    _reconstruction_would_substitute_plain,
    reconstruction_is_lossy_by_type,
    rebuild_container_from_spec,
)
from torchlens._runnable_execution import _spec_node_reconstruction_lossy


# ---- dataclass fixtures ----------------------------------------------------------------------
@dataclasses.dataclass
class PlainDC:
    x: int


@dataclasses.dataclass
class PostInitDC:
    x: int

    def __post_init__(self) -> None:
        self.derived = self.x + 1


@dataclasses.dataclass
class UserInitDC:
    x: int

    def __init__(self, x: int) -> None:
        self.x = x
        self.derived = x + 1


class _EvilNewBase:
    def __new__(cls, *args: object, **kwargs: object) -> "_EvilNewBase":
        obj = super().__new__(cls)
        object.__setattr__(obj, "secret", 1)  # __new__-computed state, dropped on substitution
        return obj


@dataclasses.dataclass
class EvilNewDC(_EvilNewBase):
    x: int


# ---- hf_model_output fixtures (name-recognized; no transformers dependency) -------------------
class InertModelOutput(collections.OrderedDict):
    """A dict-subclass ModelOutput with the inert ``dict.__new__`` -> reconstruction faithful."""


class EvilNewModelOutput(collections.OrderedDict):
    """A ModelOutput dict-subclass that OVERRIDES ``__new__`` -> reconstruction substitutes plain."""

    def __new__(cls, *args: object, **kwargs: object) -> "EvilNewModelOutput":
        obj = super().__new__(cls, *args, **kwargs)
        obj.secret = 1  # dropped on substitution
        return obj


# ---- namedtuple fixtures ---------------------------------------------------------------------
RealNT = collections.namedtuple("RealNT", ["a", "b"])


class BareTupleSubclass(tuple):
    """A tuple subclass carrying ``_fields`` + ``__slots__=()`` but a C ``tuple.__new__`` (NOT a
    Python ``FunctionType``) -> not a generated namedtuple -> reconstruction substitutes a plain
    ``tuple`` (the exact r48 secB_1 namedtuple hole)."""

    _fields = ("a",)
    __slots__ = ()


# ---- coupling meta-test corpus ---------------------------------------------------------------
_DATACLASS_CASES = [
    ("PlainDC", PlainDC, ("x",), False),
    ("PostInitDC", PostInitDC, ("x",), True),  # lossy via post_init (predicate may be False)
    ("UserInitDC", UserInitDC, ("x",), True),  # lossy via foreign_init
    ("EvilNewDC", EvilNewDC, ("x",), True),  # r49: lossy via substitution (custom __new__)
]
_HF_CASES = [
    ("InertModelOutput", InertModelOutput, ("k",), False),
    ("EvilNewModelOutput", EvilNewModelOutput, ("k",), True),  # r49: substitution (custom __new__)
]
_NAMEDTUPLE_CASES = [
    ("RealNT", RealNT, ("a", "b"), False),
    ("BareTupleSubclass", BareTupleSubclass, ("a",), True),  # r49: substitution (C __new__)
]


@pytest.mark.parametrize(
    ("kind", "name", "cls", "names"),
    [("dataclass", n, c, f) for n, c, f, _ in _DATACLASS_CASES]
    + [("hf_model_output", n, c, f) for n, c, f, _ in _HF_CASES]
    + [("namedtuple", n, c, f) for n, c, f, _ in _NAMEDTUPLE_CASES],
)
def test_substitution_implies_lossy_gate(kind: str, name: str, cls: type, names: tuple) -> None:
    """COUPLING (anti-divergence): whenever ``_reconstruction_would_substitute_plain`` is True, the
    load-time forged-flag gate MUST also be lossy -- gate == reconstruction-substitutes on every
    reconstruction kind, enforced by the shared predicate."""

    spec = _make_spec(kind, name, names)
    substitutes = _reconstruction_would_substitute_plain(cls, kind, names, spec)
    if kind == "namedtuple":
        gate_lossy = _spec_node_reconstruction_lossy(spec)
    else:
        gate_lossy = reconstruction_is_lossy_by_type(cls, names, kind, spec)
    assert (not substitutes) or gate_lossy, (
        f"{kind}/{name}: reconstruction substitutes plain but the gate says not-lossy "
        "(gate/reconstruction divergence -> silent false VERIFIED)"
    )


@pytest.mark.parametrize(
    ("kind", "name", "cls", "names", "want_lossy"),
    [("dataclass", n, c, f, w) for n, c, f, w in _DATACLASS_CASES]
    + [("hf_model_output", n, c, f, w) for n, c, f, w in _HF_CASES]
    + [("namedtuple", n, c, f, w) for n, c, f, w in _NAMEDTUPLE_CASES],
)
def test_forged_flag_recompute(
    kind: str, name: str, cls: type, names: tuple, want_lossy: bool
) -> None:
    """A forged ``lossy_reconstruction=False`` is independently recomputed to the expected verdict:
    the ``__new__`` / generated-namedtuple hole types are now lossy; plain dataclass / real
    namedtuple / inert ModelOutput stay VERIFIED-eligible (no over-trigger)."""

    spec = _make_spec(kind, name, names)
    if kind == "namedtuple":
        assert _spec_node_reconstruction_lossy(spec) is want_lossy
    else:
        assert reconstruction_is_lossy_by_type(cls, names, kind, spec) is want_lossy


# ---- predicate over-trigger pins (pure, no resolution) ---------------------------------------
def test_predicate_over_trigger_safe() -> None:
    """Over-trigger pin: a plain dataclass, a real ``collections.namedtuple``, and an inert
    dict-``__new__`` ModelOutput do NOT substitute (stay VERIFIED-eligible)."""

    assert _reconstruction_would_substitute_plain(PlainDC, "dataclass", ("x",), None) is False
    assert _reconstruction_would_substitute_plain(RealNT, "namedtuple", ("a", "b"), None) is False
    assert (
        _reconstruction_would_substitute_plain(InertModelOutput, "hf_model_output", ("k",), None)
        is False
    )
    # and the hole types DO substitute
    assert _reconstruction_would_substitute_plain(EvilNewDC, "dataclass", ("x",), None) is True
    assert (
        _reconstruction_would_substitute_plain(BareTupleSubclass, "namedtuple", ("a",), None)
        is True
    )


# ---- reconstruction actually substitutes (proves the coupling closes a REAL hole) ------------
def test_reconstruction_substitutes_plain_for_custom_new_dataclass() -> None:
    """Behavioral: ``_rebuild_container_from_spec`` returns a PLAIN ``dict`` (not ``EvilNewDC``) for
    a custom-``__new__`` dataclass -- the substitution the coupled gate now flags as lossy."""

    spec = _make_spec("dataclass", "EvilNewDC", ("x",))
    rebuilt = rebuild_container_from_spec(spec, [7])
    assert type(rebuilt) is dict  # substitution: recorded type + ``__new__`` 'secret' dropped

    plain_spec = _make_spec("dataclass", "PlainDC", ("x",))
    plain_rebuilt = rebuild_container_from_spec(plain_spec, [7])
    assert isinstance(plain_rebuilt, PlainDC)  # over-trigger control: rebuilds as its own type


def test_reconstruction_substitutes_plain_tuple_for_c_new_namedtuple() -> None:
    """Behavioral: a C-``__new__`` tuple subclass rebuilds as a PLAIN ``tuple`` (substitution);
    a real ``collections.namedtuple`` rebuilds as its own type (over-trigger control)."""

    spec = _make_spec("namedtuple", "BareTupleSubclass", ("a",))
    rebuilt = rebuild_container_from_spec(spec, [5])
    assert type(rebuilt) is tuple

    real_spec = _make_spec("namedtuple", "RealNT", ("a", "b"))
    real_rebuilt = rebuild_container_from_spec(real_spec, [1, 2])
    assert isinstance(real_rebuilt, RealNT)


def _make_spec(kind: str, qualname: str, names: tuple) -> ContainerSpec:
    """Build a forged (``lossy_reconstruction=False``) ContainerSpec resolving to a local fixture."""

    if kind == "hf_model_output":
        return ContainerSpec(
            kind=kind,
            keys=names,
            type_module=__name__,
            type_qualname=qualname,
            lossy_reconstruction=False,
        )
    return ContainerSpec(
        kind=kind,
        fields=names,
        keys=names if kind == "namedtuple" else (),
        type_module=__name__,
        type_qualname=qualname,
        lossy_reconstruction=False,
    )
