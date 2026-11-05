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

import abc
import collections
import dataclasses
import typing

import pytest

from torchlens._ir.container import (
    ContainerSpec,
    rebuild_container_from_spec,
    reconstruction_is_lossy_by_type,
)
from torchlens._runnable.call_outputs import _spec_node_reconstruction_lossy


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
    def __new__(cls, *args: object, **kwargs: object) -> _EvilNewBase:
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

    def __new__(cls, *args: object, **kwargs: object) -> EvilNewModelOutput:
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


# ---- r51 secB_1 metaclass-``__call__`` fixtures (fourth computed-dropped-state hook) ----------
class _MetaCall(type):
    """A metaclass whose ``__call__`` computes a dropped non-field extra (the r50 secB_1 hole)."""

    def __call__(cls, *args: object, **kwargs: object) -> object:
        obj = super().__call__(*args, **kwargs)
        object.__setattr__(obj, "derived", getattr(obj, "x", 0))  # tensor-derived-stand-in extra
        return obj


@dataclasses.dataclass
class MetaDC(metaclass=_MetaCall):
    """A dataclass with an INERT ``__new__`` / generated ``__init__`` but a metaclass ``__call__``:
    ``__new__``-based reconstruction bypasses the metaclass -> ``derived`` is dropped, yet none of
    the r25/r47/r49 signals fire. Lossy ONLY via the r51 metaclass signal."""

    x: int


class _MetaMOCall(type):
    def __call__(cls, *args: object, **kwargs: object) -> object:
        obj = super().__call__(*args, **kwargs)
        object.__setattr__(obj, "derived", 1)  # dropped non-key extra
        return obj


class MetaModelOutput(collections.OrderedDict, metaclass=_MetaMOCall):
    """An hf_model_output dict-subclass with inert ``dict.__new__`` + trusted OrderedDict init but a
    metaclass ``__call__`` computing a dropped extra -> lossy ONLY via the r51 metaclass signal."""


class _BenignMetaCall(type):
    """A metaclass whose ``__call__`` computes NOTHING -- still fail-closed (conservative residual)."""

    def __call__(cls, *args: object, **kwargs: object) -> object:
        return super().__call__(*args, **kwargs)


@dataclasses.dataclass
class BenignMetaDC(metaclass=_BenignMetaCall):
    """A benign custom metaclass ``__call__`` -> conservatively lossy (accepted fail-closed residual):
    the signal cannot prove a custom ``__call__`` computes nothing without invoking it."""

    x: int


@dataclasses.dataclass
class ABCDataclass(metaclass=abc.ABCMeta):
    """Over-trigger control: ``abc.ABCMeta`` defines no ``__call__`` of its own, so the MRO walk
    falls through to ``type.__call__`` -> foreign=False -> stays VERIFIED-eligible."""

    x: int


class TypingNT(typing.NamedTuple):
    """Over-trigger control: a ``typing.NamedTuple`` has metaclass ``type`` -> foreign=False."""

    a: int


# ---- coupling meta-test corpus ---------------------------------------------------------------
_DATACLASS_CASES = [
    ("PlainDC", PlainDC, ("x",), False),
    ("PostInitDC", PostInitDC, ("x",), True),  # lossy via post_init (predicate may be False)
    ("UserInitDC", UserInitDC, ("x",), True),  # lossy via foreign_init
    ("EvilNewDC", EvilNewDC, ("x",), True),  # r49: lossy via substitution (custom __new__)
    ("MetaDC", MetaDC, ("x",), True),  # r51: lossy via metaclass __call__ (substitutes=False)
    ("ABCDataclass", ABCDataclass, ("x",), False),  # r51 over-trigger control: ABCMeta -> not lossy
]
_HF_CASES = [
    ("InertModelOutput", InertModelOutput, ("k",), False),
    ("EvilNewModelOutput", EvilNewModelOutput, ("k",), True),  # r49: substitution (custom __new__)
    ("MetaModelOutput", MetaModelOutput, ("k",), True),  # r51: lossy via metaclass __call__
]
_NAMEDTUPLE_CASES = [
    ("RealNT", RealNT, ("a", "b"), False),
    ("BareTupleSubclass", BareTupleSubclass, ("a",), True),  # r49: substitution (C __new__)
]


# Kept: sole branch coverage of _ir/_container_type_probes.py, _ir/container.py; its other params stay because the list is not trimmable per id.
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


# Kept: sole branch coverage of _ir/container.py.
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
