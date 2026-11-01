"""``metadata.pkl`` reduces every Enum member by VALUE, whatever the interpreter does.

CPython 3.11.0 through 3.11.3 reduce an ``Enum`` member as ``getattr(cls, name)``
(by name); every other supported release reduces ``cls(value)``. The by-name form
changes the artifact bytes and reaches ``builtins.getattr`` in the restricted loader,
which admits the Enum CLASS as a global but never ``getattr`` on it, so a save on
those interpreters failed its own dry-run reload. The canonical metadata pickler now
emits the by-value form itself. These tests force the by-name ``__reduce_ex__`` on
every interpreter, so the contract is exercised on 3.10 and 3.12 as well as 3.11.3,
and pin that the loader's trust boundary did not move: ``getattr`` on an Enum class,
on a non-Enum class and on a non-member name all stay refused.
"""

from __future__ import annotations

import enum
import importlib
import io
import pickle
import pickletools
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io._canonical_pickle import dump_canonical_metadata
from torchlens._io._safe_unpickle import (
    _SAFE_TORCHLENS_TYPES,
    SafeBundleUnpickler,
    _safe_getattr,
)
from torchlens._trace_state import TraceState

# --------------------------------------------------------------------------- #
# The 3.11.0-3.11.3 reduction, forced onto a class so every interpreter hits it.
# --------------------------------------------------------------------------- #


def _reduce_by_name(self: enum.Enum, proto: int) -> tuple[Any, tuple[Any, ...]]:
    """``Enum.__reduce_ex__`` as CPython 3.11.0 through 3.11.3 shipped it."""

    return getattr, (self.__class__, self._name_)


def _force_by_name(monkeypatch: pytest.MonkeyPatch, *classes: type[enum.Enum]) -> None:
    for cls in classes:
        monkeypatch.setattr(cls, "__reduce_ex__", _reduce_by_name)


def _canonical_bytes(value: Any) -> bytes:
    buffer = io.BytesIO()
    dump_canonical_metadata(value, buffer)
    return buffer.getvalue()


def _global_pairs(payload: bytes) -> set[tuple[str, str]]:
    """Every ``(module, name)`` a GLOBAL / STACK_GLOBAL opcode in ``payload`` names."""

    pairs: set[tuple[str, str]] = set()
    strings: list[str] = []
    for op, arg, _pos in pickletools.genops(payload):
        if op.name in {"SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE"}:
            strings = [*strings, arg][-2:]
        elif op.name == "GLOBAL":
            pairs.add(tuple(arg.split(" ", 1)))  # type: ignore[arg-type]
        elif op.name == "STACK_GLOBAL" and len(strings) == 2:
            pairs.add((strings[0], strings[1]))
    return pairs


# --------------------------------------------------------------------------- #
# Every Enum KIND, through the writer alone (test-local classes are not loadable
# by the restricted loader, so these round trip through plain pickle).
# --------------------------------------------------------------------------- #


class PlainKind(enum.Enum):
    ONE = "one"
    TWO = "two"
    ALSO_TWO = "two"  # alias


class StrKind(str, enum.Enum):
    LEFT = "left"
    RIGHT = "right"


class IntKind(enum.IntEnum):
    LOW = 1
    HIGH = 2


class FlagKind(enum.Flag):
    READ = 1
    WRITE = 2


class IntFlagKind(enum.IntFlag):
    A = 1
    B = 2
    C = 4


_KIND_CASES: tuple[tuple[type[enum.Enum], Any], ...] = (
    (PlainKind, PlainKind.ONE),
    (PlainKind, PlainKind.ALSO_TWO),
    (StrKind, StrKind.RIGHT),
    (IntKind, IntKind.HIGH),
    (FlagKind, FlagKind.READ | FlagKind.WRITE),
    (IntFlagKind, IntFlagKind.A | IntFlagKind.C),
)


@pytest.mark.parametrize(
    ("cls", "member"),
    _KIND_CASES,
    ids=["plain", "plain-alias", "str", "int", "flag-combined", "intflag-combined"],
)
def test_every_enum_kind_pickles_by_value_regardless_of_interpreter(
    monkeypatch: pytest.MonkeyPatch, cls: type[enum.Enum], member: Any
) -> None:
    """By-name and by-value interpreters write identical bytes that load to the member."""

    expected = _canonical_bytes(member)
    _force_by_name(monkeypatch, cls)
    forced = _canonical_bytes(member)

    assert forced == expected
    assert ("builtins", "getattr") not in _global_pairs(forced)
    loaded = pickle.loads(forced)
    assert loaded is cls(member.value) and loaded == member


# --------------------------------------------------------------------------- #
# The Enum classes a real artifact carries: every one on the loader allowlist.
# --------------------------------------------------------------------------- #

_EXPECTED_ADMITTED_ENUMS = frozenset(
    {
        ("torchlens._trace_state", "TraceState"),
        ("torchlens.intervention.types", "Relationship"),
        ("torchlens.intervention.types", "ForkFieldPolicy"),
        ("torchlens._ir.container_registry", "Role"),
        ("torchlens._ir.container_registry", "Phase"),
    }
)


def _admitted_enum_classes() -> dict[tuple[str, str], type[enum.Enum]]:
    found: dict[tuple[str, str], type[enum.Enum]] = {}
    for module, qualname in _SAFE_TORCHLENS_TYPES:
        obj: Any = importlib.import_module(module)
        for part in qualname.split("."):
            obj = getattr(obj, part)
        if isinstance(obj, type) and issubclass(obj, enum.Enum):
            found[(module, qualname)] = obj
    return found


def test_admitted_enum_inventory_is_the_five_metadata_enums() -> None:
    """The loader admits exactly these Enum classes; a new one joins this test knowingly."""

    assert frozenset(_admitted_enum_classes()) == _EXPECTED_ADMITTED_ENUMS


def test_every_admitted_enum_member_round_trips_through_the_restricted_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each member of each admitted Enum loads back by identity under the by-name reduce."""

    classes = _admitted_enum_classes()
    _force_by_name(monkeypatch, *classes.values())
    for cls in classes.values():
        for member in cls.__members__.values():
            payload = _canonical_bytes(member)
            assert ("builtins", "getattr") not in _global_pairs(payload)
            assert SafeBundleUnpickler(io.BytesIO(payload)).load() is member


# --------------------------------------------------------------------------- #
# A whole bundle, saved under the 3.11.0-3.11.3 reduction, loads back.
# --------------------------------------------------------------------------- #


def _tiny_trace() -> Any:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    return tl.trace(model, torch.randn(2, 4))


@pytest.mark.smoke
def test_trace_saved_under_by_name_enum_reduce_round_trips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``Trace.save`` + ``tl.load`` succeed when the interpreter reduces Enums by name."""

    trace = _tiny_trace()
    plain = tmp_path / "plain.tlspec"
    trace.save(plain)
    _force_by_name(monkeypatch, *_admitted_enum_classes().values())
    forced = tmp_path / "forced.tlspec"
    trace.save(forced)

    assert (forced / "metadata.pkl").read_bytes() == (plain / "metadata.pkl").read_bytes()
    assert ("builtins", "getattr") not in _global_pairs((forced / "metadata.pkl").read_bytes())
    loaded = tl.load(forced)
    assert isinstance(loaded, tl.Trace)


# --------------------------------------------------------------------------- #
# The trust boundary did not move: no getattr admission for Enum classes.
# --------------------------------------------------------------------------- #


def _getattr_stream(module: str, name: str, attr: str) -> bytes:
    """``getattr(<module>.<name>, attr)`` as a REDUCE: the 3.11.0-3.11.3 member stream."""

    proto = pickle.PROTO + bytes([2])
    op = b"c" + b"builtins\ngetattr\n"
    op += b"c" + f"{module}\n{name}\n".encode()
    op += b"\x8c" + bytes([len(attr)]) + attr.encode()
    op += b"\x86R."  # TUPLE2 ; REDUCE ; STOP
    return proto + op


@pytest.mark.parametrize(
    ("module", "name", "attr"),
    [
        ("torchlens._trace_state", "TraceState", "PRISTINE"),  # Enum class, member name
        ("torchlens.intervention.types", "Relationship", "not_a_member"),  # non-member name
        ("torchlens._ir.container_registry", "Phase", "__dict__"),  # dunder off an Enum class
        ("torchlens._io", "BlobRef", "__init__"),  # admitted non-Enum class
    ],
    ids=["enum-member", "enum-non-member", "enum-dunder", "non-enum-class"],
)
def test_getattr_reduce_stays_refused(module: str, name: str, attr: str) -> None:
    """A by-name Enum stream (or any getattr off an admitted class) is still refused."""

    with pytest.raises(pickle.UnpicklingError, match="Blocked getattr"):
        SafeBundleUnpickler(io.BytesIO(_getattr_stream(module, name, attr))).load()


def test_safe_getattr_refuses_enum_classes_directly() -> None:
    with pytest.raises(pickle.UnpicklingError, match="Blocked getattr"):
        _safe_getattr(TraceState, "PRISTINE")
    with pytest.raises(pickle.UnpicklingError, match="Blocked getattr"):
        _safe_getattr(TraceState, "not_a_member")
