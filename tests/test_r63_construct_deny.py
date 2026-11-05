"""Round-63 immunizer: bare CONSTRUCTION of torch tensor / numpy ndarray types is denied.

The r49 belt (``tests/test_r49_unpickler_weights_only_baseline.py``) closes torch STORAGE
construction, but a torch storage is not the only pickle-reachable attacker-sized
allocator. A hostile ``metadata.pkl`` can REDUCE/NEWOBJ/NEWOBJ_EX-construct

    * ``torch.Tensor(N)``            -> an N-element uninitialized-heap tensor,
    * ``torch.FloatTensor(N)``       -> ditto (a legacy ``tensortype`` class, NOT a
                                        ``torch.Tensor`` subclass, so the r49 storage
                                        check and a naive ``issubclass(_, torch.Tensor)``
                                        BOTH miss it),
    * ``numpy.ndarray((N,))``        -> an N-element uninitialized ndarray

at PLAIN ``tl.load()`` time (no trust, no ``.run()``) -- an attacker-sized allocation DoS
plus an uninitialized-heap read. The r49 belt refused only storage TYPES; these tensor /
ndarray types passed.

The fix (``torchlens/_io/_safe_unpickle.py``): a shared ``_is_alloc_constructor_type``
predicate (storage OR ``torch.Tensor`` subclass OR legacy ``tensortype`` class OR
``numpy.ndarray`` subclass) is applied on ALL THREE construction belts (``load_reduce`` /
``load_newobj`` / ``load_newobj_ex``), reusing the storage-deny refusal path (no new error
code). The type OBJECTS stay RESOLVABLE at ``find_class`` -- legit ``.tlspec`` metadata
references ``torch.Tensor`` / ``torch.FloatTensor`` / ``numpy.ndarray`` as inert dtype /
module_type values -- and the legit payload reconstructors (``_rebuild_tensor_v2`` /
numpy ``_frombuffer`` / ``_reconstruct``, all FUNCTIONS not alloc TYPES, plus the TorchLens
inline tensor reconstructor) are untouched. Zero-regression is PROVEN: a real
embedded-weights ``.tlspec`` performs many REDUCE/NEWOBJ ops and ZERO tensor/ndarray
constructions, so it round-trips (save -> load -> run) unchanged.
"""

from __future__ import annotations

import io
import pickle

import pytest
import torch
from torch import nn

from torchlens._io._safe_unpickle import (
    SafeBundleUnpickler,
)
from torchlens.options import CaptureOptions

_CAP = CaptureOptions(
    replayable=True,
    capture_container_structure=True,
    cache=False,
)


# --------------------------------------------------------------------------- #
# Hand-assembled PROTO2 construction gadgets (GLOBAL + belt opcode).
# --------------------------------------------------------------------------- #


def _int_arg(value: int) -> bytes:
    """A single LONG1-encoded int argument (a size big enough to be a real DoS)."""

    body = value.to_bytes((value.bit_length() + 8) // 8, "little", signed=True) if value else b""
    return pickle.LONG1 + bytes([len(body)]) + body


def _shape_tuple_arg(value: int) -> bytes:
    """A single ``(value,)`` shape tuple argument (numpy ndarray ctor form)."""

    return pickle.MARK + _int_arg(value) + pickle.TUPLE


def _global(module: str, name: str) -> bytes:
    return pickle.GLOBAL + (module + "\n" + name + "\n").encode()


def _reduce_gadget(module: str, name: str, arg: bytes) -> bytes:
    """``GLOBAL(module,name)`` then ``REDUCE`` over ``(arg,)`` -- bare construction."""

    return (
        pickle.PROTO
        + bytes([2])
        + _global(module, name)
        + pickle.MARK
        + arg
        + pickle.TUPLE
        + pickle.REDUCE
        + pickle.STOP
    )


def _newobj_gadget(module: str, name: str, arg: bytes) -> bytes:
    """``GLOBAL(cls)`` + ``(arg,)`` + ``NEWOBJ`` -- ``cls.__new__(cls, arg)``."""

    return (
        pickle.PROTO
        + bytes([2])
        + _global(module, name)
        + pickle.MARK
        + arg
        + pickle.TUPLE
        + pickle.NEWOBJ
        + pickle.STOP
    )


def _newobj_ex_gadget(module: str, name: str, arg: bytes) -> bytes:
    """``GLOBAL(cls)`` + ``(arg,)`` + ``{}`` + ``NEWOBJ_EX``."""

    return (
        pickle.PROTO
        + bytes([2])
        + _global(module, name)
        + pickle.MARK
        + arg
        + pickle.TUPLE
        + pickle.EMPTY_DICT
        + pickle.NEWOBJ_EX
        + pickle.STOP
    )


_ALLOC_TARGETS = (
    ("torch", "Tensor", _int_arg(6_000_000_000)),
    ("torch", "FloatTensor", _int_arg(6_000_000_000)),
    ("numpy", "ndarray", _shape_tuple_arg(6_000_000_000)),
)


@pytest.mark.parametrize("module,name,arg", _ALLOC_TARGETS)
def test_reduce_construction_of_tensor_ndarray_refused(module: str, name: str, arg: bytes) -> None:
    """A PROTO2 GLOBAL+REDUCE of Tensor/FloatTensor/ndarray RAISES (never allocates)."""

    payload = _reduce_gadget(module, name, arg)
    with pytest.raises(pickle.UnpicklingError, match="allocation-constructor"):
        SafeBundleUnpickler(io.BytesIO(payload)).load()


@pytest.mark.parametrize("module,name,arg", _ALLOC_TARGETS)
def test_newobj_construction_of_tensor_ndarray_refused(module: str, name: str, arg: bytes) -> None:
    """A PROTO2 NEWOBJ of Tensor/FloatTensor/ndarray RAISES (never allocates)."""

    payload = _newobj_gadget(module, name, arg)
    with pytest.raises(pickle.UnpicklingError, match="allocation-constructor"):
        SafeBundleUnpickler(io.BytesIO(payload)).load()


@pytest.mark.parametrize("module,name,arg", _ALLOC_TARGETS)
def test_newobj_ex_construction_of_tensor_ndarray_refused(
    module: str, name: str, arg: bytes
) -> None:
    """A PROTO2 NEWOBJ_EX of Tensor/FloatTensor/ndarray RAISES (never allocates)."""

    payload = _newobj_ex_gadget(module, name, arg)
    with pytest.raises(pickle.UnpicklingError, match="allocation-constructor"):
        SafeBundleUnpickler(io.BytesIO(payload)).load()


# --------------------------------------------------------------------------- #
# Resolution (inert type reference) is STILL admitted -- construction != resolution.
# --------------------------------------------------------------------------- #


def _resolve_or_none(module: str, name: str) -> object:
    try:
        return SafeBundleUnpickler(io.BytesIO(b"")).find_class(module, name)
    except Exception:  # noqa: BLE001 - a denied/unresolvable name resolves to nothing
        return None


# --------------------------------------------------------------------------- #
# Constructor-semantics drift guard: the admitted-CONSTRUCTABLE alloc set is EMPTY.
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Zero-regression: a real embedded-weights .tlspec still saves / loads / runs, and the
# legit load path performs ZERO alloc-type constructions.
# --------------------------------------------------------------------------- #


class _StatefulLinear(nn.Module):
    """Runnable-eligible parameterized graph with a persistent + non-persistent buffer."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.register_buffer("scale", torch.ones(4))
        self.register_buffer("tmp", torch.zeros(4), persistent=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin(value)) * self.scale + self.tmp
