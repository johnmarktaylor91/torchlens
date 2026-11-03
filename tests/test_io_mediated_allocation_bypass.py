"""Round-6 whole-class immunizer: an admitted callable may not MEDIATE an allocation.

``_is_alloc_constructor_type`` refuses a REDUCE / NEWOBJ that constructs a torch storage,
a ``torch.Tensor`` subclass or a ``numpy.ndarray`` -- but it inspects the TYPE BEING
CONSTRUCTED. That makes it structurally BLIND to any allowlisted callable that constructs
one of those on its behalf: at REDUCE time the stack holds the MEDIATOR, not the
allocating type.

The reported instance (SEC1) was ``numpy._core.multiarray._reconstruct``, allowlisted in
``_SAFE_EXPLICIT_GLOBALS`` as an ordinary function. An 88-byte pickle produced a
2,000,000-element uninitialized array, and it was reachable end-to-end: written into a real
artifact's ``metadata.pkl``, ``tl.load()`` performed the allocation BEFORE any structural
check. At larger N it is roughly 16 GiB of uninitialized heap.

The auditor's closing note -- "likely a CLASS, not a singleton" -- is confirmed here. Every
``_SAFE_EXPLICIT_GLOBALS`` entry and every other surface ``find_class`` admits was assessed;
these are the mediators found, each pinned below:

  1. numpy ``_reconstruct`` (both the modern ``numpy._core.multiarray`` and legacy
     ``numpy.core.multiarray`` spellings) -> ``numpy.ndarray``.
  2. Torch tensor FACTORIES reached as REDUCE targets. ``_safe_getattr`` legitimately
     admits ``getattr(torch._C._VariableFunctionsClass, "empty")`` because an honest
     stream stores that reference as an inert VALUE -- but a following REDUCE CALLS it,
     and ``torch.empty(N)`` / ``empty_strided`` / ``zeros`` / ``rand`` allocate an
     attacker-sized tensor (``empty`` uninitialized) from a 73-byte pickle.
  3. ARGUMENT-BEARING ``torch.nn.Module`` construction. A module class is a torch-owned,
     non-storage DATA type, so the torch branch of ``find_class`` admits it as an inert
     reference -- but CONSTRUCTING it with pickle-supplied sizes runs ``__init__``, which
     allocates attacker-sized PARAMETERS: ``nn.Linear(65536, 65536)`` is ~16 GiB from a
     48-byte pickle. The rule is arg-aware ON PURPOSE -- a real artifact NEWOBJ-constructs
     a zero-argument ``nn.Identity()`` (a ``splice_module`` intervention helper), and an
     arg-blind belt refused that honest artifact.
  4. ``bytes(N)`` / ``bytearray(N)`` given an integer SIZE rather than a buffer -- a
     direct zero-fill allocator the type-keyed belt never covered.

  5. ``torch._utils._rebuild_tensor_v2`` -- the mediator the FIRST pass got wrong. It must
     stay REDUCE-invocable (it is the legitimate tensor path), so its bound is enforced at
     the SOURCE of any storage it could be handed. The legacy source, an embedded
     ``torch.storage._load_from_bytes`` blob (a nested ``torch.load`` whose legacy format
     yielded a RESIZABLE storage; measured: a 354-byte pickle produced a 100 MB tensor,
     ~282,000x), is now REFUSED BY NAME before anything resolves. The only tensor source
     left is the TorchLens inline reduction (``_inline_tensor.rebuild_inline_tensor``),
     which returns a finished tensor over a buffer-backed, non-resizable storage, so
     torch's own ``set_`` refuses to grow it for every present and future consumer while
     an exact-fit view still succeeds.

Assessed and NOT mediators (recorded so a future reader does not re-derive them):
``numpy.dtype`` (describes an itemsize, allocates no buffer), ``_frombuffer`` (bounded by
a buffer already present in the stream), ``collections.defaultdict`` (a stored
``default_factory`` is never invoked by any unpickle opcode), the pure-data builtin
value/collection constructors, and the ``operator`` root (it resolves to an inert
``_DeferredForeignCallable`` that refuses execution, so ``operator.mul(b"A", N)`` cannot
amplify). ``numpy._core.multiarray.scalar`` / ``numpy.core.multiarray.scalar`` (both
spellings) is likewise assessed and NOT a mediator: ``scalar(dtype, bytes)`` rebuilds a
SINGLE numpy scalar bounded by the fixed ``dtype.itemsize`` (at most a few dozen bytes for
the largest numpy scalar types), with no attacker-controlled count/shape argument -- unlike
``_reconstruct`` (an attacker-sized ``shape``) there is no size to amplify.

Every gate here must refuse nothing legitimate: the honest-artifact round-trip at the end
proves real ``.tlspec`` save/load is unaffected.
"""

from __future__ import annotations

import io
import pickle
import struct
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from torchlens._io._safe_unpickle import (
    _NUMPY_RECONSTRUCT_FUNCS,
    SafeBundleUnpickler,
    _alloc_refusal_reason,
    _is_alloc_constructor_type,
    _PurePythonBundleUnpickler,
)
from torchlens._io.bundle import _RenameAwareUnpickler
from torchlens.options import CaptureOptions

# torch renamed ``torch._C._TensorBase`` -> ``torch._C.TensorBase`` (the new name on
# torch>=2.4; the old spelling remains on the 2.1 floor). Feature-detect so the tensor-base
# descriptor probe below works across the whole supported range without parsing versions.
_TENSOR_BASE = getattr(torch._C, "TensorBase", None) or torch._C._TensorBase

_CAP = CaptureOptions(
    replayable=True,
    capture_container_structure=True,
    cache=False,
)

# Large enough that an actual allocation is unmistakable, small enough that a REGRESSION
# (the gate removed) does not wedge the machine running the suite.
_N = 2_000_000


@pytest.fixture(autouse=True)
def _preserve_disclosure_flags() -> Any:
    """Leave the process-wide one-time save-disclosure flags exactly as they were found.

    ``_warn_nonpersistent_buffer_disclosure_once`` and its r6 sibling fire ONCE per process.
    Tests in other files assert them with ``pytest.warns``, so a file that merely happens to
    run earlier and trip the one-shot would starve those assertions -- an order-dependent
    cross-file failure. Snapshot and restore both flags around every test here.
    """

    import torchlens._io.bundle as bundle_module

    before = (
        bundle_module._NONPERSISTENT_DISCLOSURE_WARNED,
        bundle_module._UNATTESTABLE_ACTIVATION_DISCLOSURE_WARNED,
    )
    yield
    (
        bundle_module._NONPERSISTENT_DISCLOSURE_WARNED,
        bundle_module._UNATTESTABLE_ACTIVATION_DISCLOSURE_WARNED,
    ) = before


# --------------------------------------------------------------------------- #
# Hand-assembled pickle helpers (protocol 2 opcodes, no pickler cooperation).
# --------------------------------------------------------------------------- #


def _glob(module: str, name: str) -> bytes:
    """Emit a ``GLOBAL module name`` opcode."""

    return pickle.GLOBAL + (module + "\n" + name + "\n").encode()


def _binint(value: int) -> bytes:
    """Emit a 4-byte ``BININT``."""

    return pickle.BININT + struct.pack("<i", value)


def _torch_factory_pickle(factory: str, count: int) -> bytes:
    """Assemble ``getattr(_VariableFunctionsClass, factory)((count,))``."""

    return (
        pickle.PROTO
        + bytes([2])
        + _glob("builtins", "getattr")
        + _glob("torch._C", "_VariableFunctionsClass")
        + pickle.SHORT_BINUNICODE
        + bytes([len(factory)])
        + factory.encode()
        + pickle.TUPLE2
        + pickle.REDUCE
        + pickle.MARK
        + _binint(count)
        + pickle.TUPLE
        + pickle.TUPLE1
        + pickle.REDUCE
        + pickle.STOP
    )


def _module_ctor_pickle(module: str, name: str, *args: int) -> bytes:
    """Assemble ``<module>.<name>(*args)`` -- an ``nn.Module`` construction."""

    body = b"".join(_binint(arg) for arg in args)
    return (
        pickle.PROTO
        + bytes([2])
        + _glob(module, name)
        + pickle.MARK
        + body
        + pickle.TUPLE
        + pickle.REDUCE
        + pickle.STOP
    )


def _builtin_size_pickle(name: str, count: int) -> bytes:
    """Assemble ``builtins.<name>(count)`` for ``bytes`` / ``bytearray``."""

    return (
        pickle.PROTO
        + bytes([2])
        + _glob("builtins", name)
        + pickle.MARK
        + _binint(count)
        + pickle.TUPLE
        + pickle.REDUCE
        + pickle.STOP
    )


# --------------------------------------------------------------------------- #
# 1. SEC1: numpy ``_reconstruct`` mediating an ndarray allocation.
# --------------------------------------------------------------------------- #


def test_reconstruct_belt_fires_white_box_and_fails_closed() -> None:
    """``load_reduce`` refuses the mediator on the stack, and on a malformed arg tuple."""

    helper = next(iter(_NUMPY_RECONSTRUCT_FUNCS))
    unpickler = _PurePythonBundleUnpickler(io.BytesIO(b""))
    unpickler.stack = [helper, (np.ndarray, (_N,), b"b")]  # type: ignore[attr-defined]
    with pytest.raises(pickle.UnpicklingError, match="mediated allocation"):
        unpickler.load_reduce()
    # A security belt never admits a call it cannot inspect.
    assert _alloc_refusal_reason(helper, ()) is not None
    assert _alloc_refusal_reason(helper, "not-a-tuple") is not None


# --------------------------------------------------------------------------- #
# 2. Torch tensor factories reached as REDUCE targets via ``_safe_getattr``.
# --------------------------------------------------------------------------- #


# shrink-r4: rows kept carry a sole library arc or a campaign kill; the others were proven redundant (slice 1 evidence)
@pytest.mark.parametrize(
    "factory",
    [
        "empty",
    ],
)
def test_torch_tensor_factory_reduce_refused(factory: str) -> None:
    """A REDUCE that CALLS a torch tensor factory is refused (``empty`` is uninitialized)."""

    payload = _torch_factory_pickle(factory, _N)
    assert len(payload) < 128
    with pytest.raises(pickle.UnpicklingError, match="mediated allocation"):
        _RenameAwareUnpickler(io.BytesIO(payload)).load()


# shrink-r4: rows kept carry a sole library arc or a campaign kill; the others were proven redundant (slice 1 evidence)
@pytest.mark.parametrize(
    "method",
    [
        "new_ones",
    ],
)
def test_tensor_constructor_methods_are_all_refused(method: str) -> None:
    """Ownership is read off the descriptor, so the whole ``new_*`` family is covered."""

    from torchlens._io._safe_unpickle import _safe_getattr

    descriptor = _safe_getattr(_TENSOR_BASE, method)
    assert getattr(descriptor, "__module__", None) is None, "the __module__-blind case"
    assert _alloc_refusal_reason(descriptor, (torch.zeros(1), (_N,))) is not None


# --------------------------------------------------------------------------- #
# 2b. The vetted ``_rebuild*`` family may not amplify any storage it can be handed.
#
# ``_rebuild_tensor_v2`` must stay invocable, so its bound is enforced at the SOURCE: the
# legacy embedded-storage source (``torch.storage._load_from_bytes``) is refused by name,
# and the inline tensor reduction yields a non-resizable storage, so torch's own ``set_``
# refuses to grow it.
# --------------------------------------------------------------------------- #


def _embedded_blob(value: Any, *, legacy: bool) -> bytes:
    """Serialize ``value`` as ``torch.storage._load_from_bytes`` input bytes."""

    buffer = io.BytesIO()
    torch.save(value, buffer, _use_new_zipfile_serialization=not legacy)
    return buffer.getvalue()


# --------------------------------------------------------------------------- #
# 3. ``nn.Module`` construction allocates parameters on the belt's behalf.
# --------------------------------------------------------------------------- #


# shrink-r4: rows kept carry a sole library arc or a campaign kill; the others were proven redundant (slice 1 evidence)
@pytest.mark.parametrize(
    ("module", "name", "args"),
    [
        ("torch.nn.modules.sparse", "Embedding", (4096, 4096)),
    ],
)
def test_nn_module_construction_refused(module: str, name: str, args: tuple[int, ...]) -> None:
    """Constructing a SIZED ``nn.Module`` from a tiny pickle is refused."""

    payload = _module_ctor_pickle(module, name, *args)
    assert len(payload) < 128
    with pytest.raises(pickle.UnpicklingError, match="mediated allocation"):
        _RenameAwareUnpickler(io.BytesIO(payload)).load()


def test_nn_module_rule_is_argument_bearing_only() -> None:
    """A ZERO-ARGUMENT module construction stays allowed -- real artifacts do exactly that.

    The arg-aware boundary is load-bearing. An arg-blind belt (module classes folded into
    ``_is_alloc_constructor_type``) refused a real artifact: a ``splice_module`` intervention
    helper round-trips ``torch.nn.modules.linear.Identity()``, which a pickled module
    INSTANCE always reaches as ``cls.__new__(cls)`` with an EMPTY argument tuple plus a
    BUILD. A zero-argument construction also cannot be scaled by the attacker.
    """

    assert not _is_alloc_constructor_type(nn.Linear)
    assert not _is_alloc_constructor_type(nn.Module)
    assert _alloc_refusal_reason(nn.Identity, ()) is None
    assert _alloc_refusal_reason(nn.Identity, (), {}) is None
    assert _alloc_refusal_reason(nn.Linear, (65536, 65536)) is not None
    assert _alloc_refusal_reason(nn.Linear, (), {"in_features": 65536, "out_features": 65536})
    unpickler = SafeBundleUnpickler(io.BytesIO(b""))
    assert unpickler.find_class("torch.nn.modules.linear", "Linear") is nn.Linear


def test_zero_argument_module_newobj_round_trips() -> None:
    """The exact honest shape -- ``NEWOBJ(nn.Identity, ())`` -- still loads."""

    payload = (
        pickle.PROTO
        + bytes([2])
        + _glob("torch.nn.modules.linear", "Identity")
        + pickle.EMPTY_TUPLE
        + pickle.NEWOBJ
        + pickle.STOP
    )
    loaded = _RenameAwareUnpickler(io.BytesIO(payload)).load()
    assert isinstance(loaded, nn.Identity)


# --------------------------------------------------------------------------- #
# 4. ``bytes`` / ``bytearray`` handed an integer SIZE.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ["bytes", "bytearray"])
def test_integer_sized_buffer_allocation_refused(name: str) -> None:
    """``bytes(N)`` / ``bytearray(N)`` allocate N zero bytes from ~30 pickle bytes."""

    payload = _builtin_size_pickle(name, _N)
    assert len(payload) < 64
    with pytest.raises(pickle.UnpicklingError, match="mediated allocation"):
        _RenameAwareUnpickler(io.BytesIO(payload)).load()


def test_buffer_copy_construction_still_allowed() -> None:
    """A ``bytes``/``bytearray`` built from a buffer ALREADY in the stream stays allowed."""

    assert _alloc_refusal_reason(bytes, (b"abc",)) is None
    assert _alloc_refusal_reason(bytearray, (b"abc",)) is None
    assert _alloc_refusal_reason(bytes, (_N,)) is not None
    assert _alloc_refusal_reason(bytearray, (_N,)) is not None


# --------------------------------------------------------------------------- #
# 5. Assessed NON-mediators stay admitted (the belt must not over-block).
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# 6. Honest artifacts are unaffected (the whole point of "purely additive").
# --------------------------------------------------------------------------- #
