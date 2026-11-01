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
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io._canonical_pickle import dump_canonical_metadata
from torchlens._io._inline_tensor import rebuild_inline_tensor
from torchlens._io._safe_unpickle import (
    _NUMPY_RECONSTRUCT_FUNCS,
    _SAFE_EXPLICIT_GLOBALS,
    SafeBundleUnpickler,
    _alloc_refusal_reason,
    _is_alloc_constructor_type,
)
from torchlens._io.bundle import _RenameAwareUnpickler
from torchlens.options import CaptureOptions

# torch renamed ``torch._C._TensorBase`` -> ``torch._C.TensorBase`` (the new name on
# torch>=2.4; the old spelling remains on the 2.1 floor). Feature-detect so the tensor-base
# descriptor probe below works across the whole supported range without parsing versions.
_TENSOR_BASE = getattr(torch._C, "TensorBase", None) or torch._C._TensorBase

_CAP = CaptureOptions(
    intervention_ready=True,
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


def _inline_tensor_reduce(tensor: torch.Tensor) -> bytes:
    """``rebuild_inline_tensor(bytes, dtype, shape, False, "cpu")`` as REDUCE opcodes."""

    data = tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    dtype_name = str(tensor.dtype).encode()
    shape = b"".join(_binint(dim) for dim in tensor.shape)
    return (
        _glob("torchlens._io._inline_tensor", "rebuild_inline_tensor")
        + pickle.MARK
        + pickle.BINBYTES
        + struct.pack("<I", len(data))
        + data
        + pickle.SHORT_BINUNICODE
        + bytes([len(dtype_name)])
        + dtype_name
        + pickle.MARK
        + shape
        + pickle.TUPLE
        + pickle.NEWFALSE
        + pickle.SHORT_BINUNICODE
        + bytes([3])
        + b"cpu"
        + pickle.TUPLE
        + pickle.REDUCE
    )


def _binint(value: int) -> bytes:
    """Emit a 4-byte ``BININT``."""

    return pickle.BININT + struct.pack("<i", value)


def _reconstruct_pickle(module: str, count: int) -> bytes:
    """Assemble ``_reconstruct(numpy.ndarray, (count,), b"b")`` -- the SEC1 gadget."""

    return (
        pickle.PROTO
        + bytes([2])
        + _glob(module, "_reconstruct")
        + _glob("numpy", "ndarray")
        + pickle.MARK
        + _binint(count)
        + pickle.TUPLE
        + pickle.SHORT_BINBYTES
        + bytes([1])
        + b"b"
        + pickle.TUPLE3
        + pickle.REDUCE
        + pickle.STOP
    )


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


def test_numpy_reconstruct_helpers_resolve_for_the_guard() -> None:
    """The identity set the guard matches on is non-empty and covers the allowlist."""

    assert _NUMPY_RECONSTRUCT_FUNCS, "numpy _reconstruct never resolved for the belt"
    allowlisted = {module for module, name in _SAFE_EXPLICIT_GLOBALS if name == "_reconstruct"}
    assert allowlisted == {"numpy.core.multiarray", "numpy._core.multiarray"}


@pytest.mark.smoke_cells(
    "test_reconstruct_mediated_ndarray_allocation_refused[numpy.core.multiarray]"
)
@pytest.mark.parametrize("module", ["numpy._core.multiarray", "numpy.core.multiarray"])
def test_reconstruct_mediated_ndarray_allocation_refused(module: str) -> None:
    """A tiny ``_reconstruct`` REDUCE no longer allocates an attacker-sized array."""

    payload = _reconstruct_pickle(module, _N)
    assert len(payload) < 128, "the gadget must stay tiny -- that is the amplification"
    with pytest.raises(pickle.UnpicklingError, match="mediated allocation"):
        _RenameAwareUnpickler(io.BytesIO(payload)).load()
    with pytest.raises(pickle.UnpicklingError, match="mediated allocation"):
        SafeBundleUnpickler(io.BytesIO(payload)).load()


def test_reconstruct_belt_fires_white_box_and_fails_closed() -> None:
    """``load_reduce`` refuses the mediator on the stack, and on a malformed arg tuple."""

    helper = next(iter(_NUMPY_RECONSTRUCT_FUNCS))
    unpickler = SafeBundleUnpickler(io.BytesIO(b""))
    unpickler.stack = [helper, (np.ndarray, (_N,), b"b")]  # type: ignore[attr-defined]
    with pytest.raises(pickle.UnpicklingError, match="mediated allocation"):
        unpickler.load_reduce()
    # A security belt never admits a call it cannot inspect.
    assert _alloc_refusal_reason(helper, ()) is not None
    assert _alloc_refusal_reason(helper, "not-a-tuple") is not None


def test_reconstruct_gadget_refused_end_to_end_through_tl_load(tmp_path: Path) -> None:
    """The gadget planted in a REAL artifact's metadata.pkl is refused by ``tl.load``."""

    path = tmp_path / "hostile.tlspec"
    tl.trace(nn.Linear(3, 2), torch.ones(2, 3), capture=_CAP).save(path)
    (path / "metadata.pkl").write_bytes(_reconstruct_pickle("numpy.core.multiarray", _N))

    with pytest.raises(Exception) as caught:
        tl.load(path)
    assert "mediated allocation" in str(caught.value) or "mediated allocation" in str(
        caught.value.__cause__
    )


# --------------------------------------------------------------------------- #
# 2. Torch tensor factories reached as REDUCE targets via ``_safe_getattr``.
# --------------------------------------------------------------------------- #


@pytest.mark.smoke_cells("test_torch_tensor_factory_reduce_refused[arange]")
@pytest.mark.parametrize(
    "factory", ["empty", "empty_strided", "zeros", "ones", "rand", "randn", "arange"]
)
def test_torch_tensor_factory_reduce_refused(factory: str) -> None:
    """A REDUCE that CALLS a torch tensor factory is refused (``empty`` is uninitialized)."""

    payload = _torch_factory_pickle(factory, _N)
    assert len(payload) < 128
    with pytest.raises(pickle.UnpicklingError, match="mediated allocation"):
        _RenameAwareUnpickler(io.BytesIO(payload)).load()


def test_torch_reference_still_resolves_as_an_inert_value() -> None:
    """Resolution of a torch callable REFERENCE stays allowed -- only CALLING it is refused.

    This is the load-tolerate / execute-deny split the module applies everywhere: honest
    metadata stores function-registry references, so denying the resolution would break
    real artifacts.
    """

    unpickler = SafeBundleUnpickler(io.BytesIO(b""))
    resolved = unpickler.find_class("builtins", "getattr")
    assert callable(resolved)
    assert resolved(torch._C._VariableFunctionsClass, "empty") is not None


@pytest.mark.smoke
def test_tensor_constructor_method_descriptor_reduce_refused() -> None:
    """A tensor CONSTRUCTOR METHOD reached off ``TensorBase`` cannot amplify a small tensor.

    The second half of mediator (2), and the one that survived the first fix: descriptors
    fetched off ``torch._C.TensorBase`` carry ``__module__ is None``, so a
    ``__module__``-only ownership test misses them. The full chain is REACHABLE --
    the inline tensor reduction yields a genuine (proportional, tiny) tensor from stream
    bytes, and ``getattr(TensorBase, "new_empty")(that_tensor, (N,))`` then returns an
    arbitrarily large UNINITIALIZED tensor. Measured before the fix: a 1,681-byte pickle
    produced a 16,000,000-byte tensor. Ownership is now read from ``__objclass__`` /
    ``__self__`` as well as ``__module__``.
    """

    descriptor = (
        _glob("builtins", "getattr")
        + _glob("torch._C", _TENSOR_BASE.__name__)
        + pickle.SHORT_BINUNICODE
        + bytes([9])
        + b"new_empty"
        + pickle.TUPLE2
        + pickle.REDUCE
    )
    small_tensor = _inline_tensor_reduce(torch.arange(2, dtype=torch.float32))
    payload = (
        pickle.PROTO
        + bytes([2])
        + descriptor
        + pickle.MARK
        + small_tensor
        + pickle.MARK
        + _binint(_N)
        + pickle.TUPLE
        + pickle.TUPLE
        + pickle.REDUCE
        + pickle.STOP
    )

    with pytest.raises(pickle.UnpicklingError, match="mediated allocation"):
        _RenameAwareUnpickler(io.BytesIO(payload)).load()


@pytest.mark.parametrize("method", ["new_empty", "new_zeros", "new_ones", "new_full"])
def test_tensor_constructor_methods_are_all_refused(method: str) -> None:
    """Ownership is read off the descriptor, so the whole ``new_*`` family is covered."""

    from torchlens._io._safe_unpickle import _safe_getattr

    descriptor = _safe_getattr(_TENSOR_BASE, method)
    assert getattr(descriptor, "__module__", None) is None, "the __module__-blind case"
    assert _alloc_refusal_reason(descriptor, (torch.zeros(1), (_N,))) is not None


@pytest.mark.smoke
def test_ownership_probe_decides_instead_of_crashing_on_a_hostile_owner() -> None:
    """A REDUCE target whose owner is UNHASHABLE must be DECIDED, not raise out of the belt.

    The ownership probe reads ``__objclass__`` / ``__self__`` off an attacker-reachable
    callable. Testing those with ``in`` against a frozenset raises ``TypeError`` for an
    unhashable owner (an ``ndarray``) or mis-decides for a broadcasting ``__eq__`` -- the
    exact trap this module's own numpy/bytes branch calls out. Identity comparison decides.
    """

    class _UnhashableSelf:
        __self__ = np.zeros(3)

        def __call__(self) -> None: ...

    class _UnhashableObjclass:
        __objclass__ = np.zeros(3)

        def __call__(self) -> None: ...

    assert _alloc_refusal_reason(_UnhashableSelf(), ()) is None
    assert _alloc_refusal_reason(_UnhashableObjclass(), ()) is None


def test_vetted_rebuild_reconstructors_are_still_invocable() -> None:
    """The ``torch._utils._rebuild*`` family stays REDUCE-invocable (it is the legit path)."""

    assert _alloc_refusal_reason(torch._utils._rebuild_tensor_v2, ((), 0, (1,), (1,))) is None
    assert _alloc_refusal_reason(torch._utils._rebuild_parameter, ((), False, {})) is None
    assert _alloc_refusal_reason(torch.empty, ((_N,),)) is not None


# --------------------------------------------------------------------------- #
# 2b. The vetted ``_rebuild*`` family may not amplify any storage it can be handed.
#
# ``_rebuild_tensor_v2`` must stay invocable, so its bound is enforced at the SOURCE: the
# legacy embedded-storage source (``torch.storage._load_from_bytes``) is refused by name,
# and the inline tensor reduction yields a non-resizable storage, so torch's own ``set_``
# refuses to grow it.
# --------------------------------------------------------------------------- #


def _uint8_typed_storage() -> Any:
    """Build a 4-byte typed storage without the deprecated ``Tensor.storage()`` accessor."""

    return torch.storage.TypedStorage(
        wrap_storage=torch.arange(4, dtype=torch.uint8).untyped_storage(),
        dtype=torch.uint8,
        _internal=True,
    )


def _typed_view(tensor: torch.Tensor) -> Any:
    """Wrap a tensor's (frozen) untyped storage as the typed storage ``_rebuild_tensor_v2`` takes."""

    return torch.storage.TypedStorage(
        wrap_storage=tensor.untyped_storage(), dtype=tensor.dtype, _internal=True
    )


def _embedded_blob(value: Any, *, legacy: bool) -> bytes:
    """Serialize ``value`` as ``torch.storage._load_from_bytes`` input bytes."""

    buffer = io.BytesIO()
    torch.save(value, buffer, _use_new_zipfile_serialization=not legacy)
    return buffer.getvalue()


def _rebuild_amplification_pickle(blob: bytes, count: int) -> bytes:
    """Assemble ``_rebuild_tensor_v2(_load_from_bytes(blob), 0, (count,), (1,), False, None)``."""

    return (
        pickle.PROTO
        + bytes([2])
        + _glob("torch._utils", "_rebuild_tensor_v2")
        + pickle.MARK
        + _glob("torch.storage", "_load_from_bytes")
        + pickle.MARK
        + pickle.BINBYTES
        + struct.pack("<I", len(blob))
        + blob
        + pickle.TUPLE
        + pickle.REDUCE
        + _binint(0)
        + pickle.MARK
        + _binint(count)
        + pickle.TUPLE
        + pickle.MARK
        + _binint(1)
        + pickle.TUPLE
        + pickle.NEWFALSE
        + pickle.NONE
        + pickle.TUPLE
        + pickle.REDUCE
        + pickle.STOP
    )


def test_inline_tensor_storage_is_never_resizable() -> None:
    """The inline reduction yields a buffer-backed storage torch cannot grow."""

    for dtype in (torch.uint8, torch.float32):
        itemsize = torch.empty((), dtype=dtype).element_size()
        loaded = rebuild_inline_tensor(bytes(4 * itemsize), str(dtype), (4,))
        assert not loaded.untyped_storage().resizable(), f"{dtype} stayed growable"


def test_rebuild_tensor_v2_cannot_amplify_an_inline_tensor() -> None:
    """A vetted ``_rebuild_tensor_v2`` may not GROW the storage behind an inline tensor.

    ``_N`` here is small enough that a REGRESSION does not wedge the suite.
    """

    storage = _typed_view(rebuild_inline_tensor(bytes(range(4)), "torch.uint8", (4,)))
    with pytest.raises(RuntimeError, match="(?i)resiz"):
        torch._utils._rebuild_tensor_v2(storage, 0, (_N,), (1,), False, None)
    exact_fit = torch._utils._rebuild_tensor_v2(storage, 0, (4,), (1,), False, None)
    assert exact_fit.tolist() == [0, 1, 2, 3]


@pytest.mark.parametrize("legacy", [False, True])
def test_legacy_embedded_storage_refused_by_name(
    legacy: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The former amplification carrier is refused at ``_load_from_bytes`` before torch.load."""

    def _trip(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("torch.load was entered with file bytes")

    monkeypatch.setattr(torch, "load", _trip)
    payload = _rebuild_amplification_pickle(
        _embedded_blob(_uint8_typed_storage(), legacy=legacy), _N
    )
    with pytest.raises(pickle.UnpicklingError, match=r"torch\.storage\._load_from_bytes"):
        SafeBundleUnpickler(io.BytesIO(payload)).load()


def test_rebuild_amplification_refused_end_to_end_through_tl_load(tmp_path: Path) -> None:
    """The storage-amplification gadget planted in a real artifact is refused by ``tl.load``."""

    path = tmp_path / "hostile_rebuild.tlspec"
    tl.trace(nn.Linear(3, 2), torch.ones(2, 3), capture=_CAP).save(path)
    (path / "metadata.pkl").write_bytes(
        _rebuild_amplification_pickle(_embedded_blob(_uint8_typed_storage(), legacy=True), _N)
    )
    with pytest.raises(Exception) as caught:
        tl.load(path)
    assert "_load_from_bytes" in str(caught.value) + str(caught.value.__cause__)


@pytest.mark.smoke
def test_inline_tensor_round_trips_exactly() -> None:
    """The writer's inline reduction preserves value, shape, dtype and ``requires_grad``.

    The loaded tensor is a contiguous copy (strides are not preserved); an EXACT-FIT
    ``_rebuild_tensor_v2`` view over its storage still succeeds.
    """

    for original in (
        torch.arange(6, dtype=torch.float32).reshape(2, 3),
        torch.arange(12, dtype=torch.float32).reshape(3, 4)[:, 1:3],
        torch.ones(3, requires_grad=True),
    ):
        buffer = io.BytesIO()
        dump_canonical_metadata({"t": original}, buffer)
        assert b"_load_from_bytes" not in buffer.getvalue()
        loaded = SafeBundleUnpickler(io.BytesIO(buffer.getvalue())).load()["t"]
        assert torch.equal(loaded.detach(), original.detach())
        assert loaded.shape == original.shape
        assert loaded.dtype == original.dtype
        assert loaded.requires_grad == original.requires_grad
    storage = _typed_view(rebuild_inline_tensor(bytes(range(4)), "torch.uint8", (4,)))
    exact_fit = torch._utils._rebuild_tensor_v2(storage, 0, (4,), (1,), False, None)
    assert exact_fit.tolist() == [0, 1, 2, 3]


# --------------------------------------------------------------------------- #
# 3. ``nn.Module`` construction allocates parameters on the belt's behalf.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("module", "name", "args"),
    [
        ("torch.nn.modules.linear", "Linear", (4096, 4096)),
        ("torch.nn.modules.sparse", "Embedding", (4096, 4096)),
    ],
)
def test_nn_module_construction_refused(module: str, name: str, args: tuple[int, ...]) -> None:
    """Constructing a SIZED ``nn.Module`` from a tiny pickle is refused."""

    payload = _module_ctor_pickle(module, name, *args)
    assert len(payload) < 128
    with pytest.raises(pickle.UnpicklingError, match="mediated allocation"):
        _RenameAwareUnpickler(io.BytesIO(payload)).load()


@pytest.mark.smoke
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


@pytest.mark.smoke_cells("test_integer_sized_buffer_allocation_refused[bytes]")
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


def test_assessed_non_mediators_are_not_refused() -> None:
    """Entries whose audit verdict is "cannot mediate" keep loading."""

    import collections

    for func, args in (
        (np.dtype, ("f8",)),
        (collections.OrderedDict, ()),
        (collections.defaultdict, (list,)),
        (list, ((1, 2),)),
        (tuple, ((1, 2),)),
        (set, ((1, 2),)),
        (frozenset, ((1, 2),)),
        (dict, ()),
        (complex, (1, 2)),
        (slice, (1, 2)),
        (int, ("7",)),
        (float, ("1.5",)),
        (str, (b"a",)),
        (bool, (1,)),
        (object, ()),
    ):
        assert _alloc_refusal_reason(func, args) is None, f"{func!r} wrongly refused"


def test_safe_explicit_globals_surface_is_pinned() -> None:
    """Pin the allowlist so a NEW entry forces an explicit mediation verdict.

    ``_SAFE_EXPLICIT_GLOBALS`` is the surface this whole class lives on. Every entry below
    carries an audited verdict in the module docstring; adding one without assessing it is
    exactly how SEC1 shipped, so widening the set must break this test on purpose.
    """

    assert (
        frozenset(
            {
                ("collections", "OrderedDict"),
                ("collections", "defaultdict"),
                ("collections", "Counter"),
                ("builtins", "object"),
                ("builtins", "list"),
                ("builtins", "set"),
                ("builtins", "frozenset"),
                ("builtins", "dict"),
                ("builtins", "tuple"),
                ("builtins", "bytearray"),
                ("builtins", "bytes"),
                ("builtins", "complex"),
                ("builtins", "int"),
                ("builtins", "float"),
                ("builtins", "str"),
                ("builtins", "bool"),
                ("builtins", "slice"),
                ("builtins", "Ellipsis"),
                ("torch._C", "TensorBase"),
                ("torch._C", "_TensorBase"),
                ("torch._C", "_VariableFunctionsClass"),
                ("numpy", "dtype"),
                ("numpy", "ndarray"),
                ("numpy.core.numeric", "_frombuffer"),
                ("numpy._core.numeric", "_frombuffer"),
                ("numpy.core.multiarray", "_reconstruct"),
                ("numpy._core.multiarray", "_reconstruct"),
                ("numpy.core.multiarray", "scalar"),
                ("numpy._core.multiarray", "scalar"),
            }
        )
        == _SAFE_EXPLICIT_GLOBALS
    )


# --------------------------------------------------------------------------- #
# 6. Honest artifacts are unaffected (the whole point of "purely additive").
# --------------------------------------------------------------------------- #


class _RoundTripNet(nn.Module):
    """Small graph with parameters, a persistent buffer and a non-persistent buffer."""

    def __init__(self) -> None:
        """Build byte-stable capture state."""

        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.norm = nn.BatchNorm2d(4)
        self.fc = nn.Linear(4 * 8 * 8, 5)
        self.register_buffer("gain", torch.ones(4, 1, 1), persistent=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Apply the deterministic graph."""

        value = torch.relu(self.norm(self.conv(value))) * self.gain
        return self.fc(value.flatten(1))


@pytest.mark.parametrize(
    ("level", "include_weights", "include_activations"),
    [
        ("full", False, False),
        ("runnable", False, False),
        ("runnable", True, False),
        ("runnable", True, True),
    ],
)
def test_honest_artifact_round_trip_unaffected(
    tmp_path: Path,
    level: str,
    include_weights: bool,
    include_activations: bool,
) -> None:
    """A REAL save/load round-trip still succeeds under every new belt."""

    model = _RoundTripNet().eval()
    inputs = torch.randn(2, 3, 8, 8)
    trace = tl.trace(model, inputs, capture=_CAP)
    path = tmp_path / f"honest-{level}-{include_weights}-{include_activations}.tlspec"
    kwargs: dict[str, object] = {}
    if level != "full":
        kwargs = {
            "level": level,
            "include_weights": include_weights,
            "include_activations": include_activations,
        }
    with warnings.catch_warnings():
        # The model carries a used non-persistent buffer, so a runnable save emits the
        # existing one-time privacy disclosure; it is orthogonal to this belt.
        warnings.simplefilter("ignore", UserWarning)
        trace.save(path, **kwargs)  # type: ignore[arg-type]

    loaded = tl.load(path)

    assert loaded.num_ops == trace.num_ops
    assert [str(op.label) for op in loaded.layer_list] == [str(op.label) for op in trace.layer_list]
    tl.validation.validate_tlspec(path)
