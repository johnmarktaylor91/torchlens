"""Regression tests for SEC-H1: INST/OBJ opcode allocation-belt bypass.

The ``SafeBundleUnpickler`` allocation belt (r49 storage, r63 tensor/ndarray) refuses
constructing an attacker-sized ALLOCATION type (a torch storage, a ``torch.Tensor``
subclass or legacy ``torch.<dtype>Tensor``, or a ``numpy.ndarray`` subclass) from a
crafted ``metadata.pkl`` at plain ``tl.load()`` time. Before this fix the belt was
OPCODE-SCOPED to the four dispatch overrides (BUILD / REDUCE / NEWOBJ / NEWOBJ_EX), so
the LEGACY protocol-0/1 constructing opcodes ``INST`` (``b'i'``) and ``OBJ`` (``b'o'``)
-- which both call ``klass(*args)`` through the shared ``_instantiate`` primitive rather
than an inline dispatch handler -- bypassed the belt entirely. A ~30-byte
``INST numpy.ndarray`` / ``OBJ numpy.ndarray`` / ``INST torch.FloatTensor`` stream
allocated an attacker-sized, uninitialized-heap buffer (OOM DoS + uninitialized-heap read
on a public object) BEFORE any structural check.

The fix overrides ``SafeBundleUnpickler._instantiate`` (invoked via normal MRO by the
inherited ``load_inst`` / ``load_obj``) so the SAME belt fires for INST/OBJ -- closing
the check at the CONSTRUCTION primitive, not per-opcode.

OPCODE-ENUMERATION PROOF (``test_every_constructing_opcode_is_gated``): the full set of
pickle opcodes that construct an object from attacker-controlled args is
{REDUCE, NEWOBJ, NEWOBJ_EX, INST, OBJ} (BUILD applies ``__setstate__`` / storage-rebind).
REDUCE/NEWOBJ/NEWOBJ_EX are gated by the four dispatch overrides; INST/OBJ funnel through
the overridden ``_instantiate``; BUILD is gated by ``load_build``. No constructing opcode
reaches an unbounded allocation.

This belt is a LOCKED TRIPWIRE: these tests must only ever be TIGHTENED, never weakened.
"""

from __future__ import annotations

import io
import pickle

import numpy as np
import pytest
import torch

from torchlens._io._safe_unpickle import (
    _NUMPY_RECONSTRUCT_FUNCS,
    SafeBundleUnpickler,
)

# 1,000,000-element float64 ndarray == 8 MB; the same stream scales LINEARLY (an attacker
# picks N up to ~2e9 == 16 GiB). Kept modest so that if the guard is reverted (mutation
# proof) the now-succeeding construction allocates only ~8 MB, never the DoS payload.
_N = 1_000_000
_TN = 1_000  # torch tensor element count (small; refused before construction anyway)


def _load(data: bytes) -> object:
    """Run the restricted unpickler over a crafted stream."""

    return SafeBundleUnpickler(io.BytesIO(data)).load()


# --- crafted attacker streams, one per constructing opcode -----------------------------
# INST: MARK, push args, then ``i<module>\n<name>\n`` resolves + constructs klass(*args).
# OBJ:  MARK, push class (GLOBAL ``c``), push args, ``o`` constructs args[0](*args[1:]).
_INST_NDARRAY = b"(" + (b"I%d\n" % _N) + b"inumpy\nndarray\n" + b"."
_OBJ_NDARRAY = b"(cnumpy\nndarray\n" + (b"I%d\n" % _N) + b"o."
_INST_FLOATTENSOR = b"(" + (b"I%d\n" % _TN) + b"itorch\nFloatTensor\n" + b"."
_OBJ_FLOATTENSOR = b"(ctorch\nFloatTensor\n" + (b"I%d\n" % _TN) + b"o."
# REDUCE control: bare ``numpy.ndarray(N)`` -- the belt already refused this pre-fix.
_REDUCE_NDARRAY = b"cnumpy\nndarray\n" + (b"I%d\n" % _N) + b"\x85R."
# NEWOBJ control: push cls, push ``(N,)`` (TUPLE1), NEWOBJ.
_NEWOBJ_NDARRAY = b"\x80\x02" + b"cnumpy\nndarray\n" + (b"I%d\n" % _N) + b"\x85\x81."
# Storage stays refused at find_class (opcode-independent identity gate), never resolves.
_INST_FLOATSTORAGE = b"(" + (b"I%d\n" % _TN) + b"itorch\nFloatStorage\n" + b"."
# bytes/bytearray handed an integer SIZE (a direct zero-fill allocator) via INST/OBJ.
_INST_BYTES_SIZE = b"(" + (b"I%d\n" % _N) + b"ibuiltins\nbytes\n" + b"."
_OBJ_BYTEARRAY_SIZE = b"(cbuiltins\nbytearray\n" + (b"I%d\n" % _N) + b"o."


@pytest.mark.parametrize(
    ("label", "stream"),
    [
        ("INST numpy.ndarray", _INST_NDARRAY),
        ("OBJ numpy.ndarray", _OBJ_NDARRAY),
        ("INST torch.FloatTensor", _INST_FLOATTENSOR),
        ("OBJ torch.FloatTensor", _OBJ_FLOATTENSOR),
    ],
)
def test_inst_obj_alloc_type_refused(label: str, stream: bytes) -> None:
    """SEC-H1 core: INST/OBJ construction of an alloc type is refused (no allocation)."""

    with pytest.raises(pickle.UnpicklingError):
        _load(stream)


@pytest.mark.parametrize(
    ("label", "stream"),
    [
        ("INST bytes(N)", _INST_BYTES_SIZE),
        ("OBJ bytearray(N)", _OBJ_BYTEARRAY_SIZE),
    ],
)
def test_inst_obj_integer_sized_buffer_refused(label: str, stream: bytes) -> None:
    """INST/OBJ ``bytes(N)`` / ``bytearray(N)`` integer-size allocator is refused."""

    with pytest.raises(pickle.UnpicklingError):
        _load(stream)


@pytest.mark.parametrize(
    ("label", "stream"),
    [
        ("REDUCE numpy.ndarray", _REDUCE_NDARRAY),
        ("NEWOBJ numpy.ndarray", _NEWOBJ_NDARRAY),
        ("INST torch.FloatStorage", _INST_FLOATSTORAGE),
    ],
)
def test_other_opcode_paths_still_refused(label: str, stream: bytes) -> None:
    """Regression guard: the pre-existing REDUCE/NEWOBJ belt + storage gate stay closed."""

    with pytest.raises(pickle.UnpicklingError):
        _load(stream)


def test_build_on_torch_tensor_refused() -> None:
    """BUILD (``__setstate__`` / storage-rebind) on a torch tensor instance is refused."""

    unpickler = SafeBundleUnpickler(io.BytesIO(b"."))
    # load_build reads ``self.stack[-2]`` (the instance) and ``[-1]`` (the state).
    unpickler.stack = [torch.zeros(1), {}]  # type: ignore[attr-defined]
    with pytest.raises(pickle.UnpicklingError):
        unpickler.load_build()


def test_instantiate_refuses_mediated_reconstruct() -> None:
    """INST/OBJ routing a MEDIATED allocation (numpy ``_reconstruct``) is refused.

    ``_reconstruct(numpy.ndarray, (N,), b"b")`` runs ``ndarray.__new__`` on a
    pickle-supplied shape and returns attacker-sized uninitialized memory. It is an
    allowlisted plain FUNCTION (not an alloc TYPE), so it is caught by
    ``_alloc_refusal_reason`` inside the ``_instantiate`` override rather than the
    type-keyed belt.
    """

    if not _NUMPY_RECONSTRUCT_FUNCS:  # pragma: no cover - numpy always present with torch
        pytest.skip("numpy _reconstruct helper not resolvable")
    reconstruct = next(iter(_NUMPY_RECONSTRUCT_FUNCS))
    unpickler = SafeBundleUnpickler(io.BytesIO(b"."))
    with pytest.raises(pickle.UnpicklingError):
        # args as a LIST, exactly as ``load_inst`` / ``load_obj`` hand it over.
        unpickler._instantiate(reconstruct, [np.ndarray, (_N,), b"b"])


# --- benign / do-not-over-block --------------------------------------------------------
_INST_BENIGN_LIST = b"(ibuiltins\nlist\n."
_OBJ_BENIGN_ORDEREDDICT = b"(ccollections\nOrderedDict\no."
_INST_BENIGN_BYTES_COPY = b"(" + pickle.SHORT_BINBYTES + b"\x03abc" + b"ibuiltins\nbytes\n" + b"."


def test_inst_benign_container_loads() -> None:
    """A benign INST of a non-alloc container still constructs (no over-blocking)."""

    assert _load(_INST_BENIGN_LIST) == []


def test_obj_benign_ordereddict_loads() -> None:
    """A benign OBJ of a non-alloc mapping still constructs (no over-blocking)."""

    import collections

    assert _load(_OBJ_BENIGN_ORDEREDDICT) == collections.OrderedDict()


def test_inst_bytes_buffer_copy_loads() -> None:
    """``bytes(b"abc")`` COPIES a buffer already present in the stream -- allowed."""

    assert _load(_INST_BENIGN_BYTES_COPY) == b"abc"


def test_tlspec_roundtrip_still_loads(tmp_path: object) -> None:
    """Integration benign proof: a real ``.tlspec`` save/load round-trip still works."""

    import torchlens as tl

    class _M(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = torch.nn.Linear(4, 3)
            self.bn = torch.nn.BatchNorm1d(3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(self.bn(self.lin(x)))

    model = _M().eval()
    x = torch.randn(2, 4)
    log = tl.trace(model, x, save=tl.func("relu"))
    path = str(tmp_path) + "/roundtrip.tlspec"  # type: ignore[operator]
    log.save(path)
    loaded = tl.load(path)
    assert loaded.backend == "torch"
    assert loaded.summary()


# --- structural / opcode-enumeration proof (mutation tripwire) -------------------------
def test_instantiate_override_present() -> None:
    """MUTATION TRIPWIRE: reverting the ``_instantiate`` override fails here + above."""

    assert "_instantiate" in SafeBundleUnpickler.__dict__
    assert SafeBundleUnpickler._instantiate.__qualname__ == "SafeBundleUnpickler._instantiate"


def test_every_constructing_opcode_is_gated() -> None:
    """PROOF that NO pickle construction opcode reaches an unbounded allocation.

    The complete set of opcodes that construct an object from attacker-controlled args is
    {REDUCE, NEWOBJ, NEWOBJ_EX, INST, OBJ}; BUILD applies ``__setstate__`` / a
    storage-rebind. Each is gated: REDUCE/NEWOBJ/NEWOBJ_EX/BUILD by a subclass dispatch
    override, INST/OBJ by the ``_instantiate`` override their inherited handlers funnel
    through via ``self._instantiate``.
    """

    dispatch = SafeBundleUnpickler.dispatch
    # The four dispatch-level overrides are the SUBCLASS functions, not the base ones.
    for opcode, expected in (
        (pickle.REDUCE[0], "SafeBundleUnpickler.load_reduce"),
        (pickle.NEWOBJ[0], "SafeBundleUnpickler.load_newobj"),
        (pickle.NEWOBJ_EX[0], "SafeBundleUnpickler.load_newobj_ex"),
        (pickle.BUILD[0], "SafeBundleUnpickler.load_build"),
    ):
        assert dispatch[opcode].__qualname__ == expected

    # INST/OBJ keep the BASE dispatch handlers (they call ``self._instantiate``); the
    # gate lives on the overridden shared primitive, invoked via normal MRO.
    assert dispatch[pickle.INST[0]].__qualname__ == "_Unpickler.load_inst"
    assert dispatch[pickle.OBJ[0]].__qualname__ == "_Unpickler.load_obj"
    assert "_instantiate" in SafeBundleUnpickler.__dict__
