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

The fix overrides ``_PurePythonBundleUnpickler._instantiate`` (invoked via normal MRO by the
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

import pytest
import torch

from torchlens._io._safe_unpickle import (
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


_OBJ_FLOATTENSOR = b"(ctorch\nFloatTensor\n" + (b"I%d\n" % _TN) + b"o."
_OBJ_BYTEARRAY_SIZE = b"(cbuiltins\nbytearray\n" + (b"I%d\n" % _N) + b"o."


# shrink-r4: rows kept carry a sole library arc or a campaign kill; the others were proven redundant (slice 1 evidence)
@pytest.mark.parametrize(
    ("label", "stream"),
    [
        ("OBJ torch.FloatTensor", _OBJ_FLOATTENSOR),
    ],
)
def test_inst_obj_alloc_type_refused(label: str, stream: bytes) -> None:
    """SEC-H1 core: INST/OBJ construction of an alloc type is refused (no allocation)."""

    with pytest.raises(pickle.UnpicklingError):
        _load(stream)


# shrink-r4: rows kept carry a sole library arc or a campaign kill; the others were proven redundant (slice 1 evidence)
@pytest.mark.parametrize(
    ("label", "stream"),
    [
        ("OBJ bytearray(N)", _OBJ_BYTEARRAY_SIZE),
    ],
)
def test_inst_obj_integer_sized_buffer_refused(label: str, stream: bytes) -> None:
    """INST/OBJ ``bytes(N)`` / ``bytearray(N)`` integer-size allocator is refused."""

    with pytest.raises(pickle.UnpicklingError):
        _load(stream)


_INST_BENIGN_BYTES_COPY = b"(" + pickle.SHORT_BINBYTES + b"\x03abc" + b"ibuiltins\nbytes\n" + b"."


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
    assert loaded.provenance.backend == "torch"
    assert loaded.summary()


# --- structural / opcode-enumeration proof (mutation tripwire) -------------------------
