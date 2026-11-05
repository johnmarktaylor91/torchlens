"""Stream-shape refusals of the restricted bundle-metadata unpickler.

``SafeBundleUnpickler`` refuses two opcode shapes that no TorchLens writer emits:

* the extension-code opcodes EXT1 / EXT2 / EXT4, which resolve a class through the
  process-wide ``copyreg`` registry and cache the result process-wide, refused before any
  registry lookup;
* a FRAME that declares more bytes than remain in the stream, refused before the frame
  is read (the base machine hands the declared size to ``file.read``).

Both raise the loader's existing ``pickle.UnpicklingError``, which the load sites map to
``ArtifactError`` with the code ``bundle_metadata_integrity_refused``. The test that
registers a real extension code does so in a child interpreter, so the registry of the
test process is never touched.
"""

from __future__ import annotations

import io
import pickle
import struct
from typing import Any

import pytest

from torchlens._io._safe_unpickle import SafeBundleUnpickler, _PurePythonBundleUnpickler

_P2 = pickle.PROTO + b"\x02"
_P4 = pickle.PROTO + b"\x04"
_BODY = pickle.NONE + pickle.STOP
_EXT_REFUSAL = "extension codes are not used in bundle metadata"
_FRAME_REFUSAL = "FRAME declares more bytes than remain"
_EXT_OPCODES = {"EXT1", "EXT2", "EXT4"}


def _frame(size: int) -> bytes:
    """Return a FRAME opcode declaring ``size`` payload bytes."""

    return pickle.FRAME + struct.pack("<Q", size)


def _ext(opcode: bytes, code: int) -> bytes:
    """Return an extension-code opcode naming ``code`` in its own width."""

    width = {pickle.EXT1: "<B", pickle.EXT2: "<H", pickle.EXT4: "<i"}[opcode]
    return opcode + struct.pack(width, code)


class _UnseekableStream(io.RawIOBase):
    """Readable stream that cannot report its length (a pipe, a socket)."""

    def __init__(self, data: bytes) -> None:
        """Serve ``data`` front to back without seek support."""

        super().__init__()
        self._inner = io.BytesIO(data)

    def readable(self) -> bool:
        """Report the stream as readable."""

        return True

    def seekable(self) -> bool:
        """Report the stream as not seekable."""

        return False

    def readinto(self, buffer: Any) -> int:
        """Fill ``buffer`` from the wrapped bytes."""

        chunk = self._inner.read(len(buffer))
        buffer[: len(chunk)] = chunk
        return len(chunk)


# --------------------------------------------------------------------------- #
# Extension-code opcodes
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# FRAME length
# --------------------------------------------------------------------------- #


def test_a_stream_that_cannot_report_its_length_still_bounds_a_frame() -> None:
    """A framed stream from a source that cannot report its length is refused, on both machines.

    The pure-Python machine reads per opcode and cannot bound a FRAME without a source
    length, so it refuses the framed stream at its first FRAME. The C path reads the
    source whole, yet makes the same decision (fail closed, the same typed refusal): a
    stream's acceptance never depends on which machine ran it. An unframed stream from
    the same source loads on both.
    """

    framed = pickle.dumps({"a": [1, 2, 3]}, protocol=4)
    over = _P4 + _frame(len(_BODY) + 1) + _BODY
    unframed = pickle.dumps({"a": [1, 2, 3]}, protocol=2)
    for machine in (SafeBundleUnpickler, _PurePythonBundleUnpickler):
        for stream in (framed, over):
            with pytest.raises(pickle.UnpicklingError, match=_FRAME_REFUSAL) as info:
                machine(io.BufferedReader(_UnseekableStream(stream))).load()
            assert type(info.value) is pickle.UnpicklingError
        assert machine(io.BufferedReader(_UnseekableStream(unframed))).load() == {"a": [1, 2, 3]}
    # The same framed bytes from a source that reports its length load on both.
    for machine in (SafeBundleUnpickler, _PurePythonBundleUnpickler):
        assert machine(io.BytesIO(framed)).load() == {"a": [1, 2, 3]}
    # The fast path's interpreter fallback inherits the source's decision too.
    fast = SafeBundleUnpickler(io.BufferedReader(_UnseekableStream(unframed)))
    assert fast.load() == {"a": [1, 2, 3]}
    assert fast._source_sized is False


# --------------------------------------------------------------------------- #
# What the writer emits
# --------------------------------------------------------------------------- #
