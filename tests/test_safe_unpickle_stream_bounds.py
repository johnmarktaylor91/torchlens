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
import pickletools
import struct
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._data_classes.trace import Trace
from torchlens._io._canonical_pickle import dump_canonical_metadata
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


class _ReadRecordingStream(io.BytesIO):
    """In-memory stream that records every size passed to ``read``."""

    def __init__(self, data: bytes) -> None:
        """Wrap ``data`` and start with an empty read log."""

        super().__init__(data)
        self.read_sizes: list[int] = []

    def read(self, size: int | None = -1) -> bytes:
        """Record the requested size, then read as ``io.BytesIO`` does."""

        self.read_sizes.append(-1 if size is None else size)
        return super().read(size)


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


def _opcodes(data: bytes) -> list[tuple[str, Any, int]]:
    """Return ``(name, argument, position)`` for every opcode in ``data``."""

    return [(op.name, arg, pos) for op, arg, pos in pickletools.genops(data)]


def _assert_frames_exact(data: bytes) -> None:
    """Assert every FRAME in ``data`` declares no more bytes than follow its header."""

    for name, arg, pos in _opcodes(data):
        if name == "FRAME":
            header_end = pos + 1 + struct.calcsize("<Q")
            assert arg <= len(data) - header_end, (pos, arg)


# --------------------------------------------------------------------------- #
# Extension-code opcodes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("opcode", "code"),
    [(pickle.EXT1, 0x7F), (pickle.EXT2, 0x1234), (pickle.EXT4, 0x12345678)],
    ids=["EXT1", "EXT2", "EXT4"],
)
def test_extension_opcodes_refuse_before_any_registry_lookup(
    monkeypatch: pytest.MonkeyPatch, opcode: bytes, code: int
) -> None:
    """Each extension opcode refuses without consulting the extension registry."""

    lookups: list[int] = []

    def _recording_get_extension(self: Any, ext_code: int) -> None:
        """Record a registry lookup instead of performing it."""

        lookups.append(ext_code)

    # The C machine's registry lookup is not a Python attribute; the pure-Python machine's
    # is. The reader pass refuses the opcode before either machine runs, so neither the
    # recorded lookup nor a load path is ever reached.
    monkeypatch.setattr(_PurePythonBundleUnpickler, "get_extension", _recording_get_extension)
    data = _P2 + _ext(opcode, code) + pickle.MARK + pickle.TUPLE + pickle.REDUCE + pickle.STOP
    unpickler = SafeBundleUnpickler(io.BytesIO(data))
    with pytest.raises(pickle.UnpicklingError, match=_EXT_REFUSAL):
        unpickler.load()
    assert lookups == []
    assert unpickler.load_path is None
    with pytest.raises(pickle.UnpicklingError, match=_EXT_REFUSAL):
        _PurePythonBundleUnpickler(io.BytesIO(data)).load()
    assert lookups == []


def test_a_registered_extension_code_is_refused_and_never_cached() -> None:
    """A live registry entry for an allowlisted class is still refused, and nothing is cached.

    Runs in a child interpreter: ``copyreg.add_extension`` is process-wide, so registering
    here would leak into every later test. The stock unpickler loading the same stream
    afterwards proves the codes were live.
    """

    program = textwrap.dedent(
        """
        import copyreg, io, pickle, struct
        from torchlens._io._safe_unpickle import SafeBundleUnpickler
        entries = [
            (pickle.EXT1, "<B", 0x7F, "OrderedDict"),
            (pickle.EXT2, "<H", 0x1234, "Counter"),
            (pickle.EXT4, "<i", 0x12345678, "defaultdict"),
        ]
        verdicts = []
        for opcode, width, code, name in entries:
            copyreg.add_extension("collections", name, code)
            data = (
                pickle.PROTO + b"\\x02" + opcode + struct.pack(width, code)
                + pickle.STOP
            )
            cached_before = dict(copyreg._extension_cache)
            try:
                SafeBundleUnpickler(io.BytesIO(data)).load()
            except pickle.UnpicklingError as exc:
                verdicts.append("refused" if "extension codes" in str(exc) else repr(exc))
            else:
                verdicts.append("loaded")
            same = dict(copyreg._extension_cache) == cached_before
            verdicts.append("cache-unchanged" if same else "cache-written")
            verdicts.append(pickle.loads(data).__name__)
        print(" ".join(verdicts))
        """
    )
    completed = subprocess.run(  # noqa: S603 -- fixed argv, this interpreter
        [sys.executable, "-I", "-c", program], capture_output=True, timeout=180
    )
    assert completed.returncode == 0, completed.stderr.decode()
    assert completed.stdout.decode().split() == [
        "refused",
        "cache-unchanged",
        "OrderedDict",
        "refused",
        "cache-unchanged",
        "Counter",
        "refused",
        "cache-unchanged",
        "defaultdict",
    ]


# --------------------------------------------------------------------------- #
# FRAME length
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "declared",
    [len(_BODY) + 1, 2**40, 2**64 - 1],
    ids=["one_byte_over", "a_terabyte", "the_largest_length"],
)
def test_a_frame_declaring_more_than_remains_is_refused_before_it_is_read(declared: int) -> None:
    """The declared length is checked against the bytes left before any read of the frame."""

    data = _P4 + _frame(declared) + _BODY
    stream = _ReadRecordingStream(data)
    unpickler = SafeBundleUnpickler(stream)
    with pytest.raises(pickle.UnpicklingError, match=_FRAME_REFUSAL):
        unpickler.load()
    # The loader reads the stream whole, once, before any opcode runs; no read is ever
    # sized by the declaration, and neither machine ran.
    assert stream.read_sizes == [-1]
    assert unpickler.load_path is None
    # The pure-Python machine bounds the same declaration at the opcode.
    pure_stream = _ReadRecordingStream(data)
    with pytest.raises(pickle.UnpicklingError, match=_FRAME_REFUSAL):
        _PurePythonBundleUnpickler(pure_stream).load()
    assert declared not in pure_stream.read_sizes
    assert pure_stream.tell() == len(_P4) + len(_frame(0))


def test_an_overrunning_frame_in_a_real_file_is_refused(tmp_path: Path) -> None:
    """The bound holds on a buffered file handle, the shape the bundle load sites pass."""

    path = tmp_path / "metadata.pkl"
    path.write_bytes(_P4 + _frame(2**40) + _BODY)
    with path.open("rb") as handle, pytest.raises(pickle.UnpicklingError, match=_FRAME_REFUSAL):
        SafeBundleUnpickler(handle).load()


def test_a_frame_overrunning_a_later_frame_is_refused() -> None:
    """A second frame is bounded by what remains after the first, not by the whole stream."""

    first = pickle.EMPTY_LIST + pickle.MARK + pickle.NONE
    second = pickle.APPENDS + pickle.STOP
    data = _P4 + _frame(len(first)) + first + _frame(len(second) + 1) + second
    with pytest.raises(pickle.UnpicklingError, match=_FRAME_REFUSAL):
        SafeBundleUnpickler(io.BytesIO(data)).load()
    exact = _P4 + _frame(len(first)) + first + _frame(len(second)) + second
    assert SafeBundleUnpickler(io.BytesIO(exact)).load() == [None]


def test_an_exact_frame_still_loads() -> None:
    """The check bounds the declaration; exact frames, single or many, load unchanged."""

    assert SafeBundleUnpickler(io.BytesIO(_P4 + _frame(len(_BODY)) + _BODY)).load() is None
    value = {"text": "x" * 200_000, "numbers": list(range(50_000)), "nested": [{"k": (1, 2.5)}]}
    for protocol in (4, pickle.HIGHEST_PROTOCOL):
        data = pickle.dumps(value, protocol=protocol)
        assert sum(name == "FRAME" for name, _, _ in _opcodes(data)) > 1
        assert SafeBundleUnpickler(io.BytesIO(data)).load() == value


def test_a_stream_that_cannot_report_its_length_still_bounds_a_frame() -> None:
    """A frame is bounded by the bytes actually present, with or without a seekable source.

    The loader reads its source whole before any opcode runs, so an unseekable source (a
    pipe, a socket) gives the same bound as a file: an over-declared frame refuses, an
    exact one loads. The pure-Python machine, which reads per opcode, cannot bound a frame
    without a length and refuses the framed stream outright.
    """

    framed = pickle.dumps({"a": [1, 2, 3]}, protocol=4)
    assert SafeBundleUnpickler(io.BufferedReader(_UnseekableStream(framed))).load() == {
        "a": [1, 2, 3]
    }
    over = _P4 + _frame(len(_BODY) + 1) + _BODY
    with pytest.raises(pickle.UnpicklingError, match=_FRAME_REFUSAL):
        SafeBundleUnpickler(io.BufferedReader(_UnseekableStream(over))).load()
    with pytest.raises(pickle.UnpicklingError, match=_FRAME_REFUSAL):
        _PurePythonBundleUnpickler(io.BufferedReader(_UnseekableStream(framed))).load()
    unframed = pickle.dumps({"a": [1, 2, 3]}, protocol=2)
    assert SafeBundleUnpickler(io.BufferedReader(_UnseekableStream(unframed))).load() == {
        "a": [1, 2, 3]
    }


# --------------------------------------------------------------------------- #
# What the writer emits
# --------------------------------------------------------------------------- #


def test_the_canonical_writer_frames_exactly_and_emits_no_extension_codes(
    tmp_path: Path,
) -> None:
    """The metadata writer pickles at the highest protocol with exact frames, no EXT opcodes."""

    state = {
        "labels": frozenset(f"layer_{index}" for index in range(5_000)),
        "rows": [{"name": f"op_{index}", "shape": (index, 3)} for index in range(5_000)],
        "tensor": torch.arange(6, dtype=torch.float32).reshape(2, 3),
    }
    buffer = io.BytesIO()
    dump_canonical_metadata(state, buffer)
    data = buffer.getvalue()
    names = [name for name, _, _ in _opcodes(data)]
    assert _opcodes(data)[0][:2] == ("PROTO", pickle.HIGHEST_PROTOCOL)
    assert names.count("FRAME") > 1
    assert not _EXT_OPCODES.intersection(names)
    _assert_frames_exact(data)
    path = tmp_path / "metadata.pkl"
    path.write_bytes(data)
    with path.open("rb") as handle:
        loaded = SafeBundleUnpickler(handle).load()
    assert loaded["labels"] == state["labels"] and loaded["rows"] == state["rows"]
    assert torch.equal(loaded["tensor"], state["tensor"])


def test_a_saved_bundle_metadata_has_exact_frames_and_no_extension_codes(
    tmp_path: Path,
) -> None:
    """A real ``trace.save`` writes metadata the tightened loader reads back through ``tl.load``."""

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 3))
    trace = tl.trace(model, torch.randn(2, 4))
    bundle_path = tmp_path / "bundle.tl"
    trace.save(bundle_path)
    data = (bundle_path / "metadata.pkl").read_bytes()
    assert not _EXT_OPCODES.intersection(name for name, _, _ in _opcodes(data))
    _assert_frames_exact(data)
    restored = tl.load(bundle_path)
    assert isinstance(restored, Trace)
