"""Read history rows back out of an offline wandb run's ``.wandb`` file.

wandb 0.30 ships no Python reader for its transaction log (the old
``wandb/sdk/internal/datastore.py`` went with the Go core), so the real-package
tracker tests parse it here, strictly. Layout, as wandb-core writes it:

* a 7-byte file header: ``:W&B``, the little-endian magic ``0xBEE1``, version 0;
* LevelDB-log framing in 32 KiB blocks counted from the START OF THE FILE (the
  file header occupies the first 7 bytes of block 0); each chunk has a 7-byte
  header ``<crc32 (IEEE, seeded with the type byte), length (u16), type (u8)>``
  with type FULL=1, FIRST=2, MIDDLE=3, LAST=4; a block tail shorter than a
  chunk header is zero padding;
* each reassembled payload is one serialized ``wandb.proto.Record``.

Every deviation raises: a bad magic, an unknown chunk type, a CRC mismatch,
non-zero padding, a chunk out of FIRST/MIDDLE/LAST order, or a record cut off
at end of file. wandb-core can still be closing the file when ``run.finish()``
returns, so :func:`history_rows` polls until the log parses to the end and
holds the run's ``exit`` record, and fails loudly if it never does.
"""

from __future__ import annotations

import json
import struct
import time
import zlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

_FILE_HEADER = b":W&B" + struct.pack("<HB", 0xBEE1, 0)
_BLOCK = 32768
_CHUNK_HEADER = 7
_FULL, _FIRST, _MIDDLE, _LAST = 1, 2, 3, 4
_SEED = {kind: zlib.crc32(bytes([kind])) for kind in (_FULL, _FIRST, _MIDDLE, _LAST)}


class IncompleteLogError(AssertionError):
    """The log ends inside a chunk or a multi-chunk record (still being written)."""


def _chunks(buf: bytes) -> Iterator[tuple[int, bytes]]:
    """Yield ``(type, data)`` for every chunk, verifying framing and CRCs."""

    pos = len(_FILE_HEADER)
    while pos < len(buf):
        left = _BLOCK - (pos % _BLOCK)
        if left < _CHUNK_HEADER:
            pad = buf[pos : pos + left]
            if pad.strip(b"\0"):
                raise AssertionError(f"non-zero block padding at byte {pos}: {pad!r}")
            pos += left
            continue
        if pos + _CHUNK_HEADER > len(buf):
            raise IncompleteLogError(f"chunk header cut off at byte {pos} of {len(buf)}")
        crc, length, kind = struct.unpack("<IHB", buf[pos : pos + _CHUNK_HEADER])
        if kind not in _SEED:
            raise AssertionError(f"unknown chunk type {kind} at byte {pos}")
        if _CHUNK_HEADER + length > left:
            raise AssertionError(f"chunk at byte {pos} overruns its block ({length} bytes)")
        data = buf[pos + _CHUNK_HEADER : pos + _CHUNK_HEADER + length]
        if len(data) < length:
            raise IncompleteLogError(f"chunk data cut off at byte {pos} of {len(buf)}")
        if zlib.crc32(data, _SEED[kind]) & 0xFFFFFFFF != crc:
            raise AssertionError(f"CRC mismatch for the type-{kind} chunk at byte {pos}")
        yield kind, data
        pos += _CHUNK_HEADER + length


def _payloads(buf: bytes) -> Iterator[bytes]:
    """Yield each reassembled record payload in file order."""

    if buf[: len(_FILE_HEADER)] != _FILE_HEADER:
        raise AssertionError(f"not a version-0 wandb transaction log: {buf[:7]!r}")
    pending: list[bytes] | None = None
    for kind, data in _chunks(buf):
        if kind in (_FULL, _FIRST) and pending is not None:
            raise AssertionError(f"type-{kind} chunk inside an unfinished record")
        if kind in (_MIDDLE, _LAST) and pending is None:
            raise AssertionError(f"type-{kind} chunk with no FIRST chunk before it")
        if kind == _FULL:
            yield data
        elif kind == _FIRST:
            pending = [data]
        elif kind == _MIDDLE:
            pending.append(data)  # type: ignore[union-attr]
        else:
            yield b"".join([*pending, data])  # type: ignore[misc]
            pending = None
    if pending is not None:
        raise IncompleteLogError("the log ends inside a multi-chunk record")


def records(path: Path) -> list[Any]:
    """Return every ``Record`` in the log (raises on any framing problem)."""

    from wandb.proto import wandb_internal_pb2

    parsed = []
    for payload in _payloads(path.read_bytes()):
        record = wandb_internal_pb2.Record()
        record.ParseFromString(payload)
        if record.WhichOneof("record_type") is None:
            raise AssertionError(f"record with no known record_type in {path}")
        parsed.append(record)
    return parsed


def _finished_records(path: Path, timeout: float) -> list[Any]:
    """Poll until the log parses to the end and holds the ``exit`` record."""

    deadline = time.monotonic() + timeout
    while True:
        try:
            parsed = records(path)
            state = "no exit record yet"
            if any(r.WhichOneof("record_type") == "exit" for r in parsed):
                return parsed
        except IncompleteLogError as exc:
            state = str(exc)
        if time.monotonic() > deadline:
            raise AssertionError(f"{path} never finished within {timeout}s: {state}")
        time.sleep(0.2)


def history_rows(run_dir: str | Path, timeout: float = 30.0) -> list[dict[str, Any]]:
    """Return every history row (keys joined with ``/``, JSON-decoded values)."""

    files = sorted(Path(run_dir).rglob("run-*.wandb"))
    if len(files) != 1:
        raise AssertionError(f"expected one offline .wandb file under {run_dir}, found {files}")
    return [
        {
            (item.key or "/".join(item.nested_key)): json.loads(item.value_json)
            for item in record.history.item
        }
        for record in _finished_records(files[0], timeout)
        if record.WhichOneof("record_type") == "history"
    ]
