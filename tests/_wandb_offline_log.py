"""Read history rows back out of an offline wandb run's ``.wandb`` file.

wandb 0.30 ships no Python reader for its transaction log, so the real-package
tracker tests parse it directly: a 7-byte ``:W&B`` header, then LevelDB-log
framing (32 KiB blocks; 7-byte record headers: crc32c, length, type with
FULL=1, FIRST=2, MIDDLE=3, LAST=4) around serialized ``Record`` protobufs.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Iterator
from pathlib import Path
from typing import Any

_BLOCK = 32768
_HEADER = 7


def _payloads(path: Path) -> Iterator[bytes]:
    """Yield each reassembled record payload in file order."""

    buf = path.read_bytes()
    if buf[:4] != b":W&B":
        raise AssertionError(f"{path} is not a wandb transaction log: {buf[:7]!r}")
    pos = _HEADER
    pending = b""
    while pos + _HEADER <= len(buf):
        left = _BLOCK - (pos % _BLOCK)
        if left < _HEADER:
            pos += left
            continue
        _crc, length, kind = struct.unpack("<IHB", buf[pos : pos + _HEADER])
        if kind == 0 and length == 0:
            pos += left
            continue
        data = buf[pos + _HEADER : pos + _HEADER + length]
        pos += _HEADER + length
        if kind == 1:
            yield data
        elif kind == 2:
            pending = data
        elif kind == 3:
            pending += data
        elif kind == 4:
            yield pending + data
            pending = b""


def history_rows(run_dir: str | Path) -> list[dict[str, Any]]:
    """Return every history row (keys joined with ``/``, JSON-decoded values)."""

    from wandb.proto import wandb_internal_pb2

    files = sorted(Path(run_dir).rglob("run-*.wandb"))
    if not files:
        raise AssertionError(f"no offline .wandb file under {run_dir}")
    rows: list[dict[str, Any]] = []
    for payload in _payloads(files[0]):
        record = wandb_internal_pb2.Record()
        record.ParseFromString(payload)
        if record.WhichOneof("record_type") != "history":
            continue
        row = {
            (item.key or "/".join(item.nested_key)): json.loads(item.value_json)
            for item in record.history.item
        }
        rows.append(row)
    return rows
