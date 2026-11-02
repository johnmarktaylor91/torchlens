"""Helpers for the 3.0 saved-format reader tests: a tiny traced CNN and envelope surgery.

The reader tests never hand-write a whole artifact: they save a real trace with
the writer, copy the directory, change one thing (a byte, a key, a cell, a
digest) and load the copy. ``redigest`` keeps the manifest's ``files`` and
``records`` tables true for a rewritten file, so a mutation reaches the check
under test instead of the digest check in front of it.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from torch import nn

import torchlens as tl
from torchlens._io.format3 import contract

MANIFEST = contract.MANIFEST_FILE


def tiny_model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Conv2d(3, 4, 3), nn.ReLU(), nn.Flatten(), nn.Linear(4 * 6 * 6, 5))


def tiny_input() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(2, 3, 8, 8)


def tiny_trace() -> Any:
    return tl.trace(tiny_model(), tiny_input(), replayable=True)


def save_tiny(root: Path, level: str = "portable") -> Any:
    """Trace the tiny CNN and save it under ``root`` at ``level``; return the live trace."""

    trace = tiny_trace()
    trace.save(root, level=level)
    return trace


def read_manifest(root: Path) -> dict[str, Any]:
    return json.loads((root / MANIFEST).read_text(encoding="utf-8"))


def write_manifest(root: Path, manifest: dict[str, Any]) -> None:
    (root / MANIFEST).write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
        encoding="utf-8",
    )


def copy_artifact(root: Path, target: Path) -> Path:
    shutil.copytree(root, target)
    return target


def redigest(root: Path, relpath: str) -> None:
    """Re-list ``relpath`` in the manifest with its current sha256 and size."""

    raw = (root / relpath).read_bytes()
    manifest = read_manifest(root)
    entry = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    manifest["files"][relpath] = entry
    for record in manifest.get("records", {}).values():
        if record.get("path") == relpath:
            record.update(entry)
    write_manifest(root, manifest)


def read_record(root: Path, kind: str) -> dict[str, Any]:
    return json.loads((root / contract.RECORDS_DIR / f"{kind}.json").read_text(encoding="utf-8"))


def write_record(
    root: Path, kind: str, payload: dict[str, Any], *, redigest_it: bool = True
) -> None:
    relpath = f"{contract.RECORDS_DIR}/{kind}.json"
    (root / relpath).write_text(
        json.dumps(payload, separators=(",", ":"), ensure_ascii=False), encoding="utf-8"
    )
    if redigest_it:
        redigest(root, relpath)


def set_cell(root: Path, kind: str, column: str, row: int, cell: Any) -> None:
    """Replace one cell of a record table (the trace record's ``fields`` when ``kind`` is trace)."""

    payload = read_record(root, kind)
    if kind == "trace":
        payload["fields"][column] = cell
    else:
        payload["columns"][column][row] = cell
    write_record(root, kind, payload)


def first_column(root: Path, kind: str) -> str:
    payload = read_record(root, kind)
    return next(iter(payload["fields" if kind == "trace" else "columns"]))


def flip_byte(path: Path, offset: int) -> None:
    data = bytearray(path.read_bytes())
    offset %= len(data)
    data[offset] ^= 0x01
    path.write_bytes(bytes(data))
