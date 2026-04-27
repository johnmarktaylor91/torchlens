"""Index file writers for fastlog directory bundles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .._io import TorchLensIOError
from .types import Recording


def write_pass_index(path: str | Path, recording: Recording) -> None:
    """Write ``pass_index.json`` for a fastlog bundle.

    Parameters
    ----------
    path:
        Bundle directory path.
    recording:
        Recording whose per-pass index should be persisted.

    Raises
    ------
    TorchLensIOError
        If the index file cannot be written.
    """

    pass_index: dict[str, list[str | None]] = {}
    for pass_num, indexes in recording.by_pass.items():
        pass_index[str(pass_num)] = [_record_blob_id(recording.records[index]) for index in indexes]
    _write_json(Path(path) / "pass_index.json", pass_index)


def write_label_index(path: str | Path, recording: Recording) -> None:
    """Write ``label_index.json`` for a fastlog bundle.

    Parameters
    ----------
    path:
        Bundle directory path.
    recording:
        Recording whose label index should be persisted.

    Raises
    ------
    TorchLensIOError
        If the index file cannot be written.
    """

    label_index: dict[str, list[dict[str, int | str | None]]] = {}
    for label, entries in recording.by_label.items():
        label_index[label] = [
            {
                "pass_index": pass_index,
                "record_index": record_index,
                "blob_id": _record_blob_id(recording.records[record_index]),
            }
            for pass_index, record_index in entries
        ]
    _write_json(Path(path) / "label_index.json", label_index)


def _record_blob_id(record: Any) -> str | None:
    """Return a record's persisted blob id when present."""

    blob_id = record.metadata.get("blob_id")
    return str(blob_id) if blob_id is not None else None


def _write_json(path: Path, data: Any) -> None:
    """Write one JSON file with deterministic formatting."""

    try:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except OSError as exc:
        raise TorchLensIOError(f"Failed to write fastlog index at {path}.") from exc
