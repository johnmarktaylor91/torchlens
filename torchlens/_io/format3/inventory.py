"""Inert inventory of a 3.0 save's record tables against the persisted-state contract.

The 3.0 half of :func:`torchlens._io.state_contract.inspect_state_contract`: it parses the
manifest and the JSON record tables under the bounded reader, builds no record object and
refuses no unknown column; it reports each table's columns the field bindings do not
persist, by owning record class, so a user can see everything without loading anything.
"""

from __future__ import annotations

import hashlib
import os
import stat
from importlib import import_module
from pathlib import Path
from typing import Any

from .._json import loads_bounded, read_bytes_bounded
from ..format_errors import ArtifactError
from . import contract

__all__ = ["foreign_file_note", "inspect_record_tables", "names_format3"]

_REMEDY = "re-save the trace with trace.save()"


def _read_json_file(path: Path, *, code: str, what: str) -> Any:
    """Parse one JSON file under the bounded reader; an unreadable file refuses typed."""

    try:
        return loads_bounded(read_bytes_bounded(path).decode("utf-8"))
    except (OSError, ValueError, ArtifactError) as exc:
        if isinstance(exc, FileNotFoundError) and what == "manifest":
            code = contract.REFUSAL_CODES["manifest_missing"]
        raise ArtifactError(
            f"Artifact {what} {path} could not be read: {exc}. Remedy: {_REMEDY}.",
            code=code,
            remedy=_REMEDY,
            file_path=str(path),
        ) from exc


def names_format3(root: Path) -> bool:
    """Return whether ``root``'s manifest names the 3.0 format.

    Reads the manifest under the bounded reader and nothing else. A missing,
    unreadable or non-object manifest names no format (a 2.x manifest carries
    neither ``format`` nor ``format_version``).
    """

    try:
        manifest = loads_bounded(read_bytes_bounded(root / contract.MANIFEST_FILE).decode("utf-8"))
    except (OSError, ValueError, ArtifactError):
        return False
    if not isinstance(manifest, dict):
        return False
    version = manifest.get("format_version")
    return manifest.get("format") == contract.FORMAT_NAME or (
        type(version) is int and version == contract.FORMAT_MAJOR
    )


def foreign_file_note(path: Path) -> dict[str, Any]:
    """Describe a root file the 3.0 layout does not own, never parsing it.

    Parameters
    ----------
    path:
        The foreign file (for example a root ``metadata.pkl`` planted beside a 3.0 save).

    Returns
    -------
    dict[str, Any]
        ``name``, ``size_bytes`` and ``sha256`` (the hex digest of the bytes). A
        symlink or other non-regular entry is not followed or opened: its digest is
        ``None`` and ``size_bytes`` is the entry's own size.
    """

    entry = path.lstat()
    if not stat.S_ISREG(entry.st_mode):
        return {"name": path.name, "size_bytes": entry.st_size, "sha256": None}
    digest = hashlib.sha256()
    size = 0
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"name": path.name, "size_bytes": size, "sha256": digest.hexdigest()}


def inspect_record_tables(root: Path) -> dict[str, Any]:
    """Report the unknown columns of every record table ``root``'s manifest lists.

    Parameters
    ----------
    root:
        A 3.0 artifact directory.

    Returns
    -------
    dict[str, Any]
        The :func:`~torchlens._io.state_contract.inspect_state_contract` report:
        ``record_types`` (by class qualname: ``instances``, ``unknown_fields``,
        ``ungoverned``), ``unknown_field_total`` and ``governed_record_types``.
    """

    # The column bindings live with the record classes, a higher layer than this
    # contract module; resolve them at call time, as the 2.x half resolves the record
    # classes through the field registry.
    unknown_columns = import_module(f"{__package__}._tables")._unknown_columns
    manifest = _read_json_file(
        root / contract.MANIFEST_FILE,
        code=contract.REFUSAL_CODES["manifest_unreadable"],
        what="manifest",
    )
    records = manifest.get("records") if isinstance(manifest, dict) else None
    classes = {kind: qualname for kind, qualname, _ in contract.RECORD_INDEX}
    report: dict[str, dict[str, Any]] = {}
    for kind, entry in (records if isinstance(records, dict) else {}).items():
        rows = entry.get("rows") if isinstance(entry, dict) else None
        if kind not in classes:
            report[str(kind)] = {"instances": rows, "unknown_fields": [], "ungoverned": True}
            continue
        table = _read_json_file(
            root / contract.RECORDS_DIR / f"{kind}.json",
            code=contract.REFUSAL_CODES["record_file_corrupt"],
            what="record",
        )
        names = (
            table.get("fields" if kind == "trace" else "columns")
            if isinstance(table, dict)
            else None
        )
        unknown = unknown_columns(kind, names) if isinstance(names, dict) else []
        report[classes[kind]] = {
            "instances": rows,
            "unknown_fields": sorted(unknown),
            "ungoverned": False,
        }
    governed = sorted(name for name, row in report.items() if not row["ungoverned"])
    return {
        "record_types": dict(sorted(report.items())),
        "unknown_field_total": sum(len(report[name]["unknown_fields"]) for name in governed),
        "governed_record_types": governed,
    }
