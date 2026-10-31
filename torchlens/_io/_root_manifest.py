"""Route a ``.tlspec`` directory by its root manifest before trace-bundle parsing.

A merged, fastlog or dataset-extraction directory carries a root
``manifest.json`` that is a different discriminated object from a trace
bundle's manifest (it has no tensor table), so :func:`torchlens.load` routes it
here before ``Manifest`` parsing. Every routed format first passes the one
earlier-major refusal.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

from . import _json
from .format_contract import raise_if_saved_by_older_major

NOT_ROUTED = object()
"""Sentinel: the directory is not a root-manifest format; continue to format detection."""


def _is_extraction_manifest(candidate: dict[str, Any]) -> bool:
    """Return whether a directory manifest is a dataset-extraction artifact's.

    Parameters
    ----------
    candidate:
        The parsed ``manifest.json`` mapping.

    Returns
    -------
    bool
        True for the extraction manifest schemas (v1 and v2).
    """

    from .._data_substrate.artifact import MANIFEST_SCHEMA_V1, MANIFEST_SCHEMA_V2

    return candidate.get("schema") in {MANIFEST_SCHEMA_V1, MANIFEST_SCHEMA_V2}


def _read_root_manifest(bundle_path: Path) -> dict[str, Any] | None:
    """Read a directory's root ``manifest.json`` mapping, if it has a usable one.

    Parameters
    ----------
    bundle_path:
        The ``.tlspec`` directory.

    Returns
    -------
    dict[str, Any] | None
        The parsed mapping, or None when the root manifest is absent, a
        symlink, unreadable, or not a JSON object.
    """

    merged_manifest_path = bundle_path / "manifest.json"
    if not merged_manifest_path.is_file() or merged_manifest_path.is_symlink():
        return None
    try:
        # ``read_text`` allocated the entire attacker-sized file BEFORE the
        # ceiling could refuse it; the bounded path reader reads at most
        # ``max_bytes + 1``, so discovery cannot be an allocation DoS.
        candidate = _json.read_bounded(merged_manifest_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return candidate if isinstance(candidate, dict) else None


def load_by_root_manifest(bundle_path: Path) -> Any:
    """Load a directory whose root manifest names a merged, fastlog or extraction format.

    Parameters
    ----------
    bundle_path:
        The ``.tlspec`` directory.

    Returns
    -------
    Any
        The loaded object, or :data:`NOT_ROUTED` when the root manifest is
        absent, a symlink, unreadable, or names none of these formats.
    """

    candidate = _read_root_manifest(bundle_path)
    if candidate is None:
        return NOT_ROUTED
    # One refusal for every saved format: a file an earlier major
    # release wrote is never read (no compatibility reader is carried).
    raise_if_saved_by_older_major(candidate.get("torchlens_version"), str(bundle_path))
    if candidate.get("bundle_format") == "merged-directory":
        from .._merged._artifact import load_merged

        return load_merged(bundle_path)
    if candidate.get("bundle_format") == "fastlog-directory":
        from ..recording.recover import load as load_fastlog

        return load_fastlog(bundle_path)
    if _is_extraction_manifest(candidate):
        # Resolved by name: a static import would pull the extraction
        # engine (which imports the root facade) into this module's
        # import cycle for the type checker.
        extraction = importlib.import_module("torchlens.dataset_extraction")
        return extraction.load_extraction(bundle_path)
    return NOT_ROUTED
