"""Route a ``.tlspec`` directory by its root manifest before trace-bundle parsing.

A merged, fastlog or dataset-extraction directory carries a root
``manifest.json`` that is a different discriminated object from a trace
bundle's manifest (it has no tensor table), so :func:`torchlens.load` routes it
here before ``Manifest`` parsing. A merged or fastlog directory, and any
other manifest that stamps the current format number, first passes the
release-stamp floor (``raise_if_saved_by_older_major``); an extraction
directory takes the same floor inside ``load_extraction``, after its schema
check, and a manifest stamping an older format number is left to the
format-specific refusal downstream.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

from . import _json
from .format_contract import MIN_TLSPEC_VERSION, raise_if_saved_by_older_major

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


def _release_floor_applies(candidate: dict[str, Any]) -> bool:
    """Return whether the release-stamp floor is checked at this door.

    Merged and fastlog directories have no other release check, so they always
    are. An extraction manifest is checked by ``load_extraction`` after its
    schema refusal. Any other manifest is checked only when it stamps the
    current format number: an older format number is refused downstream by the
    format-specific door, which names the format (a 2.16 ModelLog, a
    ``tlspec_version`` below the floor) more precisely than a release stamp.

    Parameters
    ----------
    candidate:
        The parsed root ``manifest.json`` mapping.

    Returns
    -------
    bool
        True when :func:`raise_if_saved_by_older_major` runs here.
    """

    if candidate.get("bundle_format") in ("merged-directory", "fastlog-directory"):
        return True
    if _is_extraction_manifest(candidate):
        return False
    stamp = candidate.get("tlspec_version")
    return isinstance(stamp, int) and stamp >= MIN_TLSPEC_VERSION


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
    # A file a release below the first format-11 release wrote is never read
    # (no compatibility reader is carried).
    if _release_floor_applies(candidate):
        from .. import __version__ as _running_version

        raise_if_saved_by_older_major(
            candidate.get("torchlens_version"), str(bundle_path), running_version=_running_version
        )
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
