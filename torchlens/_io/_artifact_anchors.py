"""Manifest <-> metadata integrity anchors (AUD-CODE 3.0b).

``manifest.json`` and ``metadata.pkl`` are written together by one writer, so
the facts both carry must agree at load or the artifact has been edited,
truncated, or spliced from two sources. The loader compares ``n_layers``, the
manifest's declared retained-op count, against the persisted ``layer_list``
length (a dropped or grafted op row). The ``tlspec_version`` stamps need no
comparison here: the manifest policy and the state decoder each refuse any
stamp other than the current one before this check runs.

Refusal code: ``bundle_manifest_metadata_mismatch`` (field names the fact).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import ArtifactError

__all__ = ["check_manifest_metadata_anchors"]

_REMEDY = (
    "re-save the artifact from its source capture with one current TorchLens "
    "version; do not hand-edit manifest.json or metadata.pkl, and never splice "
    "the two files from different bundles"
)


def _is_int(value: Any) -> bool:
    """True for a non-bool int."""

    return isinstance(value, int) and not isinstance(value, bool)


def persisted_anchor_facts(scrubbed_state: Mapping[str, Any]) -> int | None:
    """Snapshot the persisted row count from the unpickled root state.

    Taken before rehydration, which mutates the persisted ``layer_list`` in place.
    """

    rows = scrubbed_state.get("layer_list")
    return len(rows) if isinstance(rows, list) else None


def check_manifest_metadata_anchors(
    manifest: Any,
    persisted_rows: int | None,
    bundle_path: Path,
) -> None:
    """Refuse a bundle whose manifest row count contradicts its pickled metadata.

    Parameters
    ----------
    manifest:
        Parsed ``Manifest`` (``n_layers`` attribute).
    persisted_rows:
        The :func:`persisted_anchor_facts` snapshot taken before rehydration.
    bundle_path:
        Artifact directory, named in the refusal.

    Raises
    ------
    ArtifactError
        ``bundle_manifest_metadata_mismatch`` on a row-count conflict.
    """

    declared_rows = getattr(manifest, "n_layers", None)
    if _is_int(declared_rows) and persisted_rows is not None and declared_rows != persisted_rows:
        _refuse(
            f"manifest.json declares n_layers={declared_rows} but metadata.pkl carries "
            f"{persisted_rows} retained op rows",
            bundle_path=bundle_path,
            field="n_layers",
            reason="row_count_mismatch",
        )


def _refuse(message: str, *, bundle_path: Path, field: str, reason: str) -> None:
    """Raise the typed manifest/metadata anchor refusal."""

    raise ArtifactError(
        f"Bundle {bundle_path} fails its manifest/metadata integrity anchor: {message}. "
        f"Remedy: {_REMEDY}.",
        code="bundle_manifest_metadata_mismatch",
        field=field,
        reason=reason,
        remedy=_REMEDY,
        path=str(bundle_path),
    )
