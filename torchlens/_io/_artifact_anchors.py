"""Manifest <-> metadata integrity anchors (AUD-CODE 3.0b).

``manifest.json`` and ``metadata.pkl`` are written together by one writer, so
the facts both carry must agree at load or the artifact has been edited,
truncated, or spliced from two sources. The loader compares:

* the ``tlspec_version`` stamp -- the manifest's against the pickled root
  state's; the ONE lawful divergence is a ``tl.migrate`` artifact, which
  bumps the manifest stamp only and proves the lineage through the
  append-only migration witness (``tl_migration_provenance.json``);
* ``n_layers`` -- the manifest's declared retained-op count against the
  persisted ``layer_list`` length (a dropped or grafted op row).

Refusal code: ``bundle_manifest_metadata_mismatch`` (field names the fact).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import TorchLensIOError
from .compat_ledger import read_migration_witness

__all__ = ["check_manifest_metadata_anchors"]

_REMEDY = (
    "re-save the artifact from its source capture with one current TorchLens "
    "version; do not hand-edit manifest.json or metadata.pkl, and never splice "
    "the two files from different bundles"
)


def _is_int(value: Any) -> bool:
    """True for a non-bool int."""

    return isinstance(value, int) and not isinstance(value, bool)


def persisted_anchor_facts(scrubbed_state: Mapping[str, Any]) -> tuple[Any, int | None]:
    """Snapshot ``(root_stamp, persisted_row_count)`` from the unpickled root state.

    Taken BEFORE rehydration: the Trace state restore mutates the persisted
    ``layer_list`` in place (F44 injected-op rows are split back out), so the
    row count must be read from the bytes as written.
    """

    rows = scrubbed_state.get("layer_list")
    return scrubbed_state.get("tlspec_version"), (len(rows) if isinstance(rows, list) else None)


def check_manifest_metadata_anchors(
    manifest: Any,
    anchor_facts: tuple[Any, int | None],
    bundle_path: Path,
) -> None:
    """Refuse a bundle whose manifest facts contradict its pickled metadata.

    Parameters
    ----------
    manifest:
        Parsed ``Manifest`` (``tlspec_version`` / ``n_layers`` attributes).
    anchor_facts:
        The :func:`persisted_anchor_facts` snapshot taken before rehydration.
    bundle_path:
        Artifact directory (the migration witness sidecar lives beside it).

    Raises
    ------
    TorchLensIOError
        ``bundle_manifest_metadata_mismatch`` on a stamp or row-count conflict.
    """

    root_stamp, persisted_rows = anchor_facts
    manifest_stamp = getattr(manifest, "tlspec_version", None)
    if (
        isinstance(manifest_stamp, int)
        and not isinstance(manifest_stamp, bool)
        and isinstance(root_stamp, int)
        and not isinstance(root_stamp, bool)
        and root_stamp != manifest_stamp
        and not _migration_explains(bundle_path, root_stamp, manifest_stamp)
    ):
        _refuse(
            f"manifest.json declares tlspec_version={manifest_stamp} but the pickled "
            f"Trace state carries tlspec_version={root_stamp}, and no migration "
            "witness (tl_migration_provenance.json) proves that lineage",
            bundle_path=bundle_path,
            field="tlspec_version",
            reason="stamp_mismatch",
        )

    declared_rows = getattr(manifest, "n_layers", None)
    if _is_int(declared_rows) and persisted_rows is not None and declared_rows != persisted_rows:
        _refuse(
            f"manifest.json declares n_layers={declared_rows} but metadata.pkl carries "
            f"{persisted_rows} retained op rows",
            bundle_path=bundle_path,
            field="n_layers",
            reason="row_count_mismatch",
        )


def _migration_explains(bundle_path: Path, root_stamp: int, manifest_stamp: int) -> bool:
    """True iff a migration witness records exactly this root -> manifest stamp lineage."""

    try:
        witness = read_migration_witness(bundle_path)
    except TorchLensIOError:
        return False
    if not isinstance(witness, Mapping):
        return False
    origin = witness.get("origin")
    origin_stamp = origin.get("tlspec_version") if isinstance(origin, Mapping) else None
    return origin_stamp == root_stamp and witness.get("final_tlspec_version") == manifest_stamp


def _refuse(message: str, *, bundle_path: Path, field: str, reason: str) -> None:
    """Raise the typed manifest/metadata anchor refusal."""

    raise TorchLensIOError(
        f"Bundle {bundle_path} fails its manifest/metadata integrity anchor: {message}. "
        f"Remedy: {_REMEDY}.",
        code="bundle_manifest_metadata_mismatch",
        field=field,
        reason=reason,
        remedy=_REMEDY,
        path=str(bundle_path),
    )
