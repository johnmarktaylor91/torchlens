"""Validate referenced payload headers without reading tensor contents."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from safetensors import SafetensorError, safe_open

from ..format_errors import ArtifactError
from ..paths import resolve_bundle_blob_path


def validate_payload_headers(root: Path, entries: Mapping[str, Mapping[str, Any]]) -> None:
    """Check each referenced chunk and its declared keys before Trace allocation."""

    chunks: dict[str, set[str]] = {}
    for entry in entries.values():
        chunks.setdefault(entry["relative_path"], set()).add(entry["key"])
    for relative, required in chunks.items():
        path = resolve_bundle_blob_path(root, relative)
        try:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                missing = required.difference(handle.keys())
        except (ImportError, OSError, SafetensorError, ValueError) as exc:
            raise ArtifactError(
                f"Payload chunk {relative} has an unreadable safetensors header: {exc}.",
                code="bundle_load_failed",
                remedy="re-save the trace; the payload chunk is damaged or incomplete",
                file_path=str(path),
            ) from exc
        if missing:
            raise ArtifactError(
                f"Payload chunk {relative} lacks declared keys {sorted(missing)[:8]}.",
                code="bundle_load_failed",
                remedy="re-save the trace; the payload chunk and record cells disagree",
                file_path=str(path),
            )
