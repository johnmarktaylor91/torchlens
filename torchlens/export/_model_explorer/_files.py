"""File-target writer for the Model Explorer export family (memo D16).

``tl.export.model_explorer`` always emits the app's file-ingest contract:
one strict graph-collection JSON at ``path``. Node-data overlays (memo D8)
are separate vendor-format sidecar files disclosed by a small manifest --
the app ingests them through its node-data upload door, the embed registers
them from the manifest.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ...utils.display import atomic_write_text
from ._collection import to_model_explorer_dict
from ._options import ModelExplorerOptions
from ._overlay import STANDARD_OVERLAYS, build_node_data
from ._validate import validate_model_explorer_payload

__tl_layer__ = "L8"


def model_explorer(
    log: Any,
    path: str | Path,
    *,
    overlays: tuple[str, ...] | list[str] | None = None,
    **options: Any,
) -> Path:
    """Export one capture as Model Explorer graph-collection JSON.

    Parameters
    ----------
    log:
        TorchLens ``Trace`` to export.
    path:
        Destination JSON path. Overlay sidecars and the manifest land next
        to it as ``<stem>.nodedata.<provider>.json`` / ``<stem>.manifest.json``.
    overlays:
        Node-data providers to write as sidecars: overlay preset names from
        the existing closed vocabulary (``"time"``, ``"flops"``, ``"bytes"``,
        ``"nan"``, ``"magnitude"``, ``"grad_norm"``, ...), ``"field:<name>"``
        record-field sources, or ``"standard"`` for the standard packet.
    **options:
        The flat :class:`ModelExplorerOptions` keywords (``label=``,
        ``privacy_profile=``, ``strict_namespace=``, ``include_source=``,
        ``include_rolled=``, ``per_step=``, ``boundary_proxies=``,
        ``step_budget_bytes=``, ``max_step_graphs=``), forwarded verbatim to
        :func:`to_model_explorer_dict`.

    Returns
    -------
    Path
        Written collection JSON path.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = to_model_explorer_dict(log, **options)
    validate_model_explorer_payload(payload, strict=True)
    atomic_write_text(destination, json.dumps(payload, indent=2))
    if overlays:
        profile = ModelExplorerOptions(**options).privacy_profile
        _write_overlay_sidecars(log, destination, payload, tuple(overlays), profile)
    return destination


def _write_overlay_sidecars(
    log: Any,
    destination: Path,
    payload: dict[str, Any],
    overlays: tuple[str, ...],
    privacy_profile: str,
) -> None:
    """Write vendor-format node-data sidecars plus the disclosure manifest."""

    if overlays == ("standard",):
        overlays = STANDARD_OVERLAYS
    stem = destination.name.removesuffix(".json")
    manifest: dict[str, Any] = {
        "schema": "torchlens.model_explorer_manifest.v1",
        "collection": destination.name,
        "node_data": [],
    }
    for source in overlays:
        node_data, coverage = build_node_data(log, payload, source, privacy_profile=privacy_profile)
        sidecar = destination.with_name(f"{stem}.nodedata.{coverage['provider_slug']}.json")
        atomic_write_text(sidecar, json.dumps(node_data, indent=2))
        manifest["node_data"].append(
            {"file": sidecar.name, "provider": coverage["provider"], "coverage": coverage}
        )
    atomic_write_text(
        destination.with_name(f"{stem}.manifest.json"),
        json.dumps(manifest, indent=2),
    )
