"""Public I/O and administrative helpers for TorchLens."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .._io import rehydrate_nested
from .._io.bundle import cleanup_tmp, load, save
from .._run_state import RunState
from ..intervention.save import save_intervention
from ..intervention.types import InterventionSpec
from ..options import suppress_mutate_warnings
from ..user_funcs import get_model_metadata, list_logs, log_model_metadata, reset_naming_counter


def detect_tlspec_format(path: str | Path) -> str:
    """Detect the on-disk TorchLens ``.tlspec`` format.

    Detection is ordered from the newest, most explicit schema markers to older
    legacy markers. The first matching marker wins.

    Parameters
    ----------
    path:
        Directory path to inspect.

    Returns
    -------
    str
        One of ``"v2.0_unified"``, ``"v2.16_intervention_with_kind"``,
        ``"v2.16_intervention"``, ``"v2.16_modellog_portable"``, or
        ``"unknown"``.
    """

    tlspec_path = Path(path)
    manifest = _read_json_object_if_present(tlspec_path / "manifest.json")
    if manifest is not None:
        has_kind = "kind" in manifest
        has_tlspec_version = "tlspec_version" in manifest
        if has_tlspec_version and has_kind:
            return "v2.0_unified"
        if has_kind:
            return "v2.16_intervention_with_kind"

    spec = _read_json_object_if_present(tlspec_path / "spec.json")
    if spec is not None and "format_version" in spec:
        return "v2.16_intervention"

    if manifest is not None and "io_format_version" in manifest:
        return "v2.16_modellog_portable"
    return "unknown"


def _read_json_object_if_present(path: Path) -> dict[str, Any] | None:
    """Read one JSON object if the file exists and parses cleanly.

    Parameters
    ----------
    path:
        JSON file path.

    Returns
    -------
    dict[str, Any] | None
        Decoded object, or ``None`` when the file is absent or not a JSON
        object.
    """

    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def load_intervention_spec(path: str | Path) -> InterventionSpec:
    """Load an intervention spec through the canonical polymorphic loader.

    Parameters
    ----------
    path:
        Directory containing an intervention ``.tlspec``.

    Returns
    -------
    InterventionSpec
        Loaded intervention spec.

    Raises
    ------
    TypeError
        If ``path`` does not load as an intervention spec.
    """

    loaded = load(path)
    if not isinstance(loaded, InterventionSpec):
        raise TypeError("torchlens.io.load_intervention_spec expected an intervention spec.")
    return loaded


__all__ = [
    "RunState",
    "cleanup_tmp",
    "detect_tlspec_format",
    "get_model_metadata",
    "list_logs",
    "load",
    "load_intervention_spec",
    "log_model_metadata",
    "rehydrate_nested",
    "reset_naming_counter",
    "save",
    "save_intervention",
    "suppress_mutate_warnings",
]
