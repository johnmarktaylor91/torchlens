"""Public I/O and administrative helpers for TorchLens."""

from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Collection
from typing import Any

from .._io import JaxPayloadLoadHint, PayloadLoadHints, TorchLensIOError, rehydrate_nested
from .._io import _json
from .._io.bundle import cleanup_tmp, load, save
from .._trace_state import TraceState
from ..intervention.save import save_intervention
from ..intervention.types import InterventionSpec
from ..options import suppress_mutate_warnings


def list_logs() -> tuple[Any, ...]:
    """Return a snapshot of currently live TorchLens traces."""

    from .. import user_funcs

    return user_funcs.list_logs()


def reset_naming_counter(class_name: str | None = None) -> None:
    """Reset automatic TorchLens trace naming counters."""

    from .. import user_funcs

    user_funcs.reset_naming_counter(class_name)


def log_model_metadata(*args: Any, **kwargs: Any) -> Any:
    """Run metadata-only model capture through the public user_funcs surface."""

    from .. import user_funcs

    return user_funcs.log_model_metadata(*args, **kwargs)


def get_model_metadata(*args: Any, **kwargs: Any) -> Any:
    """Deprecated metadata alias routed through the public user_funcs surface."""

    from .. import user_funcs

    return user_funcs.get_model_metadata(*args, **kwargs)


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
    _reject_symlinked_metadata_path(tlspec_path)
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

    if manifest is not None and "tlspec_version" in manifest:
        return "v2.16_modellog_portable"
    return "unknown"


def _reject_symlinked_metadata_path(path: Path) -> None:
    """Reject a symlinked ``.tlspec`` format-detection path (defense-in-depth).

    ``detect_tlspec_format`` / ``inspect_tlspec`` read ``manifest.json`` /
    ``spec.json`` for format classification BEFORE the bundle loader's symlink
    guards (``torchlens._io.bundle._reject_symlink_path``) fire. A crafted
    ``.tlspec`` whose child JSON member -- or the bundle root directory -- is a
    symlink would otherwise be FOLLOWED out of the bundle at classification time
    (arbitrary-path read / DoS). Mirror the loader guards so format detection
    refuses a symlinked member too.

    Parameters
    ----------
    path:
        Bundle root or metadata-file path to validate.

    Raises
    ------
    TorchLensIOError
        If ``path`` is a symlink.
    """

    if path.is_symlink():
        raise TorchLensIOError(f"Refusing symlinked .tlspec format-detection path: {path}.")


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

    _reject_symlinked_metadata_path(path)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = _json.load_bounded(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def inspect_tlspec(path: str | Path) -> dict[str, Any]:
    """Return the parsed public manifest for a TorchLens ``.tlspec`` directory.

    Parameters
    ----------
    path:
        Directory path to inspect.

    Returns
    -------
    dict[str, Any]
        Parsed manifest object. Legacy intervention specs without a manifest
        return ``spec.json`` with ``"_inspected_from": "spec.json"``.

    Raises
    ------
    FileNotFoundError
        If no inspectable ``.tlspec`` metadata file exists.
    ValueError
        If the metadata file is not a JSON object.
    """

    tlspec_path = Path(path)
    _reject_symlinked_metadata_path(tlspec_path)
    manifest_path = tlspec_path / "manifest.json"
    manifest = _read_json_object_if_present(manifest_path)
    if manifest is not None:
        return manifest

    spec_path = tlspec_path / "spec.json"
    spec = _read_json_object_if_present(spec_path)
    if spec is not None:
        inspected = dict(spec)
        inspected["_inspected_from"] = "spec.json"
        return inspected
    raise FileNotFoundError(f"No TorchLens .tlspec manifest found at {tlspec_path}.")


def load_intervention_spec(
    path: str | Path,
    *,
    trust_custom_callables: bool = False,
    allowed_custom_callable_modules: Collection[str] | None = None,
) -> InterventionSpec:
    """Load an intervention spec through the canonical polymorphic loader.

    Parameters
    ----------
    path:
        Directory containing an intervention ``.tlspec``.
    trust_custom_callables:
        Explicit permission to import custom callables when no allowlist is
        supplied. Enable only for specs from a trusted source.
    allowed_custom_callable_modules:
        Optional allowlist of custom callable module names. When supplied,
        custom imports must be listed even if ``trust_custom_callables=True``.

    Returns
    -------
    InterventionSpec
        Loaded intervention spec.

    Raises
    ------
    TypeError
        If ``path`` does not load as an intervention spec.
    """

    loaded = load(
        path,
        trust_custom_callables=trust_custom_callables,
        allowed_custom_callable_modules=allowed_custom_callable_modules,
    )
    if not isinstance(loaded, InterventionSpec):
        raise TypeError("torchlens.io.load_intervention_spec expected an intervention spec.")
    return loaded


__all__ = [
    "TraceState",
    "JaxPayloadLoadHint",
    "PayloadLoadHints",
    "cleanup_tmp",
    "detect_tlspec_format",
    "get_model_metadata",
    "inspect_tlspec",
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
