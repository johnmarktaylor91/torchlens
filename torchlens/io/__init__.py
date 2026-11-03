"""Public I/O and administrative helpers for TorchLens."""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

from .._errors import ArgumentTypeError
from .._io import ArtifactError, JaxPayloadLoadHint, PayloadLoadHints, _json
from .._io.bundle import cleanup_tmp, load as _load
from .._io.format3 import contract as _format3_contract
from .._io.format3.reader import (  # noqa: F401  (validate_tlspec's 3.0 door)
    validate_artifact as _validate_format3_artifact,
)
from .._io.sidecar import (
    AnalysisOnlySidecar,
    SidecarFamily,
    attach_sidecar,
    list_sidecar_families,
    read_sidecar,
    register_sidecar_family,
    unregister_sidecar_family,
)
from .._trace_state import TraceState
from ..intervention.save import save_intervention
from ..intervention.types import InterventionSpec
from ..options import suppress_mutate_warnings


def list_traces() -> tuple[Any, ...]:
    """Return a snapshot of currently live TorchLens traces.

    Examples
    --------
    >>> trace = tl.trace(model, x)
    >>> any(live is trace for live in list_traces())
    True
    """

    from .. import _user_funcs as user_funcs

    return user_funcs.list_logs()


def reset_naming_counter(class_name: str | None = None) -> None:
    """Reset automatic TorchLens trace naming counters.

    Examples
    --------
    >>> reset_naming_counter("sequential")
    >>> tl.trace(model, x).name
    'sequential_1'
    >>> reset_naming_counter()
    """

    from .. import _user_funcs as user_funcs

    user_funcs.reset_naming_counter(class_name)


def detect_tlspec_format(path: str | Path) -> str:
    """Detect the on-disk TorchLens ``.tlspec`` format.

    Detection runs from the newest, most explicit markers to the oldest; the first match wins.

    Parameters
    ----------
    path:
        Directory path to inspect.

    Returns
    -------
    str
        One of ``"v3"`` (the 3.0 saved format), ``"pre_3"`` (a save written by
        TorchLens 2.x), or ``"unknown"`` (no TorchLens save markers).

    Examples
    --------
    >>> import pathlib, tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     path = pathlib.Path(tmp) / "trace.tlspec"
    ...     _ = tl.trace(model, x).save(path)
    ...     print(detect_tlspec_format(path))
    v3
    """

    tlspec_path = Path(path)
    _reject_symlinked_metadata_path(tlspec_path)
    manifest = _read_json_object_if_present(tlspec_path / "manifest.json")
    if manifest is not None and manifest.get("format") == _format3_contract.FORMAT_NAME:
        return "v3"
    return "pre_3" if _pre3_observed(tlspec_path) is not None else "unknown"


def _pre3_observed(path: Path) -> str | None:
    """Identify the 2.x marker without opening a pickle or decoding a record.

    Parameters
    ----------
    path:
        Artifact root already checked for symlinks by format detection.

    Returns
    -------
    str | None
        Marker text shared by load and validation, or no recognized 2.x marker.
    """

    from .._io.format3.validate import legacy_stamp

    _reject_symlinked_metadata_path(path)
    _reject_symlinked_metadata_path(path / "manifest.json")
    stamp = legacy_stamp(path / "manifest.json")
    if stamp is not None:
        return f"tlspec_version={stamp}"
    manifest = _read_json_object_if_present(path / "manifest.json")
    if manifest is not None and all(
        marker in manifest for marker in ("io_format_version", "n_activation_blobs")
    ):
        return (
            f"io_format_version={manifest['io_format_version']}, "
            f"n_activation_blobs={manifest['n_activation_blobs']}"
        )
    for marker in ("metadata.pkl", "bundle.json"):
        marker_path = path / marker
        _reject_symlinked_metadata_path(marker_path)
        if marker_path.exists():
            return f"{marker} at the artifact root"
    spec = _read_json_object_if_present(path / "spec.json")
    if spec is not None and "format_version" in spec:
        return f"intervention format_version={spec['format_version']}"
    if manifest is None:
        return None
    for marker in ("tlspec_version", "kind", "io_format_version", "n_activation_blobs"):
        if marker in manifest:
            return f"{marker}={manifest[marker]}, format={manifest.get('format', '<missing>')!r}"
    if "format" not in manifest and any(
        marker in manifest for marker in ("torchlens_version", "tensor_entries")
    ):
        return "manifest.json without a format key"
    return None


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
    ArtifactError
        If ``path`` is a symlink.
    """

    if path.is_symlink():
        # Same front-door cause as the bundle loader's symlink guards, so it
        # carries the same stable code a caller branches on (R65).
        raise ArtifactError(
            f"Refusing symlinked .tlspec format-detection path: {path}. "
            f"Remedy: pass the resolved bundle directory instead, e.g. "
            f"tl.load(str(Path(path).resolve())).",
            code="load_path_symlink_rejected",
        )


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

    Examples
    --------
    >>> import pathlib, tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     path = pathlib.Path(tmp) / "trace.tlspec"
    ...     _ = tl.trace(model, x).save(path)
    ...     manifest = inspect_tlspec(path)
    >>> manifest["kind"]
    'trace'
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


def read_safetensors(path: str | Path) -> Any:
    """Read a ``tl.export.safetensors`` Trace export back (DOCUMENTED-UNSTABLE).

    The file holds no pickle, so reading it never executes code. Returns a
    read-only record: ``header`` (schema, tlspec version, writer-contract
    digest, TorchLens version, capture-honesty facts), ``trace_fields`` and
    ``op_fields`` (the exported writer-contract KEEP fields, decoded to their
    original Python values), ``tensors`` (``op label -> {payload field:
    tensor}``, bit-equal to the exported payloads), and ``omitted`` (KEEP
    fields that had no exact JSON encoding).

    Parameters
    ----------
    path:
        File written by :func:`torchlens.export.safetensors`.

    Returns
    -------
    Any
        The read-only export record.

    Examples
    --------
    >>> # requires: safetensors
    >>> import pathlib, tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     path = tl.export.safetensors(tl.trace(model, x), pathlib.Path(tmp) / "t.safetensors")
    ...     record = read_safetensors(path)
    >>> record.header["schema"]
    'tl_trace_safetensors_v1'
    """

    from .._extraction.trace_safetensors import read_trace_safetensors

    return read_trace_safetensors(path)


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

    Examples
    --------
    >>> import pathlib, tempfile
    >>> fork = tl.trace(model, x).fork()
    >>> _ = fork.do(tl.func("relu"), tl.zero_ablate())
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     path = pathlib.Path(tmp) / "edit.tlspec"
    ...     _ = fork.save_intervention(path, level="portable")
    ...     spec = load_intervention_spec(path)
    >>> type(spec).__name__
    'InterventionSpec'
    """

    loaded = _load(
        path,
        trust_custom_callables=trust_custom_callables,
        allowed_custom_callable_modules=allowed_custom_callable_modules,
    )
    if not isinstance(loaded, InterventionSpec):
        raise ArgumentTypeError(
            f"load_intervention_spec loaded {type(loaded).__name__}, not InterventionSpec",
            code="artifact_kind_mismatch",
            remedy="pass the path of an intervention .tlspec artifact or use torchlens.load",
            argument="path",
            loaded_type=type(loaded).__name__,
        )
    return loaded


#: Removed public names: each raises the typed ``facade_redirect`` error naming
#: its replacement (clean break, no alias).
_REDIRECTS: dict[str, str] = {
    "get_model_metadata": "use torchlens.trace(model, x, save='none') -- a metadata-only capture is a trace that saves no payloads",
    "log_model_metadata": "use torchlens.trace(model, x, save='none') -- a metadata-only capture is a trace that saves no payloads",
}


# Runtime only: under TYPE_CHECKING a module ``__getattr__`` would make every
# name on this module type-check as Any.
if not TYPE_CHECKING:

    def __getattr__(name: str) -> Any:
        """Raise the typed teaching error for a removed torchlens.io name."""

        from .._utils.facade import resolve_facade_attr

        return resolve_facade_attr(
            owner=__name__, name=name, module_globals=globals(), redirects=_REDIRECTS
        )


def materialize_nested(
    trace: Any,
    *,
    map_location: str | torch.device = "cpu",
    payload_hints: PayloadLoadHints | Mapping[str, Any] | None = None,
) -> None:
    """Replace any remaining nested portable blob refs on a loaded model log.

    This function is a no-op unless the model log was loaded with
    ``lazy=True, materialize_nested=False``. In the default load mode, nested
    tensors are already materialized.

    Parameters
    ----------
    trace:
        Model log loaded from a portable bundle.
    map_location:
        Target device for the materialized nested tensors.
    payload_hints:
        Optional backend payload hints used during materialization.

    Examples
    --------
    >>> import pathlib, tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     path = pathlib.Path(tmp) / "trace.tlspec"
    ...     _ = tl.trace(model, x).save(path)
    ...     loaded = tl.load(path, lazy=True, materialize_nested=False)
    ...     materialize_nested(loaded)
    >>> isinstance(loaded, tl.Trace)
    True
    """

    from .._io.rehydrate import rehydrate_nested

    rehydrate_nested(trace, map_location=map_location, payload_hints=payload_hints)


__all__ = [
    "AnalysisOnlySidecar",
    "SidecarFamily",
    "TraceState",
    "JaxPayloadLoadHint",
    "PayloadLoadHints",
    "attach_sidecar",
    "cleanup_tmp",
    "detect_tlspec_format",
    "inspect_tlspec",
    "list_traces",
    "list_sidecar_families",
    "load_intervention_spec",
    "read_safetensors",
    "read_sidecar",
    "register_sidecar_family",
    "materialize_nested",
    "reset_naming_counter",
    "save_intervention",
    "suppress_mutate_warnings",
    "unregister_sidecar_family",
]
