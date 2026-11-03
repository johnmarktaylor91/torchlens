"""Shapes of the tlspec 11 fields a 3.0 manifest still carries.

The records are authoritative for everything the loader rebuilds (the op count comes from
``records.op.rows``, not ``n_layers``), but readers that open ``manifest.json`` blind still
read these fields, so a type-swapped or missing one is malformed data, never silently kept.
A ``trace`` manifest carries every :data:`.contract.MANIFEST_CARRIED_FIELDS` entry; any kind
may carry them and the :data:`.contract.MANIFEST_CARRIED_OPTIONAL_FIELDS`, and a present one
must match its shape here. The shapes are the ones the tlspec 11 reader enforced.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Final

from ..format_errors import ArtifactError
from . import contract

__all__ = ["carried_field_errors"]

#: ``bundle_format`` values a trace directory may name.
_BUNDLE_FORMATS: Final = ("directory", "fastlog-directory")
#: ``body_index`` entry fields that must be non-empty strings.
_BODY_INDEX_STRINGS: Final = ("filename", "dtype", "intended_use")
#: ``sites`` entry fields that are a string or null.
_SITE_OPTIONAL_STRINGS: Final = ("function_path", "module_path", "op_kind")
#: Save-disclosure objects: the fields each must carry, by expected kind.
_DISCLOSURE_FIELDS: Final[dict[str, tuple[tuple[str, str], ...]]] = {
    "custom_attributes_disclosure": (
        ("included", "bool"),
        ("module_count", "count"),
        ("top_level_keys", "strings"),
    ),
    "buffer_values_disclosure": (
        ("included", "bool"),
        ("buffer_count", "count"),
        ("buffer_names", "strings"),
    ),
}


def _is_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_text(value: Any) -> bool:
    return isinstance(value, str) and value != ""


def _is_strings(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


_KIND_CHECKS: Final[dict[str, Callable[[Any], bool]]] = {
    "bool": lambda value: isinstance(value, bool),
    "count": _is_count,
    "strings": _is_strings,
}


def carried_field_errors(manifest: Mapping[str, Any]) -> list[str]:
    """Every way the carried tlspec 11 fields of ``manifest`` depart from their shapes."""

    errors: list[str] = []
    if manifest.get("kind") == "trace":
        errors.extend(
            f"missing carried field {key!r}"
            for key in contract.MANIFEST_CARRIED_FIELDS
            if key not in manifest
        )
    for key in (*contract.MANIFEST_CARRIED_FIELDS, *contract.MANIFEST_CARRIED_OPTIONAL_FIELDS):
        if key in manifest:
            error = _field_error(key, manifest[key], kind=manifest.get("kind"))
            if error is not None:
                errors.append(error)
    return errors


def _field_error(key: str, value: Any, *, kind: Any) -> str | None:
    """The shape violation of one carried field, or ``None``."""

    if key in ("model_signature", "body_format"):
        return None if _is_text(value) else f"{key} must be a non-empty string"
    if key.startswith("n_") or key == "schema_version":
        return None if _is_count(value) else f"{key} must be a non-negative integer"
    if key == "bundle_format":
        return (
            None if value in _BUNDLE_FORMATS else f"bundle_format must be one of {_BUNDLE_FORMATS}"
        )
    if key in ("spec_compat_info", "intervention_compat_metadata"):
        return _compat_error(key, value, kind=kind)
    if key in _DISCLOSURE_FIELDS:
        return _disclosure_error(key, value)
    return _STRUCTURED_CHECKS[key](value)


def _compat_error(key: str, value: Any, *, kind: Any) -> str | None:
    """Intervention compatibility metadata: an object or null, and null outside an intervention."""

    if value is not None and not isinstance(value, dict):
        return f"{key} must be an object or null"
    if value is not None and kind == "trace":
        return f"{key} must be null in a trace manifest"
    return None


def _disclosure_error(key: str, value: Any) -> str | None:
    """A save disclosure: null, or an object carrying its typed fields."""

    if value is None:
        return None
    if not isinstance(value, dict):
        return f"{key} must be an object when present"
    for field, expected in _DISCLOSURE_FIELDS[key]:
        if not _KIND_CHECKS[expected](value.get(field)):
            return f"{key}.{field} must be a {expected} value"
    return None


def _body_index_error(value: Any) -> str | None:
    """``body_index``: a list of payload entries, each naming file, dtype, use and shape."""

    if not isinstance(value, list):
        return "body_index must be a list"
    for index, entry in enumerate(value):
        if not isinstance(entry, dict):
            return f"body_index[{index}] must be an object"
        if not all(_is_text(entry.get(field)) for field in _BODY_INDEX_STRINGS):
            return f"body_index[{index}] must name {_BODY_INDEX_STRINGS} as non-empty strings"
        shape = entry.get("shape")
        if not isinstance(shape, list) or not all(_is_count(dim) for dim in shape):
            return f"body_index[{index}].shape must be a list of non-negative integers"
        if not _is_count(entry.get("num_elements")):
            return f"body_index[{index}].num_elements must be a non-negative integer"
    return None


def _sites_error(value: Any) -> str | None:
    """``sites``: a list of site descriptors, each with a string ``layer_label``."""

    if not isinstance(value, list):
        return "sites must be a list"
    for index, site in enumerate(value):
        if not isinstance(site, dict) or not isinstance(site.get("layer_label"), str):
            return f"sites[{index}] must be an object with a string layer_label"
        pass_index = site.get("pass_index")
        if pass_index is not None and (
            not isinstance(pass_index, int) or isinstance(pass_index, bool)
        ):
            return f"sites[{index}].pass_index must be an integer or null"
        if any(
            site.get(field) is not None and not isinstance(site.get(field), str)
            for field in _SITE_OPTIONAL_STRINGS
        ):
            return f"sites[{index}] fields {_SITE_OPTIONAL_STRINGS} must be strings or null"
    return None


def _optional_dependencies_error(value: Any) -> str | None:
    """``optional_dependencies``: distinct non-empty strings."""

    if not isinstance(value, list) or not all(_is_text(item) for item in value):
        return "optional_dependencies must be a list of non-empty strings"
    if len(set(value)) != len(value):
        return "optional_dependencies entries must be distinct"
    return None


def _artifact_error(check: Callable[[Any], Any], value: Any) -> str | None:
    """Run a tlspec 11 validator that raises ``ArtifactError``; its message, or ``None``."""

    try:
        check(value)
    except ArtifactError as exc:
        return str(exc)
    return None


def _provenance_error(value: Any) -> str | None:
    """``provenance``: null, or the versioned capture provenance certificate."""

    if value is None:
        return None
    if not isinstance(value, dict):
        return "provenance must be an object when present"
    from ..manifest import Provenance  # deferred: the 2.x manifest module imports the package

    return _artifact_error(Provenance.from_dict, value)


def _unsupported_tensors_error(value: Any) -> str | None:
    """``unsupported_tensors``: null, or a list of string-to-string objects."""

    from ..manifest import _validate_unsupported_tensors  # deferred, as above

    return _artifact_error(_validate_unsupported_tensors, value)


_STRUCTURED_CHECKS: Final[dict[str, Callable[[Any], str | None]]] = {
    "body_index": _body_index_error,
    "sites": _sites_error,
    "optional_dependencies": _optional_dependencies_error,
    "unsupported_tensors": _unsupported_tensors_error,
    "provenance": _provenance_error,
}
