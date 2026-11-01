"""Codec of the ``.tlspec`` spec document (``spec.json``).

One encoder, ``_serialize_intervention_spec``, and one decoder,
``_deserialize_intervention_spec``, for the current format version, plus the
table of every spec ``format_version`` TorchLens has written
(``SPEC_FORMAT_VERSIONS``): which it writes today, which it refuses, and the
decode tolerances the current reader keeps for payloads older writers of the
same version produced. ``save.py`` owns the directory, tensors and manifest
around this document; ``_spec_values`` owns the value wrapper grammar.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import asdict, dataclass, fields as dataclass_fields
from pathlib import Path
from typing import Any, Literal

import torch

from .._errors import InvalidArgumentError
from ._spec_values import (
    SaveLevel,
    _deserialize_value,
    _jsonish_metadata,
    _serialize_value,
    _SerializedState,
)
from .errors import (
    OpaqueCallableError,
    ReplayPreconditionError,
)
from .types import (
    Edit,
    FireRecord,
    FrozenTargetSpec,
    HookSpec,
    InterventionSpec,
    TargetSpec,
    TargetValueSpec,
    TensorSliceSpec,
)


@dataclass(frozen=True)
class SpecFormatVersion:
    """One row of the spec ``format_version`` table.

    ``status`` is ``"current"`` for the version this module writes and reads, or
    ``"refused"`` for a version an older TorchLens wrote that no reader here
    accepts. ``decode_tolerances`` names the older payload shapes the current
    reader still accepts inside this version; each is a documented branch of the
    one decoder, never a second code path.
    """

    version: str
    status: Literal["current", "refused"]
    note: str
    decode_tolerances: tuple[str, ...] = ()


SPEC_FORMAT_VERSIONS: tuple[SpecFormatVersion, ...] = (
    SpecFormatVersion(
        "1",
        "refused",
        "written by torchlens 2.16 to 2.28; refused with artifact_version_below_floor",
    ),
    SpecFormatVersion(
        "2",
        "current",
        "the spec document this module writes",
        decode_tolerances=(
            "a __callable__ whose import path does not resolve decodes as an "
            "audit-only Edit instead of refusing",
            "__opaque_audit__ decodes as an audit-only Edit; inside a selector value "
            "it decodes as its recorded repr string",
            "a bare JSON list inside a selector value decodes as a tuple",
            "the legacy __repr__ selector-value tag decodes as its string",
        ),
    ),
)
_CURRENT_FORMAT = next(row for row in SPEC_FORMAT_VERSIONS if row.status == "current")
TLSPEC_FORMAT_VERSION = _CURRENT_FORMAT.version
_SPEC_FILE = "spec.json"


def _validate_spec_json_structure(data: dict[str, Any], spec_path: Path) -> None:
    """Validate the top-level ``spec.json`` structure before deserialization.

    WT1 A-IV item 20 (lane A08): the loader used to index straight into the
    decoded JSON, so a spec.json missing its ``intervention_spec`` object died
    with a bare ``KeyError`` and junk-typed top-level fields flowed silently
    into spec metadata. Structural violations refuse with the module's
    documented ``ReplayPreconditionError``; deeper per-record staging refusals
    are the C03 lane's continuing work.
    """

    spec_payload = data.get("intervention_spec")
    if not isinstance(spec_payload, dict):
        raise ReplayPreconditionError(
            f"{spec_path / _SPEC_FILE} is missing its 'intervention_spec' object "
            f"(got {type(spec_payload).__name__}); the file is not a TorchLens "
            "intervention spec or was hand-edited"
        )
    declared_level = data.get("save_level")
    if declared_level is not None:
        allowed = {member.value for member in SaveLevel}
        if str(declared_level) not in allowed:
            raise ReplayPreconditionError(
                f"{spec_path / _SPEC_FILE} declares unknown save_level "
                f"{declared_level!r}; expected one of {sorted(allowed)}"
            )
    for list_field in ("target_manifest", "helpers", "function_registry_keys"):
        value = data.get(list_field)
        if value is not None and not isinstance(value, list):
            raise ReplayPreconditionError(
                f"{spec_path / _SPEC_FILE} field {list_field!r} must be a list, "
                f"got {type(value).__name__}"
            )


def _validate_format_version(format_version: Any, *, path: Path | None = None) -> None:
    """Check ``format_version`` against ``SPEC_FORMAT_VERSIONS``: a refused row raises the
    one 2.x error (``artifact_version_below_floor``), an unknown version raises
    ``spec_format_version_unsupported``. ``path`` is the spec directory when the caller
    has it, for the refusal's fields."""

    row = next((r for r in SPEC_FORMAT_VERSIONS if r.version == str(format_version)), None)
    if row is not None and row.status == "refused":
        from .._io._pre3_module_paths import pre_3_artifact_error

        raise pre_3_artifact_error(
            f"intervention format_version={row.version}",
            path=None if path is None else str(path),
            code="artifact_version_below_floor",
        )
    if row is None:
        raise InvalidArgumentError(
            f"Unsupported intervention .tlspec format_version={format_version!r}; "
            f"expected {TLSPEC_FORMAT_VERSION}",
            code="spec_format_version_unsupported",
            remedy=f"use a supported intervention format version ({TLSPEC_FORMAT_VERSION})",
            format_version=str(format_version),
        )


def _serialize_intervention_spec(
    spec: InterventionSpec,
    save_level: SaveLevel,
    state: _SerializedState,
) -> dict[str, Any]:
    """Serialize an intervention spec to JSON-safe data.

    Parameters
    ----------
    spec:
        Intervention spec to serialize.
    save_level:
        Requested save level.
    state:
        Serialization tensor state.

    Returns
    -------
    dict[str, Any]
        JSON-safe spec payload.
    """

    return {
        "targets": [_target_spec_to_json(target, save_level) for target in spec.targets],
        "helper": _serialize_value(spec.helper, save_level, state),
        "value": _serialize_value(spec.value, save_level, state),
        "hook": _serialize_value(spec.hook, save_level, state),
        "target_value_specs": [
            _serialize_target_value_spec(value_spec, save_level, state)
            for value_spec in spec.target_value_specs
        ],
        "hook_specs": [
            _serialize_hook_spec(hook_spec, save_level, state) for hook_spec in spec.hook_specs
        ],
        "records": [_serialize_fire_record(record, save_level, state) for record in spec.records],
        "metadata": _jsonish_metadata(spec.metadata),
    }


def _serialize_fire_record(
    record: FireRecord,
    save_level: SaveLevel,
    state: _SerializedState,
) -> dict[str, Any]:
    """Serialize a fire record with callable-safe helper handling.

    Parameters
    ----------
    record:
        Fire record to serialize.
    save_level:
        Requested save level.
    state:
        Serialization tensor state.

    Returns
    -------
    dict[str, Any]
        JSON-safe fire-record payload.
    """

    # R10-14: ``dataclasses.asdict`` deep-copies every field value first, so a
    # graph-connected helper tensor arg raised torch's deepcopy error at save
    # time for an intermediate about to be overwritten anyway. Build the
    # payload shallowly: the two structured fields are serialized below and
    # every other FireRecord field is a scalar/string.
    data = {field.name: getattr(record, field.name) for field in dataclass_fields(record)}
    data["edit"] = _serialize_value(record.edit, save_level, state)
    data["container_path"] = _serialize_value(record.container_path, save_level, state)
    return data


def _serialize_target_value_spec(
    value_spec: TargetValueSpec,
    save_level: SaveLevel,
    state: _SerializedState,
) -> dict[str, Any]:
    """Serialize a target-value spec.

    Parameters
    ----------
    value_spec:
        Target-value spec.
    save_level:
        Requested save level.
    state:
        Serialization tensor state.

    Returns
    -------
    dict[str, Any]
        JSON-safe payload.
    """

    return {
        "site_target": _target_spec_to_json(value_spec.site_target, save_level),
        "value": _serialize_value(value_spec.value, save_level, state),
        "metadata": _jsonish_metadata(value_spec.metadata),
    }


def _serialize_hook_spec(
    hook_spec: HookSpec,
    save_level: SaveLevel,
    state: _SerializedState,
) -> dict[str, Any]:
    """Serialize a hook spec.

    Parameters
    ----------
    hook_spec:
        Hook spec.
    save_level:
        Requested save level.
    state:
        Serialization tensor state.

    Returns
    -------
    dict[str, Any]
        JSON-safe payload.
    """

    if hook_spec.metadata.get("facet_write") and save_level is not SaveLevel.AUDIT:
        # A facet-slice hook fires through a capture-bound scatter wrapper that closes
        # over the resolved FacetSpec; only the raw helper would survive serialization,
        # and a loaded spec would then apply that helper to the WHOLE home tensor
        # instead of the selected facet slice. Refuse rather than persist a spec whose
        # replay semantics silently differ from what was attached.
        facet_name = hook_spec.metadata.get("facet_name", "<unknown>")
        raise OpaqueCallableError(
            f"Cannot save a facet-slice hook (facet {facet_name!r}) at "
            f"save_level={save_level.value!r}: the slice-scatter wrapper is bound to the "
            "captured trace and cannot round-trip through a spec file. Save at "
            "level='audit' for inspection, or re-attach the facet intervention on the "
            "loaded trace."
        )
    helper = hook_spec.helper if hook_spec.helper is not None else None
    hook_value = helper if helper is not None else hook_spec.hook
    return {
        "site_target": _target_spec_to_json(hook_spec.site_target, save_level),
        "hook": _serialize_value(hook_value, save_level, state),
        "helper": _serialize_value(helper, save_level, state),
        "handle": hook_spec.handle,
        "metadata": _jsonish_metadata(hook_spec.metadata),
    }


def _deserialize_intervention_spec(
    data: dict[str, Any],
    tensors: dict[str, torch.Tensor],
    *,
    trust_custom_callables: bool = False,
    allowed_custom_callable_modules: Collection[str] | None = None,
) -> InterventionSpec:
    """Deserialize JSON-safe spec data.

    Parameters
    ----------
    data:
        Spec payload from ``spec.json``.
    tensors:
        Loaded tensor refs.
    trust_custom_callables:
        Execution-time trust for foreign import-ref callables, carried into every
        materialized ``LazyImportRef`` / import-ref helper. Defaults to fail-closed.
    allowed_custom_callable_modules:
        Optional foreign-module allowlist carried alongside the trust flag.

    Returns
    -------
    InterventionSpec
        Runtime intervention spec.
    """

    def _decode(value: Any) -> Any:
        """Decode one value under the load's trust context."""

        return _deserialize_value(
            value,
            tensors,
            trust_custom_callables=trust_custom_callables,
            allowed_custom_callable_modules=allowed_custom_callable_modules,
        )

    spec = InterventionSpec(metadata=dict(data.get("metadata", {})))
    spec.targets = [_target_spec_from_json(item) for item in data.get("targets", [])]
    spec.helper = _decode(data.get("helper"))
    spec.value = _decode(data.get("value"))
    spec.hook = _decode(data.get("hook"))
    for item in data.get("target_value_specs", []):
        spec.target_value_specs.append(
            TargetValueSpec(
                site_target=_target_spec_from_json(item["site_target"]),
                value=_decode(item.get("value")),
                metadata=dict(item.get("metadata", {})),
            )
        )
    for item in data.get("hook_specs", []):
        helper = _decode(item.get("helper"))
        hook = _decode(item.get("hook"))
        spec.hook_specs.append(
            HookSpec(
                site_target=_target_spec_from_json(item["site_target"]),
                hook=hook,
                helper=helper if isinstance(helper, Edit) else None,
                handle=item.get("handle"),
                metadata=dict(item.get("metadata", {})),
            )
        )
    spec.records = [
        _deserialize_fire_record(
            item,
            tensors,
            trust_custom_callables=trust_custom_callables,
            allowed_custom_callable_modules=allowed_custom_callable_modules,
        )
        for item in data.get("records", [])
    ]
    return spec


def _deserialize_fire_record(
    data: dict[str, Any],
    tensors: dict[str, torch.Tensor],
    *,
    trust_custom_callables: bool = False,
    allowed_custom_callable_modules: Collection[str] | None = None,
) -> FireRecord:
    """Deserialize one fire record from JSON-safe data.

    Parameters
    ----------
    data:
        JSON fire-record payload.
    tensors:
        Loaded tensor refs.
    trust_custom_callables:
        Execution-time trust for foreign import-ref callables. Defaults fail-closed.
    allowed_custom_callable_modules:
        Optional foreign-module allowlist.

    Returns
    -------
    FireRecord
        Runtime fire record.
    """

    helper = _deserialize_value(
        data.get("edit"),
        tensors,
        trust_custom_callables=trust_custom_callables,
        allowed_custom_callable_modules=allowed_custom_callable_modules,
    )
    container_path = _deserialize_value(
        data.get("container_path", ()),
        tensors,
        trust_custom_callables=trust_custom_callables,
        allowed_custom_callable_modules=allowed_custom_callable_modules,
    )
    return FireRecord(
        target_label=str(data.get("target_label", "")),
        call_label=data.get("call_label"),
        func_call_id=data.get("func_call_id"),
        container_path=tuple(container_path or ()),
        engine=data.get("engine"),
        edit=helper if isinstance(helper, Edit) else None,
        site_label=data.get("site_label"),
        timing=data.get("timing"),
        signal=data.get("signal"),
        edit_name=data.get("edit_name"),
        seed=data.get("seed"),
        determinism_note=data.get("determinism_note"),
        timestamp=data.get("timestamp"),
        backward_pass_number=data.get("backward_pass_number"),
        call_index=data.get("call_index"),
        grad_kind=data.get("grad_kind"),
        tuple_index=data.get("tuple_index"),
        replaced=data.get("replaced"),
        edge_address=_edge_address_from_json(data.get("edge_address")),
    )


def _edge_address_from_json(value: Any) -> tuple | None:
    """Rebuild a persisted fire-record ``edge_address`` occurrence address.

    WT1 A-IV item 20 (lane A08): the serializer wrote every FireRecord field
    -- ``edge_address`` included -- but the deserializer omitted it, so the
    ``(child_func_call_id, arg_kind, arg_path)`` occurrence address was
    SILENTLY DISCARDED on every load. JSON round-trips tuples as lists, so
    the address canonicalizes back to nested tuples; a present-but-malformed
    value refuses typed instead of degrading to the historical silent drop.
    """

    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ReplayPreconditionError(
            "fire-record edge_address must be a JSON array (the "
            "(child_func_call_id, arg_kind, arg_path) occurrence address), "
            f"got {type(value).__name__}"
        )

    def _canonical(item: Any) -> Any:
        """Rebuild JSON arrays as tuples, recursively (scalars pass through)."""

        if isinstance(item, (list, tuple)):
            return tuple(_canonical(part) for part in item)
        return item

    return tuple(_canonical(part) for part in value)


def _target_spec_to_json(
    target: TargetSpec | FrozenTargetSpec,
    save_level: SaveLevel,
) -> dict[str, Any]:
    """Serialize a target spec.

    Parameters
    ----------
    target:
        Target spec.
    save_level:
        Requested save level.

    Returns
    -------
    dict[str, Any]
        JSON-safe target payload.
    """

    metadata = dict(target.metadata) if isinstance(target.metadata, tuple) else target.metadata
    return {
        "selector_kind": target.selector_kind,
        "selector_value": _selector_value_to_json(target.selector_value, save_level),
        "strict": bool(target.strict),
        "slice_spec": asdict(target.slice_spec) if target.slice_spec is not None else None,
        "metadata": _jsonish_metadata(metadata),
    }


def _target_spec_from_json(data: dict[str, Any]) -> TargetSpec:
    """Deserialize a target spec.

    Parameters
    ----------
    data:
        JSON target payload.

    Returns
    -------
    TargetSpec
        Runtime target spec.
    """

    slice_data = data.get("slice_spec")
    slice_spec = TensorSliceSpec(**slice_data) if isinstance(slice_data, dict) else None
    return TargetSpec(
        selector_kind=data["selector_kind"],
        selector_value=_selector_value_from_json(data.get("selector_value")),
        strict=bool(data.get("strict", False)),
        slice_spec=slice_spec,
        metadata=dict(data.get("metadata", {})),
    )


def _selector_value_to_json(value: Any, save_level: SaveLevel) -> Any:
    """Serialize selector payloads.

    Parameters
    ----------
    value:
        Selector payload.
    save_level:
        Requested save level.

    Returns
    -------
    Any
        JSON-safe payload.
    """

    if isinstance(value, TargetSpec | FrozenTargetSpec):
        return {"__target_spec__": _target_spec_to_json(value, save_level)}
    if isinstance(value, Mapping):
        return {
            "__dict__": {
                str(key): _selector_value_to_json(item, save_level) for key, item in value.items()
            }
        }
    if isinstance(value, tuple):
        return {"__tuple__": [_selector_value_to_json(item, save_level) for item in value]}
    if isinstance(value, list):
        return {"__list__": [_selector_value_to_json(item, save_level) for item in value]}
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if callable(value):
        if save_level != SaveLevel.AUDIT:
            raise OpaqueCallableError(
                f"Callable selector payload {value!r} is non-portable and can only be saved "
                "at audit level."
            )
        return {"__opaque_audit__": {"type": type(value).__name__, "repr": repr(value)}}
    if save_level != SaveLevel.AUDIT:
        raise OpaqueCallableError(
            f"Selector payload {value!r} is non-portable and can only be saved at audit level."
        )
    return {"__opaque_audit__": {"type": type(value).__name__, "repr": repr(value)}}


def _selector_value_from_json(value: Any) -> Any:
    """Deserialize selector payloads.

    Parameters
    ----------
    value:
        JSON-safe selector payload.

    Returns
    -------
    Any
        Runtime selector payload.
    """

    if isinstance(value, dict) and "__target_spec__" in value:
        return _target_spec_from_json(value["__target_spec__"])
    if isinstance(value, dict) and "__opaque_audit__" in value:
        payload = value["__opaque_audit__"]
        return str(payload.get("repr", ""))
    if isinstance(value, dict) and "__dict__" in value:
        return {
            str(key): _selector_value_from_json(item) for key, item in value["__dict__"].items()
        }
    if isinstance(value, dict) and "__tuple__" in value:
        return tuple(_selector_value_from_json(item) for item in value["__tuple__"])
    if isinstance(value, dict) and "__list__" in value:
        return [_selector_value_from_json(item) for item in value["__list__"]]
    if isinstance(value, list):
        return tuple(_selector_value_from_json(item) for item in value)
    if isinstance(value, dict) and "__repr__" in value:
        return value["__repr__"]
    return value
