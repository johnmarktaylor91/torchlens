"""Persistence for TorchLens intervention specifications."""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import asdict, dataclass
from enum import Enum
import json
import os
from pathlib import Path
import shutil
import uuid
import warnings
from typing import Any, Literal

import torch
from safetensors.torch import load_file, save_file

from .._io.manifest import TensorEntry, sha256_of_file
from .._io.tensor_policy import Ok, is_supported_for_save
from .._io.tlspec import _TlSpecWriter
from ..ir.container import DataclassField, DictKey, HFKey, NamedField, TupleIndex
from .errors import (
    DirectActivationWriteWarning,
    DirectWriteInExecutableSaveError,
    GraphShapeMismatchError,
    MultiMatchWarning,
    OpaqueCallableInExecutableSaveError,
    ReplayPreconditionError,
    SiteResolutionError,
    UnserializableDictKeyError,
    UntrustedCallableError,
)
from .helpers import HELPER_REGISTRY_VERSION, helper_from_serialized
from .resolver import (
    function_registry_key_from_callable,
    resolve_function_registry_key,
    resolve_import_ref,
    resolve_sites,
)
from .types import (
    FireRecord,
    FrozenTargetSpec,
    FunctionRegistryKey,
    HelperSpec,
    HookSpec,
    InterventionSpec,
    TargetSpec,
    TargetValueSpec,
    TensorSliceSpec,
)

TLSPEC_FORMAT_VERSION = "2"
SUPPORTED_TLSPEC_FORMAT_VERSIONS = {"1", TLSPEC_FORMAT_VERSION}
_SPEC_FILE = "spec.json"
_MANIFEST_FILE = "manifest.json"
_README_FILE = "README.md"
_TENSOR_DIR = "tensors"
_BLOB_TENSOR_KEY = "data"


class SaveLevel(str, Enum):
    """Supported intervention spec save levels."""

    AUDIT = "audit"
    EXECUTABLE_WITH_CALLABLES = "executable_with_callables"
    PORTABLE = "portable"


@dataclass(frozen=True)
class TargetManifestDiff:
    """Diff between saved target manifest labels and a new model log."""

    matched: list[str]
    new_labels: list[str]
    missing_labels: list[str]
    selector_resolution_diffs: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class SpecCompat:
    """Compatibility result for applying a saved spec to a model log."""

    outcome: Literal["EXACT", "COMPATIBLE_WITH_CONFIRMATION", "FAIL"]
    diff: TargetManifestDiff
    targets_resolve_identically: bool


@dataclass(frozen=True)
class _SerializedState:
    """Internal state accumulated during spec serialization."""

    tensor_entries: list[TensorEntry]
    tensor_refs: dict[str, torch.Tensor]


@dataclass(frozen=True)
class LazyImportRef:
    """Callable import reference that resolves only at execution time.

    The trust context is captured from the load call that materialized this
    reference and carried until execution, so the deferred resolution enforces the
    SAME deny-by-default gate as ``resolve_function_registry_key``: a foreign module
    is never imported under the default ``trust_custom_callables=False`` load.
    """

    import_path: str
    trust_custom_callables: bool = False
    allowed_custom_callable_modules: tuple[str, ...] | None = None

    def __setstate__(self, state: object) -> None:
        """Force fail-closed trust on ANY unpickle reconstruction (self-defense).

        A ``LazyImportRef`` is materialized in-process from a load call with a trust
        context threaded through ``__init__``; it is NEVER a legitimate member of a
        pickled ``metadata.pkl`` graph. If some path (a forged bundle, a
        ``__reduce__`` gadget) reconstructs one via unpickle, its trust MUST NOT come
        from the attacker-controlled pickled fields, or a crafted
        ``LazyImportRef(import_path="os:system", trust_custom_callables=True)`` would
        REDUCE-invoke ``os.system``. Pickle restores frozen-dataclass state by
        bypassing ``__setattr__``, so this hook rebuilds the state with
        ``object.__setattr__`` and HARD-FORCES ``trust_custom_callables=False`` and
        ``allowed_custom_callable_modules=None`` regardless of the pickled values.
        The ``import_path`` is preserved (inert until ``__call__``), but resolution
        can now never be attacker-trusted -- the foreign resolver denies it.

        This is scoped to the UNPICKLE path only: the normal in-process load path
        constructs the reference through ``__init__`` (never ``__setstate__``), so a
        legitimately-trusted intervention-spec load is unaffected.

        Parameters
        ----------
        state:
            Pickled instance state (the frozen dataclass ``__dict__``, or a
            ``(dict, slots)`` 2-tuple), used only to recover ``import_path``.
        """

        payload: object = state
        if isinstance(state, tuple) and state and isinstance(state[0], dict):
            payload = state[0]
        import_path = payload.get("import_path", "") if isinstance(payload, dict) else ""
        object.__setattr__(self, "import_path", import_path if isinstance(import_path, str) else "")
        object.__setattr__(self, "trust_custom_callables", False)
        object.__setattr__(self, "allowed_custom_callable_modules", None)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Resolve and call the referenced object under its captured trust gate.

        Parameters
        ----------
        *args:
            Positional arguments forwarded to the imported callable.
        **kwargs:
            Keyword arguments forwarded to the imported callable.

        Returns
        -------
        Any
            Return value from the imported callable.
        """

        return _resolve_import_ref(
            self.import_path,
            trust_custom_callables=self.trust_custom_callables,
            allowed_custom_callable_modules=self.allowed_custom_callable_modules,
        )(*args, **kwargs)

    def __repr__(self) -> str:
        """Return a stable representation without importing the target.

        Returns
        -------
        str
            Import-reference representation.
        """

        return f"LazyImportRef({self.import_path!r})"


def save_intervention(
    log: Any,
    path: str | Path,
    *,
    level: str | SaveLevel = SaveLevel.EXECUTABLE_WITH_CALLABLES,
    allow_direct_writes: bool = False,
    overwrite: bool = False,
    _write_tensor_blob_fn: Callable[..., TensorEntry] | None = None,
) -> None:
    """Save a model log's intervention recipe to a ``.tlspec`` directory.

    Parameters
    ----------
    log:
        Trace-like object whose ``_intervention_spec`` should be persisted.
    path:
        Destination ``.tlspec`` directory path.
    level:
        Save level: ``"audit"``, ``"executable_with_callables"``, or
        ``"portable"``.
    allow_direct_writes:
        Whether executable saves may proceed when direct out writes were
        detected.
    overwrite:
        Whether an existing target directory may be replaced.
    _write_tensor_blob_fn:
        Test injection hook used to simulate tensor-write crashes.
    """

    from ..runnable import refuse_poisoned_trace

    refuse_poisoned_trace(log, "intervention export")
    save_level = _coerce_save_level(level)
    _enforce_direct_write_policy(log, save_level, allow_direct_writes=allow_direct_writes)
    target_path = Path(path)
    _reject_symlink_path(target_path, context="intervention spec target")
    tmp_path = target_path.parent / f"tmp.{uuid.uuid4().hex}"
    tensor_entries: list[TensorEntry] = []
    state = _SerializedState(tensor_entries=tensor_entries, tensor_refs={})

    try:
        if target_path.exists() and not overwrite:
            raise FileExistsError(f"Intervention spec path already exists: {target_path}")
        tmp_path.mkdir(parents=True)
        (tmp_path / _TENSOR_DIR).mkdir()

        spec = getattr(log, "_intervention_spec", None) or InterventionSpec()
        _sync_spec_records_from_log(spec, log)
        serialized_spec = _serialize_intervention_spec(spec, save_level, state)
        function_keys = _serialize_function_registry_keys(log)
        target_manifest = _build_target_manifest(log, spec, save_level)
        _write_tensor_sidecars(
            tmp_path,
            state.tensor_refs,
            tensor_entries,
            write_tensor_blob_fn=_write_tensor_blob_fn,
        )

        spec_json = {
            "format_version": TLSPEC_FORMAT_VERSION,
            "helper_registry_version": HELPER_REGISTRY_VERSION,
            "save_level": save_level.value,
            "executable": save_level != SaveLevel.AUDIT and not _spec_has_opaque(serialized_spec),
            "append_state": _append_state_for_json(log),
            "target_manifest": target_manifest,
            "helpers": _collect_helpers(serialized_spec),
            "intervention_spec": serialized_spec,
            "function_registry_keys": function_keys,
        }
        _write_json_file(tmp_path / _SPEC_FILE, spec_json)
        _TlSpecWriter.write_intervention_manifest(
            path=tmp_path / _MANIFEST_FILE,
            log=log,
            spec_json=spec_json,
            tensor_entries=tensor_entries,
            legacy_format_version=TLSPEC_FORMAT_VERSION,
            save_level=save_level.value,
        )
        _write_text_file(tmp_path / _README_FILE, _readme_text(spec_json, tensor_entries))
        _fsync_directory(tmp_path)
        if target_path.exists():
            shutil.rmtree(target_path)
        os.rename(tmp_path, target_path)
        _fsync_directory(target_path.parent)
    except Exception:
        if tmp_path.exists():
            shutil.rmtree(tmp_path, ignore_errors=True)
        raise


def load_intervention_spec(
    path: str | Path,
    *,
    trust_custom_callables: bool = False,
    allowed_custom_callable_modules: Collection[str] | None = None,
) -> InterventionSpec:
    """Load an intervention spec from a ``.tlspec`` directory.

    Parameters
    ----------
    path:
        Directory containing ``spec.json`` and tensor sidecars.
    trust_custom_callables:
        Explicit permission to import custom callables recorded in the spec
        when no allowlist is supplied. Enable only for trusted specs.
    allowed_custom_callable_modules:
        Optional allowlist of custom callable module names. When supplied,
        custom imports must be listed even if ``trust_custom_callables=True``.

    Returns
    -------
    InterventionSpec
        Loaded intervention spec with persistence metadata attached.
    """

    spec_path = Path(path)
    _reject_symlink_path(spec_path, context="intervention spec path")
    # Defense-in-depth: the spec ROOT being non-symlink does not stop a symlinked
    # CHILD member (spec.json / manifest.json) from redirecting the read out of
    # the attacker-controlled directory. Mirror the trace loader's per-member
    # symlink rejection before opening either file.
    _reject_symlink_path(spec_path / _SPEC_FILE, context="intervention spec.json")
    _reject_symlink_path(spec_path / _MANIFEST_FILE, context="intervention manifest.json")
    data = _read_json_file(spec_path / _SPEC_FILE)
    _validate_format_version(data.get("format_version"))
    manifest = _read_json_file(spec_path / _MANIFEST_FILE)
    tensor_entries = [TensorEntry.from_dict(entry) for entry in manifest.get("tensor_entries", [])]
    tensors = _load_tensor_refs(spec_path, tensor_entries)
    spec_payload = data["intervention_spec"]
    spec = _deserialize_intervention_spec(
        spec_payload,
        tensors,
        trust_custom_callables=trust_custom_callables,
        allowed_custom_callable_modules=allowed_custom_callable_modules,
    )
    metadata = dict(spec.metadata)
    metadata.update(
        {
            "format_version": data.get("format_version"),
            "helper_registry_version": data.get("helper_registry_version"),
            "save_level": data.get("save_level"),
            "executable": bool(data.get("executable", False)),
            "target_manifest": data.get("target_manifest", []),
            "function_registry_keys": data.get("function_registry_keys", []),
            "append_state": data.get("append_state", {}),
            "loaded_from_tlspec": str(spec_path),
        }
    )
    spec.metadata = metadata
    _verify_loaded_function_keys(
        data.get("function_registry_keys", []),
        trust_custom_callables=trust_custom_callables,
        allowed_custom_callable_modules=allowed_custom_callable_modules,
    )
    return spec


def _validate_format_version(format_version: Any) -> None:
    """Validate an intervention spec format version.

    Parameters
    ----------
    format_version:
        Format version value read from ``spec.json``.

    Returns
    -------
    None
        Raises when the format is unsupported.
    """

    if str(format_version) not in SUPPORTED_TLSPEC_FORMAT_VERSIONS:
        supported = ", ".join(sorted(SUPPORTED_TLSPEC_FORMAT_VERSIONS))
        raise ValueError(
            f"Unsupported intervention .tlspec format_version={format_version!r}; "
            f"expected one of {supported}."
        )


def _append_state_for_json(log: Any) -> dict[str, Any]:
    """Return append provenance fields for tlspec metadata.

    Parameters
    ----------
    log:
        Model log being saved.

    Returns
    -------
    dict[str, Any]
        JSON-safe append state summary.
    """

    append_records = [
        record
        for record in getattr(log, "state_history", [])
        if isinstance(record, dict) and record.get("op") == "append"
    ]
    return {
        "is_appended": bool(getattr(log, "is_appended", False)),
        "append_sequence_id": int(getattr(log, "_append_sequence_id", 0)),
        "append_history": list(getattr(log, "append_history", [])),
        "state_history": append_records,
    }


def check_spec_compat(spec: InterventionSpec, new_log: Any) -> SpecCompat:
    """Check whether a loaded intervention spec targets a new model log.

    Parameters
    ----------
    spec:
        Loaded or in-memory intervention spec.
    new_log:
        Model log to check.

    Returns
    -------
    SpecCompat
        Compatibility classification and target diff.
    """

    target_manifest = list(spec.metadata.get("target_manifest", []))
    graph_hash = getattr(new_log, "graph_shape_hash", None)
    all_saved: set[str] = set()
    all_resolved: set[str] = set()
    selector_diffs: dict[str, dict[str, Any]] = {}
    unresolved = False
    graph_matches = True

    for index, entry in enumerate(target_manifest):
        saved_labels = list(entry.get("resolved_labels", []))
        all_saved.update(saved_labels)
        selector = _target_spec_from_json(entry["selector"])
        selector_key = f"selector_{index}"
        saved_hash = entry.get("graph_shape_hash")
        if saved_hash != graph_hash:
            graph_matches = False
        try:
            resolved_labels = list(resolve_sites(new_log, selector, strict=True).labels())
        except SiteResolutionError as exc:
            selector_diffs[selector_key] = {
                "selector": entry["selector"],
                "saved_labels": saved_labels,
                "resolved_labels": [],
                "error": str(exc),
            }
            unresolved = True
            continue
        all_resolved.update(resolved_labels)
        if resolved_labels != saved_labels:
            selector_diffs[selector_key] = {
                "selector": entry["selector"],
                "saved_labels": saved_labels,
                "resolved_labels": resolved_labels,
            }

    matched = sorted(all_saved & all_resolved)
    new_labels = sorted(all_resolved - all_saved)
    missing_labels = sorted(all_saved - all_resolved)
    targets_identical = not selector_diffs and not new_labels and not missing_labels
    diff = TargetManifestDiff(
        matched=matched,
        new_labels=new_labels,
        missing_labels=missing_labels,
        selector_resolution_diffs=selector_diffs,
    )

    if unresolved or missing_labels:
        outcome: Literal["EXACT", "COMPATIBLE_WITH_CONFIRMATION", "FAIL"] = "FAIL"
    elif targets_identical and graph_matches:
        outcome = "EXACT"
    elif all_saved.issubset(all_resolved) or not graph_matches:
        outcome = "COMPATIBLE_WITH_CONFIRMATION"
    else:
        outcome = "FAIL"

    if outcome == "FAIL" and bool(spec.metadata.get("executable", False)) and not graph_matches:
        raise GraphShapeMismatchError(
            "Saved spec's graph_shape_hash doesn't match target log; refusing to apply at "
            "executable level."
        )
    return SpecCompat(outcome, diff, targets_identical)


def _coerce_save_level(level: str | SaveLevel) -> SaveLevel:
    """Normalize a save-level input.

    Parameters
    ----------
    level:
        String or enum save level.

    Returns
    -------
    SaveLevel
        Normalized enum value.
    """

    return level if isinstance(level, SaveLevel) else SaveLevel(level)


def _reject_symlink_path(path: Path, *, context: str) -> None:
    """Reject symlink paths before reading or writing specs.

    Parameters
    ----------
    path:
        Path to inspect.
    context:
        Human-readable path role.
    """

    if path.is_symlink():
        raise ReplayPreconditionError(f"Refusing to use symlink {context}: {path}")


def _resolve_intervention_tensor_path(spec_path: Path, relative_path: str) -> Path:
    """Resolve one manifest tensor sidecar path under the spec directory.

    This mirrors the anti-traversal / anti-symlink guarantees the portable trace
    loader gets from ``resolve_bundle_blob_path`` + ``_reject_symlink_path`` (see
    ``torchlens/_io/paths.py`` and ``torchlens/_io/bundle.py``). The intervention
    sidecar loader is reached by a DEFAULT ``tl.load`` on an attacker-controlled
    intervention-shaped ``.tlspec`` directory, so an attacker-set
    ``relative_path`` must never read a file outside ``spec_path`` and must never
    follow an in-bundle symlink. The checksum gate is NOT a defense here: the
    attacker also controls ``entry.sha256`` and can make it match the
    out-of-bundle file, so containment must be enforced BEFORE the checksum and
    the safetensors read.

    Parameters
    ----------
    spec_path:
        Trusted intervention spec directory root (already non-symlink checked).
    relative_path:
        Manifest-provided sidecar path relative to ``spec_path``.

    Returns
    -------
    Path
        Absolute, containment-checked, non-symlink sidecar path.

    Raises
    ------
    ReplayPreconditionError
        If ``relative_path`` is absolute, contains ``".."``, resolves outside
        ``spec_path``, or targets an in-bundle symlink.
    """

    candidate_rel = Path(relative_path)
    if candidate_rel.is_absolute():
        raise ReplayPreconditionError(
            f"Intervention spec rejected absolute tensor relative_path {relative_path!r}."
        )
    if ".." in candidate_rel.parts:
        raise ReplayPreconditionError(
            f"Intervention spec rejected parent traversal in tensor relative_path {relative_path!r}."
        )
    candidate = spec_path / candidate_rel
    # Reject an in-bundle symlink FILE (or symlinked final component) before we
    # resolve/read it, so a symlink that points back inside the bundle is also
    # refused rather than silently followed.
    _reject_symlink_path(candidate, context="intervention tensor sidecar")
    # Containment against the spec ROOT (not a resolved ``tensors/`` subdir),
    # so a symlinked intermediate directory that redirects the sidecar outside
    # the spec is caught here: ``candidate.resolve()`` follows the symlink out
    # and ``relative_to`` fails.
    allowed_root = spec_path.resolve()
    resolved = candidate.resolve()
    try:
        resolved.relative_to(allowed_root)
    except ValueError as exc:
        raise ReplayPreconditionError(
            "Intervention spec rejected tensor path traversal outside spec directory: "
            f"{relative_path!r}."
        ) from exc
    return resolved


def _enforce_direct_write_policy(
    log: Any,
    save_level: SaveLevel,
    *,
    allow_direct_writes: bool,
) -> None:
    """Apply the direct-write save policy.

    Parameters
    ----------
    log:
        Model log being saved.
    save_level:
        Requested save level.
    allow_direct_writes:
        Whether executable saves may proceed despite direct writes.
    """

    if not getattr(log, "_has_direct_writes", False):
        return
    if save_level == SaveLevel.AUDIT:
        warnings.warn(
            "Direct out writes are audit-only evidence in saved specs.",
            DirectActivationWriteWarning,
            stacklevel=3,
        )
        return
    if not allow_direct_writes:
        raise DirectWriteInExecutableSaveError(
            "Direct out writes cannot be saved as executable interventions. "
            "Pass allow_direct_writes=True only if the recipe semantics are intentional."
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


def _sync_spec_records_from_log(spec: InterventionSpec, log: Any) -> None:
    """Merge trace-local fire records into an intervention spec ledger.

    Parameters
    ----------
    spec:
        Mutable intervention spec about to be saved.
    log:
        Trace-like object that may hold per-op or backward-call records.

    Returns
    -------
    None
        Mutates ``spec.records`` with de-duplicated records.
    """

    merged: list[FireRecord] = []
    seen: set[tuple[Any, ...]] = set()
    for record in [*getattr(spec, "records", []), *_trace_fire_records(log)]:
        key = _fire_record_key(record)
        if key in seen:
            continue
        seen.add(key)
        merged.append(record)
    spec.records = merged


def _trace_fire_records(log: Any) -> list[FireRecord]:
    """Return all fire records materialized on a trace.

    Parameters
    ----------
    log:
        Trace-like object.

    Returns
    -------
    list[FireRecord]
        Forward and backward fire records found on the trace.
    """

    records: list[FireRecord] = []
    for layer in getattr(log, "layer_list", []) or []:
        records.extend(
            record
            for record in getattr(layer, "interventions", []) or []
            if isinstance(record, FireRecord)
        )
    for grad_fn in getattr(log, "grad_fn_logs", {}).values():
        for call in getattr(getattr(grad_fn, "calls", None), "_list", []):
            records.extend(_flatten_fire_ref(getattr(call, "intervention_fire_ref", None)))
    return records


def _flatten_fire_ref(value: Any) -> list[FireRecord]:
    """Flatten a GradFnCall fire-ref field into records.

    Parameters
    ----------
    value:
        FireRecord, tuple of records, or ``None``.

    Returns
    -------
    list[FireRecord]
        Fire records in the reference.
    """

    if isinstance(value, FireRecord):
        return [value]
    if isinstance(value, tuple):
        return [item for item in value if isinstance(item, FireRecord)]
    return []


def _fire_record_key(record: FireRecord) -> tuple[Any, ...]:
    """Return a stable de-duplication key for a fire record.

    Parameters
    ----------
    record:
        Fire record to identify.

    Returns
    -------
    tuple[Any, ...]
        Key covering direction, site, pass/call position, tuple slot, and
        helper identity.
    """

    return (
        record.direction,
        record.engine,
        record.target_label,
        record.call_label,
        record.site_label,
        record.helper_name,
        _helper_identity(record.helper),
        record.timing,
        record.backward_pass_index,
        record.call_index,
        record.grad_kind,
        record.tuple_index,
    )


def _helper_identity(helper: HelperSpec | None) -> tuple[Any, ...] | None:
    """Return a stable structural identity for a helper spec.

    Parameters
    ----------
    helper:
        Helper spec from a fire record.

    Returns
    -------
    tuple[Any, ...] | None
        Hashable helper identity, or ``None`` when no helper is attached.
    """

    if helper is None:
        return None
    return (
        helper.name,
        helper.kind,
        helper.portability,
        tuple(repr(arg) for arg in helper.args),
        tuple((key, repr(value)) for key, value in helper.kwargs),
        tuple(helper.metadata),
    )


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

    data = asdict(record)
    data["helper"] = _serialize_value(record.helper, save_level, state)
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

    helper = hook_spec.helper if hook_spec.helper is not None else None
    hook_value = helper if helper is not None else hook_spec.hook
    return {
        "site_target": _target_spec_to_json(hook_spec.site_target, save_level),
        "hook": _serialize_value(hook_value, save_level, state),
        "helper": _serialize_value(helper, save_level, state),
        "handle": hook_spec.handle,
        "metadata": _jsonish_metadata(hook_spec.metadata),
    }


# Reserved wrapper-dict keys used by ``_serialize_value`` / ``_deserialize_value`` to
# tag non-plain payloads. On load, ``_deserialize_value`` decides how to interpret a
# JSON object purely by testing for the *presence* of one of these keys, so a plain
# user dict whose key literally equals one of them would be misread (e.g. silently
# reconstructed as a ``HelperSpec`` for ``__opaque_audit__``). To make that impossible,
# ANY dict with a key in this reserved namespace is escaped through the fully-general
# ``__dict_items__`` item-list encoding instead of the plain-object encoding, so a
# genuine user key can never be mistaken for a wrapper tag. The reserved namespace is
# every ``__dunder__`` string, which also future-proofs any wrapper tag added later.
_RESERVED_WRAPPER_KEYS = frozenset(
    {
        "__tensor_ref__",
        "__helper__",
        "__callable__",
        "__output_path_component__",
        "__opaque_audit__",
        "__dict_items__",
        "__tuple_key__",
    }
)


def _is_reserved_wrapper_key(key: Any) -> bool:
    """Return ``True`` if ``key`` lives in the reserved wrapper-dict namespace.

    Any string that both starts and ends with ``"__"`` (a classic dunder) is reserved
    so it can never collide with a serializer sentinel. This covers every current
    wrapper tag (see :data:`_RESERVED_WRAPPER_KEYS`) and any future one.

    Parameters
    ----------
    key:
        Candidate dict key.

    Returns
    -------
    bool
        Whether the key must be escaped rather than emitted as a plain JSON key.
    """

    if not isinstance(key, str):
        return False
    if key in _RESERVED_WRAPPER_KEYS:
        return True
    return len(key) >= 4 and key.startswith("__") and key.endswith("__")


def _serialize_value(value: Any, save_level: SaveLevel, state: _SerializedState) -> Any:
    """Serialize tensors, helpers, callables, and JSON-safe literals.

    Parameters
    ----------
    value:
        Runtime value.
    save_level:
        Requested save level.
    state:
        Serialization tensor state.

    Returns
    -------
    Any
        JSON-safe serialized value.
    """

    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, torch.Tensor):
        tensor_id = f"value_{len(state.tensor_refs)}"
        state.tensor_refs[tensor_id] = value.detach().cpu()
        return {"__tensor_ref__": tensor_id}
    if isinstance(value, HelperSpec):
        return {"__helper__": _serialize_helper(value, save_level, state)}
    if isinstance(value, TupleIndex | DictKey | HFKey | NamedField | DataclassField):
        return _serialize_output_path_component(value, save_level, state)
    if isinstance(value, tuple):
        return [_serialize_value(item, save_level, state) for item in value]
    if isinstance(value, list):
        return [_serialize_value(item, save_level, state) for item in value]
    if isinstance(value, dict):
        # All-``str`` keys round-trip losslessly as a plain JSON object (the common
        # case, on-disk format unchanged) -- UNLESS a key collides with a reserved
        # wrapper tag (``_is_reserved_wrapper_key``), in which case emitting a plain
        # object would let ``_deserialize_value`` misread the user dict as a wrapper
        # (silent corruption for ``__opaque_audit__``, loud crashes for the rest).
        # Any non-``str`` key (int/float/tuple/...) would also be silently corrupted
        # by ``str(key)`` -- JSON objects only allow string keys. Both cases are
        # encoded as an explicit item list that preserves each key's type and value
        # through load. TorchLens never silently stringifies or mis-tags keys
        # (mirrors ``annotate``'s reject-don't-coerce rule).
        if all(type(key) is str for key in value) and not any(
            _is_reserved_wrapper_key(key) for key in value
        ):
            return {key: _serialize_value(item, save_level, state) for key, item in value.items()}
        return {
            "__dict_items__": [
                [_serialize_dict_key(key), _serialize_value(item, save_level, state)]
                for key, item in value.items()
            ]
        }
    if callable(value):
        return {"__callable__": _serialize_callable(value, save_level)}
    return _serialize_opaque(value, save_level)


def _serialize_dict_key(key: Any) -> Any:
    """Serialize a dict key preserving its type through a spec round-trip.

    ``str``/``int``/``float``/``bool``/``None`` survive as native JSON scalars;
    tuples are tagged so they reload as hashable tuples. Any other key type raises
    rather than being silently stringified.

    Parameters
    ----------
    key:
        Runtime dict key.

    Returns
    -------
    Any
        JSON-safe, type-preserving key encoding.
    """

    if key is None or type(key) in (str, int, float, bool):
        return key
    if isinstance(key, tuple):
        return {"__tuple_key__": [_serialize_dict_key(item) for item in key]}
    raise UnserializableDictKeyError(
        f"dict key {key!r} of type {type(key).__name__!r} cannot be preserved through a "
        "spec save/load round-trip; TorchLens refuses to silently stringify it. Use "
        "str/int/float/bool/None keys, or a tuple of those."
    )


def _serialize_output_path_component(
    value: TupleIndex | DictKey | HFKey | NamedField | DataclassField,
    save_level: SaveLevel,
    state: _SerializedState,
) -> dict[str, Any]:
    """Serialize a portable output-container path component.

    Parameters
    ----------
    value:
        Output path component.
    save_level:
        Requested save level.
    state:
        Serialization tensor state.

    Returns
    -------
    dict[str, Any]
        JSON-safe component payload.
    """

    if isinstance(value, TupleIndex):
        return {"__output_path_component__": "tuple_index", "index": value.index}
    if isinstance(value, DictKey):
        return {
            "__output_path_component__": "dict_key",
            "key": _serialize_value(value.key, save_level, state),
        }
    if isinstance(value, HFKey):
        return {
            "__output_path_component__": "hf_key",
            "key": _serialize_value(value.key, save_level, state),
        }
    if isinstance(value, NamedField):
        return {"__output_path_component__": "named_field", "name": value.name}
    return {"__output_path_component__": "dataclass_field", "name": value.name}


def _serialize_helper(
    helper: HelperSpec,
    save_level: SaveLevel,
    state: _SerializedState,
) -> dict[str, Any]:
    """Serialize a helper spec with save-level enforcement.

    Parameters
    ----------
    helper:
        Helper spec.
    save_level:
        Requested save level.
    state:
        Serialization tensor state.

    Returns
    -------
    dict[str, Any]
        Serialized helper payload.
    """

    portability = helper.portability
    if save_level == SaveLevel.PORTABLE and portability != "builtin":
        raise OpaqueCallableInExecutableSaveError(
            f"Portable intervention specs cannot save {portability} helper {helper.name!r}."
        )
    if save_level == SaveLevel.EXECUTABLE_WITH_CALLABLES and portability == "opaque_audit":
        raise OpaqueCallableInExecutableSaveError(
            f"Executable intervention specs cannot save opaque helper {helper.name!r}."
        )
    if portability == "import_ref":
        import_path = dict(helper.metadata).get("import_path") or _import_path_for_callable(
            helper.factory
        )
        return {"portability": "import_ref", "name": helper.name, "import_path": import_path}
    if portability == "opaque_audit":
        return {
            "portability": "opaque_audit",
            "name": helper.name,
            "repr": repr(helper),
            "batch_independent": bool(helper.batch_independent),
            "compatible_with_append": bool(helper.compatible_with_append),
        }

    return {
        "portability": "builtin",
        "name": helper.name,
        "kind": helper.kind,
        "direction": helper.direction,
        "args": [_serialize_value(arg, save_level, state) for arg in helper.args],
        "kwargs": {key: _serialize_value(value, save_level, state) for key, value in helper.kwargs},
        "metadata": _jsonish_metadata(dict(helper.metadata)),
        "batch_independent": bool(helper.batch_independent),
        "compatible_with_append": bool(helper.compatible_with_append),
    }


def _serialize_callable(value: Callable[..., Any], save_level: SaveLevel) -> dict[str, Any]:
    """Serialize a callable as import-ref or audit-only repr.

    Parameters
    ----------
    value:
        Callable to serialize.
    save_level:
        Requested save level.

    Returns
    -------
    dict[str, Any]
        Callable payload.
    """

    import_path = _import_path_for_callable(value)
    if import_path is not None and _callable_round_trips(value, import_path):
        if save_level == SaveLevel.PORTABLE:
            raise OpaqueCallableInExecutableSaveError(
                f"Portable intervention specs cannot save import-ref callable {import_path}."
            )
        return {"portability": "import_ref", "import_path": import_path, "repr": repr(value)}
    if save_level != SaveLevel.AUDIT:
        raise OpaqueCallableInExecutableSaveError(
            f"Callable {value!r} is opaque and can only be saved at audit level."
        )
    return {"portability": "opaque_audit", "repr": repr(value)}


def _serialize_opaque(value: Any, save_level: SaveLevel) -> dict[str, Any]:
    """Serialize a non-JSON object as audit-only data.

    Parameters
    ----------
    value:
        Opaque value.
    save_level:
        Requested save level.

    Returns
    -------
    dict[str, Any]
        Audit-only payload.
    """

    if save_level != SaveLevel.AUDIT:
        raise OpaqueCallableInExecutableSaveError(
            f"Value {value!r} is not portable and can only be saved at audit level."
        )
    return {"__opaque_audit__": {"type": type(value).__name__, "repr": repr(value)}}


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
                helper=helper if isinstance(helper, HelperSpec) else None,
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
        data.get("helper"),
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
        helper=helper if isinstance(helper, HelperSpec) else None,
        site_label=data.get("site_label"),
        timing=data.get("timing"),
        direction=data.get("direction"),
        helper_name=data.get("helper_name"),
        seed=data.get("seed"),
        determinism_note=data.get("determinism_note"),
        timestamp=data.get("timestamp"),
        backward_pass_index=data.get("backward_pass_index"),
        call_index=data.get("call_index"),
        grad_kind=data.get("grad_kind"),
        tuple_index=data.get("tuple_index"),
        replaced=data.get("replaced"),
    )


def _deserialize_value(
    value: Any,
    tensors: dict[str, torch.Tensor],
    *,
    trust_custom_callables: bool = False,
    allowed_custom_callable_modules: Collection[str] | None = None,
) -> Any:
    """Deserialize a value from ``spec.json``.

    Any import-ref callable materialized here captures the load's trust context so
    its deferred resolution enforces the SAME deny-by-default gate as
    ``resolve_function_registry_key``. Defaults are fail-closed: an unthreaded caller
    gets an untrusted (no-foreign-import) reference.

    Parameters
    ----------
    value:
        JSON-decoded value.
    tensors:
        Loaded tensor refs.
    trust_custom_callables:
        Execution-time trust for foreign import-ref callables.
    allowed_custom_callable_modules:
        Optional foreign-module allowlist.

    Returns
    -------
    Any
        Runtime value.
    """

    allowed_tuple: tuple[str, ...] | None = (
        None if allowed_custom_callable_modules is None else tuple(allowed_custom_callable_modules)
    )

    def _decode(item: Any) -> Any:
        """Recurse under the same trust context."""

        return _deserialize_value(
            item,
            tensors,
            trust_custom_callables=trust_custom_callables,
            allowed_custom_callable_modules=allowed_custom_callable_modules,
        )

    def _trusted_import_resolver(import_path: str) -> Callable[..., Any]:
        """Resolve an import ref under this load's trust context."""

        return _resolve_import_ref(
            import_path,
            trust_custom_callables=trust_custom_callables,
            allowed_custom_callable_modules=allowed_custom_callable_modules,
        )

    if isinstance(value, dict) and "__tensor_ref__" in value:
        return tensors[str(value["__tensor_ref__"])]
    if isinstance(value, dict) and "__helper__" in value:
        # Decode a builtin helper's args/kwargs through THIS same full codec so the
        # decoder stays in lockstep with ``_serialize_value``. The narrow
        # ``_decode_jsonish`` fallback only understood ``__tensor_ref__`` and
        # silently returned every other wrapper (``__callable__``/``__opaque_audit__``/
        # ...) as a raw dict, corrupting callable/opaque helper arguments. The import
        # resolver is bound to the load's trust context so an import-ref helper cannot
        # import a foreign module under the default untrusted load.
        return helper_from_serialized(
            value["__helper__"],
            tensor_loader=lambda tensor_id: tensors[tensor_id],
            import_resolver=_trusted_import_resolver,
            value_decoder=_decode,
        )
    if isinstance(value, dict) and "__callable__" in value:
        callable_payload = value["__callable__"]
        if callable_payload["portability"] == "import_ref":
            return LazyImportRef(
                str(callable_payload["import_path"]),
                trust_custom_callables=trust_custom_callables,
                allowed_custom_callable_modules=allowed_tuple,
            )
        return HelperSpec(
            helper_name="opaque_audit",
            portability="opaque_audit",
            metadata=(("repr", callable_payload.get("repr", "")), ("executable", False)),
        )
    if isinstance(value, dict) and "__opaque_audit__" in value:
        payload = value["__opaque_audit__"]
        return HelperSpec(
            helper_name=str(payload.get("type", "opaque_audit")),
            portability="opaque_audit",
            metadata=(("repr", payload.get("repr", "")), ("executable", False)),
        )
    if isinstance(value, dict) and "__output_path_component__" in value:
        return _deserialize_output_path_component(
            value,
            tensors,
            trust_custom_callables=trust_custom_callables,
            allowed_custom_callable_modules=allowed_custom_callable_modules,
        )
    if isinstance(value, dict) and "__dict_items__" in value:
        return {_deserialize_dict_key(key): _decode(item) for key, item in value["__dict_items__"]}
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if isinstance(value, dict):
        return {key: _decode(item) for key, item in value.items()}
    return value


def _deserialize_dict_key(key: Any) -> Any:
    """Reconstruct a dict key encoded by :func:`_serialize_dict_key`.

    Parameters
    ----------
    key:
        JSON-decoded key encoding.

    Returns
    -------
    Any
        Runtime, hashable dict key with its original type restored.
    """

    if isinstance(key, dict) and "__tuple_key__" in key:
        return tuple(_deserialize_dict_key(item) for item in key["__tuple_key__"])
    return key


def _deserialize_output_path_component(
    value: dict[str, Any],
    tensors: dict[str, torch.Tensor],
    *,
    trust_custom_callables: bool = False,
    allowed_custom_callable_modules: Collection[str] | None = None,
) -> TupleIndex | DictKey | HFKey | NamedField | DataclassField:
    """Deserialize a portable output-container path component.

    Parameters
    ----------
    value:
        JSON-safe component payload.
    tensors:
        Loaded tensor refs.
    trust_custom_callables:
        Execution-time trust for foreign import-ref callables. Defaults fail-closed.
    allowed_custom_callable_modules:
        Optional foreign-module allowlist.

    Returns
    -------
    TupleIndex | DictKey | HFKey | NamedField | DataclassField
        Runtime path component.
    """

    def _decode(item: Any) -> Any:
        """Decode a nested key under the same trust context."""

        return _deserialize_value(
            item,
            tensors,
            trust_custom_callables=trust_custom_callables,
            allowed_custom_callable_modules=allowed_custom_callable_modules,
        )

    kind = str(value.get("__output_path_component__"))
    if kind == "tuple_index":
        return TupleIndex(int(value["index"]))
    if kind == "dict_key":
        return DictKey(_decode(value.get("key")))
    if kind == "hf_key":
        return HFKey(_decode(value.get("key")))
    if kind == "named_field":
        return NamedField(str(value["name"]))
    if kind == "dataclass_field":
        return DataclassField(str(value["name"]))
    raise SiteResolutionError(f"Unsupported output path component kind {kind!r}.")


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
            raise OpaqueCallableInExecutableSaveError(
                f"Callable selector payload {value!r} is non-portable and can only be saved "
                "at audit level."
            )
        return {"__opaque_audit__": {"type": type(value).__name__, "repr": repr(value)}}
    if save_level != SaveLevel.AUDIT:
        raise OpaqueCallableInExecutableSaveError(
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


def _build_target_manifest(
    log: Any,
    spec: InterventionSpec,
    save_level: SaveLevel,
) -> list[dict[str, Any]]:
    """Build the saved target manifest for all recipe selectors.

    Parameters
    ----------
    log:
        Source model log.
    spec:
        Intervention spec.
    save_level:
        Requested save level.

    Returns
    -------
    list[dict[str, Any]]
        Manifest entries.
    """

    targets: list[TargetSpec] = []
    targets.extend(spec.targets)
    targets.extend(value_spec.site_target for value_spec in spec.target_value_specs)
    targets.extend(hook_spec.site_target for hook_spec in spec.hook_specs)
    manifest: list[dict[str, Any]] = []
    for target in targets:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", MultiMatchWarning)
                resolved = resolve_sites(log, target, strict=True)
        except SiteResolutionError as exc:
            if "Backward selectors require log_backward()" not in str(
                exc
            ) and "predicate selectors are non-portable in strict mode" not in str(exc):
                raise
            status = (
                "unresolved_nonportable"
                if "predicate selectors are non-portable in strict mode" in str(exc)
                else "unresolved_backward"
            )
            manifest.append(
                {
                    "selector": _target_spec_to_json(target, save_level),
                    "resolved_labels": [],
                    "resolved_status": status,
                    "resolution_error": str(exc),
                    "graph_shape_hash": getattr(log, "graph_shape_hash", None),
                    "_address_normalized": _normalized_address(target),
                }
            )
            continue
        manifest.append(
            {
                "selector": _target_spec_to_json(target, save_level),
                "resolved_labels": list(resolved.labels()),
                "resolved_status": "resolved",
                "graph_shape_hash": getattr(log, "graph_shape_hash", None),
                "_address_normalized": _normalized_address(target),
            }
        )
    return manifest


def _normalized_address(target: TargetSpec) -> str | None:
    """Return normalized module selector data when applicable.

    Parameters
    ----------
    target:
        Target spec.

    Returns
    -------
    str | None
        Normalized module address.
    """

    if target.selector_kind not in {"module", "in_module"}:
        return None
    return str(target.selector_value).strip(".")


def _serialize_function_registry_keys(log: Any) -> list[dict[str, Any]]:
    """Serialize function registry keys from a model log.

    Parameters
    ----------
    log:
        Model log.

    Returns
    -------
    list[dict[str, Any]]
        Function registry key entries.
    """

    entries: list[dict[str, Any]] = []
    for layer in getattr(log, "layer_list", []):
        key = _function_key_for_layer(layer)
        if key is None:
            continue
        entries.append({"layer_label": str(layer.layer_label), "key": asdict(key)})
    return entries


def _function_key_for_layer(layer: Any) -> FunctionRegistryKey | None:
    """Return or infer a layer's function registry key.

    Parameters
    ----------
    layer:
        Layer pass log-like object.

    Returns
    -------
    FunctionRegistryKey | None
        Function registry key or ``None`` for source nodes.
    """

    template = getattr(layer, "args_template", None)
    key = getattr(template, "func_id", None)
    if isinstance(key, FunctionRegistryKey):
        return key
    func = getattr(layer, "func", None)
    if func is None:
        return None
    return function_registry_key_from_callable(func)


def _verify_loaded_function_keys(
    entries: Iterable[dict[str, Any]],
    *,
    trust_custom_callables: bool,
    allowed_custom_callable_modules: Collection[str] | None,
) -> None:
    """Validate resolvable saved keys without importing untrusted foreign code.

    Load-time analysis tolerates a well-formed foreign custom key when its trust
    gate denies resolution. Malformed and otherwise unresolvable keys still fail
    closed. The foreign custom key is resolved only at execution, where the trust
    gate is enforced again.

    Parameters
    ----------
    entries:
        Serialized function key entries.
    trust_custom_callables:
        Whether foreign custom callable imports may be verified when no
        allowlist is supplied.
    allowed_custom_callable_modules:
        Optional custom callable module allowlist.
    """

    for entry in entries:
        key_data = entry.get("key", {})
        key = FunctionRegistryKey(**key_data)
        try:
            resolve_function_registry_key(
                key,
                trust_custom_callables=trust_custom_callables,
                allowed_custom_callable_modules=allowed_custom_callable_modules,
            )
        except UntrustedCallableError:
            # An untrusted custom key can be inspected without importing its module.
            # Resolution for execution applies the same trust gate and still denies
            # foreign code by default.
            continue


def _write_tensor_sidecars(
    tmp_path: Path,
    tensor_refs: dict[str, torch.Tensor],
    tensor_entries: list[TensorEntry],
    *,
    write_tensor_blob_fn: Callable[..., TensorEntry] | None,
) -> None:
    """Write safetensors sidecars for serialized tensors.

    Parameters
    ----------
    tmp_path:
        Temporary spec directory.
    tensor_refs:
        Mapping from tensor ID to tensor payload.
    tensor_entries:
        Manifest entry accumulator.
    write_tensor_blob_fn:
        Optional test injection writer.
    """

    writer = write_tensor_blob_fn or _write_tlspec_tensor_blob
    for tensor_id, tensor in tensor_refs.items():
        decision = is_supported_for_save(tensor, strict=True)
        if not isinstance(decision, Ok):
            raise ValueError(f"Unsupported tensor for intervention save {tensor_id}: {decision}")
        tensor_entries.append(
            writer(
                tmp_path=tmp_path,
                blob_id=tensor_id,
                tensor=tensor,
                kind="intervention_value",
                label=tensor_id,
            )
        )


def _write_tlspec_tensor_blob(
    *,
    tmp_path: Path,
    blob_id: str,
    tensor: torch.Tensor,
    kind: str,
    label: str,
) -> TensorEntry:
    """Write one intervention tensor sidecar.

    Parameters
    ----------
    tmp_path:
        Temporary spec directory.
    blob_id:
        Tensor identifier.
    tensor:
        Tensor payload.
    kind:
        Logical tensor kind.
    label:
        Human-readable label.

    Returns
    -------
    TensorEntry
        Manifest entry.
    """

    contiguous = tensor.contiguous()
    relative_path = Path(_TENSOR_DIR) / f"{blob_id}.safetensors"
    blob_path = tmp_path / relative_path
    save_file({_BLOB_TENSOR_KEY: contiguous}, str(blob_path))
    _fsync_file(blob_path)
    return TensorEntry(
        blob_id=blob_id,
        kind=kind,
        label=label,
        relative_path=relative_path.as_posix(),
        backend="safetensors",
        shape=[int(dim) for dim in contiguous.shape],
        dtype=str(contiguous.dtype).replace("torch.", ""),
        device_at_save=str(tensor.device),
        layout=str(contiguous.layout).replace("torch.", ""),
        bytes=int(contiguous.numel() * contiguous.element_size()),
        sha256=sha256_of_file(blob_path),
        requires_grad=bool(tensor.requires_grad),
    )


def _load_tensor_refs(spec_path: Path, entries: list[TensorEntry]) -> dict[str, torch.Tensor]:
    """Load tensor refs from safetensors sidecars.

    Parameters
    ----------
    spec_path:
        Spec directory.
    entries:
        Tensor manifest entries.

    Returns
    -------
    dict[str, torch.Tensor]
        Loaded tensors by blob ID.
    """

    tensors: dict[str, torch.Tensor] = {}
    for entry in entries:
        path = _resolve_intervention_tensor_path(spec_path, entry.relative_path)
        if sha256_of_file(path) != entry.sha256:
            raise ReplayPreconditionError(f"Tensor sidecar checksum mismatch: {entry.blob_id}")
        tensor = load_file(str(path))[_BLOB_TENSOR_KEY]
        if entry.requires_grad:
            tensor.requires_grad_(True)
        tensors[entry.blob_id] = tensor
    return tensors


def _collect_helpers(spec_payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect helper payloads for the top-level helper index.

    Parameters
    ----------
    spec_payload:
        Serialized intervention spec.

    Returns
    -------
    list[dict[str, Any]]
        Helper payloads.
    """

    helpers: list[dict[str, Any]] = []

    def visit(value: Any) -> None:
        """Append helper leaves while walking serialized data."""

        if isinstance(value, dict) and "__helper__" in value:
            helpers.append(value["__helper__"])
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(spec_payload)
    return helpers


def _spec_has_opaque(spec_payload: dict[str, Any]) -> bool:
    """Return whether serialized spec data contains audit-only opaque payloads.

    Parameters
    ----------
    spec_payload:
        Serialized spec.

    Returns
    -------
    bool
        ``True`` when opaque payloads are present.
    """

    encoded = json.dumps(spec_payload, sort_keys=True)
    return "opaque_audit" in encoded or "__opaque_audit__" in encoded


def _jsonish_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe metadata dictionary.

    Parameters
    ----------
    metadata:
        Runtime metadata.

    Returns
    -------
    dict[str, Any]
        JSON-safe metadata.
    """

    return {str(key): _metadata_value(value) for key, value in metadata.items()}


def _metadata_value(value: Any) -> Any:
    """Serialize one metadata value conservatively.

    Parameters
    ----------
    value:
        Runtime metadata value.

    Returns
    -------
    Any
        JSON-safe value.
    """

    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, tuple | list):
        return [_metadata_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _metadata_value(item) for key, item in value.items()}
    return repr(value)


def _import_path_for_callable(value: Any) -> str | None:
    """Return ``module:qualname`` for a callable when available.

    Parameters
    ----------
    value:
        Callable-like object.

    Returns
    -------
    str | None
        Import path or ``None``.
    """

    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if not module or not qualname or "<locals>" in qualname:
        return None
    return f"{module}:{qualname}"


def _callable_round_trips(value: Callable[..., Any], import_path: str) -> bool:
    """Return whether an import path resolves to the same callable.

    Parameters
    ----------
    value:
        Callable being serialized.
    import_path:
        Candidate import path.

    Returns
    -------
    bool
        ``True`` when the import path resolves identically.
    """

    try:
        # Save time verifies the identity of the user's OWN in-memory callable, whose
        # module is already imported, so trust is granted here. A callable the shared
        # gate refuses as impure (e.g. a fixed-namespace side-effecting op) simply does
        # not round-trip as an import ref and falls back to opaque/audit serialization.
        return _resolve_import_ref(import_path, trust_custom_callables=True) is value
    except (
        AttributeError,
        ImportError,
        ValueError,
        TypeError,
        UntrustedCallableError,
        ReplayPreconditionError,
    ):
        return False


def _resolve_import_ref(
    import_path: str,
    *,
    trust_custom_callables: bool = False,
    allowed_custom_callable_modules: Collection[str] | None = None,
) -> Callable[..., Any]:
    """Resolve a ``module:qualname`` import reference through the shared trust gate.

    Delegates to :func:`torchlens.intervention.resolver.resolve_import_ref` so a
    bundle-supplied import reference obeys the SAME deny-by-default contract as every
    other bundle-reachable callable resolution: fixed torch/operator namespaces and
    TorchLens-owned helpers always resolve (purity-gated), while a genuinely foreign
    module default-denies with :class:`UntrustedCallableError` and is never imported
    unless the caller opted into trust. Defaults are fail-closed.

    Parameters
    ----------
    import_path:
        Import reference in ``module:qualname`` form.
    trust_custom_callables:
        Explicit permission to import a foreign custom callable when no allowlist is
        supplied.
    allowed_custom_callable_modules:
        Optional allowlist of custom callable module names.

    Returns
    -------
    Callable[..., Any]
        Resolved callable.
    """

    return resolve_import_ref(
        import_path,
        trust_custom_callables=trust_custom_callables,
        allowed_custom_callable_modules=allowed_custom_callable_modules,
    )


def _write_json_file(path: Path, data: dict[str, Any]) -> None:
    """Write and fsync canonical JSON.

    Parameters
    ----------
    path:
        File path.
    data:
        JSON payload.
    """

    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_text_file(path: Path, text: str) -> None:
    """Write and fsync text.

    Parameters
    ----------
    path:
        File path.
    text:
        Text payload.
    """

    with path.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _read_json_file(path: Path) -> dict[str, Any]:
    """Read JSON object data.

    Parameters
    ----------
    path:
        JSON file path.

    Returns
    -------
    dict[str, Any]
        Decoded JSON object.
    """

    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ReplayPreconditionError(f"{path} must contain a JSON object")
    return data


def _fsync_file(path: Path) -> None:
    """Fsync an existing file.

    Parameters
    ----------
    path:
        File path.
    """

    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_directory(path: Path) -> None:
    """Fsync a directory when the platform allows it.

    Parameters
    ----------
    path:
        Directory path.
    """

    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _readme_text(spec_json: dict[str, Any], tensor_entries: list[TensorEntry]) -> str:
    """Build the human-readable spec README.

    Parameters
    ----------
    spec_json:
        Serialized spec payload.
    tensor_entries:
        Tensor manifest entries.

    Returns
    -------
    str
        README text.
    """

    return (
        "# TorchLens intervention spec\n\n"
        f"- format_version: {spec_json['format_version']}\n"
        f"- helper_registry_version: {spec_json['helper_registry_version']}\n"
        f"- save_level: {spec_json['save_level']}\n"
        f"- executable: {spec_json['executable']}\n"
        f"- targets: {len(spec_json['target_manifest'])}\n"
        f"- tensor_sidecars: {len(tensor_entries)}\n"
    )


__all__ = [
    "SaveLevel",
    "SpecCompat",
    "TargetManifestDiff",
    "check_spec_compat",
    "load_intervention_spec",
    "resolve_function_registry_key",
    "save_intervention",
]
