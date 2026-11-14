"""Persistence for TorchLens intervention specifications."""

from __future__ import annotations

import json
import os
import shutil
import uuid
import warnings
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .._errors import InvalidArgumentError
from .._io._durability import fsync_tree
from .._io._json import read_bounded
from .._io._staging import begin_staging, settle_failed_staging
from .._io.format3.contract import INTERVENTION_FILE
from .._io.format3.payloads import PayloadStore
from .._io.format3.recipe import (
    RecipeHeader,
    read_recipe,
    recipe_placeholders,
    trace_recipe_context,
)
from .._io.format3.writer import EnvelopeSpec, envelope_fields, write_intervention_recipe
from .._io.format_contract import below_floor_error
from .._io.format_errors import ArtifactError
from .._io.tensor_policy import Ok, is_supported_for_save
from .._io.tlspec import _TlSpecWriter
from .._runnable import refuse_poisoned_trace
from .._selection.resolver import _resolve_sites
from .._utils._record_group_table import record_member
from ..intervention._resolver_registry import (
    function_registry_key_from_callable,
    resolve_function_registry_key,
)
from ._helper_fingerprint import refuse_changed_staged_helpers
from ._rerun_predicate import spec_from_recipe
from ._save_fs_guards import _fsync_directory, _reject_symlink_path
from ._spec_codec import _target_spec_to_json, _validate_format_version
from ._spec_compat import (
    SpecCompat,
    TargetManifestDiff,
    _resolution_fanout_bound,
    _site_keys_for_table,
    check_spec_compat,
)
from ._spec_values import SaveLevel, _coerce_save_level
from ._types import (
    Edit,
    FireRecord,
    FunctionRegistryKey,
    InterventionSpec,
    TargetSpec,
)
from .errors import (
    DirectActivationWriteWarning,
    DirectWriteSaveError,
    MultiMatchWarning,
    OpaqueCallableError,
    ReplayPreconditionError,
    SiteResolutionError,
    UntrustedCallableError,
)
from .hooks import _rebound_facet_request

_MANIFEST_FILE = "manifest.json"
#: The 2.x spec document; its presence without ``intervention.json`` marks a 2.x spec.
_LEGACY_SPEC_FILE = "spec.json"
_BLOB_TENSOR_KEY = "data"


def save_intervention(
    trace: Any,
    path: str | Path,
    *,
    level: str | SaveLevel = SaveLevel.EXECUTABLE_WITH_CALLABLES,
    allow_direct_writes: bool = False,
    overwrite: bool = False,
    _write_tensor_blob_fn: Callable[..., Any] | None = None,
) -> None:
    """Save a model log's intervention recipe as a ``kind="intervention"`` artifact.

    The directory holds ``manifest.json``, ``intervention.json`` (rules and
    built-in edits as typed JSON; a user callable is a placeholder) and the
    recipe's tensors in ``blobs/recipe-<nnnn>.safetensors``.

    Parameters
    ----------
    trace:
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
        Test injection hook, called with ``tmp_path`` and ``chunks`` after the
        recipe tensors are written and before publish (simulates a crash).

    Examples
    --------
    >>> import pathlib, tempfile
    >>> edited = tl.trace(model, x,
    ...     intervene=tl.when(tl.func("relu"), tl.zero_ablate()))
    >>> with tempfile.TemporaryDirectory() as d:
    ...     path = pathlib.Path(d) / "spec.tlspec"
    ...     tl.io.save_intervention(edited, path, level="audit")
    ...     print((path / "intervention.json").exists())
    True
    """

    # The intervention-spec door has a settled-outcome gate so a
    # FAILED/aborted/unknown capture cannot export its spec (fire records,
    # target manifest) as if nothing happened. Same authority and ordering as
    # trace.save(): outcome gate first, then the poison gate.
    from .._capture.outcome import require_capture_capability

    require_capture_capability(trace, "save_analysis")

    refuse_poisoned_trace(trace, "intervention export")
    save_level = _coerce_save_level(level)
    _enforce_direct_write_policy(trace, save_level, allow_direct_writes=allow_direct_writes)
    # The recipe must be the one that produced the trace, not a later in-place edit.
    refuse_changed_staged_helpers(trace._intervention_spec, door="save_intervention")
    target_path = Path(path)
    _reject_symlink_path(target_path, context="intervention spec target")
    tmp_path = target_path.parent / f"{target_path.name}.tmp.{uuid.uuid4().hex}"
    backup_path: Path | None = None

    try:
        if target_path.exists() and not overwrite:
            raise FileExistsError(f"Intervention spec path already exists: {target_path}")
        begin_staging(target_path, tmp_path, overwrite=overwrite)

        spec = getattr(trace, "_intervention_spec", None) or InterventionSpec()
        _sync_spec_records_from_log(spec, trace)
        _refuse_unsupported_recipe_tensors(spec)
        target_manifest = _build_target_manifest(trace, spec, save_level)
        store = PayloadStore(tmp_path)
        context = {
            **trace_recipe_context(trace),
            "function_registry_keys": _serialize_function_registry_keys(trace),
        }
        header = RecipeHeader(save_level=save_level.value, **context)
        intervention_entry, disclosures = write_intervention_recipe(
            spec, tmp_path, store=store, header=header
        )
        chunks = store.close()
        if _write_tensor_blob_fn is not None:
            # Test hook: runs after the recipe and its tensors are on disk, before publish.
            _write_tensor_blob_fn(tmp_path=tmp_path, chunks=chunks)
        recipe = read_bounded(tmp_path / INTERVENTION_FILE)
        if save_level is SaveLevel.PORTABLE:
            _refuse_unportable_recipe(recipe)
        _TlSpecWriter.write_intervention_manifest(
            path=tmp_path / _MANIFEST_FILE,
            log=trace,
            recipe=recipe,
            target_manifest=target_manifest,
            save_level=save_level.value,
            envelope=dict(
                envelope_fields(
                    tmp_path,
                    EnvelopeSpec(
                        "intervention",
                        save_level.value,
                        codecs=store.codecs,
                        disclosures=disclosures,
                    ),
                ),
                intervention=intervention_entry,
            ),
        )
        fsync_tree(tmp_path)
        if target_path.exists():
            # Re-check overwrite at swap time, not just at save start (R59
            # TOCTOU): a concurrent writer could have created ``target_path``
            # after the start-of-save check, and the unconditional swap below
            # would back it aside and then destroy it (the backup is rmtree'd
            # on success). Refuse instead of live-destroying it.
            if not overwrite:
                raise FileExistsError(f"Intervention spec path already exists: {target_path}")
            backup_path = target_path.parent / f"{target_path.name}.bak.{uuid.uuid4().hex}"
            os.rename(target_path, backup_path)
        os.rename(tmp_path, target_path)
        _fsync_directory(target_path.parent)
        if backup_path is not None:
            shutil.rmtree(backup_path, ignore_errors=True)
    except BaseException as exc:
        if backup_path is not None and not target_path.exists() and backup_path.exists():
            try:
                os.rename(backup_path, target_path)
            except OSError as restore_exc:
                # Double fault: the save failed AND the restore failed, so the prior
                # spec is gone from its canonical path but still exists under the
                # backup name. Disclose it so it is recoverable rather than silently
                # stranded (no behavior change on the single-fault restore path).
                warnings.warn(
                    f"Could not restore the previous intervention spec from its backup "
                    f"after a failed save ({restore_exc}). Your prior spec is NOT lost: "
                    f"it remains at {backup_path}. Move it back to {target_path} to "
                    "recover it.",
                    UserWarning,
                    stacklevel=2,
                )
        settle_failed_staging(tmp_path, target_path, overwrite=overwrite, primary=exc)
        raise


def _spec_is_predicate_derived(spec: InterventionSpec) -> bool:
    """Return whether every hook of ``spec`` was staged by the predicate door.

    Such a spec carries only DERIVED resolved-label TARGETS.
    The recipe stamps the fact so a reload discloses "the persisted ADDRESSING is
    derived" instead of presenting derived labels as authored addressing. The
    stamp keys the DOOR, never the metadata riders (spec_where_repr /
    spec_rule_id never re-resolve).
    """

    return bool(spec.hook_specs) and all(
        dict(hook_spec.metadata).get("created_by") == "intervene_predicate"
        for hook_spec in spec.hook_specs
    )


def _refuse_unportable_recipe(recipe: Mapping[str, Any]) -> None:
    """Refuse a portable save whose written recipe needs user code to rebuild.

    Portable means it rebuilds without user code: a callable placeholder (or a
    disclosed, stringified value) leaves the recipe non-executable and breaks
    that promise.
    """

    if recipe.get("executable", False):
        return
    raise OpaqueCallableError(
        "A portable intervention spec must rebuild without user code, but this "
        "recipe holds a user callable (saved only as a placeholder) or a value "
        "the format cannot carry.",
        code="intervention_spec_not_portable",
        remedy=(
            "save at level='audit' for an audit-only record, or stage the rule with built-in edits"
        ),
    )


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
    _reject_symlink_path(spec_path / INTERVENTION_FILE, context="intervention.json")
    _reject_symlink_path(spec_path / _MANIFEST_FILE, context="intervention manifest.json")
    _refuse_2x_spec_layout(spec_path)
    manifest = _read_json_file(spec_path / _MANIFEST_FILE)
    recipe = read_recipe(spec_path)
    spec = _spec_with_recipe_metadata(recipe, spec_path)
    compat = manifest.get("intervention_compat_metadata")
    target_manifest = compat.get("target_manifest", []) if isinstance(compat, dict) else []
    if not isinstance(target_manifest, list):
        raise ArtifactError(
            f"The intervention manifest at {spec_path} carries a target_manifest that is "
            f"not an array ({type(target_manifest).__name__}).",
            code="manifest_schema_invalid",
            remedy="re-save the spec with save_intervention; do not hand-edit the manifest",
        )
    metadata = {**spec.metadata, "target_manifest": target_manifest}
    spec.metadata = metadata
    _verify_loaded_function_keys(
        recipe.get("function_registry_keys", []),
        trust_custom_callables=trust_custom_callables,
        allowed_custom_callable_modules=allowed_custom_callable_modules,
    )
    return spec


def _refuse_2x_spec_layout(spec_path: Path) -> None:
    """Refuse a torchlens 2.x intervention spec (``spec.json``, no ``intervention.json``).

    A 2.x spec is never read: format ``1`` refuses through the spec format table
    and the 2.29-2.36 format ``2`` refuses here, both with the one 2.x error
    (``artifact_version_below_floor``).
    """

    legacy = spec_path / _LEGACY_SPEC_FILE
    if (spec_path / INTERVENTION_FILE).exists() or not legacy.exists():
        return
    _reject_symlink_path(legacy, context="intervention spec.json")
    format_version = _read_json_file(legacy).get("format_version")
    _validate_format_version(format_version, path=spec_path)
    raise below_floor_error(
        observed=f"intervention format_version={format_version}",
        path=str(spec_path),
        code="artifact_version_below_floor",
    )


def _load_trace_recipe_spec(path: str | Path, trace: Any = None) -> InterventionSpec:
    """Return the staged spec a saved trace's ``intervention.json`` describes.

    The trace reader attaches it as the loaded trace's ``_intervention_spec``. A facet
    write is rebound on ``trace`` and refuses typed when it no longer reproduces there;
    ``trace=None`` leaves it unbound, so it refuses to fire.
    """

    spec = load_intervention_spec(path)
    if trace is not None:
        spec.hook_specs = [_rebound_facet_request(trace, hook, path) for hook in spec.hook_specs]
    return spec


def _spec_with_recipe_metadata(recipe: Mapping[str, Any], root: Path) -> InterventionSpec:
    """Rebuild the spec of a decoded recipe and stamp its persistence metadata."""

    spec = spec_from_recipe(recipe)
    spec.metadata = {
        **spec.metadata,
        "format_version": recipe.get("format_version"),
        "helper_registry_version": recipe.get("helper_registry_version"),
        "save_level": recipe.get("save_level"),
        "executable": bool(recipe.get("executable", False)),
        "target_manifest": [],
        "function_registry_keys": recipe.get("function_registry_keys", []),
        "append_state": recipe.get("append_state", {}),
        "loaded_from_tlspec": str(root),
        # Disclose reloads whose persisted ADDRESSING is
        # derived per-site labels (predicate-door lowering).
        "spec_derived": bool(recipe.get("spec_derived", False)),
        "recipe_placeholders": [
            {"position": item.position, "qualname": item.qualname}
            for item in recipe_placeholders(recipe)
        ],
    }
    return spec


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
        "is_appended": bool(log._provenance_is_appended),
        "append_sequence_id": int(getattr(log, "_append_sequence_id", 0)),
        "append_history": list(log._append_history),
        "state_history": append_records,
    }


def _refuse_unsupported_recipe_tensors(spec: InterventionSpec) -> None:
    """Refuse a recipe tensor the payload codec cannot store (sparse, nested, meta).

    Parameters
    ----------
    spec:
        Staged spec about to be written.

    Raises
    ------
    InvalidArgumentError
        ``intervention_tensor_unsupported`` naming the first such tensor.
    """

    stack: list[Any] = [spec.value]
    for hook_spec in spec.hook_specs:
        helper = hook_spec.helper
        if helper is not None:
            stack.extend([helper.args, [item for _, item in helper.kwargs]])
    stack.extend(value_spec.value for value_spec in spec.target_value_specs)
    while stack:
        item = stack.pop()
        if isinstance(item, torch.Tensor):
            decision = is_supported_for_save(item, strict=True)
            if not isinstance(decision, Ok):
                raise InvalidArgumentError(
                    f"Unsupported tensor in the intervention recipe: {decision}",
                    code="intervention_tensor_unsupported",
                    remedy="use dense, codec-supported tensors in the intervention spec",
                )
        elif isinstance(item, list | tuple):
            stack.extend(item)
        elif isinstance(item, dict):
            stack.extend(item.values())


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
        raise DirectWriteSaveError(
            "Direct out writes cannot be saved as executable interventions. "
            "Pass allow_direct_writes=True only if the recipe semantics are intentional."
        )


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
    for layer in log._layer_list or []:
        records.extend(
            record for record in getattr(layer, "edits", []) or [] if isinstance(record, FireRecord)
        )
    for grad_fn in log._grad_fn_logs.values():
        for call in getattr(getattr(grad_fn, "calls", None), "_list", []):
            records.extend(_flatten_fire_ref(getattr(call, "fire_record", None)))
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
        Key covering signal, site, pass/call position, tuple slot, and
        helper identity.
    """

    return (
        record.signal,
        record.engine,
        record.target_label,
        record.call_label,
        record.site_label,
        record.edit_name,
        _helper_identity(record.edit),
        record.timing,
        record.backward_pass_number,
        record.call_index,
        record.grad_kind,
        record.tuple_index,
    )


def _helper_identity(helper: Edit | None) -> tuple[Any, ...] | None:
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
    # Tensor arguments by full content: torch's repr elides large tensors, so
    # two fires of different large vectors would otherwise de-duplicate.
    from ._content_identity import render_value

    return (
        helper.name,
        helper.signal,
        helper.portability,
        tuple(render_value(arg) for arg in helper.args),
        tuple((key, render_value(value)) for key, value in helper.kwargs),
        tuple(helper.metadata),
    )


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
                resolved = _resolve_sites(
                    log,
                    target,
                    strict=True,
                    max_fanout=_resolution_fanout_bound(log),
                )
        except SiteResolutionError as exc:
            if "Backward selectors require backward()" not in str(
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
                    "graph_shape_hash": log._provenance_fingerprint,
                    "_address_normalized": _normalized_address(target),
                }
            )
            continue
        manifest.append(
            {
                "selector": _target_spec_to_json(target, save_level),
                "resolved_labels": list(resolved.labels()),
                # Site-key-primary targets: the join
                # identity is the structural site key; labels are display-only
                # disclosure (ME ordinal rule). Keyless rows (backward sites,
                # legacy captures) record None.
                "resolved_site_keys": _site_keys_for_table(log, resolved),
                "resolved_status": "resolved",
                "graph_shape_hash": log._provenance_fingerprint,
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
    for layer in log._layer_list:
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

    template = record_member(layer, "args_template", None)
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


def _read_json_file(path: Path) -> dict[str, Any]:
    """Read JSON object data from an UNTRUSTED intervention-spec directory.

    A loaded spec directory is attacker-controlled input on the same footing as a
    ``.tlspec`` bundle, so this routes through the ONE bounded reader
    (:mod:`torchlens._io._json`: byte ceiling + non-recursive depth prescan) rather
    than stdlib ``json.load``. Stdlib ``json`` answered a 10,000-deep nested array
    with an untyped ``RecursionError`` escaping ``load_intervention_spec``; the
    bounded reader refuses at the depth boundary and is re-raised here as the
    already-documented typed ``ReplayPreconditionError``.

    Parameters
    ----------
    path:
        JSON file path.

    Returns
    -------
    dict[str, Any]
        Decoded JSON object.

    Raises
    ------
    ReplayPreconditionError
        When the payload is over-size, over-nested, malformed, or not an object.
    """

    try:
        data = read_bounded(path)
    except json.JSONDecodeError as exc:
        raise ReplayPreconditionError(f"{path} is not parsable JSON ({exc})") from exc
    if not isinstance(data, dict):
        raise ReplayPreconditionError(f"{path} must contain a JSON object")
    return data


__all__ = [
    "SaveLevel",
    "SpecCompat",
    "TargetManifestDiff",
    "check_spec_compat",
    "load_intervention_spec",
    "resolve_function_registry_key",
    "save_intervention",
]
