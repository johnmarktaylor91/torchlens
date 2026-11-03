"""Persistence for TorchLens intervention specifications."""

from __future__ import annotations

import json
import os
import shutil
import uuid
import warnings
from collections.abc import Callable, Collection, Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from .._errors import InvalidArgumentError
from .._io._durability import fsync_dir, fsync_file
from .._io._json import read_bounded
from .._io.manifest import TensorEntry
from .._io.paths import reject_symlink_path
from .._io.tensor_policy import Ok, is_supported_for_save
from .._io.tlspec import _TlSpecWriter
from .._runnable import refuse_poisoned_trace
from .._selection.resolver import _resolve_sites
from .._utils.hashing import sha256_file
from ..intervention._resolver_registry import (
    function_registry_key_from_callable,
    resolve_function_registry_key,
)
from ._helper_fingerprint import refuse_changed_staged_helpers
from ._spec_codec import (
    _SPEC_FILE,
    TLSPEC_FORMAT_VERSION,
    _deserialize_intervention_spec,
    _serialize_intervention_spec,
    _target_spec_to_json,
    _validate_format_version,
    _validate_spec_json_structure,
)
from ._spec_values import (
    SaveLevel,
    _coerce_save_level,
    _SerializedState,
)
from .errors import (
    DirectActivationWriteWarning,
    DirectWriteSaveError,
    MultiMatchWarning,
    ReplayPreconditionError,
    SiteResolutionError,
    UntrustedCallableError,
)
from .helpers import HELPER_REGISTRY_VERSION
from .spec_compat import (
    SpecCompat,
    TargetManifestDiff,
    _resolution_fanout_bound,
    _site_keys_for_table,
    check_spec_compat,
)
from .types import (
    Edit,
    FireRecord,
    FunctionRegistryKey,
    InterventionSpec,
    TargetSpec,
)


def _reject_symlink_path(path: Path, *, context: str) -> None:
    """Reject symlink paths using the intervention replay error contract."""

    reject_symlink_path(
        path,
        context=context,
        exc_type=ReplayPreconditionError,
        message_prefix="Refusing to use symlink",
        trailing_period=False,
    )


_MANIFEST_FILE = "manifest.json"
_README_FILE = "README.md"
_TENSOR_DIR = "tensors"
_BLOB_TENSOR_KEY = "data"


def save_intervention(
    trace: Any,
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
        Test injection hook used to simulate tensor-write crashes.

    Examples
    --------
    >>> import pathlib, tempfile
    >>> edited = tl.trace(model, x, capture=tl.options.CaptureOptions(replayable=True),
    ...     intervene=tl.when(tl.func("relu"), tl.zero_ablate()))
    >>> with tempfile.TemporaryDirectory() as d:
    ...     path = pathlib.Path(d) / "spec.tlspec"
    ...     tl.io.save_intervention(edited, path, level="audit")
    ...     print((path / "spec.json").exists())
    True
    """

    # WT1 A-IV item 20 (lane A08): the intervention-spec door had NO settled-
    # outcome gate, so a FAILED/aborted/unknown capture could export its spec
    # (fire records, target manifest) as if nothing happened. Same N1 authority
    # and ordering as trace.save(): outcome gate first, then the poison gate.
    from .._capture.outcome import require_capture_capability

    require_capture_capability(trace, "save_analysis")

    refuse_poisoned_trace(trace, "intervention export")
    save_level = _coerce_save_level(level)
    _enforce_direct_write_policy(trace, save_level, allow_direct_writes=allow_direct_writes)
    # The recipe must be the one that produced the trace, not a later in-place edit.
    refuse_changed_staged_helpers(trace._intervention_spec, door="save_intervention")
    target_path = Path(path)
    _reject_symlink_path(target_path, context="intervention spec target")
    tmp_path = target_path.parent / f"tmp.{uuid.uuid4().hex}"
    backup_path: Path | None = None
    tensor_entries: list[TensorEntry] = []
    state = _SerializedState(tensor_entries=tensor_entries, tensor_refs={})

    try:
        if target_path.exists() and not overwrite:
            raise FileExistsError(f"Intervention spec path already exists: {target_path}")
        tmp_path.mkdir(parents=True)
        (tmp_path / _TENSOR_DIR).mkdir()

        spec = getattr(trace, "_intervention_spec", None) or InterventionSpec()
        _sync_spec_records_from_log(spec, trace)
        serialized_spec = _serialize_intervention_spec(spec, save_level, state)
        function_keys = _serialize_function_registry_keys(trace)
        target_manifest = _build_target_manifest(trace, spec, save_level)
        _write_tensor_sidecars(
            tmp_path,
            state.tensor_refs,
            tensor_entries,
            write_tensor_blob_fn=_write_tensor_blob_fn,
        )

        # C03 / leverage B7: a spec whose every hook was staged by the
        # capture-time predicate door carries only DERIVED resolved-label
        # TARGETS (the predicate door lowers a fired selector into per-site
        # label targets). Stamp the fact so a reload discloses "the persisted
        # ADDRESSING is derived" instead of presenting derived labels as
        # authored addressing. The stamp keys the DOOR, never the metadata
        # riders: since B7 predicate-door hooks may ALSO carry the rule's
        # expression provenance (spec_where_repr / spec_rule_id), but that
        # rider never re-resolves — the addressing is still derived labels.
        # attach_hooks/do spec-door entries are authored (NOT derived).
        spec_derived = bool(spec.hook_specs) and all(
            dict(hook_spec.metadata).get("created_by") == "intervene_predicate"
            for hook_spec in spec.hook_specs
        )
        spec_json = {
            "format_version": TLSPEC_FORMAT_VERSION,
            "helper_registry_version": HELPER_REGISTRY_VERSION,
            "save_level": save_level.value,
            "executable": save_level != SaveLevel.AUDIT and not _spec_has_opaque(serialized_spec),
            "append_state": _append_state_for_json(trace),
            "target_manifest": target_manifest,
            "helpers": _collect_helpers(serialized_spec),
            "intervention_spec": serialized_spec,
            "function_registry_keys": function_keys,
            "spec_derived": spec_derived,
        }
        _write_json_file(tmp_path / _SPEC_FILE, spec_json)
        _TlSpecWriter.write_intervention_manifest(
            path=tmp_path / _MANIFEST_FILE,
            log=trace,
            spec_json=spec_json,
            tensor_entries=tensor_entries,
            legacy_format_version=TLSPEC_FORMAT_VERSION,
            save_level=save_level.value,
        )
        _write_text_file(tmp_path / _README_FILE, _readme_text(spec_json, tensor_entries))
        # Blob file data is fsynced at write time (fsync_file), but directory
        # ENTRIES must be flushed bottom-up: without fsyncing tensors/ first, a
        # power crash just after publish can leave a durable spec.json whose
        # tensors/*.safetensors entries were lost (R59; siblings use fsync_tree).
        _fsync_directory(tmp_path / _TENSOR_DIR)
        _fsync_directory(tmp_path)
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
    except BaseException:
        if tmp_path.exists():
            shutil.rmtree(tmp_path, ignore_errors=True)
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
    _validate_format_version(data.get("format_version"), path=spec_path)
    _validate_spec_json_structure(data, spec_path)
    manifest = _read_json_file(spec_path / _MANIFEST_FILE)
    manifest_entries = manifest.get("tensor_entries", [])
    if not isinstance(manifest_entries, list):
        raise ReplayPreconditionError(
            f"{spec_path / _MANIFEST_FILE} field 'tensor_entries' must be a list, "
            f"got {type(manifest_entries).__name__}"
        )
    tensor_entries = [TensorEntry.from_dict(entry) for entry in manifest_entries]
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
            # C03 / leverage B7: disclose reloads whose persisted ADDRESSING
            # is derived per-site labels (predicate-door lowering); legacy
            # artifacts without the stamp settle by the same door-keyed
            # derivation over the loaded hooks (an expression-provenance
            # rider never flips the stamp).
            "spec_derived": bool(
                data.get(
                    "spec_derived",
                    bool(spec.hook_specs)
                    and all(
                        dict(hook_spec.metadata).get("created_by") == "intervene_predicate"
                        for hook_spec in spec.hook_specs
                    ),
                )
            ),
        }
    )
    spec.metadata = metadata
    _verify_loaded_function_keys(
        data.get("function_registry_keys", []),
        trust_custom_callables=trust_custom_callables,
        allowed_custom_callable_modules=allowed_custom_callable_modules,
    )
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
                # C03 site-key-primary targets (surgery Build 0a): the join
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
            raise InvalidArgumentError(
                f"Unsupported tensor for intervention save {tensor_id}: {decision}",
                code="intervention_tensor_unsupported",
                remedy="use dense, codec-supported tensors in the intervention spec",
                tensor_id=str(tensor_id),
            )
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
    if contiguous.device.type != "cpu":
        contiguous = contiguous.cpu()
    relative_path = Path(_TENSOR_DIR) / f"{blob_id}.safetensors"
    blob_path = tmp_path / relative_path
    save_file({_BLOB_TENSOR_KEY: contiguous}, str(blob_path))
    fsync_file(blob_path)
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
        sha256=sha256_file(blob_path),
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
        if sha256_file(path) != entry.sha256:
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


def _fsync_directory(path: Path) -> None:
    """Fsync a directory when the platform allows it.

    Failures propagate where directory handles exist; Windows has none and is
    skipped (``fsync_dir(strict=True)``).

    Parameters
    ----------
    path:
        Directory path.
    """

    fsync_dir(path, strict=True)


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
