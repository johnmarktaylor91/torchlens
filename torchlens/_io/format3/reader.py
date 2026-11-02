"""Reader of the 3.0 saved format: manifest first, digests, record tables, records.

The order is fixed and every step refuses typed before the next one opens a
file: :func:`read_manifest` (the 2.x test, the manifest schema, the version
window), :func:`validate_artifact` (every listed file's byte count and
SHA-256), the record tables (envelope, column set against the field bindings,
one typed decode per cell through :mod:`.values`), cross-record references
(identity strings and row indices into the tables), record construction
through the record classes' state hooks under the governed load, tensors
through the existing lazy layer, and the load validators unchanged. Nothing
here evaluates, unpickles or imports by a name from the file. Contract:
``docs/maintainers/saved_format_3.md``; design:
``docs/maintainers/saved_format_3_design.md`` (sections 2, 3, 9 and 14).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, Final, cast

import torch

from ..format_contract import BlobRef
from ..format_errors import ArtifactError
from ..paths import resolve_bundle_blobs_dir
from . import contract
from .validate import (
    BELOW_FLOOR,
    TLSPEC_KEY,
    artifact_error,
    check_version_window,
    integrity_error,
    is_int,
    manifest_schema_errors,
    pre3_error,
    read_json_file,
    read_record_table,
    record_class,
    refuse_unknown_columns,
    symlink_guard,
    verify_file_bytes,
)
from .values import (
    CELL_CODE,
    CallablePlaceholder,
    Decoder,
    PendingRecord,
    TensorRefSink,
    build_component,
    cell_error,
)

__all__ = [
    "load_bundle",
    "load_intervention_recipe",
    "load_trace",
    "read_manifest",
    "validate_artifact",
]

_CODES: Final = contract.REFUSAL_CODES
_CORRUPT: Final = _CODES["record_file_corrupt"]
#: Root entries that mark an artifact written by torchlens 2.x (pickle carrier).
PRE3_ROOT_MARKERS: Final[tuple[str, ...]] = ("metadata.pkl", "bundle.json")
_MEMBER_DIR = re.compile(r"^members/(\d+)-([A-Za-z0-9_.-]+)$")
#: The string a record of each kind is addressed by (design section 3); kinds absent here
#: are referenced by row index only.
IDENTITY_FIELDS: Final[dict[str, str]] = {
    "op": "label",
    "layer": "layer_label",
    "param": "address",
    "module": "address",
    "module_call": "label",
    "buffer": "address",
    "grad_fn": "label",
    "aten_op": "label",
}
#: Persisted fields whose live value holds record objects: field -> target record kind.
REFERENCE_FIELDS: Final[dict[str, dict[str, str]]] = {
    "trace": {
        "_layer_list": "op",
        "_layer_dict_main_keys": "op",
        "_layer_dict_all_keys": "op",
        "_orphan_logs": "op",
        "_layer_logs": "layer",
        "_grad_fn_logs": "grad_fn",
        "_backward_pass_logs": "backward_pass",
        "_source_frames": "func_call_location",
    },
    "op": {"_param_logs": "param", "code_context": "func_call_location"},
    "layer": {"_param_logs": "param"},
    "buffer": {"versions": "op"},
    "grad_fn": {"calls": "grad_fn_call"},
    "backward_pass": {"grad_fn_calls": "grad_fn_call"},
    "module_call": {"code_context": "func_call_location"},
}
#: The same for the nested value objects (by bare class name) that hold records.
COMPONENT_REFERENCE_FIELDS: Final[dict[str, dict[str, str]]] = {
    "ParamAccessor": {"_dict": "param", "_list": "param"},
    "ModuleCallAccessor": {"_dict": "module_call", "_list": "module_call"},
    "ModuleAccessor": {
        "_dict": "module",
        "_list": "module",
        "_alias_dict": "module",
        "_pass_dict": "module_call",
    },
    "OpAccessor": {"_dict": "op", "_list": "op"},
    "LayerAccessor": {"_dict": "layer", "_list": "layer"},
    "GradFnCallAccessor": {"_dict": "grad_fn_call", "_list": "grad_fn_call"},
    "HookInfo": {"source_location": "func_call_location"},
    "_PrimitiveOpProfile": {"primitive_ops": "aten_op"},
    "GradientRecord": {"owner": "op"},
}
#: Construction order: a record's state hook runs after the records it holds.
LOAD_ORDER: Final[tuple[str, ...]] = (
    "func_call_location",
    "param",
    "grad_fn_call",
    "grad_fn",
    "backward_pass",
    "aten_op",
    "module_call",
    "module",
    "op",
    "layer",
    "buffer",
)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def read_manifest(path: str | Path) -> dict[str, Any]:
    """Read and validate ``manifest.json``; the first door, before any other file is opened.

    Raises
    ------
    ArtifactTooOldError
        ``artifact_version_below_floor``: a 2.x artifact (no ``format`` key, a
        ``metadata.pkl`` or ``bundle.json`` at the root, or a version below the floor).
    ArtifactError
        ``manifest_missing``, ``manifest_unreadable``, ``manifest_not_json_object``,
        ``manifest_schema_invalid``, ``artifact_save_level_invalid``,
        ``artifact_version_above_runtime`` or ``load_path_symlink_rejected``.
    """

    root = Path(path)
    symlink_guard(root, "artifact path")
    if not root.is_dir():
        raise artifact_error(
            f"Artifact path {root} is not a directory.",
            code=_CODES["manifest_missing"],
            remedy="pass the directory trace.save() wrote",
            file_path=str(root),
        )
    manifest_path = root / contract.MANIFEST_FILE
    symlink_guard(manifest_path, "manifest")
    for marker in PRE3_ROOT_MARKERS:
        if (root / marker).exists():
            raise pre3_error(root, f"{marker} at the artifact root", code=BELOW_FLOOR)
    manifest, _ = read_json_file(manifest_path, code=_CODES["manifest_unreadable"], what="manifest")
    if not isinstance(manifest, dict):
        raise artifact_error(
            f"Artifact manifest {manifest_path} is not a JSON object.",
            code=_CODES["manifest_not_object"],
            remedy="re-save the trace with trace.save()",
            file_path=str(manifest_path),
        )
    if "format" not in manifest:
        raise pre3_error(root, "manifest.json without a format key", code=BELOW_FLOOR)
    errors = manifest_schema_errors(manifest)
    if errors:
        shown = "; ".join(errors[:8]) + (f" (+{len(errors) - 8} more)" if len(errors) > 8 else "")
        raise artifact_error(
            f"Artifact manifest {manifest_path} fails the 3.0 manifest schema: {shown}.",
            code=_CODES["manifest_invalid"],
            remedy="re-save the trace with trace.save(); the manifest was edited or corrupted",
            file_path=str(manifest_path),
            errors=tuple(errors),
        )
    check_version_window(manifest, root)
    if manifest["save_level"] not in contract.SAVE_LEVELS:
        raise artifact_error(
            f"Artifact manifest {manifest_path} names save_level {manifest['save_level']!r}; "
            f"this runtime knows {contract.SAVE_LEVELS}.",
            code=_CODES["save_level_unknown"],
            remedy="re-save the trace with trace.save() at a known level",
            file_path=str(manifest_path),
        )
    return manifest


def validate_artifact(path: str | Path) -> dict[str, Any]:
    """Validate the manifest and every file it lists; return the manifest.

    Every listed file must exist under the root with the manifest's byte count
    and SHA-256 (``bundle_metadata_integrity_refused`` otherwise). Files not
    listed are never read.
    """

    root = Path(path)
    manifest = read_manifest(root)
    for relpath, entry in manifest["files"].items():
        verify_file_bytes(root, relpath, entry)
    return manifest


# ---------------------------------------------------------------------------
# Record tables
# ---------------------------------------------------------------------------


class _Artifact:
    """One trace artifact's decoded tables and the record objects built from them."""

    def __init__(self, root: Path, manifest: Mapping[str, Any]) -> None:
        self.root = root
        self.manifest = manifest
        self.version: int = manifest[TLSPEC_KEY]
        self.sink = TensorRefSink(codecs=tuple(manifest["codecs"]), files=manifest["files"])
        self.decoder = Decoder(self.sink, defer_records=True)
        self.tables: dict[str, list[dict[str, Any]]] = {}
        self.objects: dict[str, list[Any]] = {}
        self.identities: dict[str, dict[str, int]] = {}

    # -- decode -------------------------------------------------------------

    def decode_tables(self) -> None:
        from ..._data_classes._state_adapter import state_new

        for kind in self.manifest["records"]:
            raw_rows = read_record_table(self.root, self.manifest, kind)
            if raw_rows:
                refuse_unknown_columns(kind, raw_rows[0], declared_version=self.version)
            self.tables[kind] = [
                {
                    name: self.decoder.decode(cell, at=f"{kind}[{index}].{name}")
                    for name, cell in row.items()
                }
                for index, row in enumerate(raw_rows)
            ]
            cls = record_class(kind)
            self.objects[kind] = [state_new(cls) for _ in raw_rows]
            self._index_identities(kind)

    def _index_identities(self, kind: str) -> None:
        field = IDENTITY_FIELDS.get(kind)
        if field is None:
            return
        index: dict[str, int] = {}
        for row_number, row in enumerate(self.tables[kind]):
            identity = row.get(field)
            if not isinstance(identity, str):
                raise cell_error(
                    f"{kind}[{row_number}].{field}",
                    "the identity column must hold a string",
                    code=CELL_CODE,
                )
            if identity in index:
                raise cell_error(
                    f"{kind}[{row_number}].{field}",
                    f"identity {identity!r} repeats",
                    code=CELL_CODE,
                )
            index[identity] = row_number
        self.identities[kind] = index

    # -- references ---------------------------------------------------------

    def resolve_references(self) -> None:
        for kind, rows in self.tables.items():
            targets = REFERENCE_FIELDS.get(kind, {})
            for row_number, row in enumerate(rows):
                for name in list(row):
                    at = f"{kind}[{row_number}].{name}"
                    row[name] = self._settle(row[name], at, targets.get(name))

    def _settle(self, value: Any, at: str, target: str | None) -> Any:
        """Resolve references under ``value`` and build its pending value objects."""

        if isinstance(value, PendingRecord):
            fields = COMPONENT_REFERENCE_FIELDS.get(value.cls.__name__, {})
            settled = {
                name: self._settle(item, f"{at}.{name}", fields.get(name))
                for name, item in value.fields.items()
            }
            return build_component(value.cls, settled)
        if isinstance(value, list):
            return [self._settle(item, f"{at}[{i}]", target) for i, item in enumerate(value)]
        if isinstance(value, tuple):
            return tuple(self._settle(item, f"{at}[{i}]", target) for i, item in enumerate(value))
        if isinstance(value, (set, frozenset)):
            return type(value)(self._settle(item, at, target) for item in value)
        if isinstance(value, dict):
            return {key: self._settle(item, f"{at}.{key}", target) for key, item in value.items()}
        if target is None or value is None:
            return value
        return self._lookup(value, at, target)

    def _lookup(self, reference: Any, at: str, target: str) -> Any:
        rows = self.objects.get(target)
        if rows is None:
            raise cell_error(
                at,
                f"references the {target} table, which the artifact does not carry",
                code=CELL_CODE,
            )
        if is_int(reference):
            if 0 <= reference < len(rows):
                return rows[reference]
            raise cell_error(
                at, f"row index {reference} is outside the {target} table", code=CELL_CODE
            )
        if isinstance(reference, str):
            index = self.identities.get(target, {}).get(reference)
            if index is not None:
                return rows[index]
            raise cell_error(
                at, f"{reference!r} names no row of the {target} table", code=CELL_CODE
            )
        raise cell_error(
            at, f"a {type(reference).__name__} is not a {target} reference", code=CELL_CODE
        )

    # -- construction -------------------------------------------------------

    def build_records(self) -> None:
        """Run each record's state hook, leaves first, under the governed load."""

        from ..state_contract import governed_artifact_load

        with governed_artifact_load():
            for kind in LOAD_ORDER:
                for obj, row in zip(
                    self.objects.get(kind, ()), self.tables.get(kind, ()), strict=True
                ):
                    _restore_record(obj, row, version=self.version)

    def trace_state(self) -> dict[str, Any]:
        """The scrubbed-state-shaped dict the existing rehydrate path restores."""

        state = dict(self.tables["trace"][0])
        state[TLSPEC_KEY] = self.version
        accessor = state.pop("_io_module_accessor_state", None)
        if accessor is None and "module" in self.objects:
            accessor = self._module_accessor_state()
        state["_io_module_accessor_state"] = accessor
        return state

    def _module_accessor_state(self) -> Any:
        from ..._data_classes.module import ModuleAccessor

        modules = self.objects["module"]
        module_dict = {
            row["address"]: obj for row, obj in zip(self.tables["module"], modules, strict=True)
        }
        calls = self.objects.get("module_call", [])
        pass_dict = {
            row["label"]: obj
            for row, obj in zip(self.tables.get("module_call", []), calls, strict=True)
        }
        return build_component(
            ModuleAccessor,
            {
                "_dict": module_dict,
                "_list": list(modules),
                "_pass_dict": pass_dict,
                "_alias_dict": {},
            },
        )

    def manifest_for_rehydrate(self) -> dict[str, Any]:
        return {**self.manifest, "tensors": list(self.sink.entries.values())}


def _restore_record(obj: Any, row: Mapping[str, Any], *, version: int) -> None:
    from ..._data_classes._state_adapter import state_restore

    state = dict(row)
    state[TLSPEC_KEY] = version
    setstate = getattr(type(obj), "__setstate__", None)
    if setstate is not None and setstate is not getattr(object, "__setstate__", None):
        setstate(obj, state)
        return
    state.pop(TLSPEC_KEY)
    state_restore(obj, state)


# ---------------------------------------------------------------------------
# Doors
# ---------------------------------------------------------------------------


def _require_kind(manifest: Mapping[str, Any], root: Path, expected: str) -> None:
    if manifest["kind"] != expected:
        raise artifact_error(
            f"Artifact {root} is a {manifest['kind']!r} artifact, not {expected!r}.",
            code=_CODES["kind_mismatch"],
            remedy=f"load it with the {manifest['kind']} door or pass a {expected} artifact",
            file_path=str(root),
        )


def load_trace(
    path: str | Path,
    *,
    lazy: bool = False,
    map_location: str | torch.device = "cpu",
    materialize_nested: bool = False,
    payload_hints: Any = None,
) -> Any:
    """Load one ``kind: trace`` artifact of the 3.0 saved format.

    Every record table is decoded and every reference resolved before any
    record's state hook runs; the hooks run under the governed load, so the
    load validators run exactly as they do for in-process state.
    """

    from ..rehydrate import rehydrate_trace
    from .validate import check_record_anchors, validate_records

    root = Path(path)
    manifest = validate_artifact(root)
    _require_kind(manifest, root, "trace")
    artifact = _Artifact(root, manifest)
    artifact.decode_tables()
    artifact.resolve_references()
    validate_records(artifact.tables, manifest=manifest, root=root)
    artifact.build_records()
    state = artifact.trace_state()
    layer_list = state.get("_layer_list")
    persisted_rows = len(layer_list) if isinstance(layer_list, list) else None
    blobs_dir = resolve_bundle_blobs_dir(root) if (root / contract.BLOBS_DIR).is_dir() else None
    trace = rehydrate_trace(
        state,
        artifact.manifest_for_rehydrate(),
        root,
        lazy=lazy,
        map_location=map_location,
        materialize_nested=materialize_nested,
        payload_hints=payload_hints,
        resolved_blobs_dir=blobs_dir,
    )
    check_record_anchors(manifest, persisted_rows, root)
    _finish_loaded_trace(trace, root, manifest, artifact)
    return trace


def _finish_loaded_trace(
    trace: Any, root: Path, manifest: Mapping[str, Any], artifact: _Artifact
) -> None:
    """The post-rehydrate steps every loaded trace gets (the 2.x path's, unchanged)."""

    from ..._capture._episode_ledger_anchors import reanchor_loaded_episode_evidence
    from ..bundle import _reanchor_visualizer_paths
    from ..runnable_load import attach_sparse_run_readiness

    reanchor_loaded_episode_evidence(trace)
    _reanchor_visualizer_paths(trace, root)
    trace._loaded_from_bundle = True
    trace._source_bundle_manifest_sha256 = hashlib.sha256(
        (root / contract.MANIFEST_FILE).read_bytes()
    ).hexdigest()
    trace._source_bundle_path = root
    trace._source_bundle_created_at = manifest["created_at"]
    trace._source_bundle_provenance = manifest.get("provenance")
    attach_sparse_run_readiness(trace, None)
    disclosures = manifest.get("disclosures")
    if disclosures is not None:
        _attach_advisory(trace, disclosures)
    if contract.INTERVENTION_FILE in manifest["files"]:
        trace._intervention_spec = _read_recipe(root, manifest, artifact)


def _attach_advisory(trace: Any, disclosures: Mapping[str, Any]) -> None:
    """Attach the ``unencodable_values_stringified`` advisory row (design section 5)."""

    annotations = getattr(trace, "annotations", None)
    if not isinstance(annotations, dict):
        annotations = {}
        trace.annotations = annotations
    rows = annotations.setdefault("capture_advisories", [])
    first = disclosures["first"]
    rows.append(
        {
            "kind": "unencodable_values_stringified",
            "count": disclosures["count"],
            "first_location": first[0] if first else None,
            "message": (
                f"{disclosures['count']} value(s) had no typed encoding and were saved as "
                "their repr text; the record cells name each one"
            ),
        }
    )


def load_bundle(path: str | Path, **options: Any) -> Any:
    """Load one ``kind: bundle`` artifact: ``members/<index>-<name>/`` plus ``relations.json``."""

    from ...bundle import Bundle

    root = Path(path)
    manifest = validate_artifact(root)
    _require_kind(manifest, root, "bundle")
    members = [
        (name, load_trace(member_root, **options))
        for name, member_root in _iter_members(root, manifest)
    ]
    relations: Any = None
    baseline: Any = None
    if "relations.json" in manifest["files"]:
        data, _ = read_json_file(root / "relations.json", code=_CORRUPT, what="relations")
        if not isinstance(data, dict) or not isinstance(data.get("relations", []), list):
            raise integrity_error(
                root / "relations.json",
                "relations.json must be {relations: [...], baseline?}",
                code=_CORRUPT,
            )
        relations = data.get("relations") or None
        baseline = data.get("baseline")
    bundle = Bundle(members, relations=relations, baseline=baseline)
    bundle._source_bundle_path = root
    return bundle


def _iter_members(root: Path, manifest: Mapping[str, Any]) -> Iterator[tuple[str, Path]]:
    """Members in index order, from the manifest's file listing."""

    found: dict[int, str] = {}
    for relpath in manifest["files"]:
        parts = relpath.split("/")
        if parts[0] != "members":
            continue
        match = _MEMBER_DIR.match("/".join(parts[:2]))
        if match is None:
            raise integrity_error(
                root / relpath,
                "bundle members must be members/<index>-<name>/",
                code=_CORRUPT,
            )
        index, name = int(match.group(1)), match.group(2)
        if found.setdefault(index, name) != name:
            raise integrity_error(root / relpath, f"member index {index} repeats", code=_CORRUPT)
    for index in sorted(found):
        member_root = root / "members" / f"{index}-{found[index]}"
        symlink_guard(member_root, "bundle member")
        yield found[index], member_root


# ---------------------------------------------------------------------------
# The intervention recipe (design section 8)
# ---------------------------------------------------------------------------

_RECIPE_FORMAT: Final = "torchlens.intervention_recipe"
_RECIPE_KEYS: Final = frozenset(
    {
        "format", "format_version", "helper_registry_version", "save_level", "executable",
        "spec_derived", "append_state", "rules", "helpers", "function_registry_keys",
    }
)  # fmt: skip
_RULE_KEYS: Final = frozenset({"index", "direction", "selector", "edit", "metadata"})
_EDIT_KEYS: Final = frozenset(
    {
        "name", "args", "kwargs", "portability", "signal", "is_batch_independent",
        "is_compatible_with_append", "metadata", "selection_recipe", "callable",
    }
)  # fmt: skip


def load_intervention_recipe(path: str | Path) -> Any:
    """Load the intervention recipe (``intervention.json``) of a trace or intervention artifact.

    Built-in edits come back through the helper registry (their factories
    rebuilt from name, args and kwargs); a callable placeholder loads as an
    audit-only edit and the recipe's ``placeholders`` list names it, so every
    rerun door refuses with ``run_intervention_spec_not_persisted``.
    """

    root = Path(path)
    manifest = validate_artifact(root)
    if manifest["kind"] not in ("trace", "intervention"):
        _require_kind(manifest, root, "intervention")
    if contract.INTERVENTION_FILE not in manifest["files"]:
        raise artifact_error(
            f"Artifact {root} carries no intervention recipe.",
            code=_CODES["kind_mismatch"],
            remedy="save the trace with a staged intervention, or save_intervention(...) the spec",
            file_path=str(root),
        )
    return _read_recipe(root, manifest, _Artifact(root, manifest))


def _recipe_error(root: Path, detail: str, *, code: str) -> ArtifactError:
    return integrity_error(root / contract.INTERVENTION_FILE, detail, code=code)


def _read_recipe(root: Path, manifest: Mapping[str, Any], artifact: _Artifact) -> Any:
    from ...intervention.types import InterventionSpec

    target = root / contract.INTERVENTION_FILE
    data, raw = read_json_file(target, code=_CORRUPT, what="recipe")
    listed = manifest["files"][contract.INTERVENTION_FILE]
    if len(raw) != listed["bytes"] or hashlib.sha256(raw).hexdigest() != listed["sha256"]:
        raise _recipe_error(root, "bytes differ from the manifest's digest", code=_CORRUPT)
    if not isinstance(data, dict) or set(data) != _RECIPE_KEYS:
        raise _recipe_error(
            root, f"the recipe carries exactly {sorted(_RECIPE_KEYS)}", code=_CORRUPT
        )
    if data["format"] != _RECIPE_FORMAT or data["format_version"] != contract.FORMAT_MAJOR:
        raise _recipe_error(root, f"format must be {_RECIPE_FORMAT} version 3", code=_CORRUPT)
    if not isinstance(data["rules"], list) or not isinstance(data["executable"], bool):
        raise _recipe_error(root, "rules must be an array, executable a boolean", code=_CORRUPT)
    if data["save_level"] not in contract.SAVE_LEVELS or not isinstance(data["append_state"], dict):
        raise _recipe_error(root, "save_level unknown or append_state not an object", code=_CORRUPT)
    spec = InterventionSpec(metadata={})
    placeholders: list[tuple[int | str, str]] = []
    for position, rule in enumerate(data["rules"]):
        placeholders.extend(_restore_rule(spec, rule, position, artifact, root))
    spec.metadata["recipe"] = {
        "executable": data["executable"] and not placeholders,
        "spec_derived": data["spec_derived"],
        "append_state": data["append_state"],
        "helpers": data["helpers"],
        "function_registry_keys": data["function_registry_keys"],
        "helper_registry_version": data["helper_registry_version"],
        "placeholders": placeholders,
    }
    return spec


def _restore_rule(
    spec: Any, rule: Any, position: int, artifact: _Artifact, root: Path
) -> list[tuple[int | str, str]]:
    """Stage one rule on ``spec`` as a sticky hook; return the placeholders it carries."""

    if not isinstance(rule, dict) or set(rule) != _RULE_KEYS or rule["index"] != position:
        raise _recipe_error(root, f"rule {position} must carry {sorted(_RULE_KEYS)}", code=_CORRUPT)
    if rule["direction"] not in ("forward", "backward") or not isinstance(rule["metadata"], dict):
        raise _recipe_error(root, f"rule {position} direction or metadata mistyped", code=_CORRUPT)
    selector = _decode_selector(rule["selector"], root, position)
    edit, found = _decode_edit(rule["edit"], artifact, root, position)
    metadata = {
        **artifact.decoder.decode(rule["metadata"], at=f"recipe.rules[{position}].metadata"),
        "direction": rule["direction"],
        "rule_index": position,
        "loaded_from_recipe": True,
    }
    hook = edit.factory() if edit.factory is not None else edit
    spec.add_hook(selector, hook, helper=edit, handle=f"recipe-{position}", metadata=metadata)
    return found


def _decode_selector(selector: Any, root: Path, position: int) -> Any:
    from ...intervention._spec_codec import _target_spec_from_json

    if not isinstance(selector, dict) or not isinstance(selector.get("selector_kind"), str):
        raise _recipe_error(
            root, f"rule {position} selector is not a selector object", code=_CORRUPT
        )
    try:
        return _target_spec_from_json(selector)
    except (KeyError, TypeError, ValueError) as exc:
        raise _recipe_error(
            root, f"rule {position} selector is malformed: {exc}", code=_CORRUPT
        ) from exc


def _decode_edit(
    edit: Any, artifact: _Artifact, root: Path, position: int
) -> tuple[Any, list[tuple[int | str, str]]]:
    """Rebuild a built-in edit through the helper registry; a placeholder loads audit-only."""

    from ...intervention.helpers import rebuild_builtin_helper
    from ...intervention.types import Edit

    if not isinstance(edit, dict) or not set(edit) <= _EDIT_KEYS or "name" not in edit:
        raise _recipe_error(root, f"rule {position} edit is not an edit object", code=_CORRUPT)
    at = f"recipe.rules[{position}].edit"
    fields = {key: artifact.decoder.decode(value, at=f"{at}.{key}") for key, value in edit.items()}
    name, portability = fields["name"], fields.get("portability", "builtin")
    if not isinstance(name, str) or portability not in ("builtin", "callable_placeholder"):
        raise _recipe_error(
            root, f"rule {position} edit name or portability unknown", code=_CORRUPT
        )
    args = tuple(_materialize(fields.get("args", ()), artifact, root))
    kwargs = dict(_materialize(fields.get("kwargs", {}), artifact, root))
    found = [
        (p.position, p.qualname) for p in _placeholders_in((args, kwargs, fields.get("callable")))
    ]
    if portability == "callable_placeholder" and not found:
        raise _recipe_error(
            root, f"rule {position} placeholder edit lacks its callable", code=_CORRUPT
        )
    extras: dict[str, Any] = {
        "metadata": tuple(sorted(dict(fields.get("metadata") or {}).items())),
        "signal": fields.get("signal"),
        "is_batch_independent": bool(fields.get("is_batch_independent", False)),
        "is_compatible_with_append": bool(fields.get("is_compatible_with_append", False)),
        "selection_recipe": fields.get("selection_recipe"),
    }
    if found:
        extras["metadata"] = (
            *extras["metadata"],
            ("executable", False),
            ("placeholders", tuple(found)),
        )
        audit = Edit(name=name, args=args, kwargs=tuple(kwargs.items()), portability=cast(Any, "callable_placeholder"), **extras)  # fmt: skip
        return audit, found
    try:
        rebuilt = rebuild_builtin_helper(name, args, kwargs)
    except Exception as exc:  # the registry's typed refusal names the unknown builtin
        raise _recipe_error(
            root, f"rule {position} built-in edit cannot be rebuilt: {exc}", code=_CORRUPT
        ) from exc
    return replace(rebuilt, **extras), found


def _placeholders_in(value: Any) -> Iterator[CallablePlaceholder]:
    if isinstance(value, CallablePlaceholder):
        yield value
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            yield from _placeholders_in(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _placeholders_in(item)


def _materialize(value: Any, artifact: _Artifact, root: Path) -> Any:
    """Replace the tensor references under ``value`` with tensors, eagerly, through the lazy layer."""

    from ..rehydrate import _build_lazy_tensor_ref

    if isinstance(value, BlobRef):
        index = {entry["blob_id"]: entry for entry in artifact.sink.entries.values()}
        ref = _build_lazy_tensor_ref(value, index, root, kind="out", payload_hints=None)
        if ref is None:
            raise _recipe_error(root, f"tensor {value.blob_id} has no usable entry", code=_CORRUPT)
        return ref.materialize(
            map_location="cpu", resolved_blobs_dir=resolve_bundle_blobs_dir(root)
        )
    if isinstance(value, list):
        return [_materialize(item, artifact, root) for item in value]
    if isinstance(value, tuple):
        return tuple(_materialize(item, artifact, root) for item in value)
    if isinstance(value, dict):
        return {key: _materialize(item, artifact, root) for key, item in value.items()}
    return value
