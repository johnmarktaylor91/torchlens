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

import math
import re
import sys
from collections.abc import Callable, Iterator, Mapping
from functools import cache
from pathlib import Path
from typing import Any, Final

import torch

from ..._data_classes._state_adapter import state_new, state_restore
from ..._data_classes.field_policy import StorageKind
from ..format_contract import BlobRef
from ..state_contract import governed_artifact_load
from . import contract
from .decode import (
    CELL_CODE,
    DEVICE_TEXT,
    MAX_TENSOR_NUMEL,
    STORABLE_DTYPES,
    Decoder,
    PendingRecord,
    Unencodable,
    _element_size,
    build_component,
    cell_error,
)
from .schema import record_schema
from .validate import (
    BELOW_FLOOR,
    TLSPEC_KEY,
    artifact_error,
    check_record_anchors,
    check_version_window,
    integrity_error,
    is_int,
    legacy_stamp,
    manifest_schema_errors,
    pre3_error,
    read_json_file,
    read_record_table,
    record_class,
    refuse_unknown_columns,
    symlink_guard,
    validate_records,
    verify_file_bytes,
)
from .values import RowRef, dtype_table

__all__ = [
    "DecodedArtifact",
    "attach_disclosure_advisory",
    "install_load_hooks",
    "load_bundle",
    "load_intervention_recipe",
    "load_trace",
    "read_manifest",
    "read_tables",
    "validate_artifact",
]

_CODES: Final = contract.REFUSAL_CODES
_CORRUPT: Final = _CODES["record_file_corrupt"]
#: Root entries that mark an artifact written by torchlens 2.x (pickle carrier).
PRE3_ROOT_MARKERS: Final[tuple[str, ...]] = ("metadata.pkl", "bundle.json")
_MEMBER_DIR = re.compile(r"^members/(\d+)-([A-Za-z0-9_.-]+)$")
#: The column whose string identifies a row of each kind; a repeat refuses. Records are
#: referenced by ``ref`` cells (design section 4), never by these strings.
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
_BLOB_PATH = re.compile(r"blobs/[A-Za-z0-9_.-]+\.safetensors")
_BLOB_KEY = re.compile(r"t\d{1,12}")
_TORCH_CODEC: Final = contract.PAYLOAD_CODEC_IDS[0]
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

#: Accessor-valued persisted fields. The writer stores an accessor of records
#: as the list of its member rows (``schema._edge_members``); the reader
#: rebuilds the accessor from the members' own keys before the owner's state
#: hook runs. Each entry names the member record kind, the member's key column
#: and the accessor class, named in the member class's module (``None`` where the
#: owner's state hook wraps a plain mapping itself).
_ACCESSOR_COLUMNS: Final[dict[tuple[str, str], tuple[str, str, str | None]]] = {
    ("module", "ops"): ("module_call", "call_index", None),
    ("module", "params"): ("param", "address", "ParamAccessor"),
    ("grad_fn", "calls"): ("grad_fn_call", "call_index", None),
}


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
    stamp = legacy_stamp(manifest_path)
    if stamp is not None:
        raise pre3_error(root, f"tlspec_version={stamp}", code=BELOW_FLOOR)
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
# Tensor references
# ---------------------------------------------------------------------------


class TensorRefSink:
    """Collects the tensor cells a decode meets, as ``BlobRef`` markers plus lazy-layer entries.

    The reader hands the entries to the existing rehydrate path as the
    manifest's tensor index (blob id to entry), so eager and lazy
    materialization run through ``LazyActivationRef`` on (chunk, key)
    unchanged. A structure-only cell (no ``blob`` and no ``key``) decodes to
    ``None`` and is counted. Every field is checked against the manifest: the
    codec must be listed, the chunk must be in ``files``, the key must be an
    opaque ``t<k>``; a (chunk, key) referenced twice must be declared identically.
    """

    def __init__(self, *, codecs: tuple[str, ...], files: Mapping[str, Mapping[str, Any]]) -> None:
        self.entries: dict[str, dict[str, Any]] = {}
        self.structure_only_refs = 0
        self._codecs = codecs
        self._files = files

    def add(self, ref: Mapping[str, Any], at: str) -> Any:
        shape = ref["shape"]
        if not isinstance(shape, list) or not all(is_int(d) and d >= 0 for d in shape):
            raise cell_error(
                at, "tensor shape must be an array of non-negative integers", code=CELL_CODE
            )
        if len(shape) > 64:
            raise cell_error(at, "tensor shape has more than 64 dimensions", code=CELL_CODE)
        numel = math.prod(shape)
        if numel > MAX_TENSOR_NUMEL:
            raise cell_error(
                at,
                f"tensor shape {shape!r} declares more than {MAX_TENSOR_NUMEL} elements",
                code=CELL_CODE,
            )
        device = ref["device"]
        if not isinstance(device, str) or not DEVICE_TEXT.fullmatch(device):
            raise cell_error(at, f"tensor device {device!r} is not a device string", code=CELL_CODE)
        try:
            torch.device(device)
        except (RuntimeError, TypeError, ValueError):
            raise cell_error(
                at, f"tensor device {device!r} is not a torch device", code=CELL_CODE
            ) from None
        is_complex = ref.get("complex", False)
        if not isinstance(ref["requires_grad"], bool) or not isinstance(is_complex, bool):
            raise cell_error(
                at, "tensor requires_grad and complex must be booleans", code=CELL_CODE
            )
        codec_id, codec_metadata = _split_codec(ref["codec"], at=at)
        if codec_id not in self._codecs:
            raise cell_error(
                at, f"tensor codec {codec_id!r} is not among the manifest's codecs", code=CELL_CODE
            )
        dtype_name = ref["dtype"]
        if not isinstance(dtype_name, str) or dtype_name not in dtype_table():
            raise cell_error(
                at, f"tensor dtype {dtype_name!r} is outside the dtype table", code=CELL_CODE
            )
        if dtype_name not in STORABLE_DTYPES:
            raise cell_error(
                at, f"tensor dtype {dtype_name!r} is not a storable dtype", code=CELL_CODE
            )
        blob, key = ref.get("blob"), ref.get("key")
        if blob is None and key is None:
            self.structure_only_refs += 1
            return None
        if not isinstance(blob, str) or not _BLOB_PATH.fullmatch(blob):
            raise cell_error(
                at,
                f"tensor blob {blob!r} is not a blobs/<codec>-<nnnn>.safetensors path",
                code=CELL_CODE,
            )
        if not isinstance(key, str) or not _BLOB_KEY.fullmatch(key):
            raise cell_error(at, f"tensor key {key!r} is not an opaque t<k> key", code=CELL_CODE)
        listed = self._files.get(blob)
        if not isinstance(listed, Mapping):
            raise cell_error(
                at, f"tensor blob {blob} is not listed in the manifest's files", code=CELL_CODE
            )
        declared = numel * _element_size(dtype_name) * (2 if is_complex else 1)
        if declared > int(listed["bytes"]):
            raise cell_error(
                at,
                f"tensor declares {declared} bytes but its chunk {blob} holds {listed['bytes']}",
                code=CELL_CODE,
            )
        blob_id = f"{blob}#{key}"
        declaration = (list(shape), dtype_name, device, ref["requires_grad"], codec_id)
        seen = self.entries.get(blob_id)
        if seen is not None:
            # One stored tensor may sit in several cells (an argument the op also
            # keeps as a buffer); the second reference must declare the same tensor.
            earlier = (
                seen["shape"],
                seen["dtype"],
                seen["device_at_save"],
                seen["requires_grad"],
                seen["codec"],
            )
            if earlier != declaration:
                raise cell_error(
                    at,
                    f"tensor {blob_id} is declared differently at {seen['label']}",
                    code=CELL_CODE,
                )
            return BlobRef(blob_id, seen["kind"])
        kind = _tensor_kind(at)
        backend = "torch" if codec_id == _TORCH_CODEC else "numpy"
        self.entries[blob_id] = {
            "blob_id": blob_id,
            "kind": kind,
            "label": at,
            "relative_path": blob,
            "key": key,
            "backend": backend,
            "shape": list(shape),
            "dtype": dtype_name,
            "device_at_save": device,
            "layout": "strided",
            "bytes": listed["bytes"],
            "sha256": listed["sha256"],
            "requires_grad": ref["requires_grad"],
            "logical_backend": backend,
            "codec": codec_id,
            "codec_metadata": codec_metadata,
        }
        return BlobRef(blob_id, kind)


def _split_codec(codec: Any, *, at: str) -> tuple[str, dict[str, Any] | None]:
    """Return the codec id and the backend hints of a tensor cell's ``codec`` field."""

    if isinstance(codec, str):
        return codec, None
    if isinstance(codec, dict) and isinstance(codec.get("id"), str):
        metadata = {key: value for key, value in codec.items() if key != "id"}
        if any(isinstance(v, dict) and contract.TYPED_VALUE_KEY in v for v in metadata.values()):
            raise cell_error(at, "tensor codec hints must be plain JSON", code=CELL_CODE)
        return codec["id"], metadata or None
    raise cell_error(at, "tensor codec must be a codec id or {id, ...hints}", code=CELL_CODE)


def _tensor_kind(at: str) -> str:
    """The lazy-layer kind of a tensor cell: ``grad`` under a gradient column, else ``out``."""

    column = at.split(".", 1)[1] if "." in at else at
    head = re.split(r"[.\[{<]", column, maxsplit=1)[0]
    return "grad" if "grad" in head else "out"


# ---------------------------------------------------------------------------
# Record tables
# ---------------------------------------------------------------------------


class DecodedArtifact:
    """One trace artifact's decoded tables and the record objects built from them."""

    def __init__(self, root: Path, manifest: Mapping[str, Any]) -> None:
        self.root = root
        self.manifest = manifest
        self.version: int = manifest[TLSPEC_KEY]
        self.sink = TensorRefSink(codecs=tuple(manifest["codecs"]), files=manifest["files"])
        self.decoder = Decoder(self.sink.add, defer_records=True)
        self.tables: dict[str, list[dict[str, Any]]] = {}
        self.objects: dict[str, list[Any]] = {}
        self.identities: dict[str, dict[str, int]] = {}

    # -- decode -------------------------------------------------------------

    def decode_tables(self) -> None:
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
        self.objects["trace"] = [state_new(record_class("trace"))]

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
        """Settle every ``ref`` cell to its record object and build the pending value objects."""

        for kind, rows in self.tables.items():
            for row_number, row in enumerate(rows):
                for name in list(row):
                    row[name] = self._settle(row[name], f"{kind}[{row_number}].{name}")

    def _settle(self, value: Any, at: str) -> Any:
        if isinstance(value, RowRef):
            return self._lookup(value, at)
        if isinstance(value, PendingRecord):
            fields = {k: self._settle(v, f"{at}.{k}") for k, v in value.fields.items()}
            return build_component(value.cls, fields, at=at)
        if isinstance(value, list):
            return [self._settle(item, f"{at}[{i}]") for i, item in enumerate(value)]
        if isinstance(value, tuple):
            return tuple(self._settle(item, f"{at}[{i}]") for i, item in enumerate(value))
        if isinstance(value, (set, frozenset)):
            return type(value)(self._settle(item, at) for item in value)
        if isinstance(value, dict):
            return {key: self._settle(item, f"{at}.{key}") for key, item in value.items()}
        return value

    def _lookup(self, reference: RowRef, at: str) -> Any:
        rows = self.objects.get(reference.record)
        if rows is None:
            raise cell_error(
                at, f"references the {reference.record} table, which is absent", code=CELL_CODE
            )
        if not 0 <= reference.row < len(rows):
            raise cell_error(
                at, f"row {reference.row} is outside the {reference.record} table", code=CELL_CODE
            )
        return rows[reference.row]

    # -- construction -------------------------------------------------------

    def build_records(self) -> None:
        """Run each record's state hook, leaves first, under the governed load."""

        with governed_artifact_load():
            for kind in LOAD_ORDER:
                for row_number, (obj, row) in enumerate(
                    zip(self.objects.get(kind, ()), self.tables.get(kind, ()), strict=True)
                ):
                    row = self._rebuild_accessors(kind, row_number, row)
                    _restore_record(obj, row, version=self.version, computed=computed_columns(kind))

    def _rebuild_accessors(
        self, kind: str, row_number: int, row: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        """Return ``row`` with each flattened accessor column rebuilt from its member rows."""

        rebuilt: dict[str, Any] | None = None
        for (owner, column), (member_kind, key, build) in _ACCESSOR_COLUMNS.items():
            if owner != kind or row.get(column) is None:
                continue
            at = f"{kind}[{row_number}].{column}"
            members = row[column]
            if not isinstance(members, list):
                raise cell_error(at, f"must list {member_kind} rows", code=CELL_CODE)
            member_rows = self.tables.get(member_kind, ())
            position = {
                id(obj): index for index, obj in enumerate(self.objects.get(member_kind, ()))
            }
            keyed: dict[Any, Any] = {}
            for member in members:
                index = position.get(id(member))
                if index is None:
                    raise cell_error(at, f"a member is not a {member_kind} row", code=CELL_CODE)
                member_key = member_rows[index].get(key)
                if isinstance(member_key, bool) or not isinstance(member_key, (int, str)):
                    raise cell_error(
                        at, f"member {member_kind}[{index}].{key} is not a key", code=CELL_CODE
                    )
                if member_key in keyed:
                    raise cell_error(at, f"member key {member_key!r} repeats", code=CELL_CODE)
                keyed[member_key] = member
            if rebuilt is None:
                rebuilt = dict(row)
            # The accessor class lives beside its member class; resolving it there
            # keeps the reader out of the record classes' import cycle.
            member_module = sys.modules[record_class(member_kind).__module__]
            rebuilt[column] = keyed if build is None else getattr(member_module, build)(keyed)
        return row if rebuilt is None else rebuilt

    def trace_state(self, accessor_cls: type) -> dict[str, Any]:
        """The scrubbed-state-shaped dict the existing rehydrate path restores.

        ``accessor_cls`` is ``ModuleAccessor``, passed in by the rehydration
        step because the record classes sit above this module.
        """

        state = dict(self.tables["trace"][0])
        state[TLSPEC_KEY] = self.version
        accessor = state.pop("_io_module_accessor_state", None)
        if isinstance(accessor, Unencodable):
            accessor = None
        if accessor is None and "module" in self.objects:
            accessor = self._module_accessor_state(accessor_cls)
        state["_io_module_accessor_state"] = accessor
        return state

    def _module_accessor_state(self, accessor_cls: type) -> Any:
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
            accessor_cls,
            {
                "_dict": module_dict,
                "_list": list(modules),
                "_pass_dict": pass_dict,
                "_alias_dict": {},
            },
        )

    def manifest_for_rehydrate(self) -> dict[str, Any]:
        """The manifest as the rehydrate path reads it: the tensor index from the cells."""

        manifest = {"backend": "torch", **self.manifest}
        manifest["tensors"] = list(self.sink.entries.values())
        return manifest


@cache
def computed_columns(kind: str) -> frozenset[str]:
    """Columns the writer fills from a read-only property; loaded objects recompute them."""

    return frozenset(
        name
        for name, column in record_schema(kind).items()
        if column.storage == StorageKind.COMPUTED.value
    )


def _restore_record(
    obj: Any, row: Mapping[str, Any], *, version: int, computed: frozenset[str] = frozenset()
) -> None:
    # A COMPUTED column (``Param.is_quantized``, ``module_name``, ...) is written
    # from the live property so the file reads whole; the object owns no slot for
    # it, and installing it would shadow the property with dead state the
    # rehydrate walk then cannot assign. The 2.x pickle state never carried it.
    state = {name: value for name, value in row.items() if name not in computed}
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

#: The rehydration step of each door, installed by ``torchlens._io.bundle`` (see
#: :func:`install_load_hooks`). ``trace`` receives a :class:`DecodedArtifact` and
#: the load options; ``bundle`` the loaded members, relations, baseline and root;
#: ``recipe`` the validated artifact root.
_LOAD_HOOKS: dict[str, Callable[..., Any]] = {}
_HOOK_KINDS: Final = frozenset({"trace", "bundle", "recipe"})


def install_load_hooks(**hooks: Callable[..., Any]) -> None:
    """Install the rehydration step of the ``trace``, ``bundle`` and ``recipe`` doors.

    This module reads, decodes and validates; turning decoded tables into
    ``Trace``, ``Bundle`` and ``InterventionSpec`` objects runs the record
    classes' rehydrate path, which sits above the format package in the import
    layering, so ``torchlens._io.bundle`` installs those steps when it is
    imported (``import torchlens`` does). A door called without its step
    refuses with ``bundle_load_failed`` after the artifact validation.
    """

    unknown = set(hooks) - _HOOK_KINDS
    if unknown:
        raise ValueError(f"unknown load hook(s): {sorted(unknown)}")
    _LOAD_HOOKS.update(hooks)


def _hook(kind: str, root: Path) -> Callable[..., Any]:
    try:
        return _LOAD_HOOKS[kind]
    except KeyError:
        raise artifact_error(
            f"The 3.0 {kind} loader's rehydration step is not installed; {root} was validated "
            "but not loaded.",
            code=_CODES["blob_missing"],
            remedy="import torchlens (its load door installs the step) and load again",
            file_path=str(root),
        ) from None


def _require_kind(manifest: Mapping[str, Any], root: Path, expected: str) -> None:
    if manifest["kind"] != expected:
        raise artifact_error(
            f"Artifact {root} is a {manifest['kind']!r} artifact, not {expected!r}.",
            code=_CODES["kind_mismatch"],
            remedy=f"load it with the {manifest['kind']} door or pass a {expected} artifact",
            file_path=str(root),
        )


def read_tables(root: Path, manifest: Mapping[str, Any]) -> DecodedArtifact:
    """Decode every record table, resolve every reference and run the state hooks.

    The cross-record validators run between reference resolution and
    construction; the record anchors of the manifest are checked against the
    decoded trace row. Nothing here touches the rehydrate path.
    """

    artifact = DecodedArtifact(root, manifest)
    artifact.decode_tables()
    artifact.resolve_references()
    validate_records(artifact.tables, manifest=manifest, root=root)
    artifact.build_records()
    layer_list = artifact.tables["trace"][0].get("_layer_list")
    persisted_rows = len(layer_list) if isinstance(layer_list, list) else None
    check_record_anchors(manifest, persisted_rows, root)
    return artifact


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
    load validators run exactly as they do for in-process state. The installed
    ``trace`` step then rehydrates the trace (tensors through the lazy layer,
    accessors, the recipe when ``intervention.json`` is listed).
    """

    root = Path(path)
    manifest = validate_artifact(root)
    _require_kind(manifest, root, "trace")
    artifact = read_tables(root, manifest)
    return _hook("trace", root)(
        artifact,
        lazy=lazy,
        map_location=map_location,
        materialize_nested=materialize_nested,
        payload_hints=payload_hints,
    )


def attach_disclosure_advisory(trace: Any, disclosures: Mapping[str, Any]) -> None:
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
    return _hook("bundle", root)(members, relations=relations, baseline=baseline, root=root)


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


def load_intervention_recipe(path: str | Path) -> Any:
    """Load the intervention recipe (``intervention.json``) of a trace or intervention artifact.

    Built-in edits come back through the helper registry; a callable
    placeholder loads as an audit-only edit that every rerun door refuses
    (``run_intervention_spec_not_persisted``). Grammar: ``format3.recipe``;
    object construction: the installed ``recipe`` step.
    """

    root = Path(path)
    manifest = validate_artifact(root)
    if manifest["kind"] not in ("trace", "intervention"):
        _require_kind(manifest, root, "intervention")
    if contract.INTERVENTION_FILE not in manifest["files"]:
        raise artifact_error(
            f"Artifact {root} carries no intervention recipe.",
            code=_CODES["kind_mismatch"],
            remedy="save the trace with its intervention spec, or load the trace itself",
            file_path=str(root),
        )
    return _hook("recipe", root)(root)
