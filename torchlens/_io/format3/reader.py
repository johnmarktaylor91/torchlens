"""Reader of the 3.0 saved format: manifest first, digests, record tables, records.

Every step refuses typed before the next opens a file: manifest (2.x test, schema, version
window), listed files' digests, record tables (envelope, columns, one typed decode per cell),
cross-record references, record construction through the state hooks under the governed load,
tensors through the lazy layer. Nothing here evaluates, unpickles or imports by a name from
the file. Design: ``docs/maintainers/saved_format_3_design.md`` (sections 2, 3, 9 and 14).
"""

from __future__ import annotations

import math
import re
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Final

import torch

from ..._data_classes._state_adapter import state_new
from ..format_contract import BlobRef
from ..prerelease import PRERELEASE_STATE_KEY, validate_prerelease_state
from ..state_contract import governed_artifact_load
from . import contract
from ._payload_headers import validate_payload_headers
from ._record_refs import _RecordResolver
from ._tables import read_record_table, record_class, refuse_cells, refuse_table_columns
from .codec_policy import codec_backend
from .container import iter_members, read_member_index
from .decode import (
    CELL_CODE,
    DEVICE_TEXT,
    MAX_TENSOR_NUMEL,
    STORABLE_DTYPES,
    Decoder,
    UnencodableValue,
    _element_size,
    build_component,
    cell_error,
)
from .restore import restore_record
from .validate import (
    BELOW_FLOOR,
    TLSPEC_KEY,
    artifact_error,
    check_record_anchors,
    check_runnable_slots,
    check_runtime_policy,
    check_version_window,
    is_int,
    legacy_stamp,
    manifest_schema_errors,
    pre3_error,
    read_json_file,
    symlink_guard,
    validate_records,
    verify_listed_files,
)
from .values import dtype_table

__all__ = [
    "DecodedArtifact",
    "attach_disclosure_advisory",
    "decode_checked",
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
#: The column whose string identifies a row of each kind (a repeat refuses; ``ref`` cells link).
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

#: Accessor-valued persisted fields (``schema._edge_members``), rebuilt before the owner's state
#: hook: member record kind, the member's key column, accessor class (``None``: plain mapping).
_ACCESSOR_COLUMNS: Final[dict[tuple[str, str], tuple[str, str, str | None]]] = {
    ("layer", "ops"): ("op", "pass_number", "OpAccessor"),
    ("trace", "_param_logs"): ("param", "address", "ParamAccessor"),
    ("module", "ops"): ("module_call", "call_index", None),
    ("module", "params"): ("param", "address", "ParamAccessor"),
    ("grad_fn", "calls"): ("grad_fn_call", "call_index", None),
}


# ---- Manifest --------------------------------------------------------------


def read_manifest(path: str | Path) -> dict[str, Any]:
    """Read and validate ``manifest.json``, the first door: ``ArtifactTooOldError``
    (``artifact_version_below_floor``: no ``format`` key, a 2.x root marker, a version below
    the floor) or ``ArtifactError`` with one of the ``manifest_*`` codes, ``artifact_save_level_invalid``,
    ``artifact_version_above_runtime`` or ``load_path_symlink_rejected``."""

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
    _refuse_schema_errors(manifest_path, manifest_schema_errors(manifest))
    check_version_window(manifest, root)
    check_runtime_policy(manifest)
    if manifest["save_level"] not in contract.SAVE_LEVELS:
        raise artifact_error(
            f"Artifact manifest {manifest_path} names save_level {manifest['save_level']!r}; "
            f"this runtime knows {contract.SAVE_LEVELS}.",
            code=_CODES["save_level_unknown"],
            remedy="re-save the trace with trace.save() at a known level",
            file_path=str(manifest_path),
        )
    return manifest


def _refuse_schema_errors(manifest_path: Path, errors: list[str]) -> None:
    """Raise ``manifest_schema_invalid`` naming the first eight schema violations."""

    if not errors:
        return
    shown = "; ".join(errors[:8]) + (f" (+{len(errors) - 8} more)" if len(errors) > 8 else "")
    raise artifact_error(
        f"Artifact manifest {manifest_path} fails the 3.0 manifest schema: {shown}.",
        code=_CODES["manifest_invalid"],
        remedy="re-save the trace with trace.save(); the manifest was edited or corrupted",
        file_path=str(manifest_path),
        errors=tuple(errors),
    )


class ValidatedManifest(dict[str, Any]):
    """The manifest ``validate_artifact`` returns; for a ``trace`` artifact ``decoded`` carries
    the checked tables, so the load door builds records from the one validation pass."""

    decoded: DecodedArtifact | None

    def __init__(self, manifest: Mapping[str, Any], *, decoded: DecodedArtifact | None) -> None:
        super().__init__(manifest)
        self.decoded = decoded


def validate_artifact(path: str | Path, *, expect_kind: str | None = None) -> ValidatedManifest:
    """Validate an artifact as ``tl.load`` would, short of building its record objects.

    The envelope first (manifest, every listed file's bytes and SHA-256, the runnable slot
    cross-check), then the same bounded typed-record decode, reference resolution,
    cross-record invariants and manifest anchor check the load doors run, on the artifact's
    own tables and, for a bundle, on every member's. One code path with the doors: every
    load door calls this function, so what load refuses, validation refuses, with the same
    code. ``expect_kind`` is the door's kind: a mismatch refuses before any table is read.
    """

    root = Path(path)
    manifest = _validate_envelope(root)
    if expect_kind is not None:
        _require_kind(manifest, root, expect_kind)
    decoded: DecodedArtifact | None = None
    if manifest["kind"] == "bundle":
        metadata = read_member_index(root, manifest)
        for _name, member_root in iter_members(root, manifest, metadata):
            decode_checked(member_root, _validate_envelope(member_root))
    elif manifest["kind"] == "trace":
        decoded = decode_checked(root, manifest)
    return ValidatedManifest(manifest, decoded=decoded)


def _validate_envelope(path: str | Path) -> dict[str, Any]:
    """The envelope checks every door runs before it opens a record file: the manifest, every
    listed file (bytes, SHA-256; an unlisted file is named in a warning, never read) and the
    runnable slot cross-check."""

    root = Path(path)
    manifest = read_manifest(root)
    verify_listed_files(root, manifest)
    check_runnable_slots(root, manifest)
    return manifest


# ---- Tensor references -----------------------------------------------------


class TensorRefSink:
    """Collects the tensor cells a decode meets (``BlobRef`` markers plus lazy-layer entries that
    become the tensor index); a structure-only cell decodes to ``None``; codec, chunk and
    ``t<k>`` key are checked; repeats must agree."""

    def __init__(
        self,
        *,
        codecs: tuple[str, ...],
        files: Mapping[str, Mapping[str, Any]],
        backend: str = "torch",
    ) -> None:
        self.entries: dict[str, dict[str, Any]] = {}
        self.structure_only_refs = 0
        self._codecs = codecs
        self._files = files
        #: The artifact's backend: the codec every payload cell must use.
        self.backend = backend

    def add(self, ref: Mapping[str, Any], at: str) -> Any:
        """Check one tensor reference cell's geometry and return its placeholder."""

        shape, numel, device, is_complex, dtype_name = _tensor_geometry(ref, at)
        codec_id, codec_metadata = _split_codec(ref["codec"], at=at)
        if codec_id not in self._codecs:
            raise cell_error(
                at, f"tensor codec {codec_id!r} is not among the manifest's codecs", code=CELL_CODE
            )
        blob, key = ref.get("blob"), ref.get("key")
        if blob is None and key is None:
            self.structure_only_refs += 1
            return None
        backend = codec_backend(codec_id, codec_metadata, at, manifest_backend=self.backend)
        # A complex dtype's element size already counts both parts; the real view on disk
        # holds exactly numel * element_size bytes.
        declared = numel * _element_size(dtype_name)
        listed = self._listed_chunk(blob, key, at, declared)
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

    def _listed_chunk(self, blob: Any, key: Any, at: str, declared: int) -> Mapping[str, Any]:
        """Return the manifest ``files`` entry of ``blob`` once the reference is well formed."""

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
        if declared > int(listed["bytes"]):
            raise cell_error(
                at,
                f"tensor declares {declared} bytes but its chunk {blob} holds {listed['bytes']}",
                code=CELL_CODE,
            )
        return listed


def _tensor_geometry(ref: Mapping[str, Any], at: str) -> tuple[list[int], int, str, bool, str]:
    """Check a tensor cell's shape, device, flags and dtype; return them with the element count."""

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
        raise cell_error(at, "tensor requires_grad and complex must be booleans", code=CELL_CODE)
    dtype_name = ref["dtype"]
    if not isinstance(dtype_name, str) or dtype_name not in dtype_table():
        raise cell_error(
            at, f"tensor dtype {dtype_name!r} is outside the dtype table", code=CELL_CODE
        )
    if dtype_name not in STORABLE_DTYPES:
        raise cell_error(at, f"tensor dtype {dtype_name!r} is not a storable dtype", code=CELL_CODE)
    return shape, numel, device, is_complex, dtype_name


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


# ---- Record tables ---------------------------------------------------------


class DecodedArtifact(_RecordResolver):
    """One trace artifact's decoded tables and the record objects built from them."""

    def __init__(self, root: Path, manifest: Mapping[str, Any]) -> None:
        self.root = root
        self.manifest = manifest
        self.version: int = manifest[TLSPEC_KEY]
        self.sink = TensorRefSink(
            codecs=tuple(manifest["codecs"]),
            files=manifest["files"],
            backend=manifest.get("backend", contract.DEFAULT_BACKEND),
        )
        self.decoder = Decoder(self.sink.add, defer_records=True)
        self.tables: dict[str, list[dict[str, Any]]] = {}
        self.objects: dict[str, list[Any]] = {}
        self._deferred_trace_refs = 0
        self.identities: dict[str, dict[str, int]] = {}

    # -- decode -------------------------------------------------------------

    def decode_tables(self) -> None:
        """Decode every record table the manifest lists, in record order."""

        for kind in self.manifest["records"]:
            raw_rows = read_record_table(self.root, self.manifest, kind)
            if kind == "trace" and raw_rows:
                self._check_prerelease_marker(raw_rows[0])
            if raw_rows:
                refuse_table_columns(kind, raw_rows[0], self.manifest)
            self.tables[kind] = [
                {
                    name: self.decoder.decode(cell, at=f"{kind}[{index}].{name}")
                    for name, cell in row.items()
                }
                for index, row in enumerate(raw_rows)
            ]
            refuse_cells(kind, self.tables[kind])
            cls = record_class(kind)
            self.objects[kind] = [] if kind == "trace" else [state_new(cls) for _ in raw_rows]
            self._index_identities(kind)

    def _check_prerelease_marker(self, fields: dict[str, Any]) -> None:
        """Pop the pre-release marker of a switch-on write; refuse typed unless the switch is on."""

        if PRERELEASE_STATE_KEY in fields:
            at = f"trace.{PRERELEASE_STATE_KEY}"
            marker = self.decoder.decode(fields.pop(PRERELEASE_STATE_KEY), at=at)
            validate_prerelease_state({PRERELEASE_STATE_KEY: marker}, cls_name="Trace")

    def _index_identities(self, kind: str) -> None:
        """Index a decoded table's rows by their identity column."""

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

    # -- construction -------------------------------------------------------

    def build_records(self) -> None:
        """Run each record's state hook, leaves first, under the governed load."""

        if not self.objects.get("trace"):
            self.objects["trace"] = [state_new(record_class("trace"))]
            if self._deferred_trace_refs:
                self.resolve_references()
        with governed_artifact_load():
            for kind in LOAD_ORDER:
                for row_number, (obj, row) in enumerate(
                    zip(self.objects.get(kind, ()), self.tables.get(kind, ()), strict=True)
                ):
                    cells = self._rebuild_accessors(kind, row_number, row)
                    restore_record(obj, cells, version=self.version, kind=kind)
                    # After the hook: the state contract knows the column, not its slot.
                    for column, slot in _RESTORED_COMPUTED.get(kind, {}).items():
                        if cells.get(column) is not None:
                            object.__setattr__(obj, slot, cells[column])
                    if kind == "module":
                        self._install_module_templates(cells)

    def _install_module_templates(self, cells: Mapping[str, Any]) -> None:
        """Hand a module row's templates to its first call, where the module property reads them."""

        labels = cells.get("call_labels") or []
        if not labels or all(cells.get(column) is None for column in _TEMPLATE_SLOTS):
            return
        if not hasattr(self, "_calls_by_label"):
            rows = self.tables.get("module_call", ())
            pairs = zip(rows, self.objects["module_call"], strict=True)
            self._calls_by_label = {row.get("label"): call for row, call in pairs}
        call = self._calls_by_label.get(labels[0])
        for column, slot in _TEMPLATE_SLOTS.items():
            # The call's state hook default-fills the slot with None; a None slot is unset.
            if (
                call is not None
                and cells.get(column) is not None
                and call.__dict__.get(slot) is None
            ):
                call.__dict__[slot] = cells[column]

    def _rebuild_accessors(
        self, kind: str, row_number: int, row: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        """Return ``row`` with each flattened accessor column rebuilt from its member rows."""

        # Essential complexity: member lookup, positional and keyed forms are one rebuild.
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
        """The scrubbed-state-shaped dict the rehydrate path restores (``accessor_cls`` is
        ``ModuleAccessor``, passed in because the record classes sit above this module)."""

        state = dict(self._rebuild_accessors("trace", 0, self.tables["trace"][0]))
        state[TLSPEC_KEY] = self.version
        accessor = state.pop("_io_module_accessor_state", None)
        if isinstance(accessor, UnencodableValue):
            accessor = None
        if accessor is None and "module" in self.objects:
            accessor = self._module_accessor_state(accessor_cls)
        state["_io_module_accessor_state"] = accessor
        return state

    def _module_accessor_state(self, accessor_cls: type) -> Any:
        """Return the module accessor state built from the decoded module rows."""

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

        manifest = {"backend": contract.DEFAULT_BACKEND, **self.manifest}
        manifest["tensors"] = list(self.sink.entries.values())
        return manifest


# ---- Doors -----------------------------------------------------------------

#: The rehydration step of each door, installed by ``torchlens._io.bundle`` (install_load_hooks).
_LOAD_HOOKS: dict[str, Callable[..., Any]] = {}
_HOOK_KINDS: Final = frozenset({"trace", "bundle", "recipe"})

#: Computed columns kept behind their property; a module row's templates go to its first call.
_TEMPLATE_SLOTS: Final = {
    f"forward_{k}_template": f"_forward_{k}_template" for k in ("args", "kwargs")
}
_RESTORED_COMPUTED: Final[dict[str, dict[str, str]]] = {
    "module_call": _TEMPLATE_SLOTS,
    "op": {"arg_expressions": "_arg_expressions_cache"},
}


def install_load_hooks(**hooks: Callable[..., Any]) -> None:
    """Install the rehydration step of each door (``torchlens._io.bundle`` does, at import)."""

    unknown = set(hooks) - _HOOK_KINDS
    if unknown:
        raise ValueError(f"unknown load hook(s): {sorted(unknown)}")
    _LOAD_HOOKS.update(hooks)


def _hook(kind: str, root: Path) -> Callable[..., Any]:
    """Return the installed rehydration step of a door, refusing when none is installed."""

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
    """Refuse an artifact whose manifest kind is not the door's expected kind."""

    if manifest["kind"] != expected:
        raise artifact_error(
            f"Artifact {root} is a {manifest['kind']!r} artifact, not {expected!r}.",
            code=_CODES["kind_mismatch"],
            remedy=f"load it with the {manifest['kind']} door or pass a {expected} artifact",
            file_path=str(root),
        )


def decode_checked(root: Path, manifest: Mapping[str, Any]) -> DecodedArtifact:
    """Decode every table, resolve every reference, run the cross-record validators and the
    manifest anchor check: the bounded checks shared by the load doors and ``validate_artifact``.
    No record object is built and no tensor is read."""

    artifact = DecodedArtifact(root, manifest)
    artifact.decode_tables()
    artifact.resolve_references()
    validate_records(artifact.tables, manifest=manifest, root=root, objects=artifact.objects)
    check_record_anchors(manifest, artifact.tables["trace"][0], root)
    validate_payload_headers(root, artifact.sink.entries)
    return artifact


def read_tables(root: Path, manifest: Mapping[str, Any]) -> DecodedArtifact:
    """``decode_checked`` then the state hooks. Nothing here touches the rehydrate path."""

    artifact = decode_checked(root, manifest)
    artifact.build_records()
    return artifact


def load_trace(
    path: str | Path,
    *,
    lazy: bool = False,
    map_location: str | torch.device = "cpu",
    materialize_nested: bool = False,
    payload_hints: Any = None,
) -> Any:
    """Load one ``kind: trace`` artifact: tables and references first, then the state hooks
    under the governed load, then the installed ``trace`` step (tensors, accessors, recipe)."""

    root = Path(path)
    manifest = validate_artifact(root, expect_kind="trace")
    artifact = manifest.decoded
    assert artifact is not None  # a trace manifest always decodes
    artifact.build_records()
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
    manifest = _validate_envelope(root)
    _require_kind(manifest, root, "bundle")
    # Each member goes through ``load_trace``, hence ``validate_artifact``, once.
    metadata = read_member_index(root, manifest)
    policy = options.pop("unknown_relations", "opaque")
    members = [
        (name, load_trace(member_root, **options))
        for name, member_root in iter_members(root, manifest, metadata)
    ]
    return _hook("bundle", root)(
        members,
        metadata=metadata,
        root=root,
        tlspec_version=manifest.get(TLSPEC_KEY),
        unknown_relations=policy,
    )


def load_intervention_recipe(path: str | Path) -> Any:
    """Load the recipe (``intervention.json``) of a trace or intervention artifact: built-in edits
    through the helper registry; a callable placeholder is an audit-only edit (rerun refuses)."""

    root = Path(path)
    manifest = _validate_envelope(root)
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
