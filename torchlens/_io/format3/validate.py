"""Cross-record invariants of the 3.0 saved format, run after every table is decoded.

Grammar (shape and vocabulary) is the decoder's job (:mod:`.values`); this
module holds the checks that need more than one table: the trace's containers
agree with the record tables they index, the manifest's row counts agree with
the tables, and the identity columns are complete. The record-level load
validators (``forgery_validation``, ``_load_validators``,
``_forgery_identity_facts``) still run inside each record's state hook under
the governed load, unchanged; nothing here replaces or relaxes them. Design:
``docs/maintainers/saved_format_3_design.md``, section 9.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from importlib import import_module
from pathlib import Path
from typing import Any, Final

from .._artifact_anchors import check_manifest_metadata_anchors
from .._json import loads_bounded, read_bytes_bounded
from ..field_registry import RECORD_CONTRACT_CLASSES
from ..format_contract import TLSPEC_VERSION, above_ceiling_error
from ..format_errors import ArtifactError, ArtifactTooOldError
from ..paths import reject_symlink_path
from . import contract

__all__ = [
    "check_record_anchors",
    "check_version_window",
    "pre3_error",
    "manifest_schema_errors",
    "read_json_file",
    "read_record_table",
    "refuse_unknown_columns",
    "symlink_guard",
    "validate_records",
    "verify_file_bytes",
]

_CODES: Final = contract.REFUSAL_CODES
_CORRUPT: Final = _CODES["record_file_corrupt"]
BELOW_FLOOR: Final = _CODES["format_older_than_floor"]
ABOVE_CEILING: Final = _CODES["format_newer_than_runtime"]
_CODE: Final = _CORRUPT
TLSPEC_KEY: Final = "tlspec_version"
_RECORD_PATH = re.compile(r"^records/([a-z_]+)\.json$")
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
#: Record kinds whose storage bindings live under another key.
_SCHEMA_KEY_BY_RECORD: Final[dict[str, str]] = {"aten_op": "primitive_op"}
#: Columns the trace table may carry beyond its bindings (the state contract's transport keys).
_TRACE_TRANSPORT_COLUMNS: Final = frozenset({"_io_module_accessor_state"})


def artifact_error(message: str, *, code: str, remedy: str, **fields: Any) -> ArtifactError:
    return ArtifactError(message, code=code, remedy=remedy, **fields)


def integrity_error(path: Path, detail: str, *, code: str) -> ArtifactError:
    return artifact_error(
        f"Artifact file {path} failed its integrity check: {detail}.",
        code=code,
        remedy="re-save the trace with trace.save(); the artifact was truncated or edited",
        file_path=str(path),
        detail=detail,
    )


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def symlink_guard(path: Path, context: str) -> None:
    reject_symlink_path(path, context=context, code=_CODES["symlink_path"])


def pre3_error(root: Path, observed: str, *, code: str) -> ArtifactTooOldError:
    from .._pre3_module_paths import pre_3_artifact_error

    return pre_3_artifact_error(observed, path=str(root), code=code)


def check_version_window(manifest: Mapping[str, Any], root: Path) -> None:
    major, tlspec = manifest["format_version"], manifest[TLSPEC_KEY]
    if major < contract.FORMAT_MAJOR or tlspec < contract.RECORD_CONTRACT_FLOOR:
        raise pre3_error(
            root,
            f"format_version={major}, tlspec_version={tlspec}",
            code=BELOW_FLOOR,
        )
    if major > contract.FORMAT_MAJOR:
        raise above_ceiling_error(
            observed=major,
            subject="Artifact format_version",
            path=str(root),
            code=ABOVE_CEILING,
        )
    if tlspec > TLSPEC_VERSION:
        raise above_ceiling_error(
            observed=tlspec,
            subject="Artifact tlspec_version",
            path=str(root),
            code=ABOVE_CEILING,
        )


def read_json_file(path: Path, *, code: str, what: str) -> tuple[Any, bytes]:
    """Parse one JSON file under the bounded reader; a repeated object key refuses."""

    try:
        raw = read_bytes_bounded(path)
    except FileNotFoundError:
        raise artifact_error(
            f"Artifact {what} {path} is missing.",
            code=_CODES["manifest_missing"]
            if what == "manifest"
            else _CODES["record_file_corrupt"],
            remedy="load the artifact directory trace.save() wrote, not a file inside it",
            file_path=str(path),
        ) from None
    except (OSError, ValueError, ArtifactError) as exc:
        raise artifact_error(
            f"Artifact {what} {path} could not be read: {exc}.",
            code=code,
            remedy="re-save the trace with trace.save()",
            file_path=str(path),
        ) from exc

    def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        mapping: dict[str, Any] = {}
        for key, value in pairs:
            if key in mapping:
                raise artifact_error(
                    f"Artifact {what} {path} repeats the object key {key!r}.",
                    code=code,
                    remedy="re-save the trace with trace.save(); the file was edited",
                    file_path=str(path),
                    duplicate_key=key,
                )
            mapping[key] = value
        return mapping

    try:
        data = loads_bounded(raw.decode("utf-8"), object_pairs_hook=_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise artifact_error(
            f"Artifact {what} {path} is not valid UTF-8 JSON: {exc}.",
            code=code,
            remedy="re-save the trace with trace.save()",
            file_path=str(path),
        ) from exc
    return data, raw


def _relpath_is_contained(relpath: str) -> bool:
    if not relpath or relpath.startswith("/") or "\\" in relpath or "\x00" in relpath:
        return False
    return all(part not in ("", ".", "..") for part in relpath.split("/"))


def _file_entry_errors(relpath: Any, entry: Any) -> list[str]:
    if not isinstance(relpath, str) or not _relpath_is_contained(relpath):
        return [f"files key {relpath!r} is not a plain relative path"]
    if not isinstance(entry, dict):
        return [f"files[{relpath!r}] must be an object"]
    errors = []
    if not isinstance(entry.get("sha256"), str) or not _SHA256_HEX.match(entry["sha256"]):
        errors.append(f"files[{relpath!r}].sha256 must be 64 lowercase hex characters")
    if not is_int(entry.get("bytes")) or entry["bytes"] < 0:
        errors.append(f"files[{relpath!r}].bytes must be a non-negative integer")
    return errors


def _files_table_errors(files: Any) -> list[str]:
    if not isinstance(files, dict):
        return ["files must be an object keyed by relative path"]
    errors: list[str] = []
    for relpath, entry in files.items():
        errors.extend(_file_entry_errors(relpath, entry))
    if contract.MANIFEST_FILE in files:
        errors.append("files must not list the manifest itself")
    return errors


def _records_table_errors(records: Any, files: Any, kind: Any) -> list[str]:
    if not isinstance(records, dict):
        return ["records must be an object keyed by record kind"]
    known = {name for name, _, _ in contract.RECORD_INDEX}
    errors: list[str] = []
    for record, entry in records.items():
        if record not in known:
            errors.append(f"records names unknown record kind {record!r}")
            continue
        if not isinstance(entry, dict):
            errors.append(f"records[{record!r}] must be an object")
            continue
        path = entry.get("path")
        match = _RECORD_PATH.match(path) if isinstance(path, str) else None
        if match is None or match.group(1) != record:
            errors.append(f"records[{record!r}].path must be records/{record}.json")
            continue
        if not is_int(entry.get("rows")) or entry["rows"] < 0:
            errors.append(f"records[{record!r}].rows must be a non-negative integer")
        listed = files.get(path) if isinstance(files, dict) else None
        if not isinstance(listed, dict):
            errors.append(f"records[{record!r}].path {path} is absent from files")
        elif listed.get("sha256") != entry.get("sha256") or listed.get("bytes") != entry.get(
            "bytes"
        ):
            errors.append(
                f"records[{record!r}] digest or byte count disagrees with files[{path!r}]"
            )
    if kind == "trace" and "trace" not in records:
        errors.append("records must name the trace record")
    return errors


def manifest_schema_errors(manifest: Mapping[str, Any]) -> list[str]:
    """Return every way ``manifest`` departs from the manifest table of the contract."""

    errors = [
        f"missing required field {key!r}"
        for key in contract.MANIFEST_REQUIRED_FIELDS
        if key not in manifest
    ]
    if errors:
        return errors
    if manifest["format"] != contract.FORMAT_NAME:
        errors.append(f"format is {manifest['format']!r}, expected {contract.FORMAT_NAME!r}")
    for key in ("format_version", TLSPEC_KEY):
        if not is_int(manifest[key]) or manifest[key] < 0:
            errors.append(f"{key} must be a non-negative integer")
    for key in (*contract.WRITER_IDENTITY_FIELDS, "contract_digest"):
        if not isinstance(manifest[key], str) or not manifest[key]:
            errors.append(f"{key} must be a non-empty string")
    if manifest["kind"] not in contract.ARTIFACT_KINDS:
        errors.append(f"kind {manifest['kind']!r} is not one of {contract.ARTIFACT_KINDS}")
    policy = manifest["payload_policy"]
    if not isinstance(policy, dict) or not all(
        isinstance(key, str) and isinstance(value, bool) for key, value in policy.items()
    ):
        errors.append("payload_policy must be an object of boolean flags")
    codecs = manifest["codecs"]
    if (
        not isinstance(codecs, list)
        or len(set(codecs)) != len(codecs)
        or any(codec not in contract.PAYLOAD_CODEC_IDS for codec in codecs)
    ):
        errors.append(
            f"codecs must be a list without repeats drawn from {contract.PAYLOAD_CODEC_IDS}"
        )
    errors.extend(_files_table_errors(manifest["files"]))
    errors.extend(_records_table_errors(manifest["records"], manifest["files"], manifest["kind"]))
    disclosures = manifest.get("disclosures")
    if disclosures is not None and not (
        isinstance(disclosures, dict)
        and set(disclosures) == {"count", "first"}
        and is_int(disclosures["count"])
        and disclosures["count"] > 0
        and isinstance(disclosures["first"], list)
        and len(disclosures["first"]) <= 16
        and all(isinstance(item, str) for item in disclosures["first"])
    ):
        errors.append("disclosures must be {count: positive int, first: [<=16 strings]}")
    return errors


def verify_file_bytes(root: Path, relpath: str, entry: Mapping[str, Any]) -> Path:
    """Check one listed file's containment, existence, byte count and digest."""

    target = root / relpath
    for ancestor in [target, *target.parents]:
        if ancestor == root:
            break
        symlink_guard(ancestor, f"artifact file {relpath}")
    if not target.is_file():
        raise integrity_error(target, "listed in the manifest but absent", code=_CORRUPT)
    size = target.stat().st_size
    if size != entry["bytes"]:
        raise integrity_error(
            target,
            f"byte count {size} differs from the manifest's {entry['bytes']}",
            code=_CORRUPT,
        )
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != entry["sha256"]:
        raise integrity_error(target, "SHA-256 differs from the manifest's", code=_CORRUPT)
    return target


def record_class(kind: str) -> type:
    module_name, class_name = RECORD_CONTRACT_CLASSES[kind]
    return getattr(import_module(module_name), class_name)


def read_record_table(root: Path, manifest: Mapping[str, Any], kind: str) -> list[dict[str, Any]]:
    """Read ``records/<kind>.json`` after its digest check; return one raw dict per row."""

    entry = manifest["records"][kind]
    target = root / entry["path"]
    data, raw = read_json_file(target, code=_CORRUPT, what="record file")
    listed = manifest["files"][entry["path"]]
    if len(raw) != listed["bytes"] or hashlib.sha256(raw).hexdigest() != listed["sha256"]:
        raise integrity_error(target, "bytes differ from the manifest's digest", code=_CORRUPT)
    if not isinstance(data, dict) or data.get("record") != kind:
        raise integrity_error(target, f"not a {kind!r} record object", code=_CORRUPT)
    if data.get(TLSPEC_KEY) != manifest[TLSPEC_KEY]:
        raise integrity_error(target, "tlspec_version differs from the manifest's", code=_CORRUPT)
    if kind == "trace":
        fields = data.get("fields")
        if set(data) != {"record", TLSPEC_KEY, "fields"} or not isinstance(fields, dict):
            raise integrity_error(
                target,
                "trace record must be {record, tlspec_version, fields}",
                code=_CORRUPT,
            )
        if entry["rows"] != 1:
            raise integrity_error(target, "the trace record counts one row", code=_CORRUPT)
        return [dict(fields)]
    rows, columns = data.get("rows"), data.get("columns")
    if set(data) != {"record", TLSPEC_KEY, "rows", "columns"} or not isinstance(columns, dict):
        raise integrity_error(
            target,
            "record table must be {record, tlspec_version, rows, columns}",
            code=_CORRUPT,
        )
    if rows != entry["rows"]:
        raise integrity_error(
            target,
            f"rows {rows!r} differs from the manifest's {entry['rows']}",
            code=_CORRUPT,
        )
    for name, column in columns.items():
        if not isinstance(column, list) or len(column) != rows:
            raise integrity_error(
                target,
                f"column {name!r} is not an array of {rows} cells",
                code=_CORRUPT,
            )
    return [{name: column[index] for name, column in columns.items()} for index in range(rows)]


def _unknown_columns(kind: str, names: Mapping[str, Any]) -> list[str]:
    from ..._data_classes._schema_bindings import STORAGE_BINDINGS
    from ..._data_classes.field_policy import StorageKind

    bindings = STORAGE_BINDINGS[_SCHEMA_KEY_BY_RECORD.get(kind, kind)]
    allowed_extra = _TRACE_TRANSPORT_COLUMNS if kind == "trace" else frozenset()
    unknown = []
    for name in names:
        if name in allowed_extra:
            continue
        binding = bindings.get(name)
        if binding is None or binding.kind is StorageKind.RUNTIME or name == TLSPEC_KEY:
            unknown.append(name)
    return unknown


def refuse_unknown_columns(kind: str, names: Mapping[str, Any], *, declared_version: int) -> None:
    """Refuse a column the record's bindings do not persist (``unknown_persisted_field``)."""

    unknown = _unknown_columns(kind, names)
    if not unknown:
        return
    from ..format_errors import UnknownPersistedFieldError
    from ..state_contract import _unknown_field_remedy

    cls = record_class(kind)
    raise UnknownPersistedFieldError(
        f"{cls.__name__} record table carries {len(unknown)} column(s) unknown to this "
        f"reader's persisted-state contract: {', '.join(sorted(unknown)[:12])}. Unknown "
        "columns refuse in both writer directions.",
        code=_CODES["record_key_unknown"],
        record_type=cls.__name__,
        unknown_fields=sorted(unknown),
        declared_tlspec_version=declared_version,
        runtime_tlspec_version=TLSPEC_VERSION,
        remedy=_unknown_field_remedy(declared_version),
    )


#: Trace containers whose length must equal a record table's row count.
_TRACE_CONTAINER_ROWS: Final[tuple[tuple[str, str], ...]] = (
    ("_layer_list", "op"),
    ("_layer_dict_main_keys", "op"),
    ("_layer_logs", "layer"),
    ("_grad_fn_logs", "grad_fn"),
    ("_backward_pass_logs", "backward_pass"),
)


def _invariant_error(root: Path, detail: str, *, code: str, **fields: Any) -> ArtifactError:
    return ArtifactError(
        f"Artifact {root} fails a cross-record invariant: {detail}.",
        code=code,
        remedy="re-save the trace with trace.save(); the record tables disagree with each other",
        file_path=str(root),
        detail=detail,
        **fields,
    )


def _rows_of(tables: Mapping[str, list[dict[str, Any]]], kind: str) -> int:
    return len(tables.get(kind, ()))


def _check_trace_containers(tables: Mapping[str, list[dict[str, Any]]], root: Path) -> None:
    trace = tables["trace"][0]
    for field, kind in _TRACE_CONTAINER_ROWS:
        container = trace.get(field)
        if container is None:
            continue
        if not isinstance(container, (list, tuple, dict)):
            raise _invariant_error(
                root, f"trace.{field} is not a container", field=field, code=_CODE
            )
        if len(container) != _rows_of(tables, kind):
            raise _invariant_error(
                root,
                f"trace.{field} holds {len(container)} entries but the {kind} table has "
                f"{_rows_of(tables, kind)} rows",
                field=field,
                code=_CODE,
            )


def _check_orphans_disjoint(tables: Mapping[str, list[dict[str, Any]]], root: Path) -> None:
    """Orphan ops are ops outside the retained list; the two sets never overlap."""

    trace = tables["trace"][0]
    retained = trace.get("_layer_list")
    orphans = trace.get("_orphan_logs")
    if not isinstance(retained, list) or not isinstance(orphans, (list, tuple)):
        return
    overlap = {id(op) for op in retained} & {id(op) for op in orphans}
    if overlap:
        raise _invariant_error(
            root,
            f"{len(overlap)} op row(s) appear both retained and orphaned",
            field="_orphan_logs",
            code=_CODE,
        )


def _check_manifest_rows(
    tables: Mapping[str, list[dict[str, Any]]], manifest: Mapping[str, Any], root: Path
) -> None:
    for kind, entry in manifest["records"].items():
        if entry["rows"] != _rows_of(tables, kind):
            raise _invariant_error(
                root,
                f"manifest counts {entry['rows']} {kind} rows but {_rows_of(tables, kind)} were decoded",
                field=kind,
                code=_CODE,
            )
    if "trace" in manifest["records"] and _rows_of(tables, "trace") != 1:
        raise _invariant_error(
            root, "a trace artifact carries exactly one trace row", field="trace", code=_CODE
        )


def validate_records(
    tables: Mapping[str, list[dict[str, Any]]],
    *,
    manifest: Mapping[str, Any],
    root: Path,
) -> None:
    """Run every cross-record invariant over the decoded tables; refuse on the first failure.

    Runs after the reader resolved every reference (so dangling labels and
    indices have already refused) and before any record's state hook runs.
    """

    _check_manifest_rows(tables, manifest, root)
    if "trace" in tables:
        _check_trace_containers(tables, root)
        _check_orphans_disjoint(tables, root)


class _AnchorView:
    """The attribute shape ``check_manifest_metadata_anchors`` reads, over a 3.0 manifest."""

    def __init__(self, manifest: Mapping[str, Any]) -> None:
        entry = manifest["records"].get("op")
        self.n_layers = entry["rows"] if isinstance(entry, dict) else None


def check_record_anchors(
    manifest: Mapping[str, Any], persisted_rows: int | None, root: Path
) -> None:
    """The manifest/metadata anchor check of the 2.x path, over the op table's row count."""

    check_manifest_metadata_anchors(_AnchorView(manifest), persisted_rows, root)
