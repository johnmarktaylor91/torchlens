"""The 3.0 writer: a trace or bundle to typed JSON record tables plus safetensors chunks.

Contract: ``docs/maintainers/saved_format_3.md``; design:
``docs/maintainers/saved_format_3_design.md`` (sections 2, 3, 5, 6, 8, 9).

``Trace.save`` keeps its one chokepoint (``torchlens._io.bundle.save``): the
entry refusals, the save gates, the level and payload-flag resolution, the
scrub and the atomic publish are unchanged. Payload tensors go into chunked
safetensors files through :class:`~torchlens._io.format3.payloads.PayloadStore`;
where the chokepoint used to pickle the scrubbed state it now calls
:func:`write_trace_envelope`, which lays the state out as one table per record
kind (``records/<kind>.json``: the trace as one object, every other kind
columnar, rows in live container order), digests every file and writes the
3.0 ``manifest.json``. No pickle file is written at any level. The reload
check (late.21) re-reads the written tables through the bounded JSON reader
and the typed decoder before the publish.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from collections.abc import Mapping, Sequence
from functools import cache
from importlib import import_module
from pathlib import Path
from typing import Any

import torch

from ..._data_classes._state_adapter import state_items
from .. import _json
from ..field_registry import RECORD_CONTRACT_CLASSES
from ..format_errors import ArtifactError
from ..writer_contract import writer_contract_digest
from . import reader as _reader
from .contract import (
    FORMAT_MAJOR,
    FORMAT_NAME,
    MANIFEST_FILE,
    RECORD_CONTRACT_VERSION,
    RECORD_INDEX,
    RECORDS_DIR,
)
from .payloads import PayloadStore
from .recipe import build_recipe, write_recipe
from .schema import level_columns, record_schema
from .values import ValueEncoder, decode_value, object_state, record_type_table, tagged

#: Root files the digest table leaves out: the manifest itself and the
#: staging sentinel that comes off right before publish.
_UNDIGESTED_ROOT_FILES = frozenset({MANIFEST_FILE, "PARTIAL"})
#: Prefix of the unbuilt-door refusal message (``contract.refuse_*_not_built``).
_UNBUILT_DOOR_PREFIX = "The 3.0 saved-format door "
#: Trace state keys that hold each record kind in live container order; rows
#: found only by reference follow in discovery order.
_SEED_KEYS: tuple[tuple[str, str], ...] = (
    ("_layer_list", "op"),
    ("_layer_logs", "layer"),
    ("_module_logs", "module"),
    ("_param_logs", "param"),
    ("_grad_fn_logs", "grad_fn"),
    ("_backward_pass_logs", "backward_pass"),
)
#: Disclosure paths listed in the manifest (the full list is in the cells).
_DISCLOSURE_PATHS_SHOWN = 16
#: Op keys whose absent map is written as an empty object (contract page,
#: "Values settled on the live side"); the live op is settled the same way.
_EMPTY_MAP_OP_KEYS = ("edge_substitutions", "edge_replacement_stamps")
#: Values the discovery walk never descends into.
_LEAF_TYPES = (str, bytes, bytearray, int, float, complex, bool, type(None), torch.Tensor)


@cache
def _record_kinds_by_class() -> dict[type, str]:
    """Return record class to record kind for the twelve record contracts."""

    return {
        getattr(import_module(module), name): kind
        for kind, (module, name) in RECORD_CONTRACT_CLASSES.items()
    }


def record_kind_of(value: Any) -> str | None:
    """Return the record kind of a record object, or ``None`` for any other value."""

    kinds = _record_kinds_by_class()
    kind = kinds.get(type(value))
    if kind is not None:
        return kind
    for cls in type(value).__mro__[1:]:
        if cls in kinds:
            return kinds[cls]
    return None


def _container_items(value: Any) -> list[Any]:
    """Return the members of a container value (accessors expose ``_list``)."""

    if isinstance(value, Mapping):
        return list(value.values())
    if isinstance(value, (list, tuple, set, frozenset)):
        return list(value)
    listed = getattr(value, "_list", None)
    return list(listed) if isinstance(listed, list) else []


class RecordTables:
    """The record objects of one scrubbed trace, one ordered row list per kind.

    Parameters
    ----------
    state:
        Scrubbed trace state.
    """

    def __init__(self, state: Mapping[str, Any]) -> None:
        self.state = state
        self.rows: dict[str, list[Any]] = {kind: [] for kind, _, _ in RECORD_INDEX}
        self._index: dict[int, tuple[str, int]] = {}

    def add(self, value: Any, kind: str) -> int:
        """Register a record object once; return its row index."""

        known = self._index.get(id(value))
        if known is not None:
            return known[1]
        row = len(self.rows[kind])
        self.rows[kind].append(value)
        self._index[id(value)] = (kind, row)
        return row

    def discover(self) -> None:
        """Seed the live containers, then register every record reachable by reference."""

        for key, kind in _SEED_KEYS:
            for item in _container_items(self.state.get(key)):
                if record_kind_of(item) == kind:
                    self.add(item, kind)
        pending: deque[Any] = deque(self.state.values())
        for rows in self.rows.values():
            pending.extend(rows)
        seen: set[int] = set()
        while pending:
            value = pending.popleft()
            if isinstance(value, _LEAF_TYPES) or id(value) in seen:
                continue
            seen.add(id(value))
            pending.extend(self._children(value))

    def _children(self, value: Any) -> list[Any]:
        """Register ``value`` if it is a record; return the values it holds."""

        kind = record_kind_of(value)
        if kind == "trace":
            return []
        if kind is not None:
            self.add(value, kind)
            return [item for _, item in state_items(value)]
        if isinstance(value, Mapping):
            return [*value.keys(), *value.values()]
        if isinstance(value, (list, tuple, set, frozenset)):
            return list(value)
        if type(value).__module__.startswith("torchlens") and not isinstance(value, type):
            try:
                return list(object_state(value).values())
            except (AttributeError, TypeError, ValueError):
                return []
        return []

    def reference(self, value: Any) -> dict[str, Any] | None:
        """Return the ``ref`` cell of a record object, or ``None`` for other values."""

        kind = record_kind_of(value)
        if kind is None:
            return None
        row = 0 if kind == "trace" else self.add(value, kind)
        return tagged("ref", record=kind, row=row)


def settle_live_side(trace: Any) -> None:
    """Settle the live-side keys on the live trace before it is written.

    ``contract.LIVE_SIDE_VALUE_KEYS``: a value the loader used to settle after
    loading is settled here, on the live trace, so the loaded value equals the
    live one. An absent ``edge_substitutions`` or ``edge_replacement_stamps``
    map becomes ``{}``. ``func_duration`` and ``equivalence_class`` are
    written as they are (a ``Duration`` round-trips through the typed value
    encoding; the barcode remap of ``equivalence_class`` stays the writer's
    privacy policy). ``_options_grouping_policy`` is stamped at capture by
    loop detection on every live trace, so the writer writes it unchanged.

    Parameters
    ----------
    trace:
        Live trace about to be saved.
    """

    for op in list(getattr(trace, "_layer_list", None) or ()):
        for key in _EMPTY_MAP_OP_KEYS:
            if getattr(op, key, None) is None:
                setattr(op, key, {})


def _cell(record: Any, state: Mapping[str, Any], column: str, storage: str) -> Any:
    """Return one stored value of a record (computed keys are computed if absent)."""

    if column in state:
        return state[column]
    if storage == "computed":
        try:
            return getattr(record, column)
        except (AttributeError, LookupError, RuntimeError, TypeError, ValueError):
            return None
    return None


def encode_tables(
    state: Mapping[str, Any],
    *,
    encoder_factory: Any,
    payload_policy: Mapping[str, bool],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Lay out scrubbed trace state as the twelve record tables.

    Parameters
    ----------
    state:
        Scrubbed trace state.
    encoder_factory:
        Callable taking the ``ref`` resolver and returning a ``ValueEncoder``.
    payload_policy:
        Resolved include flags (decide the optional columns).

    Returns
    -------
    tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]
        Record file payloads by kind (kinds without rows omitted) and the
        disclosures of values written as ``unencodable``.
    """

    tables = RecordTables(state)
    tables.discover()
    encoder: ValueEncoder = encoder_factory(tables.reference)
    schema = record_schema("trace")
    fields = {
        name: encoder.encode(
            state.get(name), f"trace.{name}", payload=schema[name].storage == "payload"
        )
        for name in level_columns("trace", payload_policy)
    }
    files: dict[str, dict[str, Any]] = {
        "trace": {"record": "trace", "tlspec_version": RECORD_CONTRACT_VERSION, "fields": fields}
    }
    columns = _encode_record_rows(tables, encoder, payload_policy)
    for kind, _, _ in RECORD_INDEX:
        if kind != "trace" and tables.rows[kind]:
            files[kind] = {
                "record": kind,
                "tlspec_version": RECORD_CONTRACT_VERSION,
                "rows": len(tables.rows[kind]),
                "columns": columns[kind],
            }
    return files, encoder.disclosures


def _encode_record_rows(
    tables: RecordTables, encoder: ValueEncoder, payload_policy: Mapping[str, bool]
) -> dict[str, dict[str, list[Any]]]:
    """Encode every row of every non-trace kind, column by column.

    A cell may reference a record no container held; it gets the next row of
    its table and is encoded on a later pass, so the loop runs until no new
    rows appear.
    """

    names = {kind: level_columns(kind, payload_policy) for kind, _, _ in RECORD_INDEX}
    columns: dict[str, dict[str, list[Any]]] = {
        kind: {name: [] for name in names[kind]} for kind in names
    }
    done = dict.fromkeys(names, 0)
    progressed = True
    while progressed:
        progressed = False
        for kind in names:
            if kind == "trace":
                continue
            schema = record_schema(kind)
            while done[kind] < len(tables.rows[kind]):
                row = done[kind]
                record = tables.rows[kind][row]
                state = dict(state_items(record))
                for name in names[kind]:
                    storage = schema[name].storage
                    columns[kind][name].append(
                        encoder.encode(
                            _cell(record, state, name, storage),
                            f"{kind}[{row}].{name}",
                            payload=storage == "payload",
                        )
                    )
                done[kind] += 1
                progressed = True
    return columns


def _dump(value: Any, *, sort_keys: bool = False) -> bytes:
    """Return the canonical bytes of one JSON file of the artifact."""

    text = json.dumps(
        value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=sort_keys
    )
    return text.encode("utf-8")


def _digest(data: bytes) -> dict[str, Any]:
    """Return the ``{sha256, bytes}`` entry of one file's content."""

    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def write_record_files(root: Path, files: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Write ``records/<kind>.json`` for every table; return the manifest index.

    Parameters
    ----------
    root:
        Staged artifact directory.
    files:
        Record file payloads by kind.

    Returns
    -------
    dict[str, Any]
        ``{record: {path, rows, sha256, bytes}}``.
    """

    (root / RECORDS_DIR).mkdir(exist_ok=True)
    index: dict[str, Any] = {}
    for kind, payload in files.items():
        relative = f"{RECORDS_DIR}/{kind}.json"
        data = _dump(payload)
        (root / relative).write_bytes(data)
        rows = int(payload.get("rows", 1))
        index[kind] = {"path": relative, "rows": rows, **_digest(data)}
    return index


def file_digests(root: Path) -> dict[str, dict[str, Any]]:
    """Return ``{relpath: {sha256, bytes}}`` for every file except the manifest."""

    digests: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.parent == root and path.name in _UNDIGESTED_ROOT_FILES:
            continue
        digests[path.relative_to(root).as_posix()] = _digest(path.read_bytes())
    return digests


def disclosure_summary(disclosures: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Return the manifest ``disclosures`` object, or ``None`` when there are none."""

    if not disclosures:
        return None
    return {
        "count": len(disclosures),
        "first": [str(item["at"]) for item in disclosures[:_DISCLOSURE_PATHS_SHOWN]],
    }


def envelope_fields(
    root: Path,
    *,
    kind: str,
    save_level: str,
    payload_policy: Mapping[str, bool],
    records: Mapping[str, Mapping[str, Any]],
    codecs: Sequence[str],
    disclosures: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Return the 3.0 envelope fields of a staged artifact.

    Call it last, after every other file is written: ``files`` digests what
    is on disk.

    Parameters
    ----------
    root:
        Staged artifact directory.
    kind:
        One of ``contract.ARTIFACT_KINDS``.
    save_level:
        Resolved save level.
    payload_policy:
        Resolved payload flags.
    records:
        Record table index from :func:`write_record_files`.
    codecs:
        Codec ids that wrote at least one payload.
    disclosures:
        Values written as ``unencodable``.

    Returns
    -------
    dict[str, Any]
        The envelope keys of ``contract.MANIFEST_REQUIRED_FIELDS`` beyond the
        writer identity, plus ``disclosures`` when there are any.
    """

    fields: dict[str, Any] = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_MAJOR,
        "tlspec_version": RECORD_CONTRACT_VERSION,
        "kind": kind,
        "save_level": save_level,
        "payload_policy": {name: bool(flag) for name, flag in payload_policy.items()},
        "contract_digest": writer_contract_digest(),
        "codecs": list(codecs),
        "records": {name: dict(entry) for name, entry in records.items()},
        "files": file_digests(root),
    }
    summary = disclosure_summary(disclosures)
    if summary is not None:
        fields["disclosures"] = summary
    return fields


def payload_index(tensor_refs: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, str]]:
    """Return the manifest ``tensors`` index: blob id to its chunk file and key.

    Parameters
    ----------
    tensor_refs:
        Blob id to tensor reference fields of every stored payload.

    Returns
    -------
    dict[str, dict[str, str]]
        ``{blob_id: {"blob": chunk relpath, "key": "t<k>"}}`` in blob-id order.
    """

    return {
        blob_id: {"blob": str(fields["blob"]), "key": str(fields["key"])}
        for blob_id, fields in sorted(tensor_refs.items())
        if "blob" in fields
    }


def make_encoder_factory(tensor_refs: Mapping[str, Mapping[str, Any]]) -> Any:
    """Return the encoder factory that resolves scrubbed blob references.

    Parameters
    ----------
    tensor_refs:
        Blob id to tensor reference fields for every payload the save stored.

    Returns
    -------
    Callable
        ``factory(record_ref) -> ValueEncoder``.
    """

    def tensor_ref(value: Any, at: str) -> dict[str, Any] | None:
        blob_id = getattr(value, "blob_id", None)
        if blob_id is None:
            return None
        fields = tensor_refs.get(str(blob_id))
        if fields is None:
            raise ArtifactError(
                f"The scrubbed state references payload {blob_id!r} at {at}, but the save "
                "stored no such payload.",
                code="bundle_save_failed",
                remedy="re-save; the payload selection and the metadata disagree",
            )
        return dict(fields)

    def factory(record_ref: Any) -> ValueEncoder:
        return ValueEncoder(record_ref=record_ref, tensor_ref=tensor_ref)

    return factory


def write_trace_envelope(
    root: Path,
    *,
    base_manifest: Mapping[str, Any],
    scrubbed_state: Mapping[str, Any],
    tensor_refs: Mapping[str, Mapping[str, Any]],
    codecs: Sequence[str],
    save_level: str,
    payload_policy: Mapping[str, bool],
    kind: str = "trace",
    extra_disclosures: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Write the record tables and 3.0 manifest of one staged trace.

    Parameters
    ----------
    root:
        Staged artifact directory (payload chunks already written).
    base_manifest:
        Manifest fields the save already builds (writer identity, backend,
        model signature and fingerprint, sites, runnable descriptor).
    scrubbed_state:
        Scrubbed trace state.
    tensor_refs:
        Blob id to tensor reference fields of every stored payload.
    codecs:
        Codec ids that wrote at least one payload.
    save_level:
        Resolved save level.
    payload_policy:
        Resolved payload flags.
    kind:
        Artifact kind recorded in the manifest.
    extra_disclosures:
        Disclosures of files written beside the records (the recipe).

    Returns
    -------
    dict[str, Any]
        The manifest written to ``manifest.json``.
    """

    files, disclosures = encode_tables(
        scrubbed_state,
        encoder_factory=make_encoder_factory(tensor_refs),
        payload_policy=payload_policy,
    )
    records = write_record_files(root, files)
    manifest: dict[str, Any] = dict(base_manifest)
    # The 2.x per-file tensor list becomes the blob-id index: the runnable
    # descriptor and the weight tables name payloads by blob id.
    manifest["tensors"] = payload_index(tensor_refs)
    manifest.pop("tensor_entries", None)
    manifest.update(
        envelope_fields(
            root,
            kind=kind,
            save_level=save_level,
            payload_policy=payload_policy,
            records=records,
            codecs=codecs,
            disclosures=[*disclosures, *extra_disclosures],
        )
    )
    (root / MANIFEST_FILE).write_bytes(_dump(manifest, sort_keys=True))
    return manifest


def _refuse_reload(detail: str, *, code: str) -> ArtifactError:
    """Return the save refusal for a staged artifact that does not read back."""

    return ArtifactError(
        f"The 3.0 writer produced an artifact that does not read back: {detail}",
        code=code,
        remedy="report this; the writer and the reader disagree",
    )


def reload_check(root: Path) -> None:
    """Re-read every record table of a staged artifact and decode every cell.

    One bounded JSON parse per file plus the typed decode: row counts, column
    lengths and the manifest index must agree with what was written.

    Parameters
    ----------
    root:
        Staged artifact directory.

    Raises
    ------
    ArtifactError
        ``bundle_save_failed`` when anything does not read back.
    """

    try:
        manifest = _json.read_bounded(root / MANIFEST_FILE)
    except (OSError, ValueError) as exc:
        raise _refuse_reload(f"manifest.json: {exc}", code="bundle_save_failed") from exc
    for kind, entry in dict(manifest.get("records") or {}).items():
        try:
            payload = _json.read_bounded(root / str(entry["path"]))
        except (OSError, ValueError, KeyError) as exc:
            raise _refuse_reload(f"records of {kind}: {exc}", code="bundle_save_failed") from exc
        if kind == "trace":
            decode_value(payload.get("fields"), at="trace")
            continue
        rows = payload.get("rows")
        if rows != entry.get("rows"):
            raise _refuse_reload(
                f"{kind} has {rows} rows, the manifest says {entry.get('rows')}",
                code="bundle_save_failed",
            )
        for name, cells in dict(payload.get("columns") or {}).items():
            if not isinstance(cells, list) or len(cells) != rows:
                raise _refuse_reload(
                    f"column {kind}.{name} does not have {rows} cells", code="bundle_save_failed"
                )
            decode_value(cells, at=f"{kind}.{name}")


def preflight_reload(root: Path) -> None:
    """Reload a staged artifact once before it is published (late.21).

    The writer's reload check runs first; then the 3.0 reader validates the
    artifact. Until the reader lane's ``validate_artifact`` is merged, its stub
    refusal is the only refusal this step tolerates.

    Parameters
    ----------
    root:
        Staged artifact directory.
    """

    reload_check(root)
    try:
        _reader.validate_artifact(root)
    except ArtifactError as exc:
        if not str(exc).startswith(_UNBUILT_DOOR_PREFIX):
            raise


def write_trace(
    trace: Any,
    path: str | Path,
    *,
    level: str = "portable",
    overwrite: bool = False,
    payload_policy: dict[str, bool] | None = None,
) -> None:
    """Write one trace as a 3.0 artifact directory.

    This is ``trace.save``: the save chokepoint writes the 3.0 envelope.

    Parameters
    ----------
    trace:
        Live trace to persist.
    path:
        Target directory; refused when it exists unless ``overwrite``.
    level:
        One of ``torchlens._io.format3.contract.SAVE_LEVELS``.
    overwrite:
        Replace an existing artifact atomically.
    payload_policy:
        Explicit payload flags (``include_outs`` and the rest); conflicts with
        the level refuse typed.
    """

    trace.save(path, level=level, overwrite=overwrite, **dict(payload_policy or {}))


def write_bundle(bundle: Any, path: str | Path, *, level: str = "portable") -> None:
    """Write a bundle: one member artifact per trace plus the relation table.

    This is ``bundle.save``; every member is written by :func:`write_trace`'s
    chokepoint.

    Parameters
    ----------
    bundle:
        Live bundle to persist.
    path:
        Target directory.
    level:
        Save level applied to every member.
    """

    bundle.save(path, level=level)


def write_intervention_recipe(
    spec: Any,
    path: str | Path,
    *,
    store: PayloadStore | None = None,
    save_level: str = "portable",
    helper_registry_version: Any = None,
    append_state: Mapping[str, Any] | None = None,
    function_registry_keys: Sequence[Mapping[str, Any]] = (),
    spec_derived: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Write a staged intervention spec as the typed recipe file.

    The same grammar serves the recipe inside a trace artifact and the
    separate recipe ``save_intervention`` writes (``recipe.build_recipe``).

    Parameters
    ----------
    spec:
        Staged intervention spec (rules and built-in edits).
    path:
        Artifact directory; ``intervention.json`` is written under it and the
        recipe tensors into ``blobs/recipe-<nnnn>.safetensors``.
    store:
        Payload store of the artifact, so recipe keys share its counter; the
        caller closes it. ``None`` opens and closes one here.
    save_level, helper_registry_version, append_state, function_registry_keys, spec_derived:
        Recipe header values (``recipe.build_recipe``).

    Returns
    -------
    tuple[dict[str, Any], list[dict[str, Any]]]
        The manifest ``intervention`` entry and the disclosures of values
        stored as text.
    """

    if spec is None or not hasattr(spec, "hook_specs"):
        raise ArtifactError(
            f"write_intervention_recipe needs a staged intervention spec, got {type(spec).__name__}.",
            code="bundle_save_failed",
            remedy="pass trace._intervention_spec or a loaded InterventionSpec",
        )
    root = Path(path)
    owned = store is None
    payloads = PayloadStore(root) if store is None else store
    recipe, disclosures = build_recipe(
        spec,
        store=payloads,
        save_level=save_level,
        helper_registry_version=helper_registry_version,
        append_state=dict(append_state or {}),
        function_registry_keys=function_registry_keys,
        spec_derived=spec_derived,
    )
    if owned:
        payloads.close()
    return write_recipe(root, recipe), disclosures


__all__ = [
    "RecordTables",
    "encode_tables",
    "envelope_fields",
    "file_digests",
    "payload_index",
    "preflight_reload",
    "record_kind_of",
    "record_type_table",
    "reload_check",
    "settle_live_side",
    "write_bundle",
    "write_intervention_recipe",
    "write_record_files",
    "write_trace",
    "write_trace_envelope",
]
