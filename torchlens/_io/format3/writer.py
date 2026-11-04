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

import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..._data_classes._state_adapter import state_items
from .. import _json
from ..field_registry import RECORD_CONTRACT_CLASSES
from ..format_errors import ArtifactError
from ..prerelease import (
    PRERELEASE_STATE_KEY,
    prerelease_fields_active,
    registered_prerelease_fields,
)
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
from .decode import decode_value
from .payloads import PayloadStore
from .recipe import RecipeHeader, build_recipe, write_recipe
from .schema import LiveCells, level_columns, record_kind_of, record_schema
from .values import ValueEncoder, record_type_table, tagged

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
    ("_io_module_accessor_state", "module"),
    ("_param_logs", "param"),
    ("_grad_fn_logs", "grad_fn"),
    ("_backward_pass_logs", "backward_pass"),
)
#: Disclosure paths listed in the manifest (the full list is in the cells).
_DISCLOSURE_PATHS_SHOWN = 16
#: Op keys whose absent map is written as an empty object (contract page,
#: "Values settled on the live side"); the live op is settled the same way.
_EMPTY_MAP_OP_KEYS = ("edge_substitutions", "edge_replacement_stamps")


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
        """Seed canonical containers; encoding discovers the remaining persisted edges.

        Walking arbitrary runtime state also reaches stale copies of parameter records
        after a fork. Only cells the format actually writes may add records beyond these
        canonical seeds; ``_encode_record_rows`` already follows those cells to a fixed point.
        """

        for key, kind in _SEED_KEYS:
            for item in _container_items(self.state.get(key)):
                if record_kind_of(item) == kind:
                    self.add(item, kind)

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

    for op in list(trace._layer_list or ()):
        for key in _EMPTY_MAP_OP_KEYS:
            if getattr(op, key, None) is None:
                setattr(op, key, {})


def _write_columns(record: str, payload_policy: Mapping[str, bool]) -> list[str]:
    """Add only switch-activated registrar fields to the normal persisted columns."""

    names = level_columns(record, payload_policy)
    if not prerelease_fields_active():
        return names
    owner = RECORD_CONTRACT_CLASSES[record][1]
    extra = list(registered_prerelease_fields().get(owner, ()))
    if record == "trace":
        extra.append(PRERELEASE_STATE_KEY)
    return [*names, *(name for name in extra if name not in names)]


def encode_tables(
    state: Mapping[str, Any],
    *,
    encoder_factory: Any,
    payload_policy: Mapping[str, bool],
    cells: LiveCells | None = None,
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
    cells:
        Cell resolver holding the live originals of the scrubbed copies.

    Returns
    -------
    tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]
        Record file payloads by kind (kinds without rows omitted) and the
        disclosures of values written as ``unencodable``.
    """

    cells = cells or LiveCells()
    tables = RecordTables(state)
    tables.discover()
    encoder: ValueEncoder = encoder_factory(tables.reference)
    schema = record_schema("trace")
    fields = {
        name: encoder.encode(
            cells.cell("trace", state, state, name),
            f"trace.{name}",
            payload=name in schema and schema[name].storage == "payload",
        )
        for name in _write_columns("trace", payload_policy)
    }
    files: dict[str, dict[str, Any]] = {
        "trace": {"record": "trace", "tlspec_version": RECORD_CONTRACT_VERSION, "fields": fields}
    }
    columns = _encode_record_rows(tables, encoder, payload_policy, cells)
    for kind, _, _ in RECORD_INDEX:
        if kind != "trace" and tables.rows[kind]:
            files[kind] = {
                "record": kind,
                "tlspec_version": RECORD_CONTRACT_VERSION,
                "rows": len(tables.rows[kind]),
                "columns": columns[kind],
            }
    return files, encoder.disclosures


# Complexity: fixed-point row discovery must encode newly referenced rows before the same pass ends.
def _encode_record_rows(
    tables: RecordTables,
    encoder: ValueEncoder,
    payload_policy: Mapping[str, bool],
    cells: LiveCells,
) -> dict[str, dict[str, list[Any]]]:
    """Encode every row of every non-trace kind, column by column.

    A cell may reference a record no container held; it gets the next row of
    its table and is encoded on a later pass, so the loop runs until no new
    rows appear.
    """

    names = {kind: _write_columns(kind, payload_policy) for kind, _, _ in RECORD_INDEX}
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
                    columns[kind][name].append(
                        encoder.encode(
                            cells.cell(kind, record, state, name),
                            f"{kind}[{row}].{name}",
                            payload=name in schema and schema[name].storage == "payload",
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


@dataclasses.dataclass(frozen=True)
class EnvelopeSpec:
    """What the 3.0 envelope of a staged artifact reports about its contents.

    Attributes
    ----------
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
    """

    kind: str
    save_level: str
    payload_policy: Mapping[str, bool] = dataclasses.field(default_factory=dict)
    records: Mapping[str, Mapping[str, Any]] = dataclasses.field(default_factory=dict)
    codecs: Sequence[str] = ()
    disclosures: Sequence[Mapping[str, Any]] = ()


@dataclasses.dataclass(frozen=True)
class TraceTables:
    """The scrubbed state of one staged trace and what its record cells read.

    Attributes
    ----------
    scrubbed_state:
        Scrubbed trace state.
    tensor_refs:
        Blob id to reference fields of every stored payload.
    record_pairs, include_source:
        ``(scrubbed copy, live original)`` pairs and the source flag (``LiveCells``).
    """

    scrubbed_state: Mapping[str, Any]
    tensor_refs: Mapping[str, Mapping[str, Any]]
    record_pairs: Sequence[tuple[Any, Any]] = ()
    include_source: bool = True


def envelope_fields(root: Path, envelope: EnvelopeSpec) -> dict[str, Any]:
    """Return the 3.0 envelope fields of a staged artifact.

    Call it last, after every other file is written: ``files`` digests what
    is on disk.

    Parameters
    ----------
    root:
        Staged artifact directory.
    envelope:
        Kind, save level, payload flags, record index, codecs and disclosures.

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
        "kind": envelope.kind,
        "save_level": envelope.save_level,
        "payload_policy": {name: bool(flag) for name, flag in envelope.payload_policy.items()},
        "contract_digest": writer_contract_digest(),
        "codecs": list(envelope.codecs),
        "records": {name: dict(entry) for name, entry in envelope.records.items()},
        "files": file_digests(root),
    }
    summary = disclosure_summary(envelope.disclosures)
    if summary is not None:
        fields["disclosures"] = summary
    return fields


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
        """The stored reference fields for a blob reference at ``at`` (None for any other value)."""
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
        """A value encoder for one record that resolves blob references to stored payloads."""
        return ValueEncoder(record_ref=record_ref, tensor_ref=tensor_ref)

    return factory


def write_trace_envelope(
    root: Path, *, base_manifest: Mapping[str, Any], tables: TraceTables, envelope: EnvelopeSpec
) -> dict[str, Any]:
    """Write the record tables and 3.0 manifest of one staged trace.

    Parameters
    ----------
    root, base_manifest:
        Staged artifact directory (payload chunks written); manifest fields the
        save already builds (writer identity, backend, signature, sites, runnable).
    tables:
        Scrubbed state, payload references and live record pairs to encode.
    envelope:
        Kind, save level, payload flags and codecs; its ``disclosures`` are
        those of files written beside the records (the recipe), and the record
        index and record disclosures are added here.

    Returns
    -------
    dict[str, Any]
        The manifest written to ``manifest.json``.
    """

    files, disclosures = encode_tables(
        tables.scrubbed_state,
        encoder_factory=make_encoder_factory(tables.tensor_refs),
        payload_policy=envelope.payload_policy,
        cells=LiveCells(tables.record_pairs, include_source=tables.include_source),
    )
    records = write_record_files(root, files)
    manifest: dict[str, Any] = dict(base_manifest)
    manifest.pop("run", None)
    manifest.pop("tensors", None)
    manifest.pop("tensor_entries", None)
    filled = dataclasses.replace(
        envelope, records=records, disclosures=[*disclosures, *envelope.disclosures]
    )
    manifest.update(envelope_fields(root, filled))
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

    _require_saveable("write_trace", trace, "a live Trace")
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

    _require_saveable("write_bundle", bundle, "a live Bundle")
    bundle.save(path, level=level)


def _require_saveable(door: str, value: Any, expected: str) -> None:
    """Refuse a value without a ``save`` door, typed (``bundle_save_failed``)."""

    if not callable(getattr(value, "save", None)):
        raise ArtifactError(
            f"{door} needs {expected}, got {type(value).__name__}.",
            code="bundle_save_failed",
            remedy=f"pass {expected} to {door}",
        )


def write_intervention_recipe(
    spec: Any,
    path: str | Path,
    *,
    store: PayloadStore | None = None,
    header: RecipeHeader | None = None,
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
    header:
        Recipe header values (``recipe.RecipeHeader``); ``None`` is a portable
        recipe with no trace context.

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
    recipe, disclosures = build_recipe(spec, store=payloads, header=header or RecipeHeader())
    if owned:
        payloads.close()
    return write_recipe(root, recipe), disclosures


__all__ = (
    "EnvelopeSpec",
    "RecordTables",
    "TraceTables",
    "encode_tables",
    "envelope_fields",
    "file_digests",
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
)
