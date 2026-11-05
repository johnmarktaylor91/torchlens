"""Row-backed exhaustive capture journal (capture C2b).

An exhaustive op's journal entry is a private row handle, :class:`_RowOpRecord`,
instead of an eager :class:`~torchlens._ir.op_record.OpRecord`. At commit the
backend writes the op's store cells into a capture-time ``OpRowStore`` row and
snapshots the few values the cells cannot carry exactly into a
:class:`_RowResidual` (the output ref, the function-call ref, parameter refs,
edge uses, module frames, backend semantics, the capture policy, the raw
``is_scalar_bool`` and the intervention fire results). Every one of those is
read AT COMMIT, so a later ``requires_grad`` flip or device move cannot change
the record that is built afterwards.

The handle answers the capture-time read surface (label, seq, parents, output,
module stack, function, ancestry and the flat names the live op view reads)
straight from the row and the residual. The full record is built on demand,
once, by :func:`_record_from_row`, the single inverse of the commit; after that
the handle delegates to the cached record and drops its row and residual.

Amendments never write into the row: folding a patch onto a handle returns a
new handle over the same row whose overlay holds the patched facet paths, so
the raw journal lane stays append-only and the fold is applied exactly once,
equal to ``record_with_path_updates`` on the eager record.
"""

from __future__ import annotations

from collections.abc import Container, Iterable
from typing import Any, cast

from .events import ArgTemplateRef, ParentEdge
from .op_record import (
    AncestryFacet,
    AnnotationsFacet,
    AutogradFacet,
    ControlFacet,
    GraphFacet,
    InterventionFacet,
    ModulesFacet,
    OpCore,
    OpRecord,
    ParamsFacet,
    PolicyFacet,
    TransformFacet,
    record_with_path_updates,
)
from .refs import journal_record

__all__: tuple[str, ...] = ()

#: Every name an ``OpRecord`` answers; anything else is refused without a build.
_RECORD_NAMES: frozenset[str] = frozenset(dir(OpRecord))

#: Flat read name -> the amendment facet path that overrides it.
_FLAT_PATH: dict[str, str] = {
    "parents": "core.parents",
    "output": "core.output",
    "parent_arg_positions": "graph.parent_arg_positions",
    "is_output_parent": "graph.is_output_parent",
    "predicate_matched": "policy.predicate_matched",
    "policy": "policy.policy",
    "intervention_fired": "intervention.intervention_fired",
    "is_edited": "intervention.is_edited",
    "fire_results": "intervention.fire_results",
    "record_context": "recording.record_context",
    "capture_spec": "recording.capture_spec",
    "input_ancestors": "ancestry.input_ancestors",
    "internal_source_ancestors": "ancestry.internal_source_ancestors",
    "root_ancestors": "ancestry.root_ancestors",
    "has_internal_source_ancestor": "ancestry.has_internal_source_ancestor",
}
_OVERLAY_PATHS: frozenset[str] = frozenset(_FLAT_PATH.values())


class _RowResidual:
    """Commit-time snapshot of the record values the row cells cannot carry exactly.

    Holds only objects the eager record also holds (the same refs, frames,
    parameter refs and fire results), so it adds no strong reference.
    """

    __slots__ = (
        "kind",
        "output",
        "function",
        "module_stack",
        "edge_uses",
        "params",
        "parent_params",
        "backend_semantics",
        "policy",
        "is_scalar_bool",
        "fire_results",
        "__weakref__",
    )

    def __init__(self, **values: Any) -> None:
        """Store every residual slot from keyword arguments (all are required)."""

        for name in self.__slots__[:-1]:
            object.__setattr__(self, name, values.pop(name))
        if values:
            raise TypeError(f"unknown residual fields {sorted(values)}")


class _RowJournal:
    """The capture-time row store an exhaustive journal's handles read from."""

    __slots__ = ("store", "fids")

    def __init__(self, store: Any) -> None:
        """Bind a building ``OpRowStore`` and its field-id map."""

        self.store = store
        self.fids: dict[str, int] = store.layout.fid_by_name

    def handle(self, row_index: int, rest: _RowResidual) -> _RowOpRecord:
        """Return a fresh, unamended handle over committed row ``row_index``."""

        return _RowOpRecord(self, self.store._rows[row_index], rest, {})


class _RowOpRecord:
    """Private journal handle for one exhaustive op: row cells plus residual.

    Reads the capture-time surface without building a record; any other
    ``OpRecord`` name builds the record once (:func:`_record_from_row`) and
    delegates to it from then on.
    """

    __slots__ = ("_journal", "_row", "_rest", "_patch", "_record", "_parents", "seq")

    def __init__(
        self, journal: _RowJournal, row: list[Any], rest: _RowResidual, patch: dict[str, Any]
    ) -> None:
        """Bind the row, its residual and an amendment overlay (path -> value)."""

        self._journal = journal
        self._row: list[Any] | None = row
        self._rest: _RowResidual | None = rest
        self._patch = patch
        self._record: OpRecord | None = None
        self._parents: tuple[ParentEdge, ...] | None = None
        self.seq = 0

    def __repr__(self) -> str:
        """Return a bounded repr naming the op without building its record."""

        return f"<row op record {self.label_raw!r} seq={self.seq}>"

    def _cell(self, name: str) -> Any:
        """Return one committed cell of this op's row."""

        return self._row[self._journal.fids[name]]  # type: ignore[index]

    def materialize(self) -> OpRecord:
        """Return the full record, building it once and dropping row and residual."""

        record = self._record
        if record is None:
            record = _record_from_row(self)
            self._record = record
            self._row = None
            self._rest = None
            self._parents = None
        return record

    def take_row(self) -> tuple[list[Any], OpRecord] | None:
        """Return this unamended handle's row and its built record, for ingest (capture C2c).

        The record is built first, from the row as committed, so ingest may
        then write join cells into the row and adopt it as the op's store row.
        Returns ``None`` for an amended or already built handle, which ingest
        reads through the record path instead.
        """

        row = self._row
        if row is None or self._patch:
            return None
        return row, self.materialize()

    def build_if_reading(self, row_ids: Container[int]) -> None:
        """Build the record now when this handle still reads a row in ``row_ids`` (capture C2c).

        Ingest writes into the rows it takes; a handle built first never reads
        them again.
        """

        if self._row is not None and id(self._row) in row_ids:
            self.materialize()

    def with_patch(self, items: Iterable[tuple[str, Any]]) -> Any:
        """Return the handle (or record) with ordered facet-path updates folded on."""

        if self._record is not None:
            return record_with_path_updates(self._record, items)
        patch = dict(self._patch)
        patch.update(items)
        folded = _RowOpRecord(self._journal, self._row, self._rest, patch)  # type: ignore[arg-type]
        folded.seq = self.seq
        return folded

    def __getattr__(self, name: str) -> Any:
        """Answer a record read from the row, or build the record for the rest."""

        if name in _HANDLE_SLOTS or name[:2] == "__":
            # An unset slot (copy, unpickle) or a protocol probe: never recurse into a build.
            raise AttributeError(name)
        record = self._record
        if record is not None:
            return getattr(record, name)
        getter = _FAST_READS.get(name)
        if getter is None or not _overlay_known(self._patch):
            if name not in _RECORD_NAMES:
                # The record's own strict refusal (OpRecordAttributeError), no build.
                return OpRecord.__getattr__(cast(OpRecord, self), name)
            return getattr(self.materialize(), name)
        patch = self._patch
        if patch:
            path = _FLAT_PATH.get(name)
            if path is not None and path in patch:
                return patch[path]
        return getter(self)


_HANDLE_SLOTS = frozenset(_RowOpRecord.__slots__)


def materialize_journal_ops(events: Any) -> None:
    """Swap every row handle on ``events``' op lane for its built record, in place.

    For readers of the raw journal outside ingest's replay clone (the failed
    capture's partial trace): each entry becomes the record it already stands
    for, so the lane's values are unchanged; the amendment fold is recomputed.
    """

    lane = events.op_events
    if any(type(event) is _RowOpRecord for event in lane):
        lane[:] = [journal_record(event) for event in lane]
        events._amended_fold_cache = None


def _overlay_known(patch: dict[str, Any]) -> bool:
    """Return whether every overlay path is one the fast reads honour."""

    return all(path in _OVERLAY_PATHS for path in patch)


def _parent_edges(
    parents: list[str], positions: dict[str, dict[Any, str]]
) -> tuple[ParentEdge, ...]:
    """Return the parent edges the freeze builds from parent labels and positions."""

    first: dict[str, tuple[Any, str]] = {}
    for location, label in positions["args"].items():
        first.setdefault(label, (location, "arg"))
    for location, label in positions["kwargs"].items():
        first.setdefault(label, (location, "kwarg"))
    edges = []
    for label in parents:
        location, use = first.get(label, (None, "arg"))
        edges.append(ParentEdge(parent_label_raw=label, arg_position=location, _edge_use=use))
    return tuple(edges)


def _read_parents(handle: _RowOpRecord) -> tuple[ParentEdge, ...]:
    """Return the unamended parent edges, built from the row once per handle."""

    edges = handle._parents
    if edges is None:
        row, fids = handle._row, handle._journal.fids
        edges = _parent_edges(row[fids["parents"]], row[fids["parent_arg_positions"]])  # type: ignore[index]
        handle._parents = edges
    return edges


def _read_modules(handle: _RowOpRecord) -> tuple[Any, ...]:
    """Return the ``modules`` tuple the record's modules facet (or default) carries."""

    return tuple(handle._row[handle._journal.fids["modules"]])  # type: ignore[index]


def _read_module_stack(handle: _RowOpRecord) -> Any:
    """Return the module frames, mirroring the freeze's facet presence."""

    stack = handle._rest.module_stack  # type: ignore[union-attr]
    if stack or handle._row[handle._journal.fids["modules"]]:  # type: ignore[index]
        return stack
    return ()


def _read_scalar_bool(handle: _RowOpRecord) -> Any:
    """Return the raw ``is_scalar_bool`` (the cell is ``bool``-coerced)."""

    rest = handle._rest
    if (
        rest.is_scalar_bool is not None
        or handle._row[handle._journal.fids["bool_value"]] is not None
    ):  # type: ignore[union-attr,index]
        return rest.is_scalar_bool  # type: ignore[union-attr]
    return None


def _read_transform_config(handle: _RowOpRecord) -> dict[str, object]:
    """Return the clean user transform mapping (the cell re-injects two channels)."""

    config = dict(handle._row[handle._journal.fids["transform_config"]])  # type: ignore[index]
    config.pop("_tl_annotations", None)
    config.pop("fn_code_location", None)
    return config


def _templates_from_row(row: list[Any], fids: dict[str, int]) -> ArgTemplateRef:
    """Return the argument-template ref built from a row's five template cells."""

    return ArgTemplateRef(
        saved_args=row[fids["saved_args"]],
        saved_kwargs=row[fids["saved_kwargs"]],
        args_template=row[fids["args_template"]],
        kwargs_template=row[fids["kwargs_template"]],
        has_saved_args=row[fids["has_saved_args"]],
    )


def _read_templates(handle: _RowOpRecord) -> ArgTemplateRef:
    """Return the argument-template ref built from its five cells."""

    return _templates_from_row(handle._row, handle._journal.fids)  # type: ignore[arg-type]


def _cell_reader(cell_name: str) -> Any:
    """Return a fast read answering one record name from one row cell."""

    def read(handle: _RowOpRecord) -> Any:
        """Return the bound cell of ``handle``'s row."""

        return handle._row[handle._journal.fids[cell_name]]  # type: ignore[index]

    read._cell_name = cell_name  # type: ignore[attr-defined]
    return read


def _frozen_cell_reader(cell_name: str) -> Any:
    """Return a fast read answering a frozenset record name from a set cell."""

    def read(handle: _RowOpRecord) -> frozenset[str]:
        """Return the bound set cell as a frozenset."""

        return frozenset(handle._cell(cell_name))

    return read


def _rest_reader(slot: str) -> Any:
    """Return a fast read answering one record name from one residual slot."""

    def read(handle: _RowOpRecord) -> Any:
        """Return the bound residual slot of ``handle``."""

        return getattr(handle._rest, slot)

    read._rest_slot = slot  # type: ignore[attr-defined]
    return read


def _constant_reader(value: Any) -> Any:
    """Return a fast read answering one record name with a constant."""

    def read(handle: _RowOpRecord) -> Any:
        """Return the bound constant."""

        del handle
        return value

    return read


_FAST_READS: dict[str, Any] = {
    "kind": _rest_reader("kind"),
    "label_raw": _cell_reader("_label_raw"),
    "layer_label_raw": _cell_reader("_layer_label_raw"),
    "layer_type": _cell_reader("type"),
    "raw_index": _cell_reader("raw_index"),
    "type_index": _cell_reader("type_index"),
    "step_index": _cell_reader("step_index"),
    "pass_index": _cell_reader("pass_number"),
    "parents": _read_parents,
    "output": _rest_reader("output"),
    "is_bottom_level": _constant_reader(True),
    "func_call_id": _cell_reader("func_call_id"),
    "function": _rest_reader("function"),
    "templates": _read_templates,
    "parent_arg_positions": _cell_reader("parent_arg_positions"),
    "_edge_uses": _rest_reader("edge_uses"),
    "unattributed_tensor_args": _cell_reader("unattributed_tensor_args"),
    "dropped_edge_tensor_args": _cell_reader("dropped_edge_tensor_args"),
    "is_output_parent": _cell_reader("is_output_parent"),
    "input_was_parameter": _cell_reader("input_was_parameter"),
    "equivalence_class": _cell_reader("equivalence_class"),
    "module_stack": _read_module_stack,
    "modules": _read_modules,
    "input_ancestors": _frozen_cell_reader("input_ancestors"),
    "internal_source_ancestors": _frozen_cell_reader("internal_source_ancestors"),
    "root_ancestors": _frozen_cell_reader("root_ancestors"),
    "has_internal_source_ancestor": _cell_reader("has_internal_source_ancestor"),
    "grad_fn_class_qualname": _cell_reader("grad_fn_class_qualname"),
    "is_transform": _cell_reader("is_transform"),
    "transform_kind": _cell_reader("transform_kind"),
    "transform_chain": _cell_reader("transform_chain"),
    "transform_config": _read_transform_config,
    "transform_fn_name": _cell_reader("transform_fn_name"),
    "transform_fn_qualname": _cell_reader("transform_fn_qualname"),
    "transform_fn_source": _cell_reader("transform_fn_source"),
    "is_scalar_bool": _read_scalar_bool,
    "bool_value": _cell_reader("bool_value"),
    "params": _rest_reader("params"),
    "parent_params": _rest_reader("parent_params"),
    "backend_semantics": _rest_reader("backend_semantics"),
    "policy": _rest_reader("policy"),
    "predicate_matched": _constant_reader(True),
    "tracing_finished": _cell_reader("_tracing_finished"),
    "construction_done": _cell_reader("_construction_done"),
    "record_context": _constant_reader(None),
    "capture_spec": _constant_reader(None),
    "intervention_fired": lambda handle: bool(handle._rest.fire_results),
    "is_edited": _cell_reader("is_edited"),
    "fire_results": _rest_reader("fire_results"),
    "intervention_template_ref": _constant_reader(None),
    "source_trace": _constant_reader(None),
    "source_trace_id": _constant_reader(None),
}


def _patched_read(handle: _RowOpRecord, name: str) -> Any:
    """Answer a fast name on an amended, unbuilt handle (the ``__getattr__`` overlay rule)."""

    patch = handle._patch
    if not _overlay_known(patch):
        return getattr(handle.materialize(), name)
    path = _FLAT_PATH.get(name)
    if path is not None and path in patch:
        return patch[path]
    return _FAST_READS[name](handle)


def _fast_property(name: str, read: Any) -> property:
    """Return the handle property answering ``name`` in one call, without a failed lookup.

    The order is ``__getattr__``'s: the built record, then an amendment overlay,
    then the row (a cell read inline, a residual slot inline, else ``read``).
    """

    cell_name = getattr(read, "_cell_name", None)
    rest_slot = getattr(read, "_rest_slot", None)
    if cell_name is not None:

        def get_cell(handle: _RowOpRecord) -> Any:
            """Read ``name`` from the record, the overlay, else its row cell."""

            record = handle._record
            if record is not None:
                return getattr(record, name)
            if handle._patch:
                return _patched_read(handle, name)
            return handle._row[handle._journal.fids[cell_name]]  # type: ignore[index]

        return property(get_cell, doc=f"Record read ``{name}`` (row cell ``{cell_name}``).")
    if rest_slot is not None:

        def get_rest(handle: _RowOpRecord) -> Any:
            """Read ``name`` from the record, the overlay, else its residual slot."""

            record = handle._record
            if record is not None:
                return getattr(record, name)
            if handle._patch:
                return _patched_read(handle, name)
            return getattr(handle._rest, rest_slot)

        return property(get_rest, doc=f"Record read ``{name}`` (residual ``{rest_slot}``).")

    def get(handle: _RowOpRecord) -> Any:
        """Read ``name`` from the record, the overlay, else its fast reader."""

        record = handle._record
        if record is not None:
            return getattr(record, name)
        if handle._patch:
            return _patched_read(handle, name)
        return read(handle)

    return property(get, doc=f"Record read ``{name}``.")


# The fast names are class properties, so a read is one call; ``__getattr__``
# (reached only after a failed normal lookup) keeps every other record name.
for _fast_name, _fast_read in _FAST_READS.items():
    setattr(_RowOpRecord, _fast_name, _fast_property(_fast_name, _fast_read))
del _fast_name, _fast_read


def _transform_facet(row: list[Any], fids: dict[str, int]) -> TransformFacet | None:
    """Return the transform facet the freeze builds, or ``None`` when it omits it."""

    config = dict(row[fids["transform_config"]])
    config.pop("_tl_annotations", None)
    fn_code_location = config.pop("fn_code_location", None)
    facet = TransformFacet(
        is_transform=row[fids["is_transform"]],
        transform_kind=row[fids["transform_kind"]],
        transform_chain=row[fids["transform_chain"]],
        transform_config=config,
        transform_fn_name=row[fids["transform_fn_name"]],
        transform_fn_qualname=row[fids["transform_fn_qualname"]],
        transform_fn_source=row[fids["transform_fn_source"]],
        fn_code_location=fn_code_location,
    )
    present = (
        facet.is_transform
        or facet.transform_kind is not None
        or facet.transform_chain
        or config
        or facet.transform_fn_name is not None
        or facet.transform_fn_qualname is not None
        or facet.transform_fn_source is not None
        or fn_code_location is not None
    )
    return facet if present else None


def _ancestry_facet(row: list[Any], fids: dict[str, int]) -> AncestryFacet | None:
    """Return the ancestry facet the freeze builds, or ``None`` when it omits it."""

    input_ancestors = frozenset(row[fids["input_ancestors"]])
    internal = frozenset(row[fids["internal_source_ancestors"]])
    root = frozenset(row[fids["root_ancestors"]])
    has_internal = row[fids["has_internal_source_ancestor"]]
    if not (input_ancestors or internal or root or has_internal):
        return None
    return AncestryFacet(
        input_ancestors=input_ancestors,
        internal_source_ancestors=internal,
        root_ancestors=root,
        has_internal_source_ancestor=has_internal,
    )


def _optional_facets(row: list[Any], fids: dict[str, int], rest: _RowResidual) -> dict[str, Any]:
    """Return the optional facets keyed by record attribute, ``None`` when omitted."""

    modules = tuple(row[fids["modules"]])
    qualname = row[fids["grad_fn_class_qualname"]]
    bool_value = row[fids["bool_value"]]
    annotations = dict(row[fids["annotations"]])
    is_edited = row[fids["is_edited"]]
    fire_results = rest.fire_results
    present_control = rest.is_scalar_bool is not None or bool_value is not None
    return {
        "modules_facet": (
            ModulesFacet(module_stack=rest.module_stack, modules=modules)
            if rest.module_stack or modules
            else None
        ),
        "ancestry": _ancestry_facet(row, fids),
        "autograd": AutogradFacet(grad_fn_class_qualname=qualname)
        if qualname is not None
        else None,
        "transform": _transform_facet(row, fids),
        "control": (
            ControlFacet(is_scalar_bool=rest.is_scalar_bool, bool_value=bool_value)
            if present_control
            else None
        ),
        "params_facet": (
            ParamsFacet(params=rest.params, parent_params=rest.parent_params)
            if rest.params or rest.parent_params
            else None
        ),
        "annotations_facet": AnnotationsFacet(annotations=annotations) if annotations else None,
        "intervention": (
            InterventionFacet(
                intervention_fired=bool(fire_results),
                is_edited=is_edited,
                fire_results=fire_results,
            )
            if fire_results or is_edited
            else None
        ),
    }


def _record_from_row(handle: _RowOpRecord) -> OpRecord:
    """Build the full record from a handle's row, residual and overlay (the one inverse).

    Mirrors the exhaustive freeze field for field and facet presence for facet
    presence, then applies the amendment overlay through the same fold
    primitive the eager journal uses.
    """

    row = cast(list[Any], handle._row)
    rest = cast(_RowResidual, handle._rest)
    fids = handle._journal.fids

    positions = row[fids["parent_arg_positions"]]
    core = OpCore(
        seq=handle.seq,
        kind=rest.kind,
        label_raw=row[fids["_label_raw"]],
        layer_label_raw=row[fids["_layer_label_raw"]],
        layer_type=row[fids["type"]],
        raw_index=row[fids["raw_index"]],
        type_index=row[fids["type_index"]],
        step_index=row[fids["step_index"]],
        pass_index=row[fids["pass_number"]],
        parents=_parent_edges(row[fids["parents"]], positions),
        output=rest.output,
        is_bottom_level=True,
        func_call_id=row[fids["func_call_id"]],
    )
    record = OpRecord(
        core=core,
        function=rest.function,
        templates=_templates_from_row(row, fids),
        graph=GraphFacet(
            parent_arg_positions={kind: dict(inner) for kind, inner in positions.items()},
            edge_uses=rest.edge_uses,
            unattributed_tensor_args=row[fids["unattributed_tensor_args"]],
            dropped_edge_tensor_args=row[fids["dropped_edge_tensor_args"]],
            is_output_parent=row[fids["is_output_parent"]],
            input_was_parameter=row[fids["input_was_parameter"]],
            equivalence_class=row[fids["equivalence_class"]],
        ),
        policy_facet=PolicyFacet(
            backend_semantics=rest.backend_semantics,
            policy=rest.policy,
            predicate_matched=True,
            tracing_finished=row[fids["_tracing_finished"]],
            construction_done=row[fids["_construction_done"]],
        ),
        **_optional_facets(row, fids, rest),  # type: ignore[arg-type]
    )
    if handle._patch:
        record = record_with_path_updates(record, handle._patch.items())
    return record
