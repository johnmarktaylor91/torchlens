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

from collections.abc import Iterable
from typing import Any

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
    OpRecordAttributeError,
    ParamsFacet,
    PolicyFacet,
    TransformFacet,
    record_with_path_updates,
)

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
                raise OpRecordAttributeError(
                    f"OpRecord has no attribute {name!r} (strict protocol; "
                    "use the facet paths or the legacy property names)"
                )
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


def journal_record(event: Any) -> Any:
    """Return ``event``'s full record when it is a row handle, else ``event`` itself."""

    if type(event) is _RowOpRecord:
        return event.materialize()
    return event


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
        edges = _parent_edges(handle._cell("parents"), handle._cell("parent_arg_positions"))
        handle._parents = edges
    return edges


def _read_modules(handle: _RowOpRecord) -> tuple[Any, ...]:
    """Return the ``modules`` tuple the record's modules facet (or default) carries."""

    return tuple(handle._cell("modules"))


def _read_module_stack(handle: _RowOpRecord) -> Any:
    """Return the module frames, mirroring the freeze's facet presence."""

    stack = handle._rest.module_stack  # type: ignore[union-attr]
    if stack or handle._cell("modules"):
        return stack
    return ()


def _read_scalar_bool(handle: _RowOpRecord) -> Any:
    """Return the raw ``is_scalar_bool`` (the cell is ``bool``-coerced)."""

    rest = handle._rest
    if rest.is_scalar_bool is not None or handle._cell("bool_value") is not None:  # type: ignore[union-attr]
        return rest.is_scalar_bool  # type: ignore[union-attr]
    return None


def _read_transform_config(handle: _RowOpRecord) -> dict[str, object]:
    """Return the clean user transform mapping (the cell re-injects two channels)."""

    config = dict(handle._cell("transform_config"))
    config.pop("_tl_annotations", None)
    config.pop("fn_code_location", None)
    return config


def _read_templates(handle: _RowOpRecord) -> ArgTemplateRef:
    """Return the argument-template ref built from its five cells."""

    cell = handle._cell
    return ArgTemplateRef(
        saved_args=cell("saved_args"),
        saved_kwargs=cell("saved_kwargs"),
        args_template=cell("args_template"),
        kwargs_template=cell("kwargs_template"),
        has_saved_args=cell("has_saved_args"),
    )


def _cell_reader(cell_name: str) -> Any:
    """Return a fast read answering one record name from one row cell."""

    def read(handle: _RowOpRecord) -> Any:
        """Return the bound cell of ``handle``'s row."""

        return handle._cell(cell_name)

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


def _transform_facet(cell: Any) -> TransformFacet | None:
    """Return the transform facet the freeze builds, or ``None`` when it omits it."""

    config = dict(cell("transform_config"))
    config.pop("_tl_annotations", None)
    fn_code_location = config.pop("fn_code_location", None)
    facet = TransformFacet(
        is_transform=cell("is_transform"),
        transform_kind=cell("transform_kind"),
        transform_chain=cell("transform_chain"),
        transform_config=config,
        transform_fn_name=cell("transform_fn_name"),
        transform_fn_qualname=cell("transform_fn_qualname"),
        transform_fn_source=cell("transform_fn_source"),
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


def _ancestry_facet(cell: Any) -> AncestryFacet | None:
    """Return the ancestry facet the freeze builds, or ``None`` when it omits it."""

    input_ancestors = frozenset(cell("input_ancestors"))
    internal = frozenset(cell("internal_source_ancestors"))
    root = frozenset(cell("root_ancestors"))
    has_internal = cell("has_internal_source_ancestor")
    if not (input_ancestors or internal or root or has_internal):
        return None
    return AncestryFacet(
        input_ancestors=input_ancestors,
        internal_source_ancestors=internal,
        root_ancestors=root,
        has_internal_source_ancestor=has_internal,
    )


def _optional_facets(cell: Any, rest: _RowResidual) -> dict[str, Any]:
    """Return the optional facets keyed by record attribute, ``None`` when omitted."""

    modules = tuple(cell("modules"))
    qualname = cell("grad_fn_class_qualname")
    bool_value = cell("bool_value")
    annotations = dict(cell("annotations"))
    is_edited = cell("is_edited")
    fire_results = rest.fire_results
    present_control = rest.is_scalar_bool is not None or bool_value is not None
    return {
        "modules_facet": (
            ModulesFacet(module_stack=rest.module_stack, modules=modules)
            if rest.module_stack or modules
            else None
        ),
        "ancestry": _ancestry_facet(cell),
        "autograd": AutogradFacet(grad_fn_class_qualname=qualname)
        if qualname is not None
        else None,
        "transform": _transform_facet(cell),
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

    row, fids, rest = handle._row, handle._journal.fids, handle._rest

    def cell(name: str) -> Any:
        """Return one committed cell of the row being inverted."""

        return row[fids[name]]  # type: ignore[index]

    positions = cell("parent_arg_positions")
    core = OpCore(
        seq=handle.seq,
        kind=rest.kind,  # type: ignore[union-attr]
        label_raw=cell("_label_raw"),
        layer_label_raw=cell("_layer_label_raw"),
        layer_type=cell("type"),
        raw_index=cell("raw_index"),
        type_index=cell("type_index"),
        step_index=cell("step_index"),
        pass_index=cell("pass_number"),
        parents=_parent_edges(cell("parents"), positions),
        output=rest.output,  # type: ignore[union-attr]
        is_bottom_level=True,
        func_call_id=cell("func_call_id"),
    )
    record = OpRecord(
        core=core,
        function=rest.function,  # type: ignore[union-attr]
        templates=_read_templates(handle),
        graph=GraphFacet(
            parent_arg_positions={kind: dict(inner) for kind, inner in positions.items()},
            edge_uses=rest.edge_uses,  # type: ignore[union-attr]
            unattributed_tensor_args=cell("unattributed_tensor_args"),
            dropped_edge_tensor_args=cell("dropped_edge_tensor_args"),
            is_output_parent=cell("is_output_parent"),
            input_was_parameter=cell("input_was_parameter"),
            equivalence_class=cell("equivalence_class"),
        ),
        policy_facet=PolicyFacet(
            backend_semantics=rest.backend_semantics,  # type: ignore[union-attr]
            policy=rest.policy,  # type: ignore[union-attr]
            predicate_matched=True,
            tracing_finished=cell("_tracing_finished"),
            construction_done=cell("_construction_done"),
        ),
        **_optional_facets(cell, rest),  # type: ignore[arg-type]
    )
    if handle._patch:
        record = record_with_path_updates(record, handle._patch.items())
    return record
