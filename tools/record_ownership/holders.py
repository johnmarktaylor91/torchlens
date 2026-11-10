"""The pinned holder set: every strong path from trace-owned storage to a record.

A row is one path as the audit prints it (the holder chain, then the record
kind it reaches), the conversion step that removes it, and the audit cases in
which it appears. ``tests/test_record_ownership.py`` compares this table with
the live audit, case by case: a path the audit finds that has no row is a new
holder, and a row the audit no longer finds is stale. A step that converts a
holder to row keys deletes its rows here in the same change.

Records can be handed out weakly and hold their trace only once no row is
left for that record kind. ``Op`` and ``Layer`` go first, so every path that
reaches one of them must belong to a step in ``BEFORE_ACTIVATION``.

Reading a path: ``.name`` is an attribute or slot, ``[*]`` any position of a
sequence, ``{*}`` any value of a dictionary, ``{name}`` one named entry,
``:Type`` the object the reference passes through, and ``Store.rows[field]``
a cell of that field in a row store. Paths that start at ``Trace`` are held
by a live trace; paths that start at ``TraceState`` or ``<Kind>State`` are
what pickle (and so a deep copy) carries; paths that start at a record kind
are held by that record's own state.
"""

from __future__ import annotations

from dataclasses import dataclass

OP_LIST = "op_list"
LAYER_STATE = "layer_state"
OP_ROW_CELLS = "op_row_cells"
MODULE_AND_BUFFER_CELLS = "module_and_buffer_cells"
FORK_AND_SAVED_STATE = "fork_and_saved_state"
LATER_KINDS = "later_kinds"
ACTIVATION = "activation"
# The one strong cache of records: a path may wait for ``ACTIVATION`` only through it.
SIDE_TABLE = "Trace._trace_core:TraceCore._strong_facades"

STEPS: dict[str, str] = {
    OP_LIST: "the op list, the label dictionaries and the caches of the ops door",
    LAYER_STATE: "Layer records, the layers door and each Layer's op accessor",
    OP_ROW_CELLS: "cells of a record's own row: the facet cache and gradient records",
    MODULE_AND_BUFFER_CELLS: "module cells (calls, params) and buffer versions",
    FORK_AND_SAVED_STATE: "fork translation, copy-on-write overlays and pickled state",
    LATER_KINDS: "holders of the kinds activated after Op and Layer",
    ACTIVATION: "the trace core's strong view table, removed when views become disposable",
}

BEFORE_ACTIVATION = frozenset(
    {OP_LIST, LAYER_STATE, OP_ROW_CELLS, MODULE_AND_BUFFER_CELLS, FORK_AND_SAVED_STATE}
)
ACTIVATED_FIRST = ("Op", "Layer")


@dataclass(frozen=True)
class Holder:
    """One pinned path: its conversion step and the cases it appears in.

    ``only`` names the cases the path appears in; when it is empty the path
    appears in every case except those in ``but``.
    """

    step: str
    only: tuple[str, ...] = ()
    but: tuple[str, ...] = ()

    def appears_in(self, case: str) -> bool:
        """Return whether the audit of ``case`` is expected to find this path."""

        return case in self.only if self.only else case not in self.but


HOLDERS: dict[str, Holder] = {
    "Trace._layer_dict_all_keys{*} -> Op": Holder(OP_LIST, but=("pickle_state",)),
    "Trace._layer_dict_main_keys{*} -> Op": Holder(OP_LIST, but=("pickle_state",)),
    "Trace._layer_list[*] -> Op": Holder(OP_LIST, but=("pickle_state",)),
    "Trace._op_accessor_cache[*]:TraceOpAccessor._alias_indexes:_AliasIndexes.final{*} -> Op": Holder(
        OP_LIST, only=("alias_lookup",)
    ),
    "Trace._op_accessor_cache[*]:TraceOpAccessor._alias_indexes:_AliasIndexes.layer{*}[*] -> Op": Holder(
        OP_LIST, only=("alias_lookup",)
    ),
    "Trace._op_accessor_cache[*]:TraceOpAccessor._alias_indexes:_AliasIndexes.raw{*} -> Op": Holder(
        OP_LIST, only=("alias_lookup",)
    ),
    "Trace._op_accessor_cache[*]:TraceOpAccessor._counted_cache[*] -> Op": Holder(
        OP_LIST, but=("fork_cold", "pickle_state")
    ),
    "Trace._op_accessor_cache[*]:TraceOpAccessor._dict{*} -> Op": Holder(
        OP_LIST, but=("fork_cold", "pickle_state")
    ),
    "Trace._op_accessor_cache[*]:TraceOpAccessor._layer_label_index{*}[*] -> Op": Holder(
        OP_LIST,
        but=("alias_lookup", "facets_read", "fork_cold", "pickle_state", "plain", "save_none"),
    ),
    "Trace._op_accessor_cache[*]:TraceOpAccessor._list[*] -> Op": Holder(
        OP_LIST, but=("fork_cold", "pickle_state")
    ),
    "Trace._op_accessor_cache[*]:TraceOpAccessor._raw_index_lookup{*} -> Op": Holder(
        OP_LIST, but=("fork_cold", "pickle_state")
    ),
    f"{SIDE_TABLE}{{*}}[*] -> Op": Holder(ACTIVATION, but=("pickle_state",)),
    f"{SIDE_TABLE}{{*}}[*] -> Layer": Holder(ACTIVATION, but=("pickle_state",)),
    "Module._facets_cache:FacetView._record -> Module": Holder(
        OP_ROW_CELLS, only=("deepcopied", "facets_read", "pickle_state")
    ),
    "Module._facets_cache:FacetView._structural{*}:FacetSpec.home -> Op": Holder(
        OP_ROW_CELLS, only=("deepcopied", "facets_read", "pickle_state")
    ),
    "ModuleCall._facets_cache:FacetView._record:CallBoundModule._call -> ModuleCall": Holder(
        OP_ROW_CELLS, only=("deepcopied", "facets_read", "pickle_state")
    ),
    "ModuleCall._facets_cache:FacetView._record:CallBoundModule._module -> Module": Holder(
        OP_ROW_CELLS, only=("deepcopied", "facets_read", "pickle_state")
    ),
    "ModuleCall._facets_cache:FacetView._structural{*}:FacetSpec.home -> Op": Holder(
        OP_ROW_CELLS, only=("deepcopied", "facets_read", "pickle_state")
    ),
    "Op._core:Store.rows[_facets_cache]:FacetView._record -> Op": Holder(
        OP_ROW_CELLS, only=("pickle_state",)
    ),
    "Op._core:Store.rows[_facets_cache]:FacetView._structural{*}:FacetSpec.home -> Op": Holder(
        OP_ROW_CELLS, only=("pickle_state",)
    ),
    "Op._core:Store.rows[_grad_records][*]:GradientRecord.owner -> Op": Holder(
        OP_ROW_CELLS, only=("pickle_state",)
    ),
    "Param._tl_core:Store.rows[_grad_records][*]:GradientRecord.owner -> Param": Holder(
        OP_ROW_CELLS, only=("pickle_state",)
    ),
    "Trace._trace_core:TraceCore.kind_rows{param}:Store.rows[_grad_records][*]:GradientRecord.owner -> Param": Holder(
        OP_ROW_CELLS, only=("backward", "pickled")
    ),
    "Trace._trace_core:TraceCore.ops:Store.rows[_facets_cache]:FacetView._record -> Op": Holder(
        OP_ROW_CELLS, only=("deepcopied", "facets_read")
    ),
    "Trace._trace_core:TraceCore.ops:Store.rows[_facets_cache]:FacetView._structural{*}:FacetSpec.home -> Op": Holder(
        OP_ROW_CELLS, only=("deepcopied", "facets_read")
    ),
    "Trace._trace_core:TraceCore.ops:Store.rows[_grad_records][*]:GradientRecord.owner -> Op": Holder(
        OP_ROW_CELLS, only=("backward", "pickled")
    ),
    "Module._tl_core:Store.rows[ops]:ModuleCallAccessor._dict{*} -> ModuleCall": Holder(
        MODULE_AND_BUFFER_CELLS, only=("pickle_state",)
    ),
    "Module._tl_core:Store.rows[ops]:ModuleCallAccessor._list[*] -> ModuleCall": Holder(
        MODULE_AND_BUFFER_CELLS, only=("pickle_state",)
    ),
    "Module._tl_core:Store.rows[params]:ParamAccessor._dict{*} -> Param": Holder(
        MODULE_AND_BUFFER_CELLS, only=("pickle_state",)
    ),
    "Module._tl_core:Store.rows[params]:ParamAccessor._list[*] -> Param": Holder(
        MODULE_AND_BUFFER_CELLS, only=("pickle_state",)
    ),
    "Trace._trace_core:TraceCore.kind_rows{buffer}:Store.overlay[versions][*] -> Op": Holder(
        MODULE_AND_BUFFER_CELLS, only=("fork", "fork_cold")
    ),
    "Trace._trace_core:TraceCore.kind_rows{buffer}:Store.rows[versions][*] -> Op": Holder(
        MODULE_AND_BUFFER_CELLS, but=("pickle_state",)
    ),
    "Trace._trace_core:TraceCore.kind_rows{module}:Store.rows[ops]:ModuleCallAccessor._dict{*} -> ModuleCall": Holder(
        MODULE_AND_BUFFER_CELLS, but=("fork_cold", "pickle_state")
    ),
    "Trace._trace_core:TraceCore.kind_rows{module}:Store.rows[ops]:ModuleCallAccessor._list[*] -> ModuleCall": Holder(
        MODULE_AND_BUFFER_CELLS, but=("fork_cold", "pickle_state")
    ),
    "Trace._trace_core:TraceCore.kind_rows{module}:Store.rows[params]:ParamAccessor._dict{*} -> Param": Holder(
        MODULE_AND_BUFFER_CELLS, but=("fork_cold", "pickle_state")
    ),
    "Trace._trace_core:TraceCore.kind_rows{module}:Store.rows[params]:ParamAccessor._list[*] -> Param": Holder(
        MODULE_AND_BUFFER_CELLS, but=("fork_cold", "pickle_state")
    ),
    "BufferState.versions[*] -> Op": Holder(FORK_AND_SAVED_STATE, only=("pickle_state",)),
    "GradFnState.calls:GradFnCallAccessor._dict{*} -> GradFnCall": Holder(
        FORK_AND_SAVED_STATE, only=("pickle_state",)
    ),
    "GradFnState.calls:GradFnCallAccessor._list[*] -> GradFnCall": Holder(
        FORK_AND_SAVED_STATE, only=("pickle_state",)
    ),
    "LayerState.ops:OpAccessor._dict{*} -> Op": Holder(
        FORK_AND_SAVED_STATE, only=("pickle_state",)
    ),
    "LayerState.ops:OpAccessor._list[*] -> Op": Holder(
        FORK_AND_SAVED_STATE, only=("pickle_state",)
    ),
    "ModuleState.ops:ModuleCallAccessor._dict{*} -> ModuleCall": Holder(
        FORK_AND_SAVED_STATE, only=("pickle_state",)
    ),
    "ModuleState.ops:ModuleCallAccessor._list[*] -> ModuleCall": Holder(
        FORK_AND_SAVED_STATE, only=("pickle_state",)
    ),
    "ModuleState.params:ParamAccessor._dict{*} -> Param": Holder(
        FORK_AND_SAVED_STATE, only=("pickle_state",)
    ),
    "ModuleState.params:ParamAccessor._list[*] -> Param": Holder(
        FORK_AND_SAVED_STATE, only=("pickle_state",)
    ),
    "ParamState._grad_records[*]:GradientRecord.owner -> Param": Holder(
        FORK_AND_SAVED_STATE, only=("pickle_state",)
    ),
    "Trace._trace_core:TraceCore._record_translator:_RecordTranslator._keepalive[*] -> Op": Holder(
        FORK_AND_SAVED_STATE, only=("fork", "fork_cold")
    ),
    "Trace._trace_core:TraceCore._record_translator:_RecordTranslator.map{*} -> Buffer": Holder(
        FORK_AND_SAVED_STATE, only=("fork", "fork_cold")
    ),
    "Trace._trace_core:TraceCore._record_translator:_RecordTranslator.map{*} -> Module": Holder(
        FORK_AND_SAVED_STATE, only=("fork", "fork_cold")
    ),
    "Trace._trace_core:TraceCore._record_translator:_RecordTranslator.map{*} -> ModuleCall": Holder(
        FORK_AND_SAVED_STATE, only=("fork", "fork_cold")
    ),
    "Trace._trace_core:TraceCore._record_translator:_RecordTranslator.map{*} -> Op": Holder(
        FORK_AND_SAVED_STATE, only=("fork", "fork_cold")
    ),
    "Trace._trace_core:TraceCore._record_translator:_RecordTranslator.map{*} -> Param": Holder(
        FORK_AND_SAVED_STATE, only=("fork", "fork_cold")
    ),
    "Trace._trace_core:TraceCore._record_translator:_RecordTranslator.map{*}:ModuleCallAccessor._dict{*} -> ModuleCall": Holder(
        FORK_AND_SAVED_STATE, only=("fork", "fork_cold")
    ),
    "Trace._trace_core:TraceCore._record_translator:_RecordTranslator.map{*}:ModuleCallAccessor._list[*] -> ModuleCall": Holder(
        FORK_AND_SAVED_STATE, only=("fork", "fork_cold")
    ),
    "Trace._trace_core:TraceCore._record_translator:_RecordTranslator.map{*}:ParamAccessor._dict{*} -> Param": Holder(
        FORK_AND_SAVED_STATE, only=("fork", "fork_cold")
    ),
    "Trace._trace_core:TraceCore._record_translator:_RecordTranslator.map{*}:ParamAccessor._list[*] -> Param": Holder(
        FORK_AND_SAVED_STATE, only=("fork", "fork_cold")
    ),
    "TraceState._backward_pass_logs{*}:BackwardPass._tl_core:Store.rows[grad_fn_calls][*] -> GradFnCall": Holder(
        FORK_AND_SAVED_STATE, only=("pickle_state",)
    ),
    "TraceState._grad_fn_logs{*} -> GradFn": Holder(FORK_AND_SAVED_STATE, only=("pickle_state",)),
    "TraceState._layer_dict_all_keys{*} -> Op": Holder(
        FORK_AND_SAVED_STATE, only=("pickle_state",)
    ),
    "TraceState._layer_dict_main_keys{*} -> Op": Holder(
        FORK_AND_SAVED_STATE, only=("pickle_state",)
    ),
    "TraceState._layer_list[*] -> Op": Holder(FORK_AND_SAVED_STATE, only=("pickle_state",)),
    "TraceState._layer_logs{*} -> Layer": Holder(FORK_AND_SAVED_STATE, only=("pickle_state",)),
    "TraceState._param_logs:ParamAccessor._dict{*} -> Param": Holder(
        FORK_AND_SAVED_STATE, only=("pickle_state",)
    ),
    "TraceState._param_logs:ParamAccessor._list[*] -> Param": Holder(
        FORK_AND_SAVED_STATE, only=("pickle_state",)
    ),
    "TraceState._param_logs_by_module{*}[*] -> Param": Holder(
        FORK_AND_SAVED_STATE, only=("pickle_state",)
    ),
    "TraceState._pickle_module_accessor_state:ModuleAccessor._dict{*} -> Module": Holder(
        FORK_AND_SAVED_STATE, only=("pickle_state",)
    ),
    "TraceState._pickle_module_accessor_state:ModuleAccessor._list[*] -> Module": Holder(
        FORK_AND_SAVED_STATE, only=("pickle_state",)
    ),
    "TraceState._pickle_module_accessor_state:ModuleAccessor._pass_dict{*} -> ModuleCall": Holder(
        FORK_AND_SAVED_STATE, only=("pickle_state",)
    ),
    "GradFn._tl_core:Store.rows[calls]:GradFnCallAccessor._dict{*} -> GradFnCall": Holder(
        LATER_KINDS, only=("pickle_state", "pickled")
    ),
    "GradFn._tl_core:Store.rows[calls]:GradFnCallAccessor._list[*] -> GradFnCall": Holder(
        LATER_KINDS, only=("pickle_state", "pickled")
    ),
    "Op._core:Store.rows[_param_logs][*] -> Param": Holder(LATER_KINDS, only=("pickle_state",)),
    "Op._core:Store.rows[grad_fn] -> GradFn": Holder(LATER_KINDS, only=("pickle_state",)),
    "Trace._backward_pass_logs{*}:BackwardPass._tl_core:Store.rows[grad_fn_calls][*] -> GradFnCall": Holder(
        LATER_KINDS, only=("backward", "pickled")
    ),
    "Trace._buffer_accessor:BufferAccessor._dict{*} -> Buffer": Holder(
        LATER_KINDS, but=("pickle_state",)
    ),
    "Trace._buffer_accessor:BufferAccessor._list[*] -> Buffer": Holder(
        LATER_KINDS, but=("pickle_state",)
    ),
    "Trace._grad_fn_logs{*} -> GradFn": Holder(LATER_KINDS, only=("backward", "pickled")),
    "Trace._module_call_accessor:TraceModuleCallAccessor._dict{*} -> ModuleCall": Holder(
        LATER_KINDS, but=("fork_cold", "pickle_state")
    ),
    "Trace._module_call_accessor:TraceModuleCallAccessor._list[*] -> ModuleCall": Holder(
        LATER_KINDS, but=("fork_cold", "pickle_state")
    ),
    "Trace._module_logs:ModuleAccessor._dict{*} -> Module": Holder(
        LATER_KINDS, but=("pickle_state",)
    ),
    "Trace._module_logs:ModuleAccessor._list[*] -> Module": Holder(
        LATER_KINDS, but=("pickle_state",)
    ),
    "Trace._module_logs:ModuleAccessor._pass_dict{*} -> ModuleCall": Holder(
        LATER_KINDS, but=("pickle_state",)
    ),
    "Trace._param_logs:ParamAccessor._dict{*} -> Param": Holder(LATER_KINDS, but=("pickle_state",)),
    "Trace._param_logs:ParamAccessor._list[*] -> Param": Holder(LATER_KINDS, but=("pickle_state",)),
    "Trace._param_logs_by_module{*}[*] -> Param": Holder(
        LATER_KINDS, but=("loaded", "pickle_state")
    ),
    "Trace._trace_core:TraceCore.backward_epochs[*]:BackwardEpoch.stores{grad_fn}:Store.rows[calls]:GradFnCallAccessor._dict{*} -> GradFnCall": Holder(
        LATER_KINDS, only=("backward",)
    ),
    "Trace._trace_core:TraceCore.backward_epochs[*]:BackwardEpoch.stores{grad_fn}:Store.rows[calls]:GradFnCallAccessor._list[*] -> GradFnCall": Holder(
        LATER_KINDS, only=("backward",)
    ),
    "Trace._trace_core:TraceCore.ops:Store.rows[_param_logs][*] -> Param": Holder(
        LATER_KINDS, but=("pickle_state", "pickled", "loaded")
    ),
    # A restored layer's state shares its first op's parameter list (one list
    # object); the walk now reaches that list through the trace-held layer
    # state first, so in these cases it carries this name instead of the row above.
    "Trace._layer_logs{*}{*}[*] -> Param": Holder(
        LATER_KINDS, only=("pickled", "deepcopied", "loaded")
    ),
    "Trace._trace_core:TraceCore.ops:Store.rows[grad_fn] -> GradFn": Holder(
        LATER_KINDS, only=("backward", "pickled")
    ),
    "Trace._trace_core:TraceCore.ops:Store.rows[grad_fn_handle] -> GradFn": Holder(
        LATER_KINDS, only=("pickled",)
    ),
}

# The holders known before the audit was written; the audit must keep finding
# each of them, in the named case, for as long as its row is in the table.
KNOWN: dict[str, tuple[str, str]] = {
    "op list": ("plain", "Trace._layer_list[*] -> Op"),
    "label dictionary (main keys)": ("plain", "Trace._layer_dict_main_keys{*} -> Op"),
    "label dictionary (all keys)": ("plain", "Trace._layer_dict_all_keys{*} -> Op"),
    "facet cache": (
        "facets_read",
        "Trace._trace_core:TraceCore.ops:Store.rows[_facets_cache]:FacetView._record -> Op",
    ),
    "fork translation": (
        "fork",
        "Trace._trace_core:TraceCore._record_translator:_RecordTranslator.map{*} -> Op",
    ),
    "module cells": (
        "plain",
        "Trace._trace_core:TraceCore.kind_rows{module}:Store.rows[ops]"
        ":ModuleCallAccessor._list[*] -> ModuleCall",
    ),
    "serialised state": ("pickle_state", "TraceState._layer_list[*] -> Op"),
    "buffer versions": (
        "buffers_written",
        "Trace._trace_core:TraceCore.kind_rows{buffer}:Store.rows[versions][*] -> Op",
    ),
    "gradient records": (
        "backward",
        "Trace._trace_core:TraceCore.ops:Store.rows[_grad_records][*]:GradientRecord.owner -> Op",
    ),
}


def expected(case: str) -> set[str]:
    """Return the paths the audit of ``case`` must find, no more and no fewer."""

    return {path for path, holder in HOLDERS.items() if holder.appears_in(case)}


def reached_kind(path: str) -> str:
    """Return the record kind a pinned path reaches."""

    return path.rsplit(" -> ", 1)[1]


def blocking(kinds: tuple[str, ...] = ACTIVATED_FIRST) -> dict[str, Holder]:
    """Return the rows that still reach one of ``kinds``: what blocks its activation."""

    return {path: holder for path, holder in HOLDERS.items() if reached_kind(path) in kinds}
