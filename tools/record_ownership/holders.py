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

The pickle-state paths are transient projections (step ``PICKLE_PROJECTION``):
the state dictionary exists only while pickle runs, so they block no
activation. They stay listed, and ``is_transient`` accepts them only from a
``<Kind>State`` root in the pickle-state case; the tests also check on a live
trace that nothing of the state survives the call.
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
PICKLE_PROJECTION = "pickle_projection"
# The audit case that walks the state handed to pickle, and the roots it names.
PICKLE_CASE = "pickle_state"
PICKLE_ROOT_SUFFIX = "State"
# The one strong cache of records: a path may wait for ``ACTIVATION`` only through it.
SIDE_TABLE = "Trace._trace_core:TraceCore._strong_facades"

STEPS: dict[str, str] = {
    OP_LIST: "the op list, the label dictionaries and the caches of the ops door",
    LAYER_STATE: "Layer records, the layers door and each Layer's op accessor",
    OP_ROW_CELLS: "cells of a record's own row: the facet cache and gradient records",
    MODULE_AND_BUFFER_CELLS: "module cells (calls, params) and buffer versions",
    FORK_AND_SAVED_STATE: "fork translation and copy-on-write overlays",
    LATER_KINDS: "holders of the kinds activated after Op and Layer",
    ACTIVATION: "the one strong side table of records, made weak when its kind is activated",
    PICKLE_PROJECTION: "the state handed to pickle: it holds the records only for the pickle call",
}

BEFORE_ACTIVATION = frozenset(
    {OP_LIST, LAYER_STATE, OP_ROW_CELLS, MODULE_AND_BUFFER_CELLS, FORK_AND_SAVED_STATE}
)
ACTIVATED_FIRST = ("Op", "Layer")
# Steps whose paths are not held by a trace: nothing has to convert them before
# a kind is activated, because the holder is gone when the call that made it returns.
TRANSIENT = frozenset({PICKLE_PROJECTION})


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
    "Trace._trace_core:TraceCore._strong_facades{*}{*}[*]:ModuleCallAccessor._dict{*} -> ModuleCall": Holder(
        ACTIVATION, but=("pickle_state",)
    ),
    "Trace._trace_core:TraceCore._strong_facades{*}{*}[*]:ModuleCallAccessor._list[*] -> ModuleCall": Holder(
        ACTIVATION, but=("pickle_state",)
    ),
    "Trace._trace_core:TraceCore._strong_facades{*}{*}[*]:ParamAccessor._dict{*} -> Param": Holder(
        ACTIVATION, only=("deepcopied",)
    ),
    "Trace._trace_core:TraceCore._strong_facades{*}{*}[*]:ParamAccessor._list[*] -> Param": Holder(
        ACTIVATION, only=("deepcopied",)
    ),
    f"{SIDE_TABLE}{{*}}[*] -> Op": Holder(ACTIVATION, but=("pickle_state",)),
    f"{SIDE_TABLE}{{*}}[*] -> Layer": Holder(ACTIVATION, but=("pickle_state",)),
    "Module._facets_cache:FacetView._record -> Module": Holder(
        OP_ROW_CELLS, only=("deepcopied", "facets_read", "pickle_state")
    ),
    "ModuleCall._facets_cache:FacetView._record:CallBoundModule._call -> ModuleCall": Holder(
        OP_ROW_CELLS, only=("deepcopied", "facets_read", "pickle_state")
    ),
    "ModuleCall._facets_cache:FacetView._record:CallBoundModule._module -> Module": Holder(
        OP_ROW_CELLS, only=("deepcopied", "facets_read", "pickle_state")
    ),
    "BufferState.versions[*] -> Op": Holder(PICKLE_PROJECTION, only=(PICKLE_CASE,)),
    "GradFnState.calls:GradFnCallAccessor._dict{*} -> GradFnCall": Holder(
        PICKLE_PROJECTION, only=(PICKLE_CASE,)
    ),
    "GradFnState.calls:GradFnCallAccessor._list[*] -> GradFnCall": Holder(
        PICKLE_PROJECTION, only=(PICKLE_CASE,)
    ),
    "LayerState.ops:OpAccessor._dict{*} -> Op": Holder(PICKLE_PROJECTION, only=(PICKLE_CASE,)),
    "LayerState.ops:OpAccessor._list[*] -> Op": Holder(PICKLE_PROJECTION, only=(PICKLE_CASE,)),
    "ModuleState.ops:ModuleCallAccessor._dict{*} -> ModuleCall": Holder(
        PICKLE_PROJECTION, only=(PICKLE_CASE,)
    ),
    "ModuleState.ops:ModuleCallAccessor._list[*] -> ModuleCall": Holder(
        PICKLE_PROJECTION, only=(PICKLE_CASE,)
    ),
    "ModuleState.params:ParamAccessor._dict{*} -> Param": Holder(
        PICKLE_PROJECTION, only=(PICKLE_CASE,)
    ),
    "ModuleState.params:ParamAccessor._list[*] -> Param": Holder(
        PICKLE_PROJECTION, only=(PICKLE_CASE,)
    ),
    "TraceState._backward_pass_logs{*}:BackwardPass._tl_core:Store.rows[grad_fn_calls][*] -> GradFnCall": Holder(
        PICKLE_PROJECTION, only=(PICKLE_CASE,)
    ),
    "TraceState._grad_fn_logs{*} -> GradFn": Holder(PICKLE_PROJECTION, only=(PICKLE_CASE,)),
    "TraceState._layer_dict_all_keys{*} -> Op": Holder(PICKLE_PROJECTION, only=(PICKLE_CASE,)),
    "TraceState._layer_dict_main_keys{*} -> Op": Holder(PICKLE_PROJECTION, only=(PICKLE_CASE,)),
    "TraceState._layer_list[*] -> Op": Holder(PICKLE_PROJECTION, only=(PICKLE_CASE,)),
    "TraceState._layer_logs{*} -> Layer": Holder(PICKLE_PROJECTION, only=(PICKLE_CASE,)),
    "TraceState._param_logs:ParamAccessor._dict{*} -> Param": Holder(
        PICKLE_PROJECTION, only=(PICKLE_CASE,)
    ),
    "TraceState._param_logs:ParamAccessor._list[*] -> Param": Holder(
        PICKLE_PROJECTION, only=(PICKLE_CASE,)
    ),
    "TraceState._param_logs_by_module{*}[*] -> Param": Holder(
        PICKLE_PROJECTION, only=(PICKLE_CASE,)
    ),
    "TraceState._pickle_module_accessor_state:ModuleAccessor._dict{*} -> Module": Holder(
        PICKLE_PROJECTION, only=(PICKLE_CASE,)
    ),
    "TraceState._pickle_module_accessor_state:ModuleAccessor._list[*] -> Module": Holder(
        PICKLE_PROJECTION, only=(PICKLE_CASE,)
    ),
    "TraceState._pickle_module_accessor_state:ModuleAccessor._pass_dict{*} -> ModuleCall": Holder(
        PICKLE_PROJECTION, only=(PICKLE_CASE,)
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
    "serialised state": ("pickle_state", "TraceState._layer_list[*] -> Op"),
    "gradient records": (
        "backward",
        "Trace._trace_core:TraceCore.ops:Store.rows[_grad_records][*]:GradientRecord.owner -> Op",
    ),
}


def is_transient(path: str, holder: Holder) -> bool:
    """Return whether a row names a transient projection, not something a trace keeps.

    Only the state handed to pickle qualifies: the row is in a transient step,
    appears in the pickle-state case alone, and starts at a ``<Kind>State`` root
    (the state object itself). The same holder chain reached from a live
    ``Trace`` is a different path, has no transient row, and fails as a new
    holder: a projection that outlives the pickle call is not transient.
    """

    root = path.split(".", 1)[0].split(":", 1)[0]
    return (
        holder.step in TRANSIENT
        and holder.only == (PICKLE_CASE,)
        and root.endswith(PICKLE_ROOT_SUFFIX)
    )


def expected(case: str) -> set[str]:
    """Return the paths the audit of ``case`` must find, no more and no fewer."""

    return {path for path, holder in HOLDERS.items() if holder.appears_in(case)}


def reached_kind(path: str) -> str:
    """Return the record kind a pinned path reaches."""

    return path.rsplit(" -> ", 1)[1]


def blocking(kinds: tuple[str, ...] = ACTIVATED_FIRST) -> dict[str, Holder]:
    """Return the rows that still reach one of ``kinds``: what blocks its activation."""

    return {path: holder for path, holder in HOLDERS.items() if reached_kind(path) in kinds}
