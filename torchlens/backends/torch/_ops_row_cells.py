"""Row-direct op cells: the exhaustive draft straight to store cells (capture C2a).

Today an exhaustive op's cells are produced in two hops: the commit freezes
the draft (``fields_dict``) into an ``OpRecord`` (``_op_record_from_log``) and
ingest scatters the record back into cells (``scatter_record_to_cells``).
:func:`op_row_cells` computes the same cells in one hop, without building the
record; :func:`write_op_row` appends them as one layout-ordered row.

The value contract is exact: for the same draft (fields dict, fire results,
call box), ``op_row_cells`` returns the layout projection of
``scatter_record_to_cells(_op_record_from_log(...), IngestExtras(), trace)``,
key for key and value for value (types included), evaluated at the same
moment. The layout projection is what ``Op.__init__`` stores: names backed by
``Op`` properties are not cells (``input_ops`` and the other input summaries
are dropped), ``source_trace`` is stored as the ``_source_trace_ref`` weak
reference and ``is_in_conditional_body`` as the ``_is_in_conditional_body``
cell, so every returned name is a field of ``_OP_STORE_LAYOUT`` and
:func:`write_op_row` accepts the result as is. Facet presence is mirrored: a
facet the freeze would omit reads the checked-in ``FACET_DEFAULTS`` value, as
the scatter does. The shared function-call ref goes through
``_resolve_call_function_ref`` so sibling outputs of one call share it exactly
as the freeze does (the call box is updated the same way).

The exhaustive commit uses :func:`write_draft_row`, which runs the same fill
straight into a copy of the layout's constant row template, with the call ref
the commit already resolved; ``op_row_cells`` is the dict form of that fill.
"""

from __future__ import annotations

import weakref
from collections import defaultdict
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

from ..._ir._op_row_seed import INIT_BYTES_FIELDS
from ..._ir.op_record import FACET_DEFAULTS
from ..._ir.op_record_scatter import _resolve_dtype, _write_function_ref_cells
from ..._trace_core.cell_states import LazyFields
from ..._trace_core.op_store import _MISSING
from ..._utils.container_tokens import TOKEN_EXACT, TOKEN_INERT
from ._ops_freeze_refs import _copy_parent_arg_positions, _resolve_call_function_ref
from ._relations_resolve import ANCESTRY_PARENTS, resolve_fields

if TYPE_CHECKING:
    from ..._data_classes.trace import Trace
    from ..._ir.events import FunctionCallRef
    from ..._ir.intervention import FireResult
    from ..._trace_core.op_store import OpStoreLayout

__all__ = ("op_row_cells", "write_draft_row", "write_op_row")


#: One default instance per optional facet the freeze may omit. The writer reads
#: only immutable values off them, or copies the mutable ones, so one instance
#: per process stands in for the instance the scatter builds per absent facet.
_TRANSFORM_DEFAULT = FACET_DEFAULTS["transform"]()
#: ``op.py`` ``_container_pool_key`` of ``{"_tl_annotations": {}}`` (checked in
#: tests by the container tokens' debug walk).
_BARE_TRANSFORM_CONFIG_KEY = ("d", (((str, "_tl_annotations"), ("m", ("d", ()))),))
#: Exact key, and no member the immutable pool could replace.
_BARE_FLAGS = TOKEN_EXACT | TOKEN_INERT
_ANCESTRY_DEFAULT = FACET_DEFAULTS["ancestry"]()
_CONTROL_DEFAULT = FACET_DEFAULTS["control"]()
_AUTOGRAD_DEFAULT = FACET_DEFAULTS["autograd"]()
_INTERVENTION_DEFAULT = FACET_DEFAULTS["intervention"]()
_ANNOTATIONS_DEFAULT = FACET_DEFAULTS["annotations"]()


class _NameKeys(dict[str, str]):
    """Field-id map of the dict form: every cell name is its own key."""

    def __missing__(self, name: str) -> str:
        """Return ``name`` itself, so ``row[f[name]]`` writes the dict cell ``name``."""

        return name


_NAME_KEYS = _NameKeys()


#: Constant-valued cells the scatter seeds for every op (immutable values only).
_SCALAR_CONSTANT_CELLS: tuple[tuple[str, Any], ...] = (
    ("ordinal_index", -1),
    ("label", None),
    ("short_label", None),
    ("layer_label", None),
    ("short_layer_label", None),
    ("num_passes", 1),
    ("grad", None),
    ("transformed_grad", None),
    ("has_grad", False),
    ("grad_shape", None),
    ("transformed_grad_shape", None),
    ("grad_dtype", None),
    ("transformed_grad_dtype", None),
    ("gradient_memory", 0),
    ("transformed_gradient_memory", None),
    ("grad_fn", None),
    ("multi_output_name", None),
    ("min_distance_from_input", None),
    ("max_distance_from_input", None),
    ("is_output", False),
    ("is_final_output", False),
    ("has_output_descendant", False),
    ("is_orphan", False),
    ("min_distance_to_output", None),
    ("max_distance_to_output", None),
    ("buffer_pass", None),
    ("buffer_write_kind", None),
    ("buffer_value_changed", None),
    ("buffer_replay_validated", None),
    ("buffer_source_func_name", None),
    ("is_internal_sink", False),
    ("is_terminal_bool", False),
    ("is_terminal_conditional_bool", False),
    ("conditional_context_kind", None),
    ("conditional_wrapper_kind", None),
    ("terminal_conditional_id", None),
    ("terminal_bool_for", None),
    ("_is_in_conditional_body", False),
    ("conditional_branch_depth", 0),
    ("_address_normalized", None),
    ("fx_qualpath", None),
    ("fx_call_index", 0),
    ("is_module_output", False),
    ("is_atomic_module", False),
    ("atomic_module_call", None),
)
#: Constant-container cells; each op gets fresh empty containers, as the scatter builds them.
_EMPTY_LIST_CELLS: tuple[str, ...] = (
    "lookup_keys",
    "recurrent_ops",
    "in_conditionals",
    "conditional_branch_stack",
    "conditional_entry_children",
    "conditional_then_children",
    "conditional_else_children",
    "input_to_module_calls",
    "output_of_modules",
    "output_of_module_calls",
)
_EMPTY_DICT_CELLS: tuple[str, ...] = (
    "conditional_elif_children",
    "conditional_arm_children",
)
#: Sanctioned (poolable) container cells a writer fills only when non-empty; an
#: empty one stays default-omitted and settles onto the shared pooled empty cell.
#: Each default is the exact empty value its writer used to write (the emit, or
#: the ingest join for ``out_versions_by_child`` and ``parent_param_ops``).
_OMITTED_EMPTY_CELLS: tuple[tuple[str, type], ...] = (
    ("annotations", dict),
    ("edits", list),
    ("var_names", list),
    ("func_rng_states", dict),
    ("out_versions_by_child", dict),
    ("parent_param_ops", dict),
)
#: Cells the capture writes only when non-empty: an empty one stays default-omitted
#: and the freeze settles a fresh literal per row, as every writer here built one.
#: ``internal_source_parents`` and the two dict cells are written at the call;
#: the param cells are written by ingest's join (see ``_row_ingest``).
_OMIT_EMPTY_LIST_CELLS: tuple[str, ...] = (
    "internal_source_parents",
    "parent_params",
    "_param_barcodes",
    "param_shapes",
)
_OMIT_EMPTY_DICT_CELLS: tuple[str, ...] = ("non_tensor_kwargs", "func_config")
#: The cells above that the call writes (``write_draft_row`` restores an empty one).
_CALL_OMIT_EMPTY_CELLS: tuple[str, ...] = (
    "internal_source_parents",
    "non_tensor_kwargs",
    "func_config",
)


# Cognitive complexity 17 (over 15) is essential: the container-token registration
# is inlined because a helper call would add one Python call per committed op.
def _fill_transform(
    row: Any, f: Any, trace: Any, fields_dict: Mapping[str, Any], annotations: Any
) -> None:
    """Write the transform cells, mirroring the freeze's facet presence and config split."""

    user_config = dict(fields_dict.get("transform_config") or {})
    user_config.pop("_tl_annotations", None)
    fn_code_location = user_config.pop("fn_code_location", None)
    is_transform = bool(fields_dict.get("is_transform", False))
    kind = fields_dict.get("transform_kind")
    chain = tuple(fields_dict.get("transform_chain") or ())
    fn_name = fields_dict.get("transform_fn_name")
    fn_qualname = fields_dict.get("transform_fn_qualname")
    fn_source = fields_dict.get("transform_fn_source")
    if not (
        is_transform
        or kind is not None
        or chain
        or user_config
        or fn_name is not None
        or fn_qualname is not None
        or fn_source is not None
        or fn_code_location is not None
    ):
        default = _TRANSFORM_DEFAULT
        is_transform = default.is_transform
        kind = default.transform_kind
        chain = tuple(default.transform_chain)
        user_config = dict(default.transform_config)
        fn_name = default.transform_fn_name
        fn_qualname = default.transform_fn_qualname
        fn_source = default.transform_fn_source
        fn_code_location = default.fn_code_location
    bare = not user_config
    inner = user_config["_tl_annotations"] = dict(annotations)
    if fn_code_location is not None:
        user_config["fn_code_location"] = fn_code_location
    elif bare and not inner:
        # The common cell ``{"_tl_annotations": {}}``: its content key is a
        # constant, so the freeze's pooling reads it instead of walking. The
        # registration is inlined (no call): this runs once per committed op,
        # inside the commit scope test_capture_row_commit_cost pins.
        try:
            workspace = trace._wrapper_runtime_ws
        except AttributeError:  # dropped with the capture workspaces
            workspace = None
        if workspace is not None:
            table = workspace.container_tokens
            if table is None:
                table = workspace.container_tokens = {}
            if table is not False:
                table[id(user_config)] = (user_config, _BARE_TRANSFORM_CONFIG_KEY, _BARE_FLAGS)
    row[f["is_transform"]] = is_transform
    row[f["transform_kind"]] = kind
    row[f["transform_chain"]] = chain
    row[f["transform_config"]] = user_config
    row[f["transform_fn_name"]] = fn_name
    row[f["transform_fn_qualname"]] = fn_qualname
    row[f["transform_fn_source"]] = fn_source


def _fill_facets(
    row: Any, f: Any, fields_dict: Mapping[str, Any], fire_results: tuple[FireResult, ...]
) -> Any:
    """Write the control, autograd, intervention, annotation and ancestry cells.

    Returns the annotations mapping (the transform config cell embeds a copy).
    """
    # Cyclomatic grade C on purpose: this runs once per committed op, and one flat
    # writer saves the per-op calls that split facet helpers would add.

    is_scalar_bool = fields_dict["is_scalar_bool"]
    bool_value = fields_dict["bool_value"]
    if is_scalar_bool is None and bool_value is None:
        is_scalar_bool = _CONTROL_DEFAULT.is_scalar_bool
        bool_value = _CONTROL_DEFAULT.bool_value
    qualname = fields_dict["grad_fn_class_qualname"]
    if qualname is None:
        qualname = _AUTOGRAD_DEFAULT.grad_fn_class_qualname
    is_edited = fields_dict["is_edited"]
    if not (fire_results or is_edited):
        fire_results = _INTERVENTION_DEFAULT.fire_results
        is_edited = _INTERVENTION_DEFAULT.is_edited
    annotations = dict(fields_dict.get("annotations") or {})
    if not annotations:
        annotations = dict(_ANNOTATIONS_DEFAULT.annotations)
    row[f["is_scalar_bool"]] = bool(is_scalar_bool)
    row[f["bool_value"]] = bool_value
    row[f["grad_fn_class_qualname"]] = qualname
    row[f["is_edited"]] = is_edited
    if fire_results:
        edits = [
            result.fire_record
            for result in fire_results
            if getattr(result, "fire_record", None) is not None
        ]
        if edits:
            row[f["edits"]] = edits
    if annotations:
        row[f["annotations"]] = annotations
    if ANCESTRY_PARENTS in fields_dict:
        resolve_fields(cast(dict, fields_dict))
    input_ancestors = fields_dict["input_ancestors"]
    internal_source_ancestors = fields_dict["internal_source_ancestors"]
    root_ancestors = fields_dict["root_ancestors"]
    has_internal = fields_dict["has_internal_source_ancestor"]
    if not (input_ancestors or internal_source_ancestors or root_ancestors or has_internal):
        input_ancestors = _ANCESTRY_DEFAULT.input_ancestors
        internal_source_ancestors = _ANCESTRY_DEFAULT.internal_source_ancestors
        root_ancestors = _ANCESTRY_DEFAULT.root_ancestors
        has_internal = _ANCESTRY_DEFAULT.has_internal_source_ancestor
    row[f["root_ancestors"]] = set(root_ancestors)
    row[f["has_input_ancestor"]] = bool(input_ancestors)
    row[f["input_ancestors"]] = set(input_ancestors)
    row[f["has_internal_source_ancestor"]] = has_internal
    row[f["internal_source_ancestors"]] = set(internal_source_ancestors)
    return annotations


def _fill_output(row: Any, f: Any, fields_dict: Mapping[str, Any]) -> None:
    """Write the output-ref, transformed-output, backend-semantics and pending blob cells."""

    row[f["has_saved_activation"]] = fields_dict["has_saved_activation"]
    row[f["output_device"]] = fields_dict["output_device"]
    row[f["activation_transform"]] = fields_dict["activation_transform"]
    row[f["detach_saved_activations"]] = fields_dict["detach_saved_activations"]
    row[f["visualizer_path"]] = fields_dict["visualizer_path"]
    row[f["in_multi_output"]] = fields_dict["in_multi_output"]
    row[f["multi_output_index"]] = fields_dict["multi_output_index"]
    row[f["container_path"]] = tuple(fields_dict["container_path"])
    row[f["container_spec"]] = fields_dict["container_spec"]
    if fields_dict.get("_pending_blob_id") is not None:
        row[f["_pending_blob_id"]] = fields_dict["_pending_blob_id"]
    payload = fields_dict["transformed_out"]
    if payload is None:
        row[f["transformed_out_shape"]] = None
        row[f["transformed_out_dtype"]] = None
        row[f["transformed_activation_memory"]] = None
        row[f["transformed_out"]] = None
    else:
        if fields_dict.get("_pending_transformed_out_blob_id") is not None:
            row[f["_pending_transformed_out_blob_id"]] = fields_dict[
                "_pending_transformed_out_blob_id"
            ]
        row[f["transformed_out_shape"]] = fields_dict["transformed_out_shape"]
        row[f["transformed_out_dtype"]] = _resolve_dtype(str(fields_dict["transformed_out_dtype"]))
        row[f["transformed_activation_memory"]] = fields_dict["transformed_activation_memory"]
        row[f["transformed_out"]] = payload
    semantics = fields_dict.get("backend_semantics")
    if semantics is None:
        row[f["bytes_delta_at_call"]] = fields_dict["bytes_delta_at_call"]
        row[f["bytes_peak_at_call"]] = fields_dict["bytes_peak_at_call"]
        row[f["autograd_memory"]] = fields_dict["autograd_memory"]
        row[f["num_autograd_tensors"]] = fields_dict["num_autograd_tensors"]
        row[f["grad_fn_class_name"]] = fields_dict["grad_fn_class_name"]
    else:
        row[f["bytes_delta_at_call"]] = semantics.bytes_delta_at_call
        row[f["bytes_peak_at_call"]] = semantics.bytes_peak_at_call
        row[f["autograd_memory"]] = semantics.autograd_memory
        row[f["num_autograd_tensors"]] = semantics.num_autograd_tensors
        row[f["grad_fn_class_name"]] = semantics.grad_fn_class_name


def _fill_core_graph(
    row: Any, f: Any, trace: Trace, fields_dict: Mapping[str, Any], require_event: Any = None
) -> None:
    """Write the core, graph, template and module cells.

    ``require_event`` resolves a parent label (default: the live index's).
    """

    layer_type = fields_dict["type"]
    parents = list(fields_dict["parents"])
    modules = tuple(fields_dict["modules"])
    if require_event is None:
        require_event = trace.capture_events.live_index.require_event
    row[f["_label_raw"]] = fields_dict["_label_raw"]
    row[f["_layer_label_raw"]] = fields_dict["_layer_label_raw"]
    row[f["step_index"]] = fields_dict["step_index"] or 0
    row[f["raw_index"]] = fields_dict["raw_index"]
    row[f["type"]] = layer_type
    row[f["type_index"]] = fields_dict["type_index"]
    row[f["pass_number"]] = fields_dict["pass_number"]
    row[f["_source_trace_ref"]] = weakref.ref(trace)
    row[f["_tracing_finished"]] = fields_dict["_tracing_finished"]
    row[f["_construction_done"]] = fields_dict["_construction_done"]
    row[f["save_grads"]] = fields_dict["save_grads"]
    row[f["has_saved_args"]] = fields_dict["has_saved_args"]
    row[f["saved_args"]] = fields_dict["saved_args"]
    row[f["saved_kwargs"]] = fields_dict["saved_kwargs"]
    row[f["args_template"]] = fields_dict["args_template"]
    row[f["kwargs_template"]] = fields_dict["kwargs_template"]
    row[f["unattributed_tensor_args"]] = tuple(fields_dict.get("unattributed_tensor_args") or ())
    row[f["dropped_edge_tensor_args"]] = tuple(fields_dict.get("dropped_edge_tensor_args") or ())
    row[f["equivalence_class"]] = fields_dict["equivalence_class"]
    row[f["parents"]] = parents
    row[f["parent_arg_positions"]] = _copy_parent_arg_positions(fields_dict["parent_arg_positions"])
    row[f["is_input"]] = layer_type == "input"
    row[f["input_was_parameter"]] = bool(fields_dict.get("input_was_parameter", False))
    row[f["is_output_parent"]] = fields_dict["is_output_parent"]
    row[f["is_buffer"]] = bool(fields_dict["is_input"] or fields_dict["is_buffer"]) and (
        layer_type == "buffer"
    )
    row[f["is_internal_source"]] = layer_type != "input" and not parents
    internal_source_parents: list[str] = []
    for label in parents:
        if require_event(label).has_internal_source_ancestor:
            internal_source_parents.append(label)
    row[f["internal_source_parents"]] = internal_source_parents
    row[f["module"]] = modules[-1] if modules else None
    row[f["modules"]] = list(modules)
    row[f["module_call_stack"]] = list(modules)


def _fill_row(
    row: Any,
    f: Any,
    trace: Trace,
    fields_dict: Mapping[str, Any],
    fire_results: tuple[FireResult, ...],
) -> None:
    """Write every per-op cell of one exhaustive draft but its function cells into ``row``.

    ``row`` is a layout row (``f`` the layout's field ids) or a cell dict
    (``f`` maps each name to itself). The scalar constants are not written
    here: the layout row starts from a template holding them and the dict form
    adds them first. The empty containers are not written here either: the
    layout row starts them default-omitted (the store answers a fresh one on
    first read) and the dict form adds them through ``_fill_empty_containers``.
    The same holds for an empty ``edits`` or ``annotations`` cell.
    """

    _fill_core_graph(row, f, trace, fields_dict)
    _fill_rest(row, f, trace, fields_dict, fire_results)


def _fill_rest(
    row: Any,
    f: Any,
    trace: Trace,
    fields_dict: Mapping[str, Any],
    fire_results: tuple[FireResult, ...],
) -> None:
    """Write the output, facet and transform cells (``_fill_row`` after the core graph)."""

    _fill_output(row, f, fields_dict)
    _fill_transform(row, f, trace, fields_dict, _fill_facets(row, f, fields_dict, fire_results))


def _arg_keys_default() -> defaultdict[Any, list[Any]]:
    """Return a fresh ``module_entry_arg_keys`` default."""

    return defaultdict(list)


def _fill_empty_containers(cells: dict[str, Any]) -> None:
    """Write the fresh empty container cells of one op into the dict form."""

    for name in _EMPTY_LIST_CELLS:
        cells[name] = []
    for name in _EMPTY_DICT_CELLS:
        cells[name] = {}
    cells["edits"] = []
    cells["annotations"] = {}
    cells["output_descendants"] = set()
    cells["module_entry_arg_keys"] = _arg_keys_default()


def _default_fields(layout: OpStoreLayout) -> LazyFields:
    """Register the constant-valued cells of an exhaustive draft as defaults.

    The scalar constants are shared immutable defaults; the empty containers,
    ``_grad_records`` (which ingest used to seed) and the sanctioned cells of
    ``_OMITTED_EMPTY_CELLS`` are per-row factories.
    A constant that ingest converts in place (the byte quantities of
    ``INIT_BYTES_FIELDS``) is not registered: the capture writes its value.
    """

    fields = LazyFields(layout.names)
    for name, value in _SCALAR_CONSTANT_CELLS:
        if name not in INIT_BYTES_FIELDS:
            fields.register_default(name, value)
    for name in (*_EMPTY_LIST_CELLS, *_OMIT_EMPTY_LIST_CELLS, "_grad_records"):
        fields.register_default(name, factory=list)
    for name in (*_EMPTY_DICT_CELLS, *_OMIT_EMPTY_DICT_CELLS):
        fields.register_default(name, factory=dict)
    fields.register_default("output_descendants", factory=set)
    fields.register_default("module_entry_arg_keys", factory=_arg_keys_default)
    for name, factory in _OMITTED_EMPTY_CELLS:
        fields.register_default(name, factory=factory)
    return fields


class _RowPlan:
    """One layout's field ids and its row template of default-omitted cells.

    ``var_names`` and ``rng`` are ``(field id, default marker)`` of the two
    call cells the shared function-ref writer fills, which an empty value puts
    back to default-omitted.
    """

    __slots__ = ("layout", "fids", "omit_empty", "rng", "template", "var_names")

    def __init__(self, layout: OpStoreLayout) -> None:
        """Bind ``layout`` and build its template (unset fields stay ``_MISSING``)."""

        self.layout = layout
        self.fids = layout.fid_by_name
        self.omit_empty = tuple(self.fids[name] for name in _CALL_OMIT_EMPTY_CELLS)
        template = _default_fields(layout).seed_row()
        for name, value in _SCALAR_CONSTANT_CELLS:
            if name in INIT_BYTES_FIELDS:
                template[self.fids[name]] = value
        self.template = template
        fid = self.fids["var_names"]
        self.var_names = (fid, template[fid])
        fid = self.fids["func_rng_states"]
        self.rng = (fid, template[fid])


#: Row plans by layout identity (the layout is kept by the plan, so ids are not reused).
_PLANS: dict[int, _RowPlan] = {}


def _plan_for(layout: OpStoreLayout) -> _RowPlan:
    """Return the cached row plan for ``layout``, building it on first use."""

    plan = _PLANS.get(id(layout))
    if plan is None or plan.layout is not layout:
        plan = _RowPlan(layout)
        _PLANS[id(layout)] = plan
    return plan


def write_draft_row(
    store: Any,
    trace: Trace,
    draft: Any,
    function: FunctionCallRef,
) -> int:
    """Append one exhaustive draft's row to ``store`` and return its index.

    The row holds exactly what ``write_op_row(store, op_row_cells(...))`` holds
    for the same draft and the same resolved call ref, written straight into a
    copy of the layout's template (no intermediate cell dicts), except that the
    constant-valued cells and the empty ``_CALL_OMIT_EMPTY_CELLS`` stay
    default-omitted: the store answers them on read.

    Parameters
    ----------
    store
        A building ``OpRowStore`` (or any store with ``layout`` and ``adopt_row``).
    trace
        Active capture trace (the ``source_trace`` cell and the live index).
    draft
        The op's commit snapshot: ``fields`` (the final exhaustive draft, as
        passed to ``_op_record_from_log``), ``fire_results`` (its intervention
        results) and ``require_event`` (the parent-label resolver).
    function
        The op's resolved (shared) function-call ref.

    Returns
    -------
    int
        The new row's index.
    """

    layout = store.layout
    plan = _PLANS.get(id(layout))
    if plan is None or plan.layout is not layout:
        plan = _plan_for(layout)
    template = plan.template
    row = template.copy()
    fids = plan.fids
    fields_dict = draft.fields
    _fill_core_graph(row, fids, trace, fields_dict, draft.require_event)
    _fill_rest(row, fids, trace, fields_dict, draft.fire_results)
    _write_function_ref_cells(row, fids, function)
    fid, marker = plan.var_names
    row[fid] = marker  # the function-ref writer's fresh empty list
    # The call's RNG snapshot is one dict shared by every output of the call: a
    # multi-output op keeps the object, so the alias census still sees it.
    rng = function.func_rng_states
    if rng.__class__ is dict and not rng and not fields_dict["in_multi_output"]:
        fid, marker = plan.rng
        row[fid] = marker
    for fid in plan.omit_empty:
        if not row[fid]:
            row[fid] = template[fid]
    return int(store.adopt_row(row))


def op_row_cells(
    trace: Trace,
    fields_dict: Mapping[str, Any],
    fire_results: tuple[FireResult, ...] = (),
    call_ref_box: list[FunctionCallRef] | None = None,
) -> dict[str, Any]:
    """Return one exhaustive op's layout-stored cells without building its record.

    The dict form of :func:`write_draft_row` (same fill, keyed by name).

    Parameters
    ----------
    trace
        Active capture trace (the ``source_trace`` cell and the live index).
    fields_dict
        The final exhaustive draft, as passed to ``_op_record_from_log``.
    fire_results
        Intervention results for this op.
    call_ref_box
        The per-call shared function-ref box, updated as the freeze updates it.

    Returns
    -------
    dict[str, Any]
        Layout field name to value: the layout projection of the scatter of
        the frozen record (see the module docstring).
    """

    function = _resolve_call_function_ref(cast(dict[str, Any], fields_dict), call_ref_box)
    cells: dict[str, Any] = dict(_SCALAR_CONSTANT_CELLS)
    _fill_empty_containers(cells)
    _fill_row(cells, _NAME_KEYS, trace, fields_dict, fire_results)
    _write_function_ref_cells(cells, _NAME_KEYS, function)
    return cells


def write_op_row(store: Any, cells: Mapping[str, Any]) -> int:
    """Append one layout-ordered row holding ``cells``; unset fields stay ``_MISSING``.

    Parameters
    ----------
    store
        A building ``OpRowStore`` (or any store with ``layout`` and ``adopt_row``).
    cells
        Cell name to value; every name must be a layout field.

    Returns
    -------
    int
        The new row's index.

    Raises
    ------
    KeyError
        If a cell name is not a field of the store's layout.
    """

    fid_by_name = store.layout.fid_by_name
    row: list[Any] = [_MISSING] * store.layout.n_fields
    for name, value in cells.items():
        row[fid_by_name[name]] = value
    return int(store.adopt_row(row))
