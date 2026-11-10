"""Step 15.5 (aggregate ``Layer`` records): a pending state per layer, built on first lookup.

The step no longer builds every ``Layer``. One raw scan of the op rows
(:mod:`._layer_row_scan`) groups the passes of each layer into a pending
state, stamps the op columns the step owns and indexes the ops by pass. A
layer's state is built the first time one of the layer store's doors hands
that layer out (:func:`build_layer_state`), from the op rows.

A layer with one pass and nothing to merge is written straight from the few
facts the scan kept. Every other layer runs the merge: the multi-pass
aggregate fields, the autograd sums, the multi-pass reconcile and the
conditional views, reading its passes through bound
:class:`~torchlens._data_classes._op_columns.OpColumns` getters or, for a
short layer or ops not backed by one row store, through the ``Op`` facade.

A trace with a conditional event builds every layer inside the step, before
the conditional rebind changes the op fields the layers copy. So does a trace
whose ops are not rows of one building row store.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from itertools import zip_longest
from typing import TYPE_CHECKING, Any, cast

from .._data_classes._layer_state import PENDING, LayerStore, bind_layer_builder
from .._data_classes._op_columns import OpColumns, column_rows_by_label
from .._data_classes._op_row_containers import OpSequence
from .._quantities import Bytes
from .._trace_core.op_store import OpRowStore
from .._trace_core.relation_views import convert_layer_state_views
from .._utils._record_group_table import record_member
from ..backends._constants import is_constant_op
from ._layer_row_scan import RowScan

if TYPE_CHECKING:
    from .._data_classes.layer import Layer
    from .._data_classes.trace import Trace

__all__ = ("build_layer_logs", "build_layer_state")

#: Fields the layer merge reads per op; bound once per layer on the column path.
_READ_FIELDS = (
    "has_input_ancestor",
    "io_role",
    "is_atomic_module",
    "pass_number",
    "label",
    "conditional_branch_stack",
    "in_conditionals",
    "conditional_arm_children",
    "autograd_memory",
    "num_autograd_tensors",
    "conditional_entry_children",
)
#: A layer with fewer passes than this reads the facade: binding getters costs more.
_COLUMN_READ_PASSES = 8
#: ``[Layer class, multi-pass reconcile]`` of the last step 15.5 run.
_BUILD_ARGS: list[Any] = []
#: ``[field reader]`` while one step 15.5 run builds every layer itself, else empty.
_STEP_READER: list[Any] = []


def build_layer_logs(trace: Trace, layer_class: type, reconcile: Callable[..., None]) -> bool:
    """Fill ``trace._layer_logs`` with pending layer states (step 15.5 body).

    Parameters
    ----------
    trace:
        Trace being postprocessed; ``_layer_list`` holds its per-pass ops.
    layer_class:
        The aggregate ``Layer`` class (passed in so this module stays outside
        the record import cycle).
    reconcile:
        Multi-pass reconcile applied to every merged layer after its autograd
        sums (``finalization._reconcile_multipass_layer_fields``), called as
        ``reconcile(layer, read)`` with the layer's field reader.

    Returns
    -------
    bool
        Whether the scan already reset each op's conditional fields (a trace
        with no conditional event). When ``False`` every layer is built and
        the caller runs its per-op rebind.
    """

    _BUILD_ARGS[:] = [layer_class, reconcile]
    scan = _row_scan(trace)
    if scan is None:
        _group_through_facade(trace)
        return False
    scan.run()
    trace._cost_total_autograd_memory = scan.total()
    trace._by_pass = scan.by_pass
    logs = LayerStore(trace)
    logs.adopt_pending(scan.pending_states())
    scan.core._views.layers.pad(len(scan.rows))
    trace.__dict__["_layer_logs"] = logs
    if scan.eager:
        _build_inside_step(logs, scan)
    return not scan.eager


def _build_inside_step(logs: LayerStore, scan: RowScan) -> None:
    """Build every layer now, all reading through one set of bound column getters."""

    columns = scan.columns
    ops = list(scan.trace._layer_list)  # held: the ``id()`` keys below need live records
    get = {name: columns.getter(name) for name in _READ_FIELDS}
    row_of = dict(zip(map(id, ops), scan.keys, strict=True))
    _STEP_READER[:] = [_column_reader(columns, get, row_of)]
    try:
        logs.build_pending()
    finally:
        del _STEP_READER[:]


def _row_scan(trace: Trace) -> RowScan | None:
    """Return the raw scan for ``trace``, or ``None`` when its ops are not rows of one row store."""

    ops = trace._layer_list
    core = trace.__dict__.get("_trace_core")
    if ops.__class__ is not OpSequence or not ops or core is None:
        return None
    lookup = column_rows_by_label(cast("dict[str, Any]", {0: ops[0]}))
    if lookup is None:
        return None
    columns = lookup[0]
    store = columns.store
    keys = list(list.__iter__(ops))
    if (
        store.__class__ is not OpRowStore
        or store._rows is None
        or core.ops is not store
        or ops._owner._store is not store
        or min(keys) < 0
    ):
        return None
    try:
        return RowScan(trace, core, columns, keys, rebind=not trace._graph_conditional_records)
    except ValueError:
        return None


def _group_through_facade(trace: Trace) -> None:
    """Group, stamp and build every layer through the ``Op`` facade (no row store to scan)."""

    ops = list(trace._layer_list)
    states: dict[str, dict[str, Any]] = {}
    by_pass: dict[int, list[int]] = {}
    memories = []
    for position, op in enumerate(ops):
        # backends.2 constants stamp: rule-running backends set it; torch eager is stamped here.
        op.is_constant = bool(op.is_constant) or is_constant_op(op)
        _annotate_tied(op)
        memory = record_member(op, "autograd_memory")
        if memory is not None:
            memories.append(memory)
        state = states.setdefault(op.layer_label, {PENDING: []})
        state[PENDING].append(op)
        if getattr(op, "pass_number", None) is not None:
            by_pass.setdefault(op.pass_number, []).append(position)
    trace._cost_total_autograd_memory = Bytes(sum(map(int, memories))) if memories else None
    trace._by_pass = by_pass
    logs = LayerStore(trace)
    logs.adopt_pending(states)
    trace.__dict__["_layer_logs"] = logs
    logs.build_pending()


def _annotate_tied(op: Any) -> None:
    """Note the tied parameters of one op in its ``annotations`` (facade twin of the scan's)."""

    linked_labels = []
    for param_log in record_member(op, "_param_logs"):
        for alias_address in getattr(param_log, "all_addresses", []):
            if alias_address != param_log.address:
                linked_labels.append(f"{param_log.address} → {alias_address}")
    if linked_labels:
        record_member(op, "annotations")["tied_parameter_notation"] = linked_labels


def build_layer_state(state: dict[str, Any]) -> None:
    """Build one pending layer ``state`` in place: the layer's first lookup.

    The pending value is the scan's ``(label, pass, autograd memory, autograd
    tensors)`` for a one-pass layer with nothing to merge, the arrival-ordered
    ops for a facade-grouped layer, and ``True`` for a layer to merge from its
    keyed passes. A failed build leaves the state pending.
    """

    fact = state[PENDING]
    if fact.__class__ is not tuple:
        _merge_layer_state(state, fact)
        return
    label, number, memory, count = fact
    if memory is not None:
        memory = Bytes(int(memory))
        count = 0 if count is None else count
    else:
        count = None
    accessor = state["ops"]
    views = accessor._store.dataflow_edges is not None  # the relation freeze already ran
    del state["ops"], state[PENDING]
    state["in_conditionals"] = () if views else []
    state["terminal_bool_for"] = None
    state["_is_in_conditional_body"] = False
    state["conditional_role_stacks"] = ([],) if views else [[]]
    state["conditional_branch_stack_ops"] = {(): [number]}
    state["conditional_arm_children"] = {}
    state["annotations"] = {}
    state["ops"] = accessor
    state["call_labels"] = [label]
    state["autograd_memory"] = memory
    state["total_autograd_memory"] = memory
    state["num_autograd_tensors"] = count
    state["conditional_entry_children"] = () if views else []
    state["conditional_then_children"] = () if views else []
    state["conditional_elif_children"] = {}
    state["conditional_else_children"] = () if views else []


def _merge_layer_state(state: dict[str, Any], arrivals: Any) -> None:
    """Build a pending layer state by the full merge over its passes."""

    layer_class, reconcile = _BUILD_ARGS
    accessor = state.get("ops")
    scratch: dict[str, Any] = {}
    layer = layer_class.__new__(layer_class)
    layer.__dict__ = scratch
    keyed = accessor is not None and "_order" not in accessor.__dict__
    if arrivals is True:
        arrivals = accessor.values() if keyed else [accessor[at] for at in accessor._order]
    layer_class.__init__(layer, arrivals[0])
    if keyed:
        scratch["ops"] = accessor
    if _STEP_READER:
        read = _STEP_READER[0]
    else:
        read = _pass_reader(accessor, arrivals) if keyed else _facade_reader
    for index, op in enumerate(arrivals):
        _merge_pass(layer, op, read, first=not index, keyed=keyed)
    _set_layer_autograd(layer, read)
    reconcile(layer, read)
    _rebuild_layer_log_conditional_views(layer, read)
    # The roles a layer of a trace with no conditional holds; the rebind overwrites it otherwise.
    scratch["in_conditionals"] = []
    store = getattr(scratch["ops"], "_store", None)
    if store is not None and store.dataflow_edges is not None:
        convert_layer_state_views(scratch, {}, {})
    state.clear()
    state.update(scratch)


def _merge_pass(layer: Any, op: Any, read: Any, *, first: bool, keyed: bool) -> None:
    """Merge one pass into its layer (the multi-pass rules of ``finalization._build_layer_logs``)."""

    if not first:
        if read(op, "has_input_ancestor"):
            layer._graph_has_input_ancestor = True
        role = read(op, "io_role")
        if layer._graph_io_role is not None and role is not None:
            layer._graph_io_role = _merge_io_role(layer._graph_io_role, role)
        if read(op, "is_atomic_module"):
            layer._module_context_is_atomic_module = True
    number = read(op, "pass_number")
    if not keyed:
        layer.ops[number] = op
    layer.call_labels.append(read(op, "label"))
    stack = read(op, "conditional_branch_stack")
    # ``Op.is_in_conditional_body``, read from the two stored fields it derives from.
    in_body = bool(stack) or any(role.role == "body" for role in read(op, "in_conditionals") or [])
    _merge_conditional_values(layer, in_body, stack, number, read(op, "conditional_arm_children"))


def _pass_reader(accessor: Any, passes: list[Any]) -> Any:
    """Return the field reader for a keyed layer's passes: column getters for a long layer."""

    if len(passes) < _COLUMN_READ_PASSES or column_rows_by_label({0: passes[0]}) is None:
        return _facade_reader
    columns = OpColumns(accessor._store)
    get = {name: columns.getter(name) for name in _READ_FIELDS}
    row_of = dict(zip(map(id, passes), accessor._keys[1::2], strict=True))
    return _column_reader(columns, get, row_of)


bind_layer_builder(build_layer_state)


def _facade_reader(op: Any, name: str, *default: Any) -> Any:
    """Read field ``name`` of ``op`` through the ``Op`` facade (``getattr`` semantics)."""

    return record_member(op, name, *default)


def _column_reader(
    columns: Any, get: dict[str, Callable[[int], Any]], row_of: dict[int, int]
) -> Any:
    """Return a ``getattr``-shaped ``(op, name[, default])`` reader over column getters.

    Strict reads use the getters bound in ``get``. ``None``-default reads (the
    multi-pass reconcile's ``getattr(op, name, None)``) bind a defaulted getter
    once per field and map the facade's ``AttributeError`` to the default, as
    ``getattr`` does; any other default, and any field the view does not
    model, reads the facade.
    """

    soft: dict[str, Callable[[int], Any] | None] = {}

    def read(op: Any, name: str, *default: Any) -> Any:
        """Read field ``name`` of ``op``'s row as ``getattr(op, name[, default])`` does."""

        if not default:
            return get[name](row_of[id(op)])
        if name not in soft:
            try:
                soft[name] = columns.getter(name, default=None)
            except ValueError:
                soft[name] = None
        getter = soft[name]
        if getter is None or default[0] is not None:
            return record_member(op, name, *default)
        try:
            return getter(row_of[id(op)])
        except AttributeError:
            return None

    return read


def _merge_io_role(layer_role: str, pass_role: str) -> str:
    """Return the character-wise ``io_role`` merge with ``*`` for differing characters."""

    merged = "".join(
        c if c == s else "*" for c, s in zip_longest(layer_role, pass_role, fillvalue="")
    )
    if merged.endswith("."):
        merged = merged[:-1]
    if merged.endswith("*"):
        merged = merged.rstrip("*") + "*"
    return merged


def _set_layer_autograd(layer_log: Any, read: Callable[[Any, str], Any]) -> bool:
    """Set the layer's cross-pass autograd sums; return whether any pass saved a value."""

    passes = list(layer_log.ops.values())
    pass_autograd_bytes = [
        read(op, "autograd_memory") for op in passes if read(op, "autograd_memory") is not None
    ]
    pass_autograd_tensor_counts = [
        read(op, "num_autograd_tensors")
        for op in passes
        if read(op, "num_autograd_tensors") is not None
    ]
    if pass_autograd_bytes:
        layer_log._cost_autograd_memory = Bytes(sum(int(value) for value in pass_autograd_bytes))
        layer_log._cost_total_autograd_memory = layer_log._cost_autograd_memory
        layer_log._autograd_num_autograd_tensors = sum(pass_autograd_tensor_counts)
        return True
    layer_log._cost_autograd_memory = None
    layer_log._cost_total_autograd_memory = None
    layer_log._autograd_num_autograd_tensors = None
    return False


def _append_unique_child_label(
    child_labels: list[str], seen_labels: set[str], child_label: str
) -> None:
    """Append a child label if it has not been seen yet.

    Parameters
    ----------
    child_labels:
        Ordered list being accumulated.
    seen_labels:
        Membership set mirroring ``child_labels`` (the bare list scan made
        each aggregate projection O(k^2) in its child count -- R52).
    child_label:
        Child label to append.
    """
    if child_label not in seen_labels:
        seen_labels.add(child_label)
        child_labels.append(child_label)


def _strip_pass_suffix(layer_label: str) -> str:
    """Strip the ``:call_index`` suffix from a layer label.

    Parameters
    ----------
    layer_label:
        Pass-qualified or pass-free layer label.

    Returns
    -------
    str
        Layer label without any pass suffix.
    """
    return layer_label.split(":", 1)[0]


def _merge_conditional_values(
    layer_log: Layer,
    in_body: bool,
    branch_stack: Iterable[Any],
    pass_number: int,
    arm_children: dict[int, dict[str, list[str]]],
) -> None:
    """Merge one pass's already-read conditional fields into an aggregate ``Layer``."""

    if in_body:
        layer_log._control_flow_is_in_conditional_body = True

    stack_signature = tuple(branch_stack)
    if stack_signature not in layer_log._control_flow_branch_stack_ops:
        layer_log._control_flow_role_stacks.append(list(branch_stack))
        layer_log._control_flow_branch_stack_ops[stack_signature] = []

    signature_ops = layer_log._control_flow_branch_stack_ops[stack_signature]
    if pass_number not in signature_ops:
        signature_ops.append(pass_number)
        signature_ops.sort()

    for conditional_id, branch_children in arm_children.items():
        merged_branch_children = layer_log._control_flow_arm_children.setdefault(
            conditional_id,
            {},
        )
        for branch_kind, child_labels in branch_children.items():
            merged_child_labels = merged_branch_children.setdefault(branch_kind, [])
            merged_seen = set(merged_child_labels)
            for child_label in child_labels:
                _append_unique_child_label(
                    merged_child_labels,
                    merged_seen,
                    _strip_pass_suffix(child_label),
                )


def _rebuild_layer_log_conditional_views(
    layer_log: Layer, read: Callable[[Any, str], Any] = _facade_reader
) -> None:
    """Recompute aggregate conditional compatibility views for one ``Layer``.

    Parameters
    ----------
    layer_log:
        Aggregate layer entry whose derived conditional fields are refreshed.
    read:
        ``(op, field) -> value`` reader for the passes' entry children.
    """
    layer_log._control_flow_is_in_conditional_body = any(
        len(branch_stack) > 0 for branch_stack in layer_log._control_flow_role_stacks
    )

    conditional_entry_children: list[str] = []
    entry_seen: set[str] = set()
    for _, pass_log in sorted(layer_log.ops.items()):
        for child_label in read(pass_log, "conditional_entry_children"):
            _append_unique_child_label(
                conditional_entry_children,
                entry_seen,
                _strip_pass_suffix(child_label),
            )
    layer_log._control_flow_entry_children = conditional_entry_children

    conditional_then_children: list[str] = []
    then_seen: set[str] = set()
    conditional_elif_children: dict[int, list[str]] = {}
    elif_seen_by_index: dict[int, set[str]] = {}
    conditional_else_children: list[str] = []
    else_seen: set[str] = set()
    for branch_children in layer_log._control_flow_arm_children.values():
        for child_label in branch_children.get("then", []):
            _append_unique_child_label(conditional_then_children, then_seen, child_label)
        for branch_kind, child_labels in branch_children.items():
            if not branch_kind.startswith("elif_"):
                continue
            elif_index = int(branch_kind.split("_", 1)[1])
            elif_children = conditional_elif_children.setdefault(elif_index, [])
            elif_seen = elif_seen_by_index.setdefault(elif_index, set())
            for child_label in child_labels:
                _append_unique_child_label(elif_children, elif_seen, child_label)
        for child_label in branch_children.get("else", []):
            _append_unique_child_label(conditional_else_children, else_seen, child_label)

    layer_log._control_flow_then_children = conditional_then_children
    layer_log._control_flow_elif_children = conditional_elif_children
    layer_log._control_flow_else_children = conditional_else_children
