"""Step 15.5 (build aggregate ``Layer`` records) over ``Op`` column views.

The pass groups ``trace._layer_list`` by ``layer_label`` into aggregate
``Layer`` records, merges the multi-pass aggregate fields, sums autograd
memory and refreshes the conditional views. Its per-op field reads and writes
run through bound :class:`~torchlens._data_classes._op_columns.OpColumns`
accessors whenever every op shares one row store; otherwise the facade body
runs. Both bodies visit the same ops in the same order and perform the same
reads (same short-circuits) and writes, so the stored result is identical
(checked by per-pass store and layer fingerprints over the corpus).

Layer construction, the constant-op predicates and the multi-pass parameter
addresses keep reading the ``Op`` facade: they run once per layer or once per
op through shared predicates other passes call.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterable
from itertools import zip_longest
from typing import TYPE_CHECKING, Any, cast

from .._data_classes._op_columns import column_rows_by_label
from .._quantities import Bytes
from .._utils._record_group_table import record_member
from ..backends._constants import is_constant_op

if TYPE_CHECKING:
    from .._data_classes.layer import Layer
    from .._data_classes.trace import Trace

__all__ = ("build_layer_logs",)

#: Fields step 15.5 reads per op; bound once per pass on the column path.
_READ_FIELDS = (
    "layer_label",
    "is_constant",
    "has_input_ancestor",
    "io_role",
    "is_atomic_module",
    "pass_number",
    "label",
    "conditional_branch_stack",
    "in_conditionals",
    "conditional_arm_children",
    "_param_logs",
    "annotations",
    "autograd_memory",
    "num_autograd_tensors",
    "conditional_entry_children",
)


def build_layer_logs(trace: Trace, layer_class: type, reconcile: Callable[..., None]) -> None:
    """Build ``trace._layer_logs`` and the autograd-memory total (step 15.5 body).

    Parameters
    ----------
    trace:
        Trace being postprocessed; ``_layer_list`` holds its per-pass ops.
    layer_class:
        The aggregate ``Layer`` class (passed in so this module stays outside
        the record import cycle).
    reconcile:
        Multi-pass reconcile applied to every aggregate layer after its
        autograd sums (``finalization._reconcile_multipass_layer_fields``),
        called as ``reconcile(layer, read)`` with the pass's field reader.
    """

    ops = list(trace._layer_list)  # one local list: the ``id()`` keys below need live records
    # Keyed by position, not label: reading every label would cost a facade read per op.
    lookup = column_rows_by_label(cast("dict[str, Any]", dict(enumerate(ops)))) if ops else None
    if lookup is None:
        layer_logs = _group_passes_facade(ops, layer_class)
        total = _sum_autograd(layer_logs, reconcile, _facade_reader)
    else:
        columns, rows_by_index = cast("tuple[Any, dict[int, int]]", lookup)
        get = {name: columns.getter(name) for name in _READ_FIELDS}
        row_of = {id(op): rows_by_index[index] for index, op in enumerate(ops)}
        layer_logs = _group_passes_columns(
            ops, layer_class, get, columns.setter("is_constant"), row_of
        )
        total = _sum_autograd(layer_logs, reconcile, _column_reader(columns, get, row_of))
    trace._cost_total_autograd_memory = total
    trace._layer_logs = layer_logs


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


def _group_passes_facade(ops: list[Any], layer_class: Any) -> OrderedDict[str, Any]:
    """Group per-pass ops into aggregate layers, reading the ``Op`` facade."""

    layer_logs: OrderedDict[str, Any] = OrderedDict()
    for pass_log in ops:
        no_call_label = pass_log.layer_label
        # backends.2 constants stamp: rule-running backends set it; torch eager is stamped here.
        pass_log.is_constant = bool(pass_log.is_constant) or is_constant_op(pass_log)
        if no_call_label not in layer_logs:
            layer_log = layer_logs[no_call_label] = layer_class(pass_log)
        else:
            layer_log = layer_logs[no_call_label]
            if pass_log._graph_has_input_ancestor:
                layer_log._graph_has_input_ancestor = True
            if layer_log._graph_io_role is not None and pass_log._graph_io_role is not None:
                layer_log._graph_io_role = _merge_io_role(
                    layer_log._graph_io_role, pass_log._graph_io_role
                )
            if pass_log._module_context_is_atomic_module:
                layer_log._module_context_is_atomic_module = True
        layer_log.ops[pass_log.pass_number] = pass_log
        layer_log.call_labels.append(pass_log.label)
        _merge_layer_log_conditional_fields(layer_log, pass_log)
    return layer_logs


def _group_passes_columns(
    ops: list[Any],
    layer_class: Any,
    get: dict[str, Callable[[int], Any]],
    set_constant: Callable[[int, Any], None],
    row_of: dict[int, int],
) -> OrderedDict[str, Any]:
    """Group per-pass ops into aggregate layers, reading bound column getters.

    Same reads, short-circuits and writes as :func:`_group_passes_facade`.
    """

    layer_label, is_constant = get["layer_label"], get["is_constant"]
    io_role, pass_number = get["io_role"], get["pass_number"]
    layer_logs: OrderedDict[str, Any] = OrderedDict()
    for pass_log in ops:
        row = row_of[id(pass_log)]
        no_call_label = layer_label(row)
        set_constant(row, bool(is_constant(row)) or is_constant_op(pass_log))
        if no_call_label not in layer_logs:
            layer_log = layer_logs[no_call_label] = layer_class(pass_log)
        else:
            layer_log = layer_logs[no_call_label]
            if get["has_input_ancestor"](row):
                layer_log._graph_has_input_ancestor = True
            if layer_log._graph_io_role is not None and io_role(row) is not None:
                layer_log._graph_io_role = _merge_io_role(layer_log._graph_io_role, io_role(row))
            if get["is_atomic_module"](row):
                layer_log._module_context_is_atomic_module = True
        number = pass_number(row)
        layer_log.ops[number] = pass_log
        layer_log.call_labels.append(get["label"](row))
        stack = get["conditional_branch_stack"](row)
        # ``Op.is_in_conditional_body``, read from the two stored fields it derives from.
        in_body = bool(stack) or any(
            role.role == "body" for role in get["in_conditionals"](row) or []
        )
        _merge_conditional_values(
            layer_log, in_body, stack, number, get["conditional_arm_children"](row)
        )
    return layer_logs


def _sum_autograd(
    layer_logs: OrderedDict[str, Any],
    reconcile: Callable[..., None],
    read: Callable[[Any, str], Any],
) -> Bytes | None:
    """Annotate tied parameters, sum autograd memory per layer and refresh views.

    Returns
    -------
    Bytes | None
        Trace-wide autograd memory, or ``None`` when no pass saved any.
    """

    autograd_memory = 0
    has_autograd_saved_value = False
    for layer_log in layer_logs.values():
        for pass_log in layer_log.ops.values():
            linked_labels = []
            for param_log in read(pass_log, "_param_logs"):
                # Tied parameters list every aliasing address in ``all_addresses`` (primary
                # first); co-occurring distinct tensors (``co_parent_params``) are not tying.
                for alias_address in getattr(param_log, "all_addresses", []):
                    if alias_address == param_log.address:
                        continue
                    linked_labels.append(f"{param_log.address} → {alias_address}")
            if linked_labels:
                read(pass_log, "annotations")["tied_parameter_notation"] = linked_labels
        if _set_layer_autograd(layer_log, read):
            autograd_memory += layer_log._cost_autograd_memory
            has_autograd_saved_value = True
        reconcile(layer_log, read)
        _rebuild_layer_log_conditional_views(layer_log, read)
    return Bytes(autograd_memory) if has_autograd_saved_value else None


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


def _merge_layer_log_conditional_fields(
    layer_log: Layer,
    pass_log: Any,
) -> None:
    """Merge one pass's conditional metadata into an aggregate ``Layer``.

    Parameters
    ----------
    layer_log:
        Aggregate layer entry being updated.
    pass_log:
        Per-pass layer entry contributing conditional metadata.
    """
    _merge_conditional_values(
        layer_log,
        pass_log._control_flow_is_in_conditional_body,
        pass_log._control_flow_branch_stack,
        pass_log.pass_number,
        pass_log._control_flow_arm_children,
    )


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
