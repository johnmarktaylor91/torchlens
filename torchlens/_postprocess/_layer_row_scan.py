"""Raw row scan behind step 15.5: group the passes of each layer.

One pass over the op rows of a building row store. It groups the passes of
each layer into a pending layer state, stamps the op columns step 15.5 owns
(``is_constant``, the tied-parameter annotation and, for a trace with no
conditional event, the two conditional fields the rebind resets), indexes the
ops by pass and collects the autograd memory total.

The common path reads and writes the row lists directly, with no call per op.
A cell in any other shape is read through its column getter, which answers
exactly as the ``Op`` facade does.
"""

from __future__ import annotations

import weakref
from array import array
from typing import Any

from .._data_classes._layer_accessors import OpAccessor
from .._data_classes._layer_state import PENDING
from .._quantities import Bytes
from .._trace_core.cell_states import LazyCell
from ..backends._constants import CONSTANT_FUNC_NAME, is_constant_op

__all__ = ("RowScan",)

#: Fields the scan reads, in the order ``RowScan.fids`` unpacks them.
_FIELDS = (
    "layer_label",
    "pass_number",
    "label",
    "is_constant",
    "func_name",
    "parents",
    "_param_logs",
    "autograd_memory",
    "num_autograd_tensors",
    "conditional_branch_stack",
    "in_conditionals",
    "conditional_arm_children",
    "conditional_entry_children",
    "annotations",
    "terminal_bool_for",
)
#: Fields the scan writes straight into the row.
_RAW_WRITES = ("is_constant", "in_conditionals", "terminal_bool_for")
_UNSET_CONSTANT = (type(None), bool)


class RowScan:
    """One scan of a trace's op rows: groups, pass index and autograd total.

    Parameters
    ----------
    trace:
        Trace being postprocessed.
    core:
        The trace core that owns ``columns.store``.
    columns:
        Column view over the building row store.
    keys:
        Store row of every op, in ``trace._layer_list`` order.
    rebind:
        Whether the trace has no conditional event, so the scan itself resets
        each op's ``in_conditionals`` and ``terminal_bool_for``.
    """

    def __init__(self, trace: Any, core: Any, columns: Any, keys: list[int], rebind: bool) -> None:
        """Bind the scan to one trace and its building row store."""

        store = columns.store
        self.trace, self.core, self.columns, self.keys = trace, core, columns, keys
        self.rebind = rebind
        #: Whether every layer must be built before the caller's own rebind.
        self.eager = not rebind
        self.rows = store._rows
        self.fids = tuple(store.layout.fid_by_name[name] for name in _FIELDS)
        for name in _RAW_WRITES:
            columns.setter(name)  # raises ``ValueError`` for a field only the facade may write
        self.getters: dict[str, Any] = {}
        self.groups: dict[str, list[int]] = {}
        #: Rows of the ops whose layer state needs no merge (see ``_settle``).
        self.simple: dict[int, bool] = {}
        self.by_pass: dict[int, list[int]] = {}
        self.memories: list[Any] = []

    def read(self, name: str, key: int) -> Any:
        """Return field ``name`` of row ``key`` as the facade reads it."""

        getter = self.getters.get(name)
        if getter is None:
            getter = self.getters[name] = self.columns.getter(name)
        return getter(key)

    def run(self) -> None:
        """Scan the op rows (no call per op unless a cell has an unusual shape)."""

        self._group()
        if self.rebind:
            self._settle()

    def _group(self) -> None:
        """Group passes by layer, stamp ``is_constant``, index passes, collect autograd memory."""

        rows, groups, by_pass, memories = self.rows, self.groups, self.by_pass, self.memories
        f_label, f_pass, _full, f_const, f_name, f_parents, f_params, f_memory = self.fids[:8]
        for position, key in enumerate(self.keys):
            row = rows[key]
            label, number = row[f_label], row[f_pass]
            if label.__class__ is not str or number.__class__ is not int:
                label, number = self.read("layer_label", key), self.read("pass_number", key)
            if row[f_const] is not True:
                if (
                    row[f_const].__class__ in _UNSET_CONSTANT
                    and row[f_name].__class__ is str
                    and row[f_name] != CONSTANT_FUNC_NAME
                    and row[f_parents].__class__ is list
                    and row[f_parents]
                ):
                    row[f_const] = False  # an op with a parent is never a constant
                else:
                    self.stamp_constant(position, key)
            if row[f_params].__class__ is not list or row[f_params]:
                self.annotate_tied(key)
            memory = row[f_memory]
            if memory is not None:
                if memory.__class__ is not Bytes:
                    memory = self.read("autograd_memory", key)
                if memory is not None:
                    memories += (memory,)
            if label in groups:
                groups[label] += (number, key)
            else:
                groups[label] = [number, key]
            if number in by_pass:
                by_pass[number] += (position,)
            else:
                by_pass[number] = [position]

    def _settle(self) -> None:
        """Reset the conditional fields of each op of a trace with no conditional event.

        Leaves the cells as the layer build and the rebind left them, and
        notes in ``simple`` each op whose layer state needs no merge.
        """

        rows, simple = self.rows, self.simple
        f_full, f_memory, f_count, f_stack, f_roles, f_arms, f_entry, f_notes, f_terminal = (
            self.fids[2],
            *self.fids[7:],
        )
        for key in self.keys:
            row = rows[key]
            stack, entry, terminal = row[f_stack], row[f_entry], row[f_terminal]
            if stack.__class__ is LazyCell and stack.source is list:
                stack = row[f_stack] = []
            if entry.__class__ is LazyCell and entry.source is list:
                entry = row[f_entry] = []
            if (
                stack.__class__ is list
                and not stack
                and entry.__class__ is list
                and not entry
                and row[f_arms].__class__ is dict
                and not row[f_arms]
                and (
                    terminal is None or (terminal.__class__ is LazyCell and terminal.source is None)
                )
                and row[f_notes].__class__ is dict
                and "collective" not in row[f_notes]
                and row[f_full].__class__ is str
                and (row[f_memory] is None or row[f_memory].__class__ is Bytes)
                and (row[f_count] is None or row[f_count].__class__ is int)
            ):
                simple[key] = True
            else:
                self.settle_unusual(key)
            row[f_roles] = []
            if not self.eager:
                row[f_terminal] = None

    def stamp_constant(self, position: int, key: int) -> None:
        """Stamp ``is_constant`` of one op through the constant-op predicate."""

        stored = self.read("is_constant", key)
        op = self.trace._layer_list[position]
        self.columns.setter("is_constant")(key, bool(stored) or is_constant_op(op))

    def annotate_tied(self, key: int) -> None:
        """Note the tied parameters of one op in its ``annotations``."""

        linked_labels = []
        for param_log in self.read("_param_logs", key):
            # Tied parameters list every aliasing address in ``all_addresses`` (primary
            # first); co-occurring distinct tensors (``co_parent_params``) are not tying.
            for alias_address in getattr(param_log, "all_addresses", []):
                if alias_address == param_log.address:
                    continue
                linked_labels.append(f"{param_log.address} → {alias_address}")
        if linked_labels:
            self.read("annotations", key)["tied_parameter_notation"] = linked_labels

    def settle_unusual(self, key: int) -> None:
        """Read one op's conditional cells as the layer build does; note a set terminal.

        An op that already names the conditional it is the terminal bool of
        keeps that value for the layer build to copy, so every layer is then
        built before the caller's rebind.
        """

        for name in ("conditional_branch_stack", "conditional_arm_children"):
            self.read(name, key)
        self.read("conditional_entry_children", key)
        if self.read("terminal_bool_for", key) is not None:
            self.eager = True

    def total(self) -> Bytes | None:
        """Return the trace-wide autograd memory, or ``None`` when no pass saved any."""

        return Bytes(sum(map(int, self.memories))) if self.memories else None

    def pending_states(self) -> dict[str, dict[str, Any]]:
        """Return one pending layer state per layer, in first-seen order."""

        trace_ref = weakref.ref(self.trace)
        core_ref, store, rows = self.core.weak_self(), self.columns.store, self.rows
        simple = {} if self.eager else self.simple
        f_full, f_memory, f_count = self.fids[2], self.fids[7], self.fids[8]
        states: dict[str, dict[str, Any]] = {}
        for label, pairs in self.groups.items():
            accessor = OpAccessor.__new__(OpAccessor)
            keyed = accessor.__dict__
            keyed["_keys"] = array("q", pairs)
            keyed["_core_ref"] = core_ref
            keyed["_store"] = store
            fact: Any = True
            if pairs[2:3]:
                if pairs[0::2] != sorted(set(pairs[0::2])):
                    _insert_in_arrival_order(accessor, pairs)
            elif pairs[1] in simple:
                row = rows[pairs[1]]
                fact = (row[f_full], pairs[0], row[f_memory], row[f_count])
            states[label] = {"_source_trace_ref": trace_ref, "ops": accessor, PENDING: fact}
        return states


def _insert_in_arrival_order(accessor: Any, pairs: list[int]) -> None:
    """Key passes that arrived out of pass order, keeping the arrival order on the accessor."""

    accessor.__dict__["_keys"] = array("q")
    for at in range(0, len(pairs), 2):
        accessor._insert(pairs[at], pairs[at + 1])
