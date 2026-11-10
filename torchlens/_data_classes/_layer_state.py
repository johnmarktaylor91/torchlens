"""Trace-held Layer state: one state mapping per layer, with Layer views on demand.

A trace keeps each layer's state as one dictionary (the layer's user
overrides, deletion tombstones, build-phase snapshots and ``annotations``
all live in it, exactly as they did in the Layer's own ``__dict__``). A
``Layer`` object is a view whose ``__dict__`` *is* that dictionary, so a view
built later reads and writes the same state. The store holds the state; the
view objects are held by the trace core's strong view table until views
become disposable.
"""

from __future__ import annotations

import weakref
from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, cast

#: The ``Layer`` class, bound by ``layer.py`` at import so this module stays a leaf.
_LAYER_CLASS: list[type] = []


def bind_layer_class(layer_class: type) -> None:
    """Bind the ``Layer`` class that views are built from."""

    _LAYER_CLASS[:] = [layer_class]


def view_slots(core: Any, kind: str) -> list[Any]:
    """Return the core's strong view list for ``kind`` (indexed by row), creating it."""

    table = cast("dict[Any, Any]", core._strong_facades)
    slots = table.get(kind)
    if slots is None:
        slots = table[kind] = []
    return cast("list[Any]", slots)


def hold_view(core: Any, kind: str, index: int, view: Any, *, replace: bool = False) -> bool:
    """Hold ``view`` at ``index`` of the core's view list; return whether it is held there."""

    slots = view_slots(core, kind)
    if index >= len(slots):
        slots.extend([None] * (index + 1 - len(slots)))
    current = slots[index]
    if current is None or replace:
        slots[index] = view
        return True
    return current is view


def _layer_index(state: dict[str, Any]) -> tuple[Any, int]:
    """Return ``(core, index)`` for a layer state keyed into a live core, else ``(None, -1)``.

    A layer's index is the store row of its first-pass op, read from the
    layer's own op accessor, so the state carries no extra key.
    """

    accessor_state: Any = getattr(state.get("ops"), "__dict__", None)
    keys = accessor_state.get("_keys") if accessor_state is not None else None
    if not keys or keys[1] < 0:
        return None, -1
    core_ref = accessor_state.get("_core_ref")
    core = core_ref() if core_ref is not None else None
    return (core, keys[1]) if core is not None else (None, -1)


class LayerStore(OrderedDict[str, Any]):
    """Mapping from layer label to Layer view over trace-held layer state.

    The mapping's own entries are ``label -> state dictionary``; every read
    hands out the Layer view of that state. It subclasses ``OrderedDict`` so
    the trace field keeps its container type for every ``isinstance`` reader.
    """

    __slots__ = ("_held", "_labels", "_trace_ref")

    def __init__(self, trace: Any = None) -> None:
        """Create an empty store owned by ``trace``."""

        OrderedDict.__init__(self)
        # Views of layers that cannot be keyed into a core yet (no core, or
        # ops not bound to it): held here until ``settle`` can move them.
        self._held: dict[str, Any] | None = None
        self._labels: tuple[str, ...] | None = None
        self._trace_ref = weakref.ref(trace) if trace is not None else None

    def __getitem__(self, label: str) -> Any:
        """Return the Layer view for ``label``."""

        state = dict.__getitem__(self, label)
        held = self._held
        if held is not None:
            view = held.get(label)
            if view is not None:
                return view
        accessor_state: Any = getattr(state.get("ops"), "__dict__", None)
        keys = accessor_state.get("_keys") if accessor_state is not None else None
        if keys and keys[1] >= 0:
            core_ref = accessor_state.get("_core_ref")
            core = core_ref() if core_ref is not None else None
            if core is not None:
                slots = core._strong_facades.get("layer")
                if slots is not None and keys[1] < len(slots):
                    view = slots[keys[1]]
                    if view is not None and view.__dict__ is state:
                        return view
        return self._build(label, state)

    def get(self, label: Any, default: Any = None) -> Any:
        """Return the Layer view for ``label``, or ``default``."""

        return self[label] if label in self else default

    def __setitem__(self, label: str, layer: Any) -> None:
        """Adopt ``layer``: its state becomes trace-held and it becomes the label's view."""

        state = vars(layer)
        if label not in self:
            self._labels = None
        OrderedDict.__setitem__(self, label, state)
        self._place(label, state, layer)

    def __delitem__(self, label: str) -> None:
        """Drop one layer's state and its view."""

        state = dict.__getitem__(self, label)
        OrderedDict.__delitem__(self, label)
        self._labels = None
        self._drop_view(label, state)

    def __eq__(self, other: object) -> bool:
        """Compare as the ordered mapping of Layer views."""

        if isinstance(other, LayerStore):
            other = other.as_ordered_dict()
        return bool(self.as_ordered_dict() == other)

    def __ne__(self, other: object) -> bool:
        """Compare as the ordered mapping of Layer views."""

        return not self == other

    def __repr__(self) -> str:
        """Render as the ordered mapping this store stands in for."""

        return repr(self.as_ordered_dict())

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle as the plain ordered mapping of Layer records (the saved shape)."""

        return (OrderedDict, (), None, None, iter(self.items()))

    def values(self) -> Any:
        """Return every Layer view in insertion order."""

        self.settle()
        return [self[label] for label in self]

    def items(self) -> Any:
        """Return ``(label, Layer view)`` pairs in insertion order."""

        self.settle()
        return [(label, self[label]) for label in self]

    def clear(self) -> None:
        """Drop every layer's state and view."""

        for label, state in list(dict.items(self)):
            self._drop_view(label, state)
        OrderedDict.clear(self)
        self._held = None
        self._labels = None

    def copy(self) -> Any:
        """Return a plain ordered mapping of the Layer views."""

        return self.as_ordered_dict()

    def pop(self, label: Any, *default: Any) -> Any:
        """Remove ``label`` and return its Layer view (or the default)."""

        if label not in self:
            if default:
                return default[0]
            raise KeyError(label)
        view = self[label]
        del self[label]
        return view

    def popitem(self, last: bool = True) -> tuple[str, Any]:
        """Remove and return the last (or first) ``(label, Layer view)`` pair."""

        if not self:
            raise KeyError("dictionary is empty")
        label = next(reversed(self)) if last else next(iter(self))
        return label, self.pop(label)

    def setdefault(self, label: Any, default: Any = None) -> Any:
        """Return the Layer view for ``label``, adopting ``default`` when absent."""

        if label not in self:
            self[label] = default
        return self[label]

    def as_ordered_dict(self) -> OrderedDict[str, Any]:
        """Return a plain ordered mapping of the Layer views."""

        return OrderedDict(self.items())

    def labels(self) -> tuple[str, ...]:
        """Return the labels in insertion order (cached until the label set changes)."""

        labels = self._labels
        if labels is None or len(labels) != len(self):
            labels = self._labels = tuple(self)
        return labels

    def settle(self) -> None:
        """Move held views into the trace core once their ops are keyed into it.

        A layer adopted before its ops were bound to the trace's core (a
        restored trace before its core is rebuilt, a layer inserted before its
        first pass) is held here. Once the core's op store is frozen its op
        accessors are re-keyed; any layer whose first pass is keyed moves.
        """

        held = self._held
        if not held:
            return
        ref = self._trace_ref
        trace = ref() if ref is not None else None
        core = trace.__dict__.get("_trace_core") if trace is not None else None
        if core is None:
            return
        frozen = bool(getattr(core.ops, "frozen", False))
        for label, view in list(held.items()):
            state = dict.get(self, label)
            if state is None:
                held.pop(label, None)
                continue
            rekey = getattr(state.get("ops"), "_rekey", None) if frozen else None
            if rekey is not None:
                rekey(core)
            if _layer_index(state)[0] is not None:
                self._place(label, state, view)
        if not self._held:
            self._held = None

    def _build(self, label: str, state: dict[str, Any]) -> Any:
        """Build a Layer view over ``state`` and hold it."""

        layer_class: Any = _LAYER_CLASS[0]
        view = layer_class.__new__(layer_class)
        view.__dict__ = state
        self._place(label, state, view)
        return view

    def _place(self, label: str, state: dict[str, Any], view: Any) -> None:
        """Hold ``view`` in its core's view table, else in this store."""

        core, index = _layer_index(state)
        held = self._held
        if core is not None:
            hold_view(core, "layer", index, view, replace=True)
            if held is not None and held.pop(label, None) is not None and not held:
                self._held = None
            return
        if held is None:
            held = self._held = {}
        held[label] = view

    def _drop_view(self, label: str, state: dict[str, Any]) -> None:
        """Release the view held for a removed layer state."""

        held = self._held
        if held is not None:
            held.pop(label, None)
        core, index = _layer_index(state)
        if core is None:
            return
        slots = core._strong_facades.get("layer")
        if slots is not None and index < len(slots):
            view = slots[index]
            if view is not None and view.__dict__ is state:
                slots[index] = None


def settle_layer_logs(trace: Any) -> bool:
    """Settle the trace's layer store after its core is rebuilt; return ``True``.

    The constant result lets the size-capped rehydrate function end with one
    ``return`` that both settles the store and reports a rebuilt core.
    """

    logs = getattr(trace, "_layer_logs", None)
    if logs.__class__ is LayerStore:
        logs.settle()
    return True


class LayerSequence(Sequence[Any]):
    """Positional view of a :class:`LayerStore` (Layer views in insertion order)."""

    __slots__ = ("_store",)

    def __init__(self, store: LayerStore) -> None:
        """Bind the sequence to its store."""

        self._store = store

    def __len__(self) -> int:
        """Return the number of layers."""

        return len(self._store)

    def __getitem__(self, index: Any) -> Any:
        """Return the Layer view at ``index`` (or the list for a slice)."""

        store = self._store
        labels = store.labels()
        if isinstance(index, slice):
            return [store[label] for label in labels[index]]
        return store[labels[index]]

    def __iter__(self) -> Iterator[Any]:
        """Iterate the Layer views in insertion order."""

        return iter(self._store.values())

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle as a plain list of Layer records."""

        return (list, (list(self),))


def adopt_layer_logs(trace: Any, value: Any) -> Any:
    """Return ``value`` as the trace's :class:`LayerStore` (non-mappings pass through)."""

    if value.__class__ is LayerStore or not isinstance(value, Mapping):
        return value
    layers = list(value.items())
    if not all(isinstance(getattr(layer, "__dict__", None), dict) for _, layer in layers):
        return value
    store = LayerStore(trace)
    for label, layer in layers:
        store[label] = layer
    return store


class LayerLogsField:
    """Data descriptor for ``Trace._layer_logs``: every assigned mapping becomes a store."""

    __slots__ = ()

    def __get__(self, trace: Any, owner: Any = None) -> Any:
        """Return the trace's layer store, adopting a raw restored mapping first."""

        if trace is None:
            return self
        try:
            logs = trace.__dict__["_layer_logs"]
        except KeyError:
            raise AttributeError("_layer_logs") from None
        if logs.__class__ is LayerStore:
            return logs
        logs = trace.__dict__["_layer_logs"] = adopt_layer_logs(trace, logs)
        return logs

    def __set__(self, trace: Any, value: Any) -> None:
        """Adopt an assigned mapping of Layer records."""

        trace.__dict__["_layer_logs"] = adopt_layer_logs(trace, value)

    def __delete__(self, trace: Any) -> None:
        """Remove the layer store."""

        try:
            del trace.__dict__["_layer_logs"]
        except KeyError:
            raise AttributeError("_layer_logs") from None


def trace_state_copy(trace: Any) -> dict[str, Any]:
    """Return a copy of the trace's ``__dict__`` with the layer store in its saved shape."""

    state = cast("dict[str, Any]", trace.__dict__.copy())
    logs = state.get("_layer_logs")
    if logs.__class__ is LayerStore:
        state["_layer_logs"] = logs.as_ordered_dict()
    return state
