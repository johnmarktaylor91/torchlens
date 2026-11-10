"""Trace-held Layer state: one state mapping per layer, with Layer views on demand.

A trace keeps each layer's state as one dictionary (the layer's user
overrides, deletion tombstones, build-phase snapshots and ``annotations``
all live in it, exactly as they did in the Layer's own ``__dict__``). A
``Layer`` object is a view whose ``__dict__`` *is* that dictionary, so a view
built later reads and writes the same state. The store holds the state; the
view objects are cached in the trace core's view table
(``_trace_core/view_tables.py``): strongly while the trace is built, weakly
afterwards, when a view nobody kept is built again from the state.
"""

from __future__ import annotations

import weakref
from collections import OrderedDict, deque
from collections.abc import Iterator, Mapping, Sequence
from itertools import repeat
from typing import Any, ClassVar, cast

from .._trace_core.view_tables import DEAD, OP_KIND, WeakViews
from ._layer_ops_door import install as install_ops_door

#: The ``Layer`` class, bound by ``layer.py`` at import so this module stays a leaf.
_LAYER_CLASS: list[type] = []


def bind_layer_class(layer_class: type) -> None:
    """Bind the ``Layer`` class that views are built from, and install its ``ops`` door."""

    _LAYER_CLASS[:] = [layer_class]
    install_ops_door(layer_class)


def hold_view(core: Any, kind: str, index: int, view: Any, *, replace: bool = False) -> bool:
    """Cache ``view`` at ``index`` of the core's view table for ``kind``; return whether it is there.

    A row another live view holds is left to it unless ``replace`` is set.
    """

    tables = core._views
    slots = tables.ops if kind == OP_KIND else tables.layers
    if slots.__class__ is WeakViews:
        # A weak table is asked for the live view only (a plain read builds
        # one). Written out, with no call per row: a fork holds every layer here.
        weak: Any = slots
        try:
            current = weak.refs[index]()
        except IndexError:
            grow = index + 1 - len(weak)
            weak.refs.extend(repeat(DEAD, grow))
            list.extend(weak, repeat(None, grow))
            current = None
    else:
        if index >= len(slots):
            slots.pad(index + 1)
        current = slots[index]
    if current is None or replace:
        slots[index] = view
        return True
    return bool(current is view)


def _weak_layer_views(slots: Any, states: list[dict[str, Any]]) -> list[Any]:
    """Return the Layer view of each state from a weak table, building the ones nobody kept.

    The work per layer runs in C, so reading every layer of a built trace
    costs a fixed number of calls. Raises ``KeyError`` for a layer that is not
    keyed into the table (the caller then takes the per-label path).
    """

    rows = [state["ops"].__dict__["_keys"][1] for state in states]
    if rows and min(rows) < 0:
        raise KeyError("unkeyed layer")
    views = cast("list[Any]", slots.many(rows))
    stale = [
        at
        for at, (view, state) in enumerate(zip(views, states, strict=True))
        if view is None or view.__dict__ is not state
    ]
    if stale:
        built: list[Any] = list(map(object.__new__, repeat(_LAYER_CLASS[0], len(stale))))
        deque(map(object.__setattr__, built, repeat("__dict__"), [states[at] for at in stale]), 0)
        slots.adopt([rows[at] for at in stale], built)
        for at, view in zip(stale, built, strict=True):
            views[at] = view
    return views


def ops_with_layers_kept(trace: Any) -> Any:
    """Pair each op of ``trace`` with one list holding every Layer view of the trace.

    For a loop that looks layers up again and again (a table export): the
    views stay alive while the loop runs, so each lookup meets the live view
    instead of building a new one.
    """

    logs = trace.__dict__.get("_layer_logs")
    return zip(trace._layer_list, repeat(list(logs.values()) if logs else ()))


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
    A store adopted from a plain ``dict`` is a :class:`LayerDictStore`, whose
    plain form (pickle, copy, repr, equality) is a ``dict`` again.
    """

    __slots__ = ("_held", "_labels", "_trace_ref")

    #: The container this store stands in for: its plain form has this type.
    plain_type: ClassVar[type[dict[str, Any]]] = OrderedDict

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
        try:
            accessor_state = state["ops"].__dict__
            slots = accessor_state["_core_ref"]()._views.layers
            refs, row = slots.refs, accessor_state["_keys"][1]
            view = slots[row] if refs is None else refs[row]()  # no call for a live view
            if view.__dict__ is state:
                return view
        except (KeyError, IndexError, AttributeError, TypeError):
            pass  # not keyed into a live core, or its view is gone: build it
        return self._build(label, state)

    def get(self, label: Any, default: Any = None) -> Any:
        """Return the Layer view for ``label``, or ``default``.

        The keyed view is looked up here directly (one dict read, no nested
        call); anything else takes the ``__getitem__`` path.
        """

        try:
            state = dict.__getitem__(self, label)
        except KeyError:
            return default
        if self._held is None:
            try:
                accessor_state = state["ops"].__dict__
                slots = accessor_state["_core_ref"]()._views.layers
                refs, row = slots.refs, accessor_state["_keys"][1]
                view = slots[row] if refs is None else refs[row]()  # no call for a live view
                if view.__dict__ is state:
                    return view
            except (KeyError, IndexError, AttributeError, TypeError):
                pass  # not keyed into a live core, or its view is gone: build it
        return self[label]

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
            other = other.plain()
        return bool(self.plain() == other)

    def __ne__(self, other: object) -> bool:
        """Compare as the ordered mapping of Layer views."""

        return not self == other

    def __repr__(self) -> str:
        """Render as the plain mapping this store stands in for."""

        return repr(self.plain())

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle as the plain mapping of Layer records it was given (the saved shape)."""

        return (self.plain_type, (), None, None, iter(self.items()))

    def values(self) -> Any:
        """Return every Layer view in insertion order."""

        if self._held is None:
            # Every layer is keyed: read the views straight from the core's
            # view list, keeping only a view that is over this store's state.
            try:
                ref: Any = self._trace_ref
                slots = ref().__dict__["_trace_core"]._views.layers
                if slots.__class__ is WeakViews:
                    return _weak_layer_views(slots, list(dict.values(self)))
                views = [
                    view
                    for state in dict.values(self)
                    for view in (slots[state["ops"].__dict__["_keys"][1]],)
                    if view.__dict__ is state
                ]
                if len(views) == len(self):
                    return views
            except (KeyError, IndexError, AttributeError, TypeError):
                pass  # no core or an unkeyed layer: take the per-label path
        self.settle()
        return [self[label] for label in self]

    def items(self) -> Any:
        """Return ``(label, Layer view)`` pairs in insertion order."""

        return list(zip(self, self.values()))

    def clear(self) -> None:
        """Drop every layer's state and view."""

        for label, state in list(dict.items(self)):
            self._drop_view(label, state)
        OrderedDict.clear(self)
        self._held = None
        self._labels = None

    def copy(self) -> Any:
        """Return the plain mapping of the Layer views."""

        return self.plain()

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

    def plain(self) -> dict[str, Any]:
        """Return the plain mapping (of :attr:`plain_type`) from label to Layer view."""

        return self.plain_type(self.items())

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
        slots = core._views.layers
        if index < len(slots):
            view = slots[index]
            if view is not None and view.__dict__ is state:
                slots[index] = None


class LayerDictStore(LayerStore):
    """The layer store adopted from a plain ``dict`` (a loaded trace's layers).

    It behaves as :class:`LayerStore`; only its plain form is a ``dict``, so a
    loaded trace pickles, copies and compares its layers as the base did.
    """

    __slots__ = ()

    plain_type: ClassVar[type[dict[str, Any]]] = dict


#: Every layer store class (``__class__`` membership costs no call).
STORE_CLASSES = (LayerStore, LayerDictStore)


def settle_layer_logs(trace: Any) -> bool:
    """Settle the trace's layer store after its core is rebuilt; return ``True``.

    The constant result lets the size-capped rehydrate function end with one
    ``return`` that both settles the store and reports a rebuilt core.
    """

    logs = trace.__dict__.get("_layer_logs")
    if logs.__class__ in STORE_CLASSES:
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
    """Return ``value`` as the trace's layer store (non-mappings pass through).

    An ``OrderedDict`` (the live trace's container) becomes a
    :class:`LayerStore`; any other mapping (a loaded trace's ``dict``) becomes a
    :class:`LayerDictStore`, so the store keeps the container type it was given.
    """

    if value.__class__ in STORE_CLASSES or not isinstance(value, Mapping):
        return value
    layers = list(value.items())
    if not all(isinstance(getattr(layer, "__dict__", None), dict) for _, layer in layers):
        return value
    store = (LayerStore if isinstance(value, OrderedDict) else LayerDictStore)(trace)
    for label, layer in layers:
        store[label] = layer
    return store


def install_layer_logs(trace: Any, value: Any) -> None:
    """Set ``trace._layer_logs`` to ``value`` adopted as a layer store.

    The store lives in the trace's instance dictionary with no descriptor, so
    a read of ``trace._layer_logs`` costs no call; every writer of the field
    goes through here instead.
    """

    trace.__dict__["_layer_logs"] = adopt_layer_logs(trace, value)


def adopt_restored_layer_logs(trace: Any) -> Any:
    """Adopt the raw mapping a state restore put in ``trace.__dict__``; return the field."""

    logs = trace.__dict__["_layer_logs"] = adopt_layer_logs(trace, trace.__dict__["_layer_logs"])
    return logs


def trace_state_copy(trace: Any) -> dict[str, Any]:
    """Return a copy of the trace's ``__dict__`` with the layer store in its saved shape."""

    state = cast("dict[str, Any]", trace.__dict__.copy())
    logs = state.get("_layer_logs")
    if logs.__class__ in STORE_CLASSES:
        state["_layer_logs"] = logs.plain()
    return state
