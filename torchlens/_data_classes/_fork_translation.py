"""Fork record translation by kind and row.

A fork reads its parent's cells copy-on-write, and a parent record met in a
cell must read as the fork's record. The translator names a parent record by
the store it is bound to and its row, and finds the fork's record for the same
kind and row through the fork's own doors. It holds no record of either trace:
only store identities (weakly), row numbers, positions in the fork's op list,
and weak references to the fork's non-``Op`` records, which the fork's doors
keep alive. The one thing it keeps is the fork's copy of an object that has no
row at all (a code location), which no door of the fork holds.

While the fork is being built the translator also carries ``map``, the
identity table from each parent object to its fork shell that the build's
copies resolve through; ``seal`` drops it when the build ends.
"""

from __future__ import annotations

import weakref
from collections.abc import Iterable
from typing import Any

from .._trace_core.op_store import DetachedOpStore
from .._trace_core.record_rows import CORE_KEY, ROW_KEY
from .._utils._record_group_table import record_member
from ._accessor_base import Accessor
from ._state_adapter import state_new

_OBJECT_SETATTR = object.__setattr__
_OBJECT_GETATTRIBUTE = object.__getattribute__
_OP_KIND = "op"


def _bound_op(op: Any, op_cls: type, store: Any, row: int) -> bool:
    """Return whether ``op`` is an ``op_cls`` record bound to ``store`` at ``row``."""

    if op.__class__ is not op_cls:
        return False
    try:
        return (
            _OBJECT_GETATTRIBUTE(op, "_core") is store and _OBJECT_GETATTRIBUTE(op, "_row") == row
        )
    except AttributeError:
        return False


class _Unbound:
    """Stands for the ``Op`` class until the fork builder binds it."""


class RowTranslator:
    """Map parent records to the fork's records by kind and row.

    The fork builder binds the three class attributes in a subclass (this
    module stays outside the record classes' import cycle): the ``Op`` class,
    the column of an op's owning-trace reference, and the walk over a trace's
    op containers.

    Installed as ``OpStoreView.record_translator``: consulted on every COW
    cell read so record references inside stored containers and hydrated
    fact blocks resolve to the FORK's facades. ``Accessor`` instances stored
    in cells (``Module.ops``/``Module.params``) are rebuilt lazily with their
    members translated, memoized weakly by identity.

    A parent record bound to one of the parent's row stores is named by
    ``(kind, row)``; the store's identity is checked through a weak
    reference, so a recycled ``id`` can never mistranslate. A parent record
    with a store of its own (a detached duplicate) is named by its identity
    under a weak-reference guard, and a detached parent ``Op``, which allows
    no weak reference, by its label through the fork's label door.
    """

    __slots__ = (
        "__weakref__",
        "_copies",
        "_fork_core_ref",
        "_fork_ref",
        "_guards",
        "_miss_classes",
        "_op_positions",
        "_parent_cls",
        "_parent_ref",
        "_record_classes",
        "_shells",
        "_stores",
        "_views",
        "map",
    )

    _op_cls: type = _Unbound
    _op_trace_ref_fid: int = -1

    @staticmethod
    def _iter_ops(trace: Any) -> Iterable[Any]:
        """Walk every op container of ``trace`` (bound by the fork builder)."""

        return ()

    def __init__(self, fork_ref: Any = None, parent: Any = None, fork_core: Any = None) -> None:
        """Create an empty translator owned by the fork behind ``fork_ref``.

        ``fork_core`` is the fork's core when the fork keeps the translator
        after the build (a fork over sealed storage); without it the
        translator serves the build alone and records nothing but ``map``.
        """

        # Build-time identity table (parent object id -> fork shell). Every
        # parent object is alive for the whole build, so identity is exact
        # there; ``seal`` drops the table.
        self.map: dict[int, Any] | None = {}
        # Classes whose instances can never translate (tensors, dtypes,
        # plain metadata): one set lookup short-circuits the miss path.
        # Record classes and the parent Trace's class are never added.
        self._miss_classes: set[type] = set()
        self._record_classes: set[type] = set()
        self._parent_cls: type | None = None if parent is None else type(parent)
        self._fork_ref = fork_ref
        # Weak fork core: a module's cached keyed buffer accessor rebinds to its buffer table.
        self._fork_core_ref: Any = None if fork_core is None else weakref.ref(fork_core)
        # Weak parent-trace reference: a parent-trace value met in copied
        # state translates to the fork WITHOUT the translator pinning the
        # parent (GC parity).
        self._parent_ref: Any = None if parent is None else weakref.ref(parent)
        # id(parent store) -> (weak store reference, kind); ``None`` when the
        # translator is not kept after the build.
        self._stores: dict[int, tuple[Any, str]] | None = None
        # kind -> weak reference to the fork's view of that kind's store.
        self._views: dict[str, Any] = {}
        # (kind, row), or id(parent object) for a parent with no shared
        # store, -> weak reference to the fork's record.
        self._shells: dict[Any, Any] = {}
        # id(parent object) -> weak reference to it, for the identity names.
        self._guards: dict[int, Any] = {}
        # id(parent object with no row, a code location) -> the fork's copy.
        self._copies: dict[int, Any] = {}
        # op row -> position in the fork's op list (checked on every use).
        self._op_positions: dict[int, int] = {}
        if fork_core is not None and parent is not None:
            self._register_stores(parent.__dict__.get("_trace_core"), fork_core)

    def _register_stores(self, core: Any, fork_core: Any) -> None:
        """Record the parent's row stores, and the root stores under them, by kind."""

        stores: dict[int, tuple[Any, str]] = {}
        pairs = [(_OP_KIND, core.ops, fork_core.ops)]
        pairs.extend(
            (kind, store, fork_core.kind_rows.get(kind)) for kind, store in core.kind_rows.items()
        )
        for kind, store, view in pairs:
            if store is not None:
                stores[id(store)] = (weakref.ref(store), kind)
            if view is not None:
                # A fork of a fork reads the root store's cells, which can
                # still hold the root trace's records at the same rows.
                stores[id(view.base)] = (weakref.ref(view.base), kind)
                self._views[kind] = weakref.ref(view)
        self._stores = stores

    def install(self, fork_core: Any) -> None:
        """Anchor the translator on the fork core and bind it to every view.

        The views hold the translator WEAKLY (a retained fork Op reaching the
        fork Trace through the translator would root the whole fork graph and
        disarm last-owner payload eviction); the fork core is its strong
        anchor, so translation lives exactly as long as the fork itself.
        """

        fork_core._record_translator = self
        for view in fork_core.store_views():
            view.record_translator = self

    def building(self) -> dict[int, Any]:
        """Return the build's identity table (parent object id -> fork shell)."""

        if self.map is None:
            raise RuntimeError("the fork build that owned this translator has ended")
        return self.map

    def seal(self) -> None:
        """End the build: drop the identity table and its fork shells."""

        self.map = None

    def add(self, parent_record: Any, shell: Any) -> None:
        """Register one parent record -> fork shell mapping."""

        key = id(parent_record)
        building = self.map
        if building is not None:
            building[key] = shell
        stores = self._stores
        cls = parent_record.__class__
        if stores is None or cls is self._op_cls:
            # Not kept after the build, or an Op: a store-bound Op is found
            # by its row and a detached one by its label, with no entry.
            return
        self._record_classes.add(cls)
        store = parent_record.__dict__.get(CORE_KEY)
        entry = stores.get(id(store)) if store is not None else None
        if entry is not None:
            self._shells[(entry[1], parent_record.__dict__[ROW_KEY])] = weakref.ref(shell)
            return
        self._guards[key] = weakref.ref(parent_record)
        if store is None:
            # No row names this object and no door of the fork holds its
            # copy, so the copy lives here for as long as the fork does.
            self._copies[key] = shell
        else:
            self._shells[key] = weakref.ref(shell)

    def _remember_accessor(self, accessor: Any, clone: Any) -> None:
        """Memoize one rebuilt accessor without holding it."""

        key = id(accessor)
        if self.map is not None:
            self.map[key] = clone
        if self._stores is not None:
            self._guards[key] = weakref.ref(accessor)
            self._shells[key] = weakref.ref(clone)

    def _by_identity(self, value: Any) -> Any:
        """Return the live fork object memoized for ``value`` itself, or ``None``."""

        key = id(value)
        guard = self._guards.get(key)
        if guard is None:
            return None
        if guard() is not value:
            # The original parent object died and its id was recycled by
            # ``value``: drop the stale entry.
            del self._guards[key]
            self._shells.pop(key, None)
            self._copies.pop(key, None)
            return None
        copied = self._copies.get(key)
        if copied is not None:
            return copied
        shell_ref = self._shells.get(key)
        return shell_ref() if shell_ref is not None else None

    def _member(self, item: Any) -> Any:
        """Return the fork-side member for one accessor member."""

        building = self.map
        if building is not None:
            return building.get(id(item), item)
        mapped = self(item)
        return item if mapped is None else mapped

    def _rebuild_accessor(self, accessor: Any) -> Any:
        """Rebuild one cell-stored accessor with translated members."""

        if self.map is None:
            known = self._by_identity(accessor)
            if known is not None:
                return known
        if "_rows" in accessor.__dict__:
            from ._buffer_rows import fork_module_accessor

            core_ref = self._fork_core_ref
            fork = self._fork_ref() if self._fork_ref is not None else None
            core = core_ref() if core_ref is not None else None
            keyed = fork_module_accessor(accessor, core, fork, self._member)
            self._remember_accessor(accessor, keyed)
            return keyed
        accessor_cls: Any = type(accessor)
        clone = accessor_cls.__new__(accessor_cls)
        clone.__dict__.update(accessor.__dict__)
        member = self._member
        clone._dict = {key: member(item) for key, item in accessor._dict.items()}
        clone._list = [member(item) for item in accessor._list]
        if "_source_ref" in clone.__dict__ and self._fork_ref is not None:
            clone._source_ref = self._fork_ref
        self._remember_accessor(accessor, clone)
        return clone

    def _op_at_row(self, fork: Any, row: int) -> Any:
        """Return the fork's Op for one row of its op view."""

        view_ref = self._views.get(_OP_KIND)
        view = view_ref() if view_ref is not None else None
        if view is None:
            return None
        op_cls = self._op_cls
        ops = fork.__dict__.get("_layer_list") or ()
        position = self._op_positions.get(row)
        if (
            position is not None
            and position < len(ops)
            and _bound_op(ops[position], op_cls, view, row)
        ):
            return ops[position]
        # The op list moved (or this is the first lookup): index it again.
        positions: dict[int, int] = {}
        for index, op in enumerate(ops):
            if op.__class__ is op_cls and _OBJECT_GETATTRIBUTE(op, "_core") is view:
                positions.setdefault(_OBJECT_GETATTRIBUTE(op, "_row"), index)
        self._op_positions = positions
        position = positions.get(row)
        if position is not None:
            return ops[position]
        for op in self._iter_ops(fork):
            if _bound_op(op, op_cls, view, row):
                return op
        # No fork door holds an Op for this row (it was removed from the
        # fork). An Op is its view and row, so a new one reads the same.
        shell: Any = state_new(op_cls)
        _OBJECT_SETATTR(shell, "_core", view)
        _OBJECT_SETATTR(shell, "_row", row)
        return shell

    def _detached_op(self, fork: Any, value: Any, store: Any) -> Any:
        """Return the fork's duplicate of one detached parent Op, by label."""

        owner = store.cell_get(_OBJECT_GETATTRIBUTE(value, "_row"), self._op_trace_ref_fid)
        parent_ref = self._parent_ref
        if owner is not parent_ref:
            parent = parent_ref() if parent_ref is not None else None
            if parent is None or not callable(owner) or owner() is not parent:
                return None
        label = record_member(value, "label", None)
        if label.__class__ is not str:
            return None
        candidate = (fork.__dict__.get("_layer_dict_main_keys") or {}).get(label)
        if candidate.__class__ is not self._op_cls:
            return None
        if _OBJECT_GETATTRIBUTE(candidate, "_core").__class__ is not DetachedOpStore:
            return None
        return candidate

    def _fork_op(self, value: Any) -> Any:
        """Return the fork's Op for one parent Op, or ``None``."""

        stores = self._stores
        fork = self._fork_ref() if self._fork_ref is not None else None
        if stores is None or fork is None:
            return None
        try:
            store = _OBJECT_GETATTRIBUTE(value, "_core")
        except AttributeError:
            return None
        entry = stores.get(id(store))
        if entry is not None:
            if entry[1] == _OP_KIND and entry[0]() is store:
                return self._op_at_row(fork, _OBJECT_GETATTRIBUTE(value, "_row"))
            return None
        if store.__class__ is DetachedOpStore:
            return self._detached_op(fork, value, store)
        return None

    def _fork_record(self, value: Any) -> Any:
        """Return the fork's record for one parent record that is not an Op."""

        stores = self._stores
        store = value.__dict__.get(CORE_KEY)
        entry = stores.get(id(store)) if stores is not None and store is not None else None
        if entry is None or entry[0]() is not store:
            return self._by_identity(value)
        kind = entry[1]
        key = (kind, value.__dict__[ROW_KEY])
        shell_ref = self._shells.get(key)
        if shell_ref is None:
            return None
        shell = shell_ref()
        if shell is None:
            # No fork door holds this record any more. A record over a view is
            # its view and row, so a new one reads the same cells.
            view_ref = self._views.get(kind)
            view = view_ref() if view_ref is not None else None
            if view is None:
                return None
            shell = state_new(type(value))
            shell.__dict__[CORE_KEY] = view
            shell.__dict__[ROW_KEY] = key[1]
            self._shells[key] = weakref.ref(shell)
        return shell

    def _other(self, value: Any, cls: type) -> Any:
        """Translate the parent trace or a cell-stored accessor; learn a miss class."""

        parent_ref = self._parent_ref
        if parent_ref is not None and value is parent_ref():
            return self._fork_ref() if self._fork_ref is not None else None
        if isinstance(value, Accessor):
            return self._rebuild_accessor(value)
        if (
            cls is not self._parent_cls
            and cls is not self._op_cls
            and cls not in self._record_classes
        ):
            # ``isinstance`` returned False, so no instance of ``cls`` can
            # ever rebuild as an accessor; only the parent's own class must
            # keep taking the identity check above.
            self._miss_classes.add(cls)
        return None

    def __call__(self, value: Any) -> Any:
        """Return the fork-side object for ``value``, or ``None``."""

        building = self.map
        if building is not None:
            mapped = building.get(id(value))
            if mapped is not None:
                return mapped
            cls = value.__class__
            if cls in self._miss_classes:
                return None
            return self._other(value, cls)
        cls = value.__class__
        if cls in self._miss_classes:
            return None
        if cls is self._op_cls:
            return self._fork_op(value)
        if cls in self._record_classes:
            return self._fork_record(value)
        return self._other(value, cls)
