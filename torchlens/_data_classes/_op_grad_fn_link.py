"""The ``Op`` cells that link an op to its ``GradFn`` record, held as keys.

``Op.autograd.grad_fn`` (and, on a restored trace, ``grad_fn_handle``) names a
``GradFn`` record. The op store holds a key to that record instead of the
record, so the store keeps no ``GradFn`` object; a read builds the record
from the key (``_grad_fn_views``). Any other value in these cells (``None``,
a live autograd node) is stored and read as before.

``install_op_links`` is called once, by a module that has the ``Op`` class;
this module imports no record class.
"""

from __future__ import annotations

from typing import Any

from .._trace_core.op_store import _MISSING, PooledCell
from .._utils._record_group_table import OP_STORAGE_ATTRS
from ._grad_fn_views import GradFnKey, link_of, linked
from ._op_row_release import register_row_release_property

#: The stored ``Op`` fields that can name a ``GradFn`` record.
LINK_FIELDS = ("grad_fn", "grad_fn_handle")


class OpGradFnLink:
    """Stored ``Op`` field whose ``GradFn`` value is kept as a key.

    It reads, writes and deletes its row cell exactly as the plain stored
    field descriptor it replaces does, except for a ``GradFn`` record.
    """

    __slots__ = ("_core_get", "_fid", "_name", "_row_get")

    #: Raw cell classes a column read must hand to this descriptor.
    COLUMN_SLOW_CELLS = (PooledCell, GradFnKey)

    def __init__(self, name: str, fid: int, core_get: Any, row_get: Any) -> None:
        """Bind the field name, its column id and the two ``Op`` slot readers."""

        self._name = name
        self._fid = fid
        self._core_get = core_get
        self._row_get = row_get

    def __repr__(self) -> str:
        """Return a diagnostic representation naming the backed field."""

        return f"<Op field descriptor {self._name!r}>"

    def __get__(self, op: Any, owner: Any = None) -> Any:
        """Read the cell, building the ``GradFn`` record a key names."""

        if op is None:
            return self
        store = self._core_get(op)
        row = self._row_get(op)
        value = store.cell_get(row, self._fid)
        if value.__class__ is GradFnKey:
            return linked(value)
        if value is _MISSING:
            name = self._name
            raise AttributeError(
                f"{type(op).__name__!r} object has no attribute {name!r}",
                name=name,
                obj=op,
            )
        if value.__class__ is PooledCell:
            value = value.hydrate()
            store.cell_set(row, self._fid, value)
        return value

    def __set__(self, op: Any, value: Any) -> None:
        """Write the cell, keying a ``GradFn`` record."""

        self._core_get(op).cell_set(self._row_get(op), self._fid, link_of(value))

    def __delete__(self, op: Any) -> None:
        """Delete the cell; an unset cell raises like an unset slot."""

        if not self._core_get(op).cell_del(self._row_get(op), self._fid):
            raise AttributeError(self._name)


def install_op_links(op_class: type) -> None:
    """Replace the two link descriptors on ``op_class`` (once, at import)."""

    namespace = vars(op_class)
    core_get = namespace["_core"].__get__
    row_get = namespace["_row"].__get__
    for name in LINK_FIELDS:
        attr = OP_STORAGE_ATTRS.get(name, name)
        plain = namespace[attr]
        setattr(op_class, attr, OpGradFnLink(plain._name, plain._fid, core_get, row_get))
        register_row_release_property(attr)
