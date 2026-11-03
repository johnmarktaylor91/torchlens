"""Direct row release for op removal (the facade-husk shortcut).

The facade husk reads every stored ``Op`` field (skipping the unset ones) and
deletes each one it read. For an exact ``Op`` whose every stored-field class
attribute is a generated field descriptor (all share ``_OpField.__delete__``,
and each read raises ``AttributeError`` exactly on an unset cell) or a
registered plain-facade property, that deletes exactly the row's set cells,
which ``store.release_row`` does without the per-field dispatch and without
hydrating pooled or shared-fact cells it is about to drop.

``op.py`` binds the ``Op`` facade here once its field descriptors exist
(:func:`bind_op_facade`); backends that install a property over a stored
field declare it (:func:`register_row_release_property`). Standard library
only, so ``op.py`` imports it without an import cycle.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

#: The bound facade, set once by ``op.py``: ``(Op class, its live class
#: namespace, stored field names, generated field descriptor classes)``.
_OP_FACADE: list[tuple[type, Any, tuple[str, ...], frozenset[type]]] = []

#: Class-level ``Op`` properties declared plain stored-cell facades (see
#: :func:`register_row_release_property`).
_ROW_RELEASE_PROPERTIES: set[Any] = set()


def bind_op_facade(
    op_class: type, slot_names: Iterable[str], field_classes: frozenset[type]
) -> None:
    """Bind the ``Op`` class, its stored field names and its descriptor classes.

    Parameters
    ----------
    op_class:
        The exact ``Op`` facade class.
    slot_names:
        Stored field names in layout order.
    field_classes:
        Generated field descriptor classes that share ``_OpField.__delete__``.
    """

    _OP_FACADE[:] = [(op_class, vars(op_class), tuple(slot_names), field_classes)]


def register_row_release_property(name: str) -> None:
    """Declare the class property ``Op.<name>`` a plain facade over its stored cell.

    Contract: its getter raises ``AttributeError`` exactly when the backing
    cell is unset, and its deleter only deletes that cell. Op removal may then
    release the op's row directly instead of reading and deleting every field
    through the facade (:func:`release_op_row`). Register after the property
    is installed: the live class attribute is what gets recorded.

    Parameters
    ----------
    name:
        Stored field name whose ``Op`` class attribute is the property.
    """

    _ROW_RELEASE_PROPERTIES.add(_OP_FACADE[0][1][name])


def release_op_row(op: Any, store: Any, row: int) -> bool:
    """Delete every set cell of ``op``'s row directly, when that equals the facade husk.

    Anything outside the module contract (a subclass, a patched or
    unregistered stored-field attribute, a store without ``release_row``)
    returns ``False`` and the caller keeps the facade path.

    Parameters
    ----------
    op:
        Op being husked.
    store:
        The op's bound row store.
    row:
        The op's row id in ``store``.

    Returns
    -------
    bool
        Whether the row was released here.
    """

    op_class, namespace, slot_names, field_classes = _OP_FACADE[0]
    if op.__class__ is not op_class:
        return False
    release_row = getattr(store.__class__, "release_row", None)
    if release_row is None:
        return False
    properties = _ROW_RELEASE_PROPERTIES
    for name in slot_names:
        attr = namespace.get(name)
        if attr.__class__ in field_classes or attr in properties:
            continue
        return False
    release_row(store, row)
    return True
