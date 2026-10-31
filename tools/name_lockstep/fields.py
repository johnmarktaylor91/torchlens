"""Stored-field facts about TorchLens classes, for inventory member rows.

A class's stored fields are declared in its ``_FIELD_POLICY`` table or, for a record
facade, in its row layout (``_TL_LAYOUT``). Module constants listing the same names are
the field's tables, which the codemod renames with the field.
"""

from __future__ import annotations

import dataclasses
import sys
from collections.abc import Mapping

from name_lockstep import definitions


def _in_package(module_name: str, package: str) -> bool:
    """Return whether ``module_name`` is ``package`` or one of its submodules."""

    return module_name == package or module_name.startswith(f"{package}.")


def policy_fields(cls: type) -> list[str]:
    """Return the user-facing stored fields a class declares in its own ``_FIELD_POLICY``.

    Parameters
    ----------
    cls
        A TorchLens class.

    Returns
    -------
    list[str]
        Field names in table order; empty when the class has no such table.
    """

    table = vars(cls).get("_FIELD_POLICY")
    if not isinstance(table, Mapping):
        return []
    return [
        name
        for name, policy in table.items()
        if isinstance(name, str) and getattr(policy, "user_facing", True)
    ]


def layout_fields(cls: type) -> list[str]:
    """Return the stored fields a record-facade class keeps in its row layout (``_TL_LAYOUT``)."""

    names = getattr(vars(cls).get("_TL_LAYOUT"), "names", None)
    if not isinstance(names, tuple) or not all(isinstance(name, str) for name in names):
        return []
    return list(names)


def init_fields(cls: type) -> frozenset[str]:
    """Return the fields a dataclass or named tuple takes as constructor keywords."""

    if dataclasses.is_dataclass(cls):
        return frozenset(field.name for field in dataclasses.fields(cls) if field.init)
    fields = getattr(cls, "_fields", None)
    if issubclass(cls, tuple) and isinstance(fields, tuple):
        return frozenset(str(name) for name in fields)
    return frozenset()


def field_tables(cls: type, package: str) -> list[str]:
    """Return the module constants that list a class's field order, as ``module:NAME``.

    A constant counts when it is a list or tuple equal to the class's ``_FIELD_POLICY``
    field order or its record-facade layout, and its module's source assigns it (an
    import of it does not count).

    Parameters
    ----------
    cls
        A class with a ``_FIELD_POLICY`` table.
    package
        Top-level package name; only its imported modules are searched.

    Returns
    -------
    list[str]
        Sorted ``module:NAME`` entries.
    """

    orders = [order for order in (policy_fields(cls), layout_fields(cls)) if order]
    found: set[str] = set()
    for module_name, module in list(sys.modules.items()):
        if not orders or not _in_package(module_name, package) or module is None:
            continue
        matches = [
            name
            for name, value in vars(module).items()
            if isinstance(value, (list, tuple)) and list(value) in orders
        ]
        tree = definitions.module_tree(module_name) if matches else None
        assigned = definitions.assigned_names(tree) if tree is not None else set()
        found.update(f"{module_name}:{name}" for name in matches if name in assigned)
    return sorted(found)
