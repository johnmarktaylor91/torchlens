"""Shared-fact column blocks: the FunctionCall / ParamAlias group tables.

One fact family holds the call-level (or param-level) container facts of a
finished trace ONCE per group instead of once per op row. At the core freeze
point each declared field converts exactly once:

* the **call** family groups rows by ``func_call_id`` (sol's
  ``FunctionCallGroup``): every sibling output of one wrapped call shares the
  group's canonical immutable snapshot of ``code_context``,
  ``non_tensor_pos_args``, ``non_tensor_kwargs``, ``func_non_tensor_args``,
  ``func_config``, and ``arg_names``;
* the **param** family groups rows by value (the ``ParamAliasGroup`` block
  scoped to what value interning does not already share): ``param_shapes``.

Canonical snapshots are immutable (`tuple`, or a tuple of items for a dict)
and interned through the trace's shared view pool, so equal facts collapse
across groups too (every op captured from one source line shares ONE
``code_context`` snapshot). Member cells hold the ``_FACT`` sentinel; the
facade descriptor hydrates the exact public container type for THAT row on
first read and caches it back, so reads are identity-stable, per-row in-place
mutation stays isolated (today's fresh-container-per-row semantics exactly),
and uninspected rows retain no per-row container.

Like value interning, the conversion is differential by construction: a group member whose
value does not equal the group's first-seen value (or whose equality probe
raises) keeps its explicit cell — the block is authoritative only where it
provably reproduces the public surface. A direct write to one Op replaces the
sentinel with a per-row cell value and never mutates the shared block.
"""

from __future__ import annotations

from typing import Any

from .op_store import _FACT, _MISSING

#: Call-family fields (grouped by ``func_call_id``) mapped to their exact
#: public container type, which hydration must reproduce per row.
OP_CALL_FACT_FIELDS: dict[str, type] = {
    "code_context": list,
    "non_tensor_pos_args": list,
    "non_tensor_kwargs": dict,
    "func_non_tensor_args": list,
    "func_config": dict,
    "arg_names": tuple,
}

#: Param-family fields (grouped by value; the ParamAlias block).
OP_PARAM_FACT_FIELDS: dict[str, type] = {
    "param_shapes": list,
}

#: Every fact field's public type, for the facade descriptor installer.
OP_FACT_FIELDS: dict[str, type] = {**OP_CALL_FACT_FIELDS, **OP_PARAM_FACT_FIELDS}


class FactFamily:
    """One shared-fact column block.

    Parameters
    ----------
    fields:
        Field-name to public-container-type mapping for this family.
    """

    __slots__ = ("fields", "group_of_row", "columns")

    def __init__(self, fields: dict[str, type]) -> None:
        """Create an empty family for ``fields``."""

        self.fields = fields
        self.group_of_row: dict[tuple[str, int], int] = {}
        self.columns: dict[str, list[Any]] = {name: [] for name in fields}

    def __len__(self) -> int:
        """Return the number of group rows."""

        return max((len(column) for column in self.columns.values()), default=0)


class FactBlocks:
    """Per-store registry of fact families with row hydration."""

    __slots__ = ("families", "family_of_field")

    def __init__(self) -> None:
        """Create an empty registry."""

        self.families: dict[str, FactFamily] = {}
        self.family_of_field: dict[str, str] = {}

    def add_family(self, name: str, fields: dict[str, type]) -> FactFamily:
        """Register one family and index its fields."""

        family = FactFamily(fields)
        self.families[name] = family
        for field_name in fields:
            self.family_of_field[field_name] = name
        return family

    def hydrate(self, row: int, field_name: str) -> Any:
        """Return the public-typed value of one ``_FACT`` cell.

        Mutable public types (``list``/``dict``) hydrate a FRESH shallow copy
        of the canonical snapshot — the per-row isolation contract; immutable
        ``tuple`` fields return the shared canonical itself.
        """

        family = self.families[self.family_of_field[field_name]]
        canonical = family.columns[field_name][family.group_of_row[(field_name, row)]]
        public_type = family.fields[field_name]
        if public_type is tuple:
            return canonical
        return public_type(canonical)


def _canonical_snapshot(value: Any, public_type: type) -> Any:
    """Return the immutable canonical form of one staging container."""

    if public_type is dict:
        return tuple(value.items())
    if value.__class__ is tuple:
        return value
    return tuple(value)


def convert_fact_cells(store: Any, pool: dict[Any, Any]) -> None:
    """Convert shared-fact staging cells of a building-phase op store.

    Runs at the relation freeze point (``freeze_trace_relation_views``).
    Rows group by ``func_call_id`` for the call family and by value for the
    param family; within a call group every member must equal the first-seen
    canonical (probed defensively — an unequal or unprobeable member keeps
    its explicit cell).

    Parameters
    ----------
    store:
        Building-phase ``OpRowStore``; no-op once frozen.
    pool:
        Shared view-intern pool (canonicals dedup across groups and against
        the relation views).
    """

    from .relation_views import intern_view

    rows = store.rows_building()
    if rows is None or store.fact_blocks is not None:
        return
    fid_by_name = store.layout.fid_by_name
    blocks = FactBlocks()

    call_family = blocks.add_family("call", OP_CALL_FACT_FIELDS)
    call_fid = fid_by_name.get("func_call_id")
    call_fids = {
        name: fid for name in OP_CALL_FACT_FIELDS if (fid := fid_by_name.get(name)) is not None
    }
    if call_fid is not None and call_fids:
        group_by_call: dict[int, int] = {}
        for row, row_cells in enumerate(rows):
            call_id = row_cells[call_fid]
            if call_id.__class__ is not int:
                continue
            gid = group_by_call.get(call_id)
            if gid is None:
                gid = len(group_by_call)
                group_by_call[call_id] = gid
                for name, fid in call_fids.items():
                    column = call_family.columns[name]
                    value = row_cells[fid]
                    if value.__class__ is OP_CALL_FACT_FIELDS[name]:
                        canonical = intern_view(
                            _canonical_snapshot(value, OP_CALL_FACT_FIELDS[name]), pool
                        )
                        column.append(canonical)
                        row_cells[fid] = _FACT
                        call_family.group_of_row[(name, row)] = gid
                    else:
                        column.append(_MISSING)
                continue
            # Sibling member: convert only where it provably equals the
            # group canonical (differential discipline, like value interning).
            for name, fid in call_fids.items():
                canonical = call_family.columns[name][gid]
                if canonical is _MISSING:
                    continue
                value = row_cells[fid]
                if value.__class__ is not OP_CALL_FACT_FIELDS[name]:
                    continue
                try:
                    matches = _canonical_snapshot(value, OP_CALL_FACT_FIELDS[name]) == canonical
                except Exception:
                    matches = False
                if matches is True:
                    row_cells[fid] = _FACT
                    call_family.group_of_row[(name, row)] = gid

    param_family = blocks.add_family("param", OP_PARAM_FACT_FIELDS)
    param_fids = {
        name: fid for name in OP_PARAM_FACT_FIELDS if (fid := fid_by_name.get(name)) is not None
    }
    for name, fid in param_fids.items():
        column = param_family.columns[name]
        gid_by_canonical: dict[Any, int] = {}
        for row, row_cells in enumerate(rows):
            value = row_cells[fid]
            if value.__class__ is not OP_PARAM_FACT_FIELDS[name]:
                continue
            try:
                canonical = intern_view(
                    _canonical_snapshot(value, OP_PARAM_FACT_FIELDS[name]), pool
                )
                gid = gid_by_canonical.get(canonical)
            except TypeError:
                # Unhashable member values: keep the explicit cell.
                continue
            if gid is None:
                gid = len(column)
                gid_by_canonical[canonical] = gid
                column.append(canonical)
            row_cells[fid] = _FACT
            param_family.group_of_row[(name, row)] = gid

    store.fact_blocks = blocks
