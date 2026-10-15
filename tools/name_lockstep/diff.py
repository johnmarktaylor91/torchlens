"""Diff two surface inventories into new, removed and renamed public names."""

from __future__ import annotations

import dataclasses
from typing import Any

Rows = dict[str, dict[str, Any]]


@dataclasses.dataclass(frozen=True)
class Rename:
    """One public name that moved from ``old_id`` to ``new_id``.

    Attributes
    ----------
    old_id, new_id
        Stable row ids on the base and head sides.
    old_name, new_name
        Leaf spellings.
    basis
        ``declared`` (from a rename map) or ``fingerprint`` (same owner, kind, signature
        and docstring line, unique on both sides).
    """

    old_id: str
    new_id: str
    old_name: str
    new_name: str
    basis: str


@dataclasses.dataclass(frozen=True)
class SurfaceDiff:
    """New, removed and renamed public names between two inventories.

    Attributes
    ----------
    new
        Head rows with no base counterpart, sorted by id.
    removed
        Base rows with no head counterpart, sorted by id.
    renamed
        Pairs of a removed and a new row that are one name respelled.
    """

    new: tuple[dict[str, Any], ...]
    removed: tuple[dict[str, Any], ...]
    renamed: tuple[Rename, ...]

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary.

        Returns
        -------
        dict[str, Any]
            ``counts``, ``new``, ``removed`` and ``renamed`` lists.
        """

        return {
            "counts": {
                "new": len(self.new),
                "removed": len(self.removed),
                "renamed": len(self.renamed),
            },
            "new": [_brief(row) for row in self.new],
            "removed": [_brief(row) for row in self.removed],
            "renamed": [dataclasses.asdict(rename) for rename in self.renamed],
        }


def _brief(row: dict[str, Any]) -> dict[str, Any]:
    """Return the identifying fields of a row.

    Parameters
    ----------
    row
        An inventory row.

    Returns
    -------
    dict[str, Any]
        id, kind, name and path.
    """

    return {key: row.get(key) for key in ("id", "kind", "name", "path")}


def renamed_id(row: dict[str, Any], new_name: str) -> str:
    """Return the id a row would carry after its leaf is respelled.

    Parameters
    ----------
    row
        An inventory row (needs ``id``, ``kind`` and ``name``).
    new_name
        The proposed leaf spelling.

    Returns
    -------
    str
        The respelled id.
    """

    old_id, old_name = row["id"], row["name"]
    if row["kind"] == "param":
        return f"param:{new_name}"
    if not old_id.endswith(old_name):
        raise ValueError(f"row id {old_id!r} does not end with its name {old_name!r}")
    return old_id[: -len(old_name)] + new_name


def _declared_renames(
    base: Rows, head: Rows, removed: set[str], new: set[str], declared: dict[str, str]
) -> list[Rename]:
    """Pair rows through a declared map of old id to proposed leaf or new id.

    Parameters
    ----------
    base, head
        Inventories keyed on id.
    removed, new
        Unpaired ids; paired ids are discarded from both sets.
    declared
        Old id to proposed leaf spelling or full new id.

    Returns
    -------
    list[Rename]
        Declared renames found on both sides.
    """

    found: list[Rename] = []
    for old_id, target in sorted(declared.items()):
        if old_id not in removed:
            continue
        new_id = target if ":" in target else renamed_id(base[old_id], target)
        if new_id in new:
            found.append(
                Rename(old_id, new_id, base[old_id]["name"], head[new_id]["name"], "declared")
            )
            removed.discard(old_id)
            new.discard(new_id)
    return found


def _fingerprint_renames(base: Rows, head: Rows, removed: set[str], new: set[str]) -> list[Rename]:
    """Pair rows whose spelling-free fingerprint is unique on both sides.

    Parameters
    ----------
    base, head
        Inventories keyed on id.
    removed, new
        Unpaired ids; paired ids are discarded from both sets.

    Returns
    -------
    list[Rename]
        Fingerprint renames.
    """

    def index(rows: Rows, ids: set[str]) -> dict[tuple[str, str], list[str]]:
        groups: dict[tuple[str, str], list[str]] = {}
        for row_id in ids:
            fingerprint = rows[row_id].get("fingerprint")
            if fingerprint:
                groups.setdefault((rows[row_id]["kind"], fingerprint), []).append(row_id)
        return groups

    old_groups, new_groups = index(base, removed), index(head, new)
    found: list[Rename] = []
    for key in sorted(set(old_groups) & set(new_groups)):
        if len(old_groups[key]) == 1 and len(new_groups[key]) == 1:
            old_id, new_id = old_groups[key][0], new_groups[key][0]
            found.append(
                Rename(old_id, new_id, base[old_id]["name"], head[new_id]["name"], "fingerprint")
            )
            removed.discard(old_id)
            new.discard(new_id)
    return found


def diff_inventories(base: Rows, head: Rows, declared: dict[str, str] | None = None) -> SurfaceDiff:
    """Diff two inventories keyed on stable row ids.

    Parameters
    ----------
    base, head
        Inventories keyed on id (``load_inventory``).
    declared
        Optional rename map: old id to proposed leaf spelling or full new id. Declared
        pairs win over fingerprint pairs.

    Returns
    -------
    SurfaceDiff
        New, removed and renamed names.
    """

    removed = set(base) - set(head)
    new = set(head) - set(base)
    renamed = _declared_renames(base, head, removed, new, declared or {})
    renamed += _fingerprint_renames(base, head, removed, new)
    return SurfaceDiff(
        new=tuple(head[row_id] for row_id in sorted(new)),
        removed=tuple(base[row_id] for row_id in sorted(removed)),
        renamed=tuple(sorted(renamed, key=lambda rename: rename.old_id)),
    )


def proposals_to_map(rows: Rows) -> dict[str, str]:
    """Return the rename map carried by rows with a ``proposed`` spelling.

    Parameters
    ----------
    rows
        Name-table rows keyed on id.

    Returns
    -------
    dict[str, str]
        Old id to proposed leaf, for rows whose proposal differs from their name.
    """

    return {
        row_id: str(row["proposed"])
        for row_id, row in rows.items()
        if row.get("proposed") and row["proposed"] != row["name"]
    }
