"""Label-membership group tables with cached immutable views.

One group row per distinct membership group (equivalence class, recurrence
group). Every member op's cell holds THE one shared :class:`GroupRef` for its
group, so a read is O(1): the ref resolves to the group's cached immutable
view (``frozenset`` for unordered groups, ``tuple`` for ordered ones). This
replaces the historical copy-on-read barrier (a fresh ``set``/``list`` per
read, O(group) each): the view is immutable, so sharing one object across
every member and every reader is alias-safe by construction.

The views are LIVE through the one sanctioned mutation path — removal scrub
calls :meth:`MembershipGroups.replace` ONCE per group, so every member
reflects the new membership on its next read without per-op rebinds and
without breaking the sharing.
"""

from __future__ import annotations

from typing import Any


class MembershipGroups:
    """Group rows for one membership family.

    Parameters
    ----------
    view_type:
        Public immutable view type for this family — ``frozenset`` for
        unordered membership (equivalence), ``tuple`` for ordered membership
        (recurrence).
    """

    __slots__ = ("_source_tables", "_views", "view_type")

    def __init__(self, view_type: type) -> None:
        """Create an empty group table."""

        self.view_type = view_type
        self._views: list[Any] = []
        # Ancestor tables this table was cloned from (fork chains): pinned so
        # a GroupRef bound to ANY ancestor — the root refs live in shared
        # base storage — translates to this fork's clone by identity, and no
        # recycled id can mistranslate.
        self._source_tables: tuple[MembershipGroups, ...] = ()

    def __len__(self) -> int:
        """Return the number of group rows."""

        return len(self._views)

    def add(self, members: Any) -> int:
        """Append one group row and return its group id."""

        group_id = len(self._views)
        self._views.append(self.view_type(members))
        return group_id

    def view(self, group_id: int) -> Any:
        """Return the group's cached immutable membership view."""

        return self._views[group_id]

    def replace(self, group_id: int, members: Any) -> None:
        """Rebind one group's membership (the removal-scrub path).

        Every holder of the group's :class:`GroupRef` observes the new
        membership on its next read — the LIVE semantics of the views.
        """

        self._views[group_id] = self.view_type(members)


class GroupRef:
    """One shared cell value binding a member row to its group.

    Exactly ONE instance exists per group; every member cell holds it. That
    makes the ref itself a stable identity token for one group (the
    equivalence-symmetry invariant memoizes on it) and keeps per-member
    storage at one machine word.
    """

    __slots__ = ("groups", "group_id")

    def __init__(self, groups: MembershipGroups, group_id: int) -> None:
        """Bind the owning table and row."""

        self.groups = groups
        self.group_id = group_id

    def view(self) -> Any:
        """Return the group's current immutable membership view."""

        return self.groups.view(self.group_id)

    def __repr__(self) -> str:
        """Return a compact debugging form."""

        return f"GroupRef({self.groups.view_type.__name__}, {self.group_id})"
