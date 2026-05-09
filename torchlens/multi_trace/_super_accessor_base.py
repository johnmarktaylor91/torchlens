"""Generic base class for Bundle-side Super accessors."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Generic, TypeVar

from ._super_base import Super

T = TypeVar("T")
S = TypeVar("S", bound=Super[Any])


class SuperAccessor(Generic[T, S]):
    """Generic dict-like accessor for aligned Bundle member objects."""

    def __init__(self, bundle: Any, *, super_cls: type[S]) -> None:
        """Initialize the accessor.

        Parameters
        ----------
        bundle:
            Bundle instance that owns the member traces.
        super_cls:
            Super subclass constructed for successful lookups.
        """

        self._bundle = bundle
        self._super_cls = super_cls

    def __getitem__(self, label: str) -> S:
        """Return a cross-member Super view for ``label``.

        Parameters
        ----------
        label:
            Label to resolve in each bundle member.

        Returns
        -------
        S
            Cross-member Super view.

        Raises
        ------
        KeyError
            If the label resolves in no bundle members.
        """

        members: dict[str, T] = {}
        for name, trace in self._bundle.members.items():
            resolved = self._resolve_in_member(trace, label)
            if resolved is not None:
                members[name] = resolved
        if members:
            super_view = self._super_cls.from_members(label, members)
            super_view._bundle_member_names = list(self._bundle.members)
            return super_view

        suggestions = self._suggest(label)
        if suggestions:
            suggestion_str = ", ".join(repr(item) for item in suggestions)
            raise KeyError(f"Label {label!r} not found. Did you mean {suggestion_str}?")
        raise KeyError(f"Label {label!r} not found.")

    def __contains__(self, label: object) -> bool:
        """Return whether ``label`` resolves in at least one bundle member.

        Parameters
        ----------
        label:
            Candidate label.

        Returns
        -------
        bool
            Whether the label resolves in at least one member.
        """

        if not isinstance(label, str):
            return False
        return any(
            self._resolve_in_member(trace, label) is not None
            for trace in self._bundle.members.values()
        )

    def __iter__(self) -> Iterator[str]:
        """Iterate the sorted union of labels across all bundle members.

        Returns
        -------
        Iterator[str]
            Iterator over distinct labels in lexicographic order.
        """

        return iter(self.keys())

    def __len__(self) -> int:
        """Return the number of distinct labels across bundle members.

        Returns
        -------
        int
            Count of labels in the cross-member union.
        """

        return len(self.keys())

    def keys(self) -> list[str]:
        """Return distinct labels across bundle members in sorted order.

        Returns
        -------
        list[str]
            Sorted cross-member label union.
        """

        labels: set[str] = set()
        for trace in self._bundle.members.values():
            labels.update(self._labels_in_member(trace))
        return sorted(labels)

    def _resolve_in_member(self, trace: Any, label: str) -> T | None:
        """Resolve ``label`` within one member trace.

        Parameters
        ----------
        trace:
            Bundle member trace.
        label:
            Candidate label.

        Returns
        -------
        T | None
            Resolved object, or ``None`` when not found.
        """

        raise NotImplementedError

    def _suggest(self, label: str) -> list[str]:
        """Return lookup suggestions for ``label``.

        Parameters
        ----------
        label:
            Lookup label that did not resolve.

        Returns
        -------
        list[str]
            Candidate labels.
        """

        return []

    def _labels_in_member(self, trace: Any) -> list[str]:
        """Return candidate labels from one member trace.

        Parameters
        ----------
        trace:
            Bundle member trace.

        Returns
        -------
        list[str]
            Candidate labels from the member's layer accessor.
        """

        layer_accessor = getattr(trace, "layers", None)
        if layer_accessor is None:
            return []
        return [str(label) for label in layer_accessor.keys()]

    def __repr__(self) -> str:
        """Return a compact accessor summary.

        Returns
        -------
        str
            Representation.
        """

        keys = self.keys()
        preview = keys[:5]
        suffix = "..." if len(keys) > 5 else ""
        return f"{type(self).__name__} with {len(keys)} labels: {preview}{suffix}"
