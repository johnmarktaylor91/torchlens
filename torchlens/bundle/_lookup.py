"""Bundle site lookup: ``at``, the supergraph, structural consistency, accessor descriptors."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ..intervention._topology.topology import build_supergraph

if TYPE_CHECKING:
    from ..intervention._topology.topology import Supergraph
    from . import Bundle

__all__ = [
    "bundle_at",
    "ensure_supergraph",
    "is_structurally_consistent",
]


def bundle_at(self: Bundle, site: Any) -> Any:
    """Return the cross-member Super view for ``site`` (gate attached)."""

    view = self._label_view(site) if isinstance(site, str) else self._site_view(site)
    names = list(getattr(view, "members", {}))
    pairs = [(left, right) for index, left in enumerate(names) for right in names[index + 1 :]]
    view._compare_gate = functools.partial(self._require_comparable, "at.compare", pairs)
    return view


def ensure_supergraph(self: Bundle) -> Supergraph:
    """Build and cache the supergraph lazily."""

    if self._supergraph is None:
        self._supergraph = build_supergraph(list(self._members.values()), list(self._members))
    return self._supergraph


def is_structurally_consistent(self: Bundle) -> bool:
    """Whether every member has an equal ``graph_shape_hash`` value."""

    hashes = {member._provenance_fingerprint for member in self._members.values()}
    return len(hashes) == 1


class _BundleAccessorClassView:
    """Class-level placeholder for Bundle accessor properties."""


class _BundleAccessorProperty:
    """Property that exposes Bundle accessors without inflating method counts."""

    def __init__(self, accessor_cls: type[Any]) -> None:
        """Initialize the property.

        Parameters
        ----------
        accessor_cls:
            Accessor class to instantiate for Bundle instances.
        """

        self._accessor_cls = accessor_cls
        self._class_view = _BundleAccessorClassView()

    def __get__(
        self,
        instance: Bundle | None,
        owner: type[Bundle] | None = None,
    ) -> Any:
        """Return a class view or an accessor on instances.

        Parameters
        ----------
        instance:
            Bundle instance, or ``None`` for class access.
        owner:
            Owning Bundle class.

        Returns
        -------
        Any
            Class view for class access, otherwise the instantiated accessor.
        """

        if instance is None:
            return self._class_view
        return self._accessor_cls(instance)


class _BundleStructuralProperty:
    """Descriptor exposing Bundle predicates without counting as a method."""

    def __init__(self, getter: Callable[[Bundle], Any]) -> None:
        """Initialize the descriptor.

        Parameters
        ----------
        getter:
            Instance getter to call for Bundle instances.
        """

        self._getter = getter
        self.__doc__ = getter.__doc__

    def __get__(self, instance: Bundle | None, owner: type[Bundle]) -> Any:
        """Return the descriptor on classes or computed value on instances.

        Parameters
        ----------
        instance:
            Bundle instance, or ``None`` for class access.
        owner:
            Owning Bundle class.

        Returns
        -------
        Any
            Descriptor for class access, otherwise the computed property value.
        """

        if instance is None:
            return self
        return self._getter(instance)
