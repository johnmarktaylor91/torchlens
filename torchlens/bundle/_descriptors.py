"""Bundle class descriptors: accessor and structural properties that do not count as methods."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from . import Bundle


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
