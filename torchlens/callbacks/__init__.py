"""Callback integration namespace with lazy Lightning support.

Attribute access resolves through the shared five-step facade order
(``torchlens.utils.facade``); the ``lightning`` integration module itself is
import-inert without the foreign peer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING as _TYPE_CHECKING

if _TYPE_CHECKING:
    from types import ModuleType

_CALLBACK_MODULES = {"lightning"}


def __getattr__(name: str) -> ModuleType:
    """Import callback integrations lazily through the five-step facade order.

    Parameters
    ----------
    name:
        Callback integration module name.

    Returns
    -------
    ModuleType
        Imported callback module.

    Raises
    ------
    AttributeError
        Per the five-step contract; unknown names raise plain
        ``AttributeError``.
    """

    from ..utils.facade import resolve_facade_attr

    resolved: ModuleType = resolve_facade_attr(
        owner=__name__,
        name=name,
        module_globals=globals(),
        submodules=_CALLBACK_MODULES,
    )
    return resolved


def __dir__() -> list[str]:
    """Return visible callback namespace members.

    Returns
    -------
    list[str]
        Sorted callback module names plus real public globals.
    """

    from ..utils.facade import facade_dir

    return facade_dir(globals(), _CALLBACK_MODULES)


__all__ = ["lightning"]

# ``from __future__ import annotations`` binds ``annotations`` as a reachable
# module attribute; nothing reads the binding (the future feature is a
# compile-time flag), so unbind it -- the root facade's own idiom.
del annotations
