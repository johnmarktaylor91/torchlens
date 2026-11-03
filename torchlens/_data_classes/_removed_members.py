"""Typed refusals for removed public members of the record classes.

``Op`` and ``AtenOp`` inherit :class:`RemovedMemberMixin`. Its ``__getattr__``
runs only after normal lookup has failed, so it costs nothing on the read
path; a removed public spelling raises the typed ``facade_redirect`` error
naming what replaced it, and every other miss raises the plain
``AttributeError`` it always did.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .._utils.facade import refuse_removed_member

_GPU_KERNELS_REMEDY = (
    "GPU kernel recording is not public in this release; a capture switch for it "
    "is planned for a later release. Saved artifacts keep their kernel annotation."
)

#: Removed public member spellings per record class name -> remedy text.
REMOVED_MEMBERS: Mapping[str, Mapping[str, str]] = {
    "Op": {"gpu_kernels": _GPU_KERNELS_REMEDY},
    "AtenOp": {"gpu_kernels": _GPU_KERNELS_REMEDY},
}


class RemovedMemberMixin:
    """Refuse removed public members by name, typed, on a lookup miss."""

    __slots__ = ()

    def __getattr__(self, name: str) -> Any:
        """Raise the typed redirect for a removed member, else a plain miss.

        Parameters
        ----------
        name:
            Attribute that normal lookup did not find.

        Raises
        ------
        AttributeError
            Always; ``FacadeTeachingError`` (an ``AttributeError``) for a
            removed public spelling.
        """

        owner = type(self).__name__
        if not name.startswith("_"):
            for klass in type(self).__mro__:
                removed = REMOVED_MEMBERS.get(klass.__name__)
                if removed is not None:
                    refuse_removed_member(owner, name, removed)
        raise AttributeError(f"{owner!r} object has no attribute {name!r}", name=name, obj=self)


__all__ = ["REMOVED_MEMBERS", "RemovedMemberMixin"]
