"""Field-wise deep copy for ``tl.Edit`` (the ``Edit`` dataclass).

``copy.deepcopy`` of an edit copies every field but shares the hook factory by
identity (functions are deepcopy-atomic), instead of routing through the
factory-dropping pickle path of ``Edit.__getstate__``.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .types import Edit


def deepcopy_edit(edit: Edit, memo: dict[int, Any]) -> Edit:
    """Return a field-wise deep copy of one edit.

    Parameters
    ----------
    edit:
        Edit to copy.
    memo:
        The ``copy.deepcopy`` memo; the clone is registered before its fields
        are copied so self-references resolve to it.

    Returns
    -------
    Edit
        A new edit of the same class whose fields are deep copies, with the
        hook factory shared by identity.
    """

    cls = type(edit)
    clone = cls.__new__(cls)
    memo[id(edit)] = clone
    for key, value in edit.__dict__.items():
        object.__setattr__(clone, key, copy.deepcopy(value, memo))
    return clone
