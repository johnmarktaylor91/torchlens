"""Operator-name mutation signatures shared by capture and saved-output bookkeeping.

A leaf: it imports nothing from torchlens, so record modules can read these facts
without importing the torch backend.
"""

from __future__ import annotations

_SETTER_MUTATION_FUNC_NAMES = frozenset({"__setitem__", "__delitem__"})
"""Setter-style in-place ops whose names do NOT end in ``_`` (used only as the no-baseline
is_inplace fallback). They mutate their target but log a reconstructed output tensor, so the
version-baseline signal is unavailable and the operator name is the honest mutation signature.
"""

_INPLACE_AUGMENTED_ASSIGNMENT_DUNDER_EXCLUSIONS = frozenset(
    {"__index__", "__init__", "__init_subclass__", "__int__", "__invert__", "__iter__"}
)
"""Non-mutating ``__i*`` dunders excluded from augmented-assignment classification.

Unary dunders such as ``__invert__`` intentionally stay out of this set.
"""


def _is_inplace_augmented_assignment_dunder(name: str) -> bool:
    """Return whether ``name`` denotes an in-place augmented-assignment dunder.

    Parameters
    ----------
    name:
        Function name recorded by the torch eager wrapper.

    Returns
    -------
    bool
        True for ``__i*`` mutation dunders except known non-mutating dunders.
    """

    return name.startswith("__i") and name not in _INPLACE_AUGMENTED_ASSIGNMENT_DUNDER_EXCLUSIONS
