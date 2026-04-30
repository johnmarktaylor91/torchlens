"""Hook parsing and hook-spec ownership for future TorchLens interventions."""

from collections.abc import Callable
from typing import Any

from .errors import _not_implemented


def normalize_hook(fn: Callable[..., Any]) -> Any:
    """Normalize a future user hook callable into a hook specification.

    Parameters
    ----------
    fn:
        User callable to normalize.

    Returns
    -------
    Any
        Reserved hook specification.

    Raises
    ------
    NotImplementedError
        Always raised until Phase 3 implements hook specifications.
    """

    return _not_implemented("normalize_hook", "Phase 3")


__all__ = ["normalize_hook"]
