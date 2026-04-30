"""Future single Bundle ownership for TorchLens intervention comparisons."""

from typing import Any

from .errors import _not_implemented


class Bundle:
    """Placeholder for the future unified intervention Bundle type."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize a future Bundle.

        Parameters
        ----------
        *args:
            Reserved positional construction arguments.
        **kwargs:
            Reserved keyword construction arguments.

        Raises
        ------
        NotImplementedError
            Always raised until Phase 9 replaces TraceBundle behavior.
        """

        _not_implemented("Bundle", "Phase 9")


__all__ = ["Bundle"]
