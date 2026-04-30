"""Runtime context ownership for future TorchLens intervention execution."""

from typing import Any

from .errors import _not_implemented


def do(log: Any, *args: Any, **kwargs: Any) -> Any:
    """Apply a future one-shot intervention operation to a model log.

    Parameters
    ----------
    log:
        ModelLog-like object that will eventually receive the operation.
    *args:
        Reserved positional arguments for future site and hook/value inputs.
    **kwargs:
        Reserved keyword arguments for future engine dispatch.

    Returns
    -------
    Any
        Reserved operation result.

    Raises
    ------
    NotImplementedError
        Always raised until Phase 8b implements the dispatch operation.
    """

    return _not_implemented("do", "Phase 8b")


__all__ = ["do"]
