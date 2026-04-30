"""Full model rerun engine ownership for future TorchLens interventions."""

from typing import Any

from .errors import _not_implemented


def rerun(log: Any, model: Any, *args: Any, **kwargs: Any) -> Any:
    """Run the future full-forward rerun engine.

    Parameters
    ----------
    log:
        ModelLog-like object to mutate through rerun.
    model:
        Model object for future rerun execution.
    *args:
        Reserved positional arguments.
    **kwargs:
        Reserved keyword arguments.

    Returns
    -------
    Any
        Reserved rerun result.

    Raises
    ------
    NotImplementedError
        Always raised until Phase 7 implements rerun.
    """

    return _not_implemented("rerun", "Phase 7")


__all__ = ["rerun"]
