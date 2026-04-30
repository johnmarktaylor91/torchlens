"""Saved-DAG replay engine ownership for future TorchLens interventions."""

from typing import Any

from .errors import _not_implemented


def replay(log: Any, *args: Any, **kwargs: Any) -> Any:
    """Run the future saved-DAG replay engine.

    Parameters
    ----------
    log:
        ModelLog-like object to mutate through replay.
    *args:
        Reserved positional arguments.
    **kwargs:
        Reserved keyword arguments.
    """

    return _not_implemented("replay", "Phase 6")


def replay_from(log: Any, site: Any, *args: Any, **kwargs: Any) -> Any:
    """Run the future saved-DAG replay engine from selected origins.

    Parameters
    ----------
    log:
        ModelLog-like object to mutate through replay.
    site:
        Future origin selector or site shorthand.
    *args:
        Reserved positional arguments.
    **kwargs:
        Reserved keyword arguments.
    """

    return _not_implemented("replay_from", "Phase 6")


__all__ = ["replay", "replay_from"]
