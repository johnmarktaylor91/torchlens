"""Module training flags saved and restored around shape-inference probes.

Extracted from ``_infer_input_shape`` (file-cap ratchet): the search, the probes
and the verification traces run every module in ``eval()`` and put each flag back
afterwards, so the caller's model leaves in the mode it came in.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from torch import nn


def _training_states(model: nn.Module) -> dict[nn.Module, bool]:
    """Capture training flags for all modules.

    Parameters
    ----------
    model:
        Model whose module states are captured.

    Returns
    -------
    dict[nn.Module, bool]
        Training flags keyed by module object.
    """

    return {module: module.training for module in model.modules()}


def _restore_training_states(states: dict[nn.Module, bool]) -> None:
    """Restore module training flags.

    Parameters
    ----------
    states:
        Training flags from :func:`_training_states`.
    """

    for module, training in states.items():
        module.train(training)


@contextmanager
def training_states_restored(model: nn.Module) -> Iterator[None]:
    """Restore every module's training flag of ``model`` when the block exits.

    Parameters
    ----------
    model:
        Model whose module states are captured on entry and restored on exit.
    """

    states = _training_states(model)
    try:
        yield
    finally:
        _restore_training_states(states)
