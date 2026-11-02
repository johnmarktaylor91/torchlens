"""The private scope that lets a capture build a gradient path through frozen parameters.

A plain capture never changes a Parameter's ``requires_grad``. An analysis that
backpropagates through the graph of its own capture (``tl.mechinterp`` edge
scores) runs that capture inside :func:`frozen_parameter_gradients`; model
preparation and late registration consult :func:`_forces_frozen_parameter_gradients`
to set the flag on frozen floating leaves for the session and restore it at
cleanup. Moved out of ``model_prep`` (file-size ceiling).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from ..._data_classes.trace import Trace

_FROZEN_PARAMETER_GRADIENTS: ContextVar[bool] = ContextVar(
    "torchlens_frozen_parameter_gradients", default=False
)


@contextmanager
def frozen_parameter_gradients() -> Iterator[None]:
    """Let captures in this scope build a gradient path through frozen parameters.

    A plain capture never changes a parameter's ``requires_grad``, so a frozen
    model fed a non-differentiable input (token ids) records no autograd graph,
    exactly as eager. An analysis that backpropagates through the graph of its
    OWN capture (``tl.mechinterp`` edge scores) needs that graph, so inside this
    scope capture sets ``requires_grad=True`` on frozen floating leaf parameters
    for the session (``Param.is_trainable`` still reports the user's flag) and
    restores the flags at session cleanup. torch may then pick different kernels
    for that forward, as it would for an eager forward with trainable weights.

    Yields
    ------
    None
        Control for the analysis's capture call.
    """

    token = _FROZEN_PARAMETER_GRADIENTS.set(True)
    try:
        yield
    finally:
        _FROZEN_PARAMETER_GRADIENTS.reset(token)


def _forces_frozen_parameter_gradients(trace: Trace, param: nn.Parameter) -> bool:
    """Return whether this capture sets ``requires_grad`` on frozen ``param``.

    True only inside ``frozen_parameter_gradients()``, outside ``backward_ready``
    (which keeps the user's flags by contract), for a frozen floating or complex
    leaf (integer and bool Parameters are never gradient-capable; a non-leaf's
    flag is read-only).
    """

    return (
        _FROZEN_PARAMETER_GRADIENTS.get()
        and not param.requires_grad
        and param.is_leaf
        and not trace._options_backward_ready
        and (torch.is_floating_point(param) or torch.is_complex(param))
    )
