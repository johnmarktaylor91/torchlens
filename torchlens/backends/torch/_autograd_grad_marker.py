"""Markers for TorchLens's in-forward ``autograd.grad`` and ``autograd.backward`` recorders.

A leaf module (no package imports) so validation modules can recognise the
recorder callable without importing the boundary module, whose helpers reach
the torch wrappers.
"""

from __future__ import annotations

from typing import Any

AUTOGRAD_GRAD_RECORDER_ATTR = "__tl_autograd_grad_recorder__"
"""Marker set on the TorchLens callable that records an in-forward autograd.grad."""

AUTOGRAD_BACKWARD_RECORDER_ATTR = "__tl_autograd_backward_recorder__"
"""Marker set on the TorchLens callable that records an in-forward autograd.backward."""


def is_autograd_grad_recorder(func: Any) -> bool:
    """Return whether ``func`` is TorchLens's in-forward autograd.grad recorder.

    Parameters
    ----------
    func:
        An op's recorded callable.

    Returns
    -------
    bool
        ``True`` only for the recorder callable built at capture time.
    """

    return getattr(func, AUTOGRAD_GRAD_RECORDER_ATTR, False) is True


def is_autograd_backward_recorder(func: Any) -> bool:
    """Return whether ``func`` is TorchLens's in-forward autograd.backward recorder.

    Parameters
    ----------
    func:
        An op's recorded callable.

    Returns
    -------
    bool
        ``True`` only for a recorder callable built at capture time.
    """

    return getattr(func, AUTOGRAD_BACKWARD_RECORDER_ATTR, False) is True
