"""Unwrap a functional-collective async tensor for TorchLens tensor metadata.

Kept apart from ``_tensor_label`` so the label hot path resolves the wrapper class through the
compat chokepoint without ``_tensor_label`` importing it.
"""

from __future__ import annotations

from typing import Any

from ._torch_compat import get_async_collective_tensor_type


def _async_collective_elem(t: Any) -> Any | None:
    """Return an AsyncCollectiveTensor's inner ``.elem``, else ``None``.

    Parameters
    ----------
    t : Any
        Candidate tensor-like object.

    Returns
    -------
    Any | None
        The plain inner tensor when ``t`` is a funcol ACT wrapper; ``None``
        for every other value or on builds without functional collectives.

    Notes
    -----
    The ACT class resolves through the compat chokepoint
    (``get_async_collective_tensor_type``), which stays ``sys.modules``-
    deferred: this runs on the per-tensor label hot path and never pays the
    ``torch.distributed`` import on plain captures.
    """

    act_cls = get_async_collective_tensor_type()
    if act_cls is not None and isinstance(t, act_cls):
        return t.elem
    return None
