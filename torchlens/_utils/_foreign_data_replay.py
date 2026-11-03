"""Replay of a resolved foreign data round trip (the ``foreigndata`` op).

A zero-copy (``view``) or copying (``copy``) round trip through numpy, DLPack or
``tolist`` in plain forward code is a pure function of the departed tensor and a few
literals. Capture logs those literals as the op's arguments, so validation replays the
op through the generic path and a portable runnable artifact resolves this function by
its import path like the other TorchLens synthetic callables
(``torchlens._io._runnable_synthetic``).

Artifacts are untrusted input. Every mode reads a fresh copy of the departed tensor's
elements whose storage holds exactly those elements, so torch's own checks bound it:
``as_strided`` refuses negative sizes, strides and offsets and any view past the
storage, indexing refuses an out-of-range index, ``Tensor.to`` refuses a non-dtype, and
an unknown mode or a wrong literal count fails the lookup or the unpacking.

Modes (the second argument):

* ``"strided"`` ``(size, stride, offset)``: the output reads the departed tensor's
  elements, in its logical (row-major) order, through a strided view;
* ``"gather"`` ``(size, index)``: the output element ``i`` is the departed tensor's
  logical element ``index[i]`` (a view no stride pattern describes);
* ``"copy"`` ``(dtype, device)``: the output holds the departed tensor's values in a
  new tensor of that dtype and device.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


def _strided(elements: torch.Tensor, size: Any, stride: Any, offset: Any) -> torch.Tensor:
    """Return the strided view ``(size, stride, offset)`` over the logical elements."""

    return elements.as_strided(torch.Size(size), tuple(stride), offset)


def _gather(elements: torch.Tensor, size: Any, index: Any) -> torch.Tensor:
    """Return the logical elements at ``index``, shaped ``size``."""

    positions = torch.as_tensor(index, dtype=torch.int64, device=elements.device)
    return elements[positions].reshape(torch.Size(size))


def _copy(elements: torch.Tensor, dtype: Any, device: Any) -> torch.Tensor:
    """Return the logical elements cast to ``dtype`` on ``device`` (shape restored by caller)."""

    return elements.to(device=torch.device(device), dtype=dtype)


_REPLAYS: dict[str, Callable[..., torch.Tensor]] = {
    "strided": _strided,
    "gather": _gather,
    "copy": _copy,
}
"""Replay mode to its implementation over the departed tensor's logical elements."""

FOREIGN_DATA_REPLAY_MODES = tuple(_REPLAYS)
"""Accepted replay modes, in the order the module docstring describes them."""


def foreigndata(source: torch.Tensor, mode: str, *literals: Any) -> torch.Tensor:
    """Re-run one resolved foreign data round trip from the departed tensor.

    Parameters
    ----------
    source:
        The departed tensor (the op's only tensor argument).
    mode:
        One of :data:`FOREIGN_DATA_REPLAY_MODES`.
    *literals:
        The mode's recorded literals (see the module docstring).

    Returns
    -------
    torch.Tensor
        The tensor the host constructor produced, recomputed from ``source``.
    """

    replay = _REPLAYS[mode]
    elements = source.detach().reshape(-1).clone()
    out = replay(elements, *literals)
    return out.reshape(source.shape) if mode == "copy" else out
