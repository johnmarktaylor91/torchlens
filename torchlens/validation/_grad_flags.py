"""Captured ``requires_grad`` flags of replay arguments.

Two torch-only helpers shared by the grad-fidelity replay and the
storage-context replay: give a private replay tensor the ``requires_grad`` its
captured argument had, and read a saved snapshot's captured flags. Kept apart
from ``_replay_grad_fidelity`` so modules that need only these two import no
``core`` dependency.
"""

from __future__ import annotations

from typing import Any

import torch


def _mirror_captured_requires_grad(
    value: torch.Tensor, requires_grad: bool, leaf: bool
) -> torch.Tensor:
    """Give a private replay tensor the ``requires_grad`` its captured argument had.

    ATen's backend choice can depend on whether an input requires grad (on
    macOS arm64 a depthwise 3x3 conv picks ``Slow2d`` for a grad-requiring
    weight and ``Winograd3x3Depthwise`` otherwise; the two accumulate in a
    different order). A replay that hands the op detached copies can therefore
    call a different kernel than the captured run did, so each replay tensor
    requires grad exactly when the captured argument did. This makes replay
    more faithful; it widens no comparison.

    A captured non-leaf (a weight derived from a parameter, any op output) is
    rebuilt as a non-leaf with the same values and strides, so an in-place op
    that was legal on the captured value stays legal on the replay value. The
    autograd graph this creates is owned by the replay arguments and output
    alone and is released when the check that built them returns.

    Parameters
    ----------
    value:
        Replay tensor owned by validation (a fresh clone, never a saved payload).
    requires_grad:
        Whether the captured argument at this slot required grad.
    leaf:
        Whether the captured argument was an autograd leaf.

    Returns
    -------
    torch.Tensor
        ``value`` unchanged when no grad is needed (or its dtype cannot require
        grad), else a grad-requiring leaf or non-leaf with equal values.
    """

    if not requires_grad or value.requires_grad:
        return value
    if not (value.is_floating_point() or value.is_complex()):
        return value
    with torch.enable_grad():
        grad_leaf = value.detach().requires_grad_(True)
        return grad_leaf if leaf else grad_leaf.clone()


def _snapshot_grad_flags(snapshot: Any) -> tuple[bool, bool]:
    """Return the captured ``(requires_grad, is_leaf)`` of one saved argument snapshot.

    Snapshots are copies taken with grad attached, so ``snapshot.requires_grad``
    is the captured argument's flag. Snapshots are copies of copies, so their
    autograd graph cannot tell a captured leaf from a captured user ``clone()``
    of one; a slot without a producing op is therefore rebuilt as a non-leaf,
    the direction in which every in-place op legal at capture stays legal.

    Parameters
    ----------
    snapshot:
        Saved argument value at one slot.

    Returns
    -------
    tuple[bool, bool]
        ``(False, False)`` for a non-tensor or a snapshot that needs no grad.
    """

    if not isinstance(snapshot, torch.Tensor) or not snapshot.requires_grad:
        return False, False
    return True, False
