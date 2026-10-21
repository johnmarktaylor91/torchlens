"""Non-uniform perturbation probe for shift-invariant children.

Every minimal and unit retry step moves ALL entries of a parent by the same
amount, which is a null probe for a shift-invariant child: ``softmax(x + c)
== softmax(x)``, and likewise ``log_softmax`` and the normalizations. For a
saturated (one-hot) softmax only the random draw could then move the argmax,
and an unlucky draw false-FAILed the real edge. ``lift_min_entry_values``
backs the ``lift_min_entry`` retry rung in ``core.py``.
"""

import math

import torch


def lift_min_entry_values(tensor: torch.Tensor) -> torch.Tensor | None:
    """Return a floating tensor whose smallest entry is lifted above its largest.

    Lifting the minimum finite entry to ``max + max(max - min, 1)`` makes it
    the strict maximum of every slice that contains it, along any dim, so a
    softmax over a dim of size >= 2 changes its output there. The lifted
    entry exceeds the parent's maximum by its own value range (or the 1.0 unit
    step), the scale of the default draw and the unit steps, so it never
    "confirms" an fp-swamped edge with an unrealistically large magnitude.

    Parameters
    ----------
    tensor:
        Saved parent tensor values.

    Returns
    -------
    torch.Tensor or None
        Tensor of the same shape and dtype differing in exactly one entry, or
        ``None`` when the probe cannot apply: a non-floating or empty tensor,
        no finite entry, a lifted value the dtype cannot represent, or one the
        dtype rounds back onto the saved entry.
    """

    if not tensor.is_floating_point() or tensor.numel() == 0:
        return None
    finite = torch.isfinite(tensor)
    if not bool(finite.any()):
        return None
    finite_or_high = torch.where(finite, tensor, torch.full_like(tensor, math.inf)).reshape(-1)
    flat_index = int(finite_or_high.argmin())
    low = float(finite_or_high[flat_index])
    high = float(torch.where(finite, tensor, torch.full_like(tensor, -math.inf)).max())
    lifted_value = high + max(high - low, 1.0)
    if not math.isfinite(lifted_value) or lifted_value > torch.finfo(tensor.dtype).max:
        return None
    coordinates = []
    for size in reversed(tensor.shape):
        coordinates.append(flat_index % size)
        flat_index //= size
    lifted = tensor.detach().clone()
    lifted[tuple(reversed(coordinates))] = lifted_value
    if torch.equal(lifted, tensor):
        return None
    return lifted
