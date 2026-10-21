"""Non-uniform perturbation probe for shift-invariant children.

Every minimal and unit retry step moves ALL entries of a parent by the same
amount, which is a null probe for a shift-invariant child: ``softmax(x + c)
== softmax(x)``, and likewise ``log_softmax`` and the normalizations. For a
saturated (one-hot) softmax only the random draw could then move the argmax,
and an unlucky draw false-FAILed the real edge. ``lift_min_entry_values``
backs the ``lift_min_entry`` retry rung in ``core.py``; this module also owns
the value-step rung list, so the probe and its place in the order live together.
"""

import math
from collections.abc import Callable

import torch

LIFT_MIN_ENTRY_STRATEGY = "lift_min_entry"

#: The value-step retry rungs every child gets, in order (``core.py``'s
#: ``_value_retry_strategies`` appends the bool and discretizing ladders). The
#: four steps shift every entry equally, a null probe for a shift-invariant
#: child such as a saturated softmax; ``lift_min_entry`` is the non-uniform
#: rung that can move its argmax.
VALUE_STEP_STRATEGIES: tuple[str, ...] = (
    "step_up",
    "step_down",
    "unit_step_up",
    "unit_step_down",
    LIFT_MIN_ENTRY_STRATEGY,
)


def lift_min_entry_values(
    tensor: torch.Tensor,
    fallback: Callable[[torch.Tensor, str], torch.Tensor],
) -> torch.Tensor:
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
    fallback:
        ``core._directional_step_perturb``; called as ``fallback(tensor,
        "step_up")`` when the probe cannot apply.

    Returns
    -------
    torch.Tensor
        Tensor of the same shape and dtype differing in exactly one entry, or
        the minimal ``step_up`` when the probe cannot apply: a non-floating or
        empty tensor, no finite entry, a lifted value the dtype cannot
        represent, or one the dtype rounds back onto the saved entry.
    """

    if not tensor.is_floating_point() or tensor.numel() == 0:
        return fallback(tensor, "step_up")
    finite = torch.isfinite(tensor)
    if not bool(finite.any()):
        return fallback(tensor, "step_up")
    finite_or_high = torch.where(finite, tensor, torch.full_like(tensor, math.inf)).reshape(-1)
    flat_index = int(finite_or_high.argmin())
    low = float(finite_or_high[flat_index])
    high = float(torch.where(finite, tensor, torch.full_like(tensor, -math.inf)).max())
    lifted_value = high + max(high - low, 1.0)
    if not math.isfinite(lifted_value) or lifted_value > torch.finfo(tensor.dtype).max:
        return fallback(tensor, "step_up")
    coordinates = []
    for size in reversed(tensor.shape):
        coordinates.append(flat_index % size)
        flat_index //= size
    lifted = tensor.detach().clone()
    lifted[tuple(reversed(coordinates))] = lifted_value
    if torch.equal(lifted, tensor):
        return fallback(tensor, "step_up")
    return lifted
