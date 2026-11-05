"""Non-uniform perturbation probes for children that uniform steps cannot move.

Every minimal and unit retry step moves ALL entries of a parent by the same
amount, which is a null probe for a shift-invariant child: ``softmax(x + c)
== softmax(x)``, and likewise ``log_softmax`` and the normalizations. For a
saturated (one-hot) softmax only the random draw could then move the argmax,
and an unlucky draw false-FAILed the real edge. ``lift_min_entry_values``
backs the ``lift_min_entry`` retry rung in ``core.py``; this module also owns
the value-step rung list, so the probe and its place in the order live together.
The same holds for integer and bool parents (integers swap their smallest and
largest entries, which moves an ``argmax`` over token ids without leaving the
saved values, and bools flip one entry of a counted mask), for grouping children (the
``unique`` family, whose counts only a merge of two entries moves) and for
children that read a parent's shape only (a one-entry trim).
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

MERGE_ADJACENT_ENTRY_STRATEGY = "merge_adjacent_entry"
TRIM_LAST_ENTRY_STRATEGY = "trim_last_entry"

#: The non-uniform rungs ``core._directional_step_perturb`` hands to
#: :func:`non_uniform_probe_values`.
NON_UNIFORM_STRATEGIES = frozenset(
    {LIFT_MIN_ENTRY_STRATEGY, MERGE_ADJACENT_ENTRY_STRATEGY, TRIM_LAST_ENTRY_STRATEGY}
)

#: ``embedding_bag`` outputs after the pooled embeddings: ``offset2bag``,
#: ``bag_size`` and ``max_indices``. Outside ``mode="max"`` they are built
#: from the offsets and the NUMBER of indices, never the index values.
_EMBEDDING_BAG_AUX_OUTPUTS = frozenset({1, 2, 3})

#: Children that group equal entries: their counts and inverse outputs read how
#: entries partition into equal runs or classes, which every shift, the lift
#: and the integer swap preserve when the entries are distinct (``unique_consecutive([0, 1,
#: 2])`` keeps counts ``[1, 1, 1]``). Only a merge of two entries moves them.
_GROUPING_FUNC_NAMES = frozenset(
    {
        "unique",
        "_unique",
        "_unique2",
        "unique_dim",
        "unique_consecutive",
        "unique_dim_consecutive",
    }
)


def value_step_strategies(layer: object) -> list[str]:
    """Return the value-step retry rungs for ``layer``'s child op, in order.

    Every child gets :data:`VALUE_STEP_STRATEGIES`; a grouping child (the
    ``unique`` family) also gets the ``merge_adjacent_entry`` rung.
    """

    strategies = list(VALUE_STEP_STRATEGIES)
    if str(getattr(layer, "func_name", "") or "") in _GROUPING_FUNC_NAMES:
        strategies.append(MERGE_ADJACENT_ENTRY_STRATEGY)
    if reads_parent_shape_only(layer):
        strategies.append(TRIM_LAST_ENTRY_STRATEGY)
    return strategies


def reads_parent_shape_only(layer: object) -> bool:
    """Return whether ``layer``'s output is built from its parents' shapes, not values.

    Two outputs qualify: ``_shape_as_tensor`` (``torch.onnx.operators.
    shape_as_tensor``), whose output IS the parent's shape, and the
    ``embedding_bag`` outputs after the pooled embeddings, which count index
    entries per bag. No value perturbation can move them, so they get the
    ``trim_last_entry`` rung: a parent one entry shorter keeps the call valid
    and moves the output wherever the shape edge is real.
    """

    func_name = str(getattr(layer, "func_name", "") or "")
    if func_name == "_shape_as_tensor":
        return True
    return (
        func_name == "embedding_bag"
        and getattr(layer, "multi_output_index", None) in _EMBEDDING_BAG_AUX_OUTPUTS
    )


def non_uniform_probe_values(
    tensor: torch.Tensor,
    strategy: str,
    fallback: Callable[[torch.Tensor, str], torch.Tensor],
) -> torch.Tensor:
    """Return the values of one non-uniform rung (:data:`NON_UNIFORM_STRATEGIES`)."""

    if strategy == MERGE_ADJACENT_ENTRY_STRATEGY:
        return merge_adjacent_entry_values(tensor, fallback)
    if strategy == TRIM_LAST_ENTRY_STRATEGY:
        return trim_last_entry_values(tensor, fallback)
    return lift_min_entry_values(tensor, fallback)


def trim_last_entry_values(
    tensor: torch.Tensor,
    fallback: Callable[[torch.Tensor, str], torch.Tensor],
) -> torch.Tensor:
    """Return ``tensor`` without the last entry of its last dim (a shape probe).

    Only :func:`reads_parent_shape_only` children get this rung. A scalar or a
    tensor whose last dim is already empty has no entry to drop and takes the
    minimal ``step_up``, which cannot move a shape-only output.
    """

    if tensor.ndim == 0 or tensor.shape[-1] == 0:
        return fallback(tensor, "step_up")
    return tensor.detach().narrow(-1, 0, tensor.shape[-1] - 1).clone()


def merge_adjacent_entry_values(
    tensor: torch.Tensor,
    fallback: Callable[[torch.Tensor, str], torch.Tensor],
) -> torch.Tensor:
    """Return a tensor whose first entry that differs from its predecessor copies it.

    In flattened order, the first entry unequal to the one before it takes that
    entry's value: two runs (or two classes) merge by one entry, so the counts
    and inverse of a grouping child move wherever the edge is real. A tensor
    whose entries are all equal instead gets its last entry changed by the
    minimal ``step_up``, which splits the single run. Values stay inside the
    saved range, so the call stays valid wherever the saved one was.

    Parameters
    ----------
    tensor:
        Saved parent tensor values.
    fallback:
        ``core._directional_step_perturb``; supplies the split step and the
        minimal ``step_up`` for an empty or single-entry tensor.

    Returns
    -------
    torch.Tensor
        Tensor of the same shape and dtype differing in exactly one entry, or
        the minimal ``step_up`` for an empty or single-entry tensor.
    """

    if tensor.numel() < 2:
        return fallback(tensor, "step_up")
    flat = tensor.detach().reshape(-1)
    differs = torch.nonzero(flat[1:] != flat[:-1])
    if differs.numel():
        position = int(differs[0, 0]) + 1
        return _set_one_entry(tensor, position, flat[position - 1].item(), fallback)
    last = flat.numel() - 1
    split = fallback(flat[last:].clone(), "step_up")
    return _set_one_entry(tensor, last, split[0].item(), fallback)


def lift_min_entry_values(
    tensor: torch.Tensor,
    fallback: Callable[[torch.Tensor, str], torch.Tensor],
) -> torch.Tensor:
    """Return a tensor whose smallest entry is lifted above its largest.

    Integer tensors swap their smallest and largest entries instead, and bool
    tensors flip one entry (see the helpers below). Lifting the minimum finite entry to ``max + max(max - min, 1)`` makes it
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
        Tensor of the same shape and dtype differing in exactly one entry (two
        for the integer swap), or the minimal ``step_up`` when the probe cannot
        apply: a complex or empty tensor, a constant integer tensor, no finite
        entry, a lifted value the dtype cannot represent, or one the dtype
        rounds back onto the saved entry. ``core._directional_step_perturb``
        hands uint16/32/64 parents over as their signed views, so they take the
        integer swap on the signed bit patterns.
    """

    if tensor.numel() == 0:
        return fallback(tensor, "step_up")
    if tensor.dtype is torch.bool:
        return _flip_one_bool_entry(tensor, fallback)
    if _is_swapped_integer_dtype(tensor.dtype):
        return _swap_integer_extremes(tensor, fallback)
    if not tensor.is_floating_point():
        return fallback(tensor, "step_up")
    return _lift_float_min_entry(tensor, fallback)


def _lift_float_min_entry(
    tensor: torch.Tensor,
    fallback: Callable[[torch.Tensor, str], torch.Tensor],
) -> torch.Tensor:
    """Lift a float tensor's smallest finite entry above its largest (see ``lift_min_entry_values``)."""

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
    return _set_one_entry(tensor, flat_index, lifted_value, fallback)


#: Integer dtypes that take the min/max swap. The wide unsigned types arrive as
#: their signed views (``core._directional_step_perturb``), so they swap too.
_SWAPPED_INTEGER_DTYPES = frozenset(
    {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
)


def _is_swapped_integer_dtype(dtype: torch.dtype) -> bool:
    """Return whether the probe handles ``dtype`` as an integer (the min/max swap)."""

    return dtype in _SWAPPED_INTEGER_DTYPES


def _swap_integer_extremes(
    tensor: torch.Tensor,
    fallback: Callable[[torch.Tensor, str], torch.Tensor],
) -> torch.Tensor:
    """Return an integer tensor with its first smallest and first largest entries swapped.

    The integer form of the lift. The smallest entry takes the largest value
    and the largest takes the smallest, so an order-reading child (``argmax``,
    ``argsort``, ``sort`` indices, ``topk`` indices) sees its first maximum
    move while every entry stays inside the set of saved values. That keeps the
    probe inside any index domain the saved call already satisfied: an id
    parent of ``embedding_bag``, ``take`` or ``__getitem__`` never receives a
    value above the saved maximum, so no table bound is crossed (on CUDA an
    out-of-range id is a fatal device-side assert, not an absorbed raise).
    Every uniform step preserves the order of an integer parent (token ids
    feeding ``argmax`` for an end-of-sequence position), so without this rung
    only the random draw could move such a child. A constant tensor has nothing
    to swap and takes the minimal ``step_up``, a rung the ladder already ran.
    """

    flat = tensor.reshape(-1)
    low_index = int(flat.argmin())
    high_index = int(flat.argmax())
    low = flat[low_index].item()
    high = flat[high_index].item()
    if low == high:
        return fallback(tensor, "step_up")
    swapped = _set_one_entry(tensor, low_index, high, fallback)
    return _set_one_entry(swapped, high_index, low, fallback)


def _flip_one_bool_entry(
    tensor: torch.Tensor,
    fallback: Callable[[torch.Tensor, str], torch.Tensor],
) -> torch.Tensor:
    """Return a bool tensor with its first ``False`` entry set (or, if none, its first cleared).

    The bool form of the lift: it changes the true count by exactly one. Every
    other bool retry is ``logical_not``, which keeps the count of a balanced
    mask (``[False, True]``), so a counting child (``sum``, ``count_nonzero``)
    never moved under any retry.
    """

    flat = tensor.reshape(-1)
    false_positions = torch.nonzero(~flat)
    if false_positions.numel():
        return _set_one_entry(tensor, int(false_positions[0, 0]), True, fallback)
    return _set_one_entry(tensor, 0, False, fallback)


def _set_one_entry(
    tensor: torch.Tensor,
    flat_index: int,
    value: float | bool,
    fallback: Callable[[torch.Tensor, str], torch.Tensor],
) -> torch.Tensor:
    """Return a copy of ``tensor`` with the entry at ``flat_index`` set to ``value``."""

    coordinates = []
    for size in reversed(tensor.shape):
        coordinates.append(flat_index % size)
        flat_index //= size
    lifted = tensor.detach().clone()
    lifted[tuple(reversed(coordinates))] = value
    if torch.equal(lifted, tensor):
        return fallback(tensor, "step_up")
    return lifted
