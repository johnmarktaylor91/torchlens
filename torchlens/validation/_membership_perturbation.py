"""Membership-toggling parent perturbation for ``torch.isin``.

``isin(elements, test_elements)`` is a set-membership test: its output moves
only when an element starts or stops matching one of the test values. The
generic integer perturbation draws from the saved range (or from ``0..10``
for a one-valued tensor) and the bool-predicate battery substitutes 0, 1, -1
and the dtype extremes; for a token id tested against an end-of-sequence id
(``isin(input_ids[:, -1], eos_ids)`` in a generate loop) none of those values
is a test value, so a real edge looked insensitive. The values built here flip
the membership of the perturbed operand by construction and are checked to do
so; the perturbation check itself is unchanged and still requires the replayed
output to differ from the saved one, so a recorded parent that does not feed
the op still fails.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from .._data_classes.op import Op

_SLOT_NAMES = ("elements", "test_elements")


def isin_membership_toggle_values(
    layer: Op, parent_label: str, parent_values: torch.Tensor
) -> torch.Tensor | None:
    """Return a perturbation of an ``isin`` operand that flips membership.

    Parameters
    ----------
    layer:
        The ``isin`` op being replayed.
    parent_label:
        Label of the parent selected for perturbation.
    parent_values:
        The saved value of that parent.

    Returns
    -------
    torch.Tensor or None
        For the ``elements`` operand, a tensor in which every element's
        membership in ``test_elements`` is flipped; for the ``test_elements``
        operand, a constant tensor that changes the membership of at least one
        element. ``None`` when the slot is unknown, an operand is empty or not
        numeric, ``assume_unique=True`` (a constant operand would break that
        promise), or no flipping value is representable; the caller then uses
        the generic perturbation.
    """

    slot = _isin_slot_of(layer, parent_label)
    kwargs = dict(getattr(layer, "saved_kwargs", None) or {})
    if slot is None or kwargs.get("assume_unique"):
        return None
    other = _isin_operand(layer, kwargs, 1 - slot)
    if other is None or parent_values.numel() == 0 or other.numel() == 0:
        return None
    if parent_values.is_complex() or other.is_complex():
        return None
    with torch.no_grad():
        if slot == 0:
            return _toggled_elements(parent_values, other)
        return _toggled_test_elements(parent_values, other)


def _isin_slot_of(layer: Op, parent_label: str) -> int | None:
    """Return 0 for the ``elements`` operand, 1 for ``test_elements``, else None."""

    positions = getattr(layer, "parent_arg_positions", None) or {}
    for position in (0, 1):
        if positions.get("args", {}).get(position) == parent_label:
            return position
        if positions.get("kwargs", {}).get(_SLOT_NAMES[position]) == parent_label:
            return position
    return None


def _isin_operand(layer: Op, kwargs: dict[str, Any], slot: int) -> torch.Tensor | None:
    """Return the saved operand in ``slot`` as a tensor (a Python scalar is wrapped)."""

    args = tuple(getattr(layer, "saved_args", None) or ())
    value = args[slot] if slot < len(args) else kwargs.get(_SLOT_NAMES[slot])
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, (bool, int, float)):
        return torch.tensor(value)
    return None


def _toggled_elements(elements: torch.Tensor, test: torch.Tensor) -> torch.Tensor | None:
    """Send members to a non-member value and non-members to a test value."""

    member = _value_in(test, elements.dtype, elements.device, want_member=True, pool=test)
    outsider = _value_in(test, elements.dtype, elements.device, want_member=False, pool=test)
    if member is None or outsider is None:
        return None
    was_member = torch.isin(elements, test.to(elements.device))
    toggled = torch.where(was_member, outsider, member)
    if not bool((torch.isin(toggled, test.to(elements.device)) != was_member).all()):
        return None
    return toggled


def _toggled_test_elements(test: torch.Tensor, elements: torch.Tensor) -> torch.Tensor | None:
    """Return a constant test set that changes at least one element's membership."""

    elements = elements.to(test.device)
    was_member = torch.isin(elements, test)
    value = _value_in(
        elements, test.dtype, test.device, want_member=not bool(was_member.any()), pool=elements
    )
    if value is None:
        return None
    toggled = value.expand(test.shape).clone()
    if bool(torch.equal(torch.isin(elements, toggled), was_member)):
        return None
    return toggled


def _value_in(
    reference: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
    *,
    want_member: bool,
    pool: torch.Tensor,
) -> torch.Tensor | None:
    """Return a 0-d ``dtype`` value that is (or is not) a member of ``reference``.

    Candidates for a member are the values of ``pool`` itself; for a
    non-member, one past the extremes of ``reference`` and the small constants.
    Every candidate is cast to ``dtype`` and its membership re-checked, so a
    cast that changes the value is never trusted.
    """

    reference = reference.to(device)
    if want_member:
        candidates = list(pool.reshape(-1)[:8])
    else:
        lo, hi = reference.min(), reference.max()
        candidates = [hi + 1, lo - 1, torch.tensor(0), torch.tensor(1)]
    for candidate in candidates:
        try:
            value = candidate.to(device=device, dtype=dtype)
        except (RuntimeError, OverflowError):
            continue
        if bool(torch.isin(value, reference)) is want_member:
            return value
    return None
