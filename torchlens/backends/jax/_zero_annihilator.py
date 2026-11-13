"""Zero-annihilator value proof for the JAX backend's perturbation oracle.

Torch parity (``validation/_exemptions.py``: ``_multiplicative_zero_annihilator_decision``
and ``_matmul_zero_annihilator_decision``, ledger code ``multiplicative_zero_annihilator``):
when no perturbation candidate moved a ``mul`` or ``dot_general`` output, the edge
is credited only if its insensitivity is exact arithmetic, ``x * 0 == 0``. Every
condition reads saved values with no tolerance:

- the primitive is ``mul`` or ``dot_general`` with exactly two operands;
- every data parent fills exactly one operand slot (``x * x`` is never credited);
- the other operand's saved value is exactly zero at every element (non-empty);
- the saved output is exactly zero at every element (non-empty), so a non-finite
  parent (``inf * 0`` is NaN) or a corrupted payload is never credited.

The proof runs only after the replay matched and every candidate failed; it never
replaces the replay. Anything outside it still fails ``perturbation_insensitive``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

#: The torch reason code this proof shares (an existing code, not a new one).
ZERO_ANNIHILATOR_REASON = "multiplicative_zero_annihilator"

#: Bilinear primitives whose zero operand annihilates the other operand.
_ANNIHILATOR_PRIMITIVES = frozenset({"mul", "dot_general"})


def _is_exact_zero_array(value: Any) -> bool:
    """Return whether ``value`` is a non-empty numeric array that is zero at every element."""

    import jax.numpy as jnp

    try:
        array = jnp.asarray(value)
    except (TypeError, ValueError):
        return False
    if str(array.dtype).startswith("key<") or array.size == 0:
        return False
    try:
        return bool(jnp.all(array == 0))
    except TypeError:  # payloads with no value equality
        return False


def _zero_annihilator_justification(
    primitive: str,
    positions_by_parent: Mapping[str, Sequence[int]],
    inputs: Sequence[Any],
    saved_outputs: Sequence[Any],
) -> str | None:
    """Return the proof's justification when every data parent is annihilated, else None.

    Parameters
    ----------
    primitive
        The equation's JAX primitive name.
    positions_by_parent
        Operand slots each data parent fills, keyed by parent label.
    inputs
        Replay inputs derived from the saved trace payloads.
    saved_outputs
        Saved child output payloads.

    Returns
    -------
    str | None
        A justification naming each annihilated slot, or None when any
        condition of the module docstring fails.
    """

    if primitive not in _ANNIHILATOR_PRIMITIVES or len(inputs) != 2:
        return None
    if len(saved_outputs) != 1 or not _is_exact_zero_array(saved_outputs[0]):
        return None
    if not positions_by_parent:
        return None
    notes: list[str] = []
    for parent_label, positions in sorted(positions_by_parent.items()):
        if len(positions) != 1 or positions[0] not in (0, 1):
            return None
        other = 1 - positions[0]
        if not _is_exact_zero_array(inputs[other]):
            return None
        notes.append(f"{parent_label} at invars[{positions[0]}]: invars[{other}] is all zero")
    return f"{primitive} operand annihilated by a saved all-zero co-operand: " + "; ".join(notes)
