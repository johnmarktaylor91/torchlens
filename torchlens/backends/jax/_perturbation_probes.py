"""Perturbation candidates for the JAX backend's parent-influence oracle.

The uniform steps every parent gets, then the probes that run only while an edge
still looks dead, each mirroring a torch validator rung: the non-uniform
single-entry lift, the binary-extrema crossing value, and the bool-child battery
and comparand straddle. Every candidate is evidence only: a child output that no
candidate moves still fails ``perturbation_insensitive``.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .jaxpr import JaxEquationCapture

#: Binary extrema primitives whose dominated operand admits the crossing probe.
_BINARY_EXTREMA_PRIMITIVES = {"max": True, "min": False}


def _candidate_stream(
    capture: JaxEquationCapture,
    inputs: Sequence[Any],
    positions: Sequence[int],
    *,
    bool_child: bool,
) -> Iterator[Any]:
    """Yield perturbation candidates for one parent, cheapest and most uniform first.

    The uniform steps of :func:`_perturb_candidates` come first. The probes after
    them mirror the torch validator's later rungs and run only while the edge
    still looks dead: the non-uniform single-entry lift (``validation/_shift_probe.py``,
    which moves an ``argmax`` or a shift-invariant child), the binary-extrema
    crossing value (``binary_extrema_crossing_probe``), and the bool-child
    substitutes. Every probe is evidence only: a child no candidate moves fails.
    """

    import jax.numpy as jnp

    parent = inputs[positions[0]]
    yield from _perturb_candidates(parent)
    lifted = _lift_min_entry(jnp.asarray(parent))
    if lifted is not None:
        yield lifted
    upward = _BINARY_EXTREMA_PRIMITIVES.get(str(capture.primitive))
    if upward is not None and len(inputs) == 2 and len(positions) == 1:
        beyond = _beyond_extreme(jnp.asarray(parent), jnp.asarray(inputs[1 - positions[0]]), upward)
        if beyond is not None:
            yield beyond
    if bool_child:
        yield from _bool_child_substitutes(capture, inputs, positions)


def _lift_min_entry(parent: Any) -> Any | None:
    """Return ``parent`` with its smallest finite entry lifted above its largest, else None.

    The lifted value is ``max + max(max - min, 1)``: the strict maximum of every
    slice that holds it, at the scale of the parent's own range (torch parity:
    ``lift_min_entry_values``). None for a non-float, empty or all non-finite
    parent, or when the lifted value is not representable.
    """

    import jax.numpy as jnp

    if not jnp.issubdtype(parent.dtype, jnp.floating) or parent.size == 0:
        return None
    finite = jnp.isfinite(parent)
    if not bool(jnp.any(finite)):
        return None
    flat = jnp.reshape(jnp.where(finite, parent, jnp.inf), (-1,))
    index = int(jnp.argmin(flat))
    low, high = float(flat[index]), float(jnp.max(jnp.where(finite, parent, -jnp.inf)))
    lifted_value = high + max(high - low, 1.0)
    if not math.isfinite(lifted_value) or lifted_value > float(jnp.finfo(parent.dtype).max):
        return None
    lifted = jnp.reshape(jnp.reshape(parent, (-1,)).at[index].set(lifted_value), parent.shape)
    return None if bool(jnp.array_equal(lifted, parent)) else lifted


def _beyond_extreme(parent: Any, other: Any, upward: bool) -> Any | None:
    """Return ``parent`` filled with one value past every element of ``other``, else None.

    Torch parity (``_beyond_extreme`` in ``validation/_influence_probes.py``): a
    full magnitude past a float extreme, one step past an integer one; None when
    no crossing value is representable.
    """

    import jax.numpy as jnp

    numeric = jnp.issubdtype(parent.dtype, jnp.floating) or jnp.issubdtype(
        parent.dtype, jnp.integer
    )
    if not numeric or other.size == 0 or jnp.issubdtype(other.dtype, jnp.complexfloating):
        return None
    extreme = (jnp.max(other) if upward else jnp.min(other)).astype(parent.dtype)
    if jnp.issubdtype(parent.dtype, jnp.floating):
        step = jnp.maximum(jnp.abs(extreme), jnp.asarray(1.0, dtype=parent.dtype))
        value = extreme + step if upward else extreme - step
        return jnp.full_like(parent, value) if bool(jnp.isfinite(value)) else None
    info = jnp.iinfo(parent.dtype)
    if int(extreme) == (info.max if upward else info.min):
        return None
    return jnp.full_like(parent, int(extreme) + (1 if upward else -1))


#: Elementwise comparison primitives whose bool output admits the comparand
#: straddle (torch parity: ``validation/_influence_probes.py``).
_COMPARISON_PRIMITIVES = frozenset({"eq", "ne", "lt", "le", "gt", "ge"})


def _all_bool_outputs(saved_outputs: tuple[Any, ...]) -> bool:
    """Return whether every saved output of an equation is a bool array."""

    import jax.numpy as jnp

    return bool(saved_outputs) and all(
        getattr(output, "dtype", None) == jnp.bool_ for output in saved_outputs
    )


def _bool_child_substitutes(
    capture: JaxEquationCapture, inputs: Sequence[Any], positions: Sequence[int]
) -> tuple[Any, ...]:
    """Return threshold-crossing substitutes for one parent of a bool-output equation.

    A bool child is a threshold op, so the additive steps of
    :func:`_perturb_candidates` routinely fail to cross it (``mask != 0`` on an
    all-nonzero mask, the ``x != x`` NaN test). As the torch validator does
    for bool children, the parent is also replaced by a dtype battery (zero,
    one, sign, non-finite and extreme values) and, for a two-operand
    comparison where the parent fills one operand, by the comparand and its
    two adjacent representable values. A substitute that moves the output
    proves the edge transmits value; when every substitute leaves the output
    at its saved value the check still fails.

    Parameters
    ----------
    capture
        Captured JAX equation metadata.
    inputs
        Replay inputs derived from trace graph parents.
    positions
        Primitive argument positions the perturbed parent fills.

    Returns
    -------
    tuple[Any, ...]
        Substitute values with the parent's shape and dtype.
    """

    import jax.numpy as jnp

    parent = jnp.asarray(inputs[positions[0]])
    substitutes = list(_bool_probe_battery(parent))
    if (
        str(capture.primitive) in _COMPARISON_PRIMITIVES
        and len(inputs) == 2
        and len(positions) == 1
    ):
        other = jnp.asarray(inputs[1 - positions[0]])
        substitutes.extend(_straddle_substitutes(parent, other))
    return tuple(substitutes)


def _bool_probe_battery(parent: Any) -> tuple[Any, ...]:
    """Return dtype-directed substitutes that flip a value-transmitting predicate."""

    import jax.numpy as jnp

    dtype = parent.dtype
    if dtype == jnp.bool_:
        return (jnp.zeros_like(parent), jnp.ones_like(parent), jnp.logical_not(parent))
    if jnp.issubdtype(dtype, jnp.complexfloating):
        return (
            jnp.zeros_like(parent),
            jnp.ones_like(parent),
            jnp.full_like(parent, 1j),
            jnp.full_like(parent, complex(float("nan"), 0.0)),
        )
    if jnp.issubdtype(dtype, jnp.floating):
        return (
            jnp.zeros_like(parent),
            jnp.ones_like(parent),
            -jnp.ones_like(parent),
            -parent,
            jnp.full_like(parent, jnp.nan),
            jnp.full_like(parent, jnp.inf),
            jnp.full_like(parent, -jnp.inf),
            jnp.full_like(parent, jnp.finfo(dtype).max / 2),
        )
    if jnp.issubdtype(dtype, jnp.integer):
        info = jnp.iinfo(dtype)
        battery = [
            jnp.zeros_like(parent),
            jnp.ones_like(parent),
            jnp.full_like(parent, info.max),
            jnp.full_like(parent, info.min),
        ]
        if info.min < 0:
            battery.append(-jnp.ones_like(parent))
        return tuple(battery)
    return ()


def _straddle_substitutes(parent: Any, other: Any) -> tuple[Any, ...]:
    """Return the comparand and its two adjacent representable values, shaped as ``parent``."""

    import jax.numpy as jnp

    dtype = parent.dtype
    numeric = jnp.issubdtype(dtype, jnp.floating) or jnp.issubdtype(dtype, jnp.integer)
    if not numeric or not (
        jnp.issubdtype(other.dtype, jnp.floating) or jnp.issubdtype(other.dtype, jnp.integer)
    ):
        return ()
    try:
        base = jnp.broadcast_to(other.astype(dtype), parent.shape)
    except ValueError:  # the comparand broadcasts wider than the parent: no straddle
        return ()
    if jnp.issubdtype(dtype, jnp.floating):
        if not bool(jnp.all(jnp.isfinite(base))):
            return ()
        try:
            return (
                base,
                jnp.nextafter(base, jnp.full_like(base, -jnp.inf)),
                jnp.nextafter(base, jnp.full_like(base, jnp.inf)),
            )
        except (TypeError, NotImplementedError):  # no nextafter for this float dtype
            return (base,)
    one = jnp.ones_like(base)
    return (base, base - one, base + one)


def _fold_in_keys(value: Any, array: Any) -> Any:
    """Return ``value``'s typed PRNG key(s) with ``1`` folded into each key."""

    import jax
    import jax.numpy as jnp

    if not array.shape:
        return jax.random.fold_in(value, 1)
    flat = jnp.reshape(array, (-1,))
    folded = jax.vmap(lambda key: jax.random.fold_in(key, 1))(flat)
    return jnp.reshape(folded, array.shape)


def _perturb_candidates(value: Any) -> tuple[Any, ...]:
    """Return deterministic perturbation candidates for a JAX value.

    Parameters
    ----------
    value
        Parent payload to perturb.

    Returns
    -------
    tuple[Any, ...]
        Perturbed values with the same shape and dtype family.
    """

    import jax.numpy as jnp

    array = jnp.asarray(value)
    if str(array.dtype).startswith("key<"):
        return (_fold_in_keys(value, array),)
    if array.dtype == jnp.bool_:
        return (jnp.logical_not(array),)
    if jnp.issubdtype(array.dtype, jnp.integer):
        info = jnp.iinfo(array.dtype)
        base_candidates = [
            array + jnp.asarray(1, dtype=array.dtype),
            array - jnp.asarray(1, dtype=array.dtype),
            jnp.zeros_like(array),
            jnp.full_like(array, info.max),
        ]
        if info.min < 0:
            base_candidates.append(jnp.full_like(array, info.min))
        return (
            *base_candidates,
            jnp.bitwise_xor(array, jnp.asarray(info.max, dtype=array.dtype)),
        )
    if jnp.issubdtype(array.dtype, jnp.complexfloating):
        magnitude = jnp.max(jnp.abs(array)) + jnp.asarray(1.0, dtype=array.real.dtype)
        return (
            array + jnp.asarray(magnitude + 0.125j, dtype=array.dtype),
            array - jnp.asarray(magnitude + 0.125j, dtype=array.dtype),
        )
    if jnp.issubdtype(array.dtype, jnp.floating):
        magnitude = jnp.max(jnp.abs(array)) + jnp.asarray(1.0, dtype=array.dtype)
        candidates: tuple[Any, ...] = (array + magnitude, array - magnitude)
        if not bool(jnp.all(jnp.isfinite(array))):
            # A NaN or inf parent absorbs any additive perturbation; finite
            # replacements are the only candidates that can move the child.
            candidates = (*candidates, jnp.zeros_like(array), jnp.ones_like(array))
        return candidates
    return (array,)
