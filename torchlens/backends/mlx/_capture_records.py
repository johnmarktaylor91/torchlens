"""Capture records and payload helpers for MLX replay validation.

Extracted from ``validation`` so that module stays under the file cap; the
public spellings stay importable from ``torchlens.backends.mlx.validation``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .._validation_shared import float_replay_tolerances


class _ReplaySlot:
    """Sentinel standing in for a labeled array leaf in a capture template.

    Replay never reads the stored value of a labeled leaf — it substitutes
    the DECLARED parent's saved payload — so retaining the emit-time array
    only pinned every intermediate activation for the lifetime of the trace.
    The sentinel keeps the container structure and flatten order intact
    (paddle's ``_template_value`` retention model).
    """

    __slots__ = ()

    def __repr__(self) -> str:
        """Return the sentinel's stable display form."""

        return "<mlx-replay-slot>"


REPLAY_SLOT = _ReplaySlot()


@dataclass(frozen=True)
class MLXOpCapture:
    """One captured MLX call retained for live replay validation.

    Parameters
    ----------
    labels_raw:
        Raw TorchLens labels reserved for the call's output arrays, in
        emission order.
    op_name:
        Wrapped operation name.
    func:
        Original (unwrapped) callable invoked by the wrapper.
    args:
        Positional argument templates: labeled array leaves are replaced by
        ``REPLAY_SLOT`` (replay sources them from saved parent payloads);
        unlabeled leaves (parameters, constants) keep their emit-time values.
    kwargs:
        Keyword argument templates with the same slotting.
    output:
        Unused legacy slot retained for constructor compatibility; the
        expected replay values come from the trace's saved payloads.
    arg_leaf_labels:
        Per positional argument, the raw parent label of each array leaf in
        deterministic flatten order (``None`` for unlabeled leaves such as
        parameters).
    kwarg_leaf_labels:
        The same per-leaf parent labels for keyword arguments, keyed by
        keyword name.
    interventions:
        Declared genuine user interventions as ``(output_leaf_index,
        hook_identity)`` pairs. Mirrored into the emit-time inventory
        fingerprint, so a post-hoc claim cannot steer hook re-application.
    appliers:
        Runtime hook appliers as ``(output_leaf_index, callable)`` pairs,
        used by replay to reproduce the declared substitution. Never part of
        the fingerprint; a record whose declared interventions lack a
        matching applier fails closed.
    """

    labels_raw: tuple[str, ...]
    op_name: str
    func: Any
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    output: Any = None
    arg_leaf_labels: tuple[tuple[str | None, ...], ...] = ()
    kwarg_leaf_labels: dict[str, tuple[str | None, ...]] = field(default_factory=dict)
    interventions: tuple[tuple[int, str], ...] = ()
    appliers: tuple[tuple[int, Any], ...] = ()


def build_capture_template(
    value: Any,
    leaf_labels: tuple[str | None, ...],
) -> Any:
    """Return ``value`` with labeled array leaves replaced by ``REPLAY_SLOT``.

    Parameters
    ----------
    value:
        Emit-time argument value.
    leaf_labels:
        Recorded per-leaf parent labels for ``value`` in flatten order.

    Returns
    -------
    Any
        Template retaining structure and unlabeled leaves only.
    """

    cursor = iter(leaf_labels)

    def _slot(node: Any) -> Any:
        """Replace each labeled array leaf with ``REPLAY_SLOT``, preserving structure."""

        if _is_mlx_array(node):
            label = next(cursor, None)
            return node if label is None else REPLAY_SLOT
        # Exact container types only: an ``mlx.nn.Module`` is a dict subclass
        # and must stay the module object (its ``__call__`` reads attributes).
        if type(node) is tuple:
            return tuple(_slot(item) for item in node)
        if type(node) is list:
            return [_slot(item) for item in node]
        if type(node) is dict:
            return {key: _slot(item) for key, item in node.items()}
        return node

    return _slot(value)


def _is_mlx_array(value: Any) -> bool:
    """Return whether ``value`` is an ``mlx.core.array``.

    Parameters
    ----------
    value:
        Candidate object.

    Returns
    -------
    bool
        True for MLX arrays.
    """

    try:
        import mlx.core as mx
    except ImportError:
        return False
    return isinstance(value, mx.array)


def _iter_output_arrays(output: Any) -> tuple[Any, ...]:
    """Flatten an MLX call output into its array leaves.

    Parameters
    ----------
    output:
        Raw output object (array, tuple/list of arrays, or other).

    Returns
    -------
    tuple[Any, ...]
        Array leaves in deterministic order.
    """

    if _is_mlx_array(output):
        return (output,)
    if isinstance(output, (tuple, list)):
        leaves: list[Any] = []
        for item in output:
            leaves.extend(_iter_output_arrays(item))
        return tuple(leaves)
    return ()


def _payloads_close(a: Any, b: Any) -> bool:
    """Return whether two MLX arrays match within dtype-aware tolerance.

    Parameters
    ----------
    a, b:
        Arrays to compare.

    Returns
    -------
    bool
        True when shapes/dtypes agree and values are close.
    """

    a_np = np.asarray(a)
    b_np = np.asarray(b)
    if a_np.shape != b_np.shape or a_np.dtype != b_np.dtype:
        return False
    if a_np.dtype.kind in ("f", "c"):
        # Eps-derived per-dtype band, shared with every sibling oracle
        # (b5-opus R17-1: this ladder had hardcoded 1e-5 for EVERY >=4-byte
        # float while the replay oracle went eps-derived, leaving the fp64
        # validation band ~5e8 of fp64's own ULPs; ``np.finfo`` reports
        # component precision for complex).
        rtol, atol = float_replay_tolerances(np.finfo(a_np.dtype))
        return bool(np.allclose(a_np, b_np, rtol=rtol, atol=atol, equal_nan=True))
    return bool(np.array_equal(a_np, b_np))


def _perturb_candidates(value: Any) -> tuple[Any, ...]:
    """Return perturbed variants of one MLX array argument.

    Parameters
    ----------
    value:
        MLX array to perturb.

    Returns
    -------
    tuple[Any, ...]
        Candidate replacement arrays.
    """

    import mlx.core as mx

    value_np = np.asarray(value)
    if value_np.dtype.kind == "f":
        # The magnitude shifts mirror the paddle/tf/tinygrad tripwires: a
        # saturating call (ReLU over inputs all below -0.5) ignores the small
        # shift and the doubling, so a healthy capture would read UNPROVED.
        # A replay that ignores its argument stays UNPROVED under every one.
        magnitude = float(np.max(np.abs(value_np))) + 1.0 if value_np.size else 1.0
        shift = mx.array(magnitude, dtype=value.dtype)
        return (
            value + mx.array(0.5, dtype=value.dtype),
            value * 2,
            value + shift,
            value - shift,
        )
    if value_np.dtype.kind in ("i", "u"):
        return (value + 1, mx.zeros_like(value))
    if value_np.dtype.kind == "b":
        return (mx.logical_not(value),)
    return ()
