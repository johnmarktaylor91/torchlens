"""Source-site to target-site mapping for ``patch_from`` (Patchscopes, SelfIE).

``patch_from(source, source_site=..., positions=...)`` reads the donor value at
one named site of the source trace (any layer, not only the site being edited)
and writes chosen source positions into chosen target positions along one
axis, so a hidden state read at layer ``l`` and position ``i`` of one prompt can
be placed at layer ``l'`` and position ``j`` of another prompt of a different
length. DOCUMENTED-UNSTABLE spelling.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping
from typing import Any

import torch

from .._errors import InvalidArgumentError
from .errors import HookValueError

PositionPairs = tuple[tuple[int, int], ...]


def normalize_positions(positions: Any) -> PositionPairs | None:
    """Normalize a ``positions=`` argument to ``(source, target)`` pairs.

    Parameters
    ----------
    positions:
        ``None`` (patch the whole value), an ``int`` (the same position on
        both sides), a mapping ``{source_position: target_position}``, or an
        iterable of ``(source_position, target_position)`` pairs. Negative
        positions count from the end, as in Python indexing.

    Returns
    -------
    PositionPairs | None
        Ordered pairs, or ``None`` for a whole-value patch.

    Raises
    ------
    InvalidArgumentError
        When a position is not an integer or the mapping is empty.
    """

    if positions is None:
        return None
    if isinstance(positions, bool):
        pairs: list[Any] = [positions]
    elif isinstance(positions, int):
        pairs = [(positions, positions)]
    elif isinstance(positions, Mapping):
        pairs = list(positions.items())
    elif isinstance(positions, Iterable):
        pairs = list(positions)
    else:
        pairs = [positions]
    normalized: list[tuple[int, int]] = []
    for pair in pairs:
        if not (
            isinstance(pair, tuple)
            and len(pair) == 2
            and all(isinstance(item, int) and not isinstance(item, bool) for item in pair)
        ):
            raise InvalidArgumentError(
                f"patch_from(positions=...) entries must be (source, target) integer "
                f"pairs; received {pair!r}",
                code="patch_positions_invalid",
                remedy="pass an int, a {source_position: target_position} dict, or a "
                "list of (source_position, target_position) pairs",
                argument="positions",
            )
        normalized.append((int(pair[0]), int(pair[1])))
    if not normalized:
        raise InvalidArgumentError(
            "patch_from(positions=...) is empty, so the patch would write nothing",
            code="patch_positions_invalid",
            remedy="pass at least one (source_position, target_position) pair, or "
            "positions=None to patch the whole value",
            argument="positions",
        )
    return tuple(normalized)


def resolve_source_site(source: Any, source_site: Any) -> tuple[str, Any]:
    """Resolve the one donor op a ``source_site=`` names on the source trace.

    Parameters
    ----------
    source:
        Source trace.
    source_site:
        A label (pass-qualified on multi-pass layers) or a selector that
        resolves to exactly one captured op of the source trace.

    Returns
    -------
    tuple[str, Any]
        ``(site_label, op)`` of the donor.

    Raises
    ------
    HookValueError
        ``patch_source_site_unresolved`` when the site names no op, or names
        more than one (a donor is never guessed).
    """

    if isinstance(source_site, str):
        op = source._layer_dict_all_keys.get(source_site)
        ops: tuple[Any, ...] = () if op is None else (op,)
    else:
        from .errors import MultiMatchWarning

        with warnings.catch_warnings():
            # Several matches are this refusal's subject, not a warning.
            warnings.simplefilter("ignore", MultiMatchWarning)
            sites = source.resolve_sites(source_site, max_fanout=len(source._layer_list) + 1)
        ops = tuple(sites)
    candidates = [op for op in ops if int(getattr(op, "num_passes", 1) or 1) <= 1 or _is_pass(op)]
    if len(ops) != 1 or len(candidates) != 1:
        raise HookValueError(
            f"patch_from(source_site={source_site!r}) must name exactly one captured op "
            f"of the source trace; it names {len(ops)}"
            + (" (a multi-pass layer: address one pass as 'label:pass')" if ops else "")
            + ". Remedy: pass one pass-qualified label, or a selector narrowed to one op.",
            code="patch_source_site_unresolved",
            site=str(source_site),
        )
    op = candidates[0]
    return str(getattr(op, "label", None) or getattr(op, "layer_label", source_site)), op


def _is_pass(op: Any) -> bool:
    """Return whether ``op`` is one pass (an ``Op``) rather than a multi-pass ``Layer``."""

    return type(op).__name__ == "Op"


def apply_position_map(
    out: torch.Tensor,
    donor: torch.Tensor,
    pairs: PositionPairs,
    axis: int,
    *,
    site_label: str,
) -> torch.Tensor:
    """Write ``donor[..., s, ...]`` into ``out[..., t, ...]`` for each pair.

    Positions not named keep the target's own value. A donor batch of one is
    broadcast across the target batch; every other non-position dimension
    must match.

    Parameters
    ----------
    out:
        Target activation at the edited site.
    donor:
        Source activation at the source site (detached).
    pairs:
        ``(source_position, target_position)`` pairs.
    axis:
        Position axis, the same on both tensors.
    site_label:
        Donor site label, for error text.

    Returns
    -------
    torch.Tensor
        A new tensor; ``out`` is never mutated.

    Raises
    ------
    HookValueError
        ``patch_positions_out_of_range`` for a position or axis outside either
        tensor, ``patch_positions_shape_mismatch`` when the sliced shapes do
        not broadcast.
    """

    if not (-out.dim() <= axis < out.dim() and -donor.dim() <= axis < donor.dim()):
        raise HookValueError(
            f"patch_from(position_axis={axis}) is outside the target ({out.dim()}-d) or "
            f"donor ({donor.dim()}-d) value at {site_label!r}",
            code="patch_positions_out_of_range",
            site=site_label,
        )
    result = out.clone()
    for source_position, target_position in pairs:
        if not (
            -donor.shape[axis] <= source_position < donor.shape[axis]
            and -out.shape[axis] <= target_position < out.shape[axis]
        ):
            raise HookValueError(
                f"patch_from position pair ({source_position}, {target_position}) is outside "
                f"the donor length {donor.shape[axis]} or target length {out.shape[axis]} "
                f"on axis {axis} at {site_label!r}",
                code="patch_positions_out_of_range",
                site=site_label,
            )
        target_slice = result.select(axis, target_position)
        source_slice = donor.select(axis, source_position).to(dtype=result.dtype)
        try:
            target_slice.copy_(source_slice.expand_as(target_slice))
        except RuntimeError as exc:
            raise HookValueError(
                f"patch_from donor slice {tuple(source_slice.shape)} at {site_label!r} does "
                f"not fit the target slice {tuple(target_slice.shape)}",
                code="patch_positions_shape_mismatch",
                site=site_label,
            ) from exc
    return result
