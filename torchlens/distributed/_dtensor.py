"""DTensor dual-geometry extraction: logical identity for sharded state.

Merge-ranks tier (b), DTensor param identity. A DTensor's ``.shape`` is the
LOGICAL shape while the rank physically holds only a shard; every TorchLens
surface that talks about distributed tensor state must declare BOTH
geometries explicitly so no reader ever guesses which one it is looking at
(design-merge-ranks-c v5, 1.7 amendment 2). This module is the one extraction
point:

* the ``dtensor`` / ``tensor_parallel`` findings and ``tl.compat.report``
  rows attach per-site dual geometry, so a refused TP/FSDP2 model is refused
  with its parameters precisely IDENTIFIED (logical shape, placements, shard
  offset) instead of silently mis-counted;
* C0 boundary ``RoleEntry`` dual-geometry slots and the C2 rank-local capture
  substrate consume the same helper when DTensor values flow through capture.

Everything degrades gracefully: any probe failure yields ``None`` fields,
never an exception, because this code runs inside capture-entry detection.
"""

from __future__ import annotations

from typing import Any

__all__ = ["dtensor_dual_geometry"]


def _as_int_list(value: Any) -> list[int] | None:
    """Coerce an iterable to a list of ints, or ``None`` when it is not one."""

    try:
        return [int(item) for item in value]
    except Exception:
        return None


def dtensor_dual_geometry(value: Any) -> dict[str, Any] | None:
    """Extract the dual (logical + physical) geometry of a DTensor.

    Parameters
    ----------
    value:
        Candidate DTensor. Anything without DTensor-shaped attributes returns
        ``None``.

    Returns
    -------
    dict[str, Any] | None
        Plain-data geometry record: ``logical_shape``, ``local_shape``,
        ``placements`` (display strings), ``mesh_shape``, ``mesh_coords``
        (this rank's coordinates, ``None`` off-mesh), ``shard_offset`` (this
        shard's start indices in the logical tensor, ``None`` when
        unprovable), ``logical_numel``, and ``local_numel``. All fields are
        best-effort-``None``; the record itself is ``None`` only when the
        value is not a DTensor.
    """

    placements = getattr(value, "placements", None)
    mesh = getattr(value, "device_mesh", None)
    if placements is None or mesh is None:
        return None

    logical_shape = _as_int_list(getattr(value, "shape", None))
    local_shape: list[int] | None = None
    local_numel: int | None = None
    try:
        local = value.to_local()
        local_shape = _as_int_list(local.shape)
        local_numel = int(local.numel())
    except Exception:
        pass

    mesh_shape = _as_int_list(getattr(mesh, "shape", None))
    mesh_coords: list[int] | None = None
    try:
        coordinate = mesh.get_coordinate()
        mesh_coords = _as_int_list(coordinate) if coordinate is not None else None
    except Exception:
        pass

    shard_offset: list[int] | None = None
    # r-b4 R26-2: the geometry helper routes through _torch_compat
    # (HAS_DTENSOR_SHARD_GEOMETRY), so "the private API moved" flips a named
    # flag visible in doctor/compat instead of collapsing into the same silent
    # None as a legitimately off-mesh rank.
    from .._utils._torch_compat import get_dtensor_shard_geometry_fn

    compute_local_shape_and_global_offset = get_dtensor_shard_geometry_fn()
    if compute_local_shape_and_global_offset is not None:
        try:
            computed_local, global_offset = compute_local_shape_and_global_offset(
                tuple(logical_shape or ()), mesh, tuple(placements)
            )
            shard_offset = _as_int_list(global_offset)
            if local_shape is None:
                local_shape = _as_int_list(computed_local)
        except Exception:
            # An off-mesh rank (or geometry the helper rejects): the offset is
            # genuinely unprovable and stays None rather than guessed.
            pass

    logical_numel = None
    if logical_shape is not None:
        logical_numel = 1
        for dim in logical_shape:
            logical_numel *= dim

    return {
        "logical_shape": logical_shape,
        "local_shape": local_shape,
        "placements": [repr(placement) for placement in placements],
        "mesh_shape": mesh_shape,
        "mesh_coords": mesh_coords,
        "shard_offset": shard_offset,
        "logical_numel": logical_numel,
        "local_numel": local_numel,
    }


#: The one distributed-scope marker value (DOCUMENTED-UNSTABLE pending its S2
#: amendment; census plan 3.1). A fully replicated DTensor is NOT shard-local.
RANK_LOCAL_SHARD = "rank_local_shard"


def shard_local_placements(placements: Any) -> bool:
    """Return whether a placements sequence marks a value shard-local.

    THE ONE sharded predicate (census plan 3.3(c)): a PURE function over the
    placements limb, evaluable identically on live torch placement objects at
    capture time and on persisted ``repr`` strings at load time, so the
    marker setter and the future load-coherence validator can never drift.
    Any non-Replicate placement anywhere marks shard-local; all-Replicate
    (or empty/absent) does not -- a fully replicated DTensor's local values
    equal its logical values, so presenting it as shard-local would
    over-label (census red N5).

    Parameters
    ----------
    placements:
        A sequence of live DTensor placement objects OR their persisted
        ``repr`` strings; ``None``/empty is not shard-local.

    Returns
    -------
    bool
        ``True`` when any placement is non-Replicate.
    """

    if not placements:
        return False
    for placement in placements:
        if isinstance(placement, str):
            if not placement.startswith("Replicate"):
                return True
        elif type(placement).__name__ != "Replicate":
            return True
    return False


def value_marks_shard_local(value: Any) -> bool:
    """Return whether a live tensor value would set the shard-local marker.

    Classification is by the sharded predicate over the value's OWN declared
    placements, never by dual-geometry record presence (census plan 3.2(4b):
    a placements-bearing value with no reachable mesh still classifies).

    Parameters
    ----------
    value:
        Candidate tensor-like object (DTensor or anything else).

    Returns
    -------
    bool
        ``True`` when the value carries a non-Replicate DTensor placement.
    """

    return shard_local_placements(getattr(value, "placements", None))
