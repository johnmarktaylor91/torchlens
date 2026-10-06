"""Destination-overwrite coverage proofs for indexed writes.

Pure facts about whether an index-based write (``__setitem__``, ``index_put``,
``scatter``) overwrites every element of its destination exactly once, so the
original destination values cannot reach the output. ``exemptions.py`` combines
these facts with the perturbed-parent slot checks in its custom exemption checks
and posthoc overwrite decision; the checks themselves stay there.

Split from ``validation/exemptions.py`` along this seam (R43 file-size ratchet);
the proofs are unchanged.
"""

from typing import Any

import torch

from ..data_classes.op import Op


def _setitem_index_targets_are_unique(index: Any) -> bool:
    """Return whether a ``__setitem__`` index cannot duplicate write targets.

    Parameters
    ----------
    index:
        Index argument supplied to ``Tensor.__setitem__``.

    Returns
    -------
    bool
        True for basic indexing and for verified unique tensor/list advanced
        indices. Duplicate advanced indices can make ``selected.numel()`` equal
        the destination size while leaving some destination elements live.
    """

    components = index if isinstance(index, tuple) else (index,)
    for component in components:
        if isinstance(component, list):
            try:
                component = torch.as_tensor(component)
            except (TypeError, ValueError):
                return False
        if isinstance(component, torch.Tensor):
            if component.dtype == torch.bool:
                continue
            if not _tensor_is_integer_index(component):
                return False
            flattened = component.reshape(-1)
            if int(flattened.numel()) != int(torch.unique(flattened).numel()):
                return False
            continue
        if component is None or component is Ellipsis or isinstance(component, (slice, int)):
            continue
        return False
    return True


def _index_positions_cover_destination_exactly(
    destination: torch.Tensor,
    index: Any,
) -> bool:
    """Return whether ``destination[index]`` addresses every element exactly once.

    Value-based uniqueness (``torch.unique`` on raw index values) is blind to
    negative-index aliasing: ``0`` and ``-2`` are distinct VALUES that address
    the SAME position on a length-2 dim, so a "fully overwritten" proof counted
    a full overwrite while an element survived with its prior value. Indexing
    an identity-POSITION tensor with the saved index makes torch's own indexing
    semantics normalize negatives, slices, ellipsis, and boolean masks exactly;
    requiring the selected positions to be unique and to number the whole
    destination is the exact single-coverage proof.

    Parameters
    ----------
    destination:
        Destination tensor of the write.
    index:
        Saved index argument (``__setitem__`` index or ``index_put`` indices
        tuple).

    Returns
    -------
    bool
        True only when the index selects each destination position exactly
        once and selects all of them. Any indexing failure returns False so
        callers fail closed.
    """

    try:
        positions = torch.arange(destination.numel(), device=destination.device).reshape(
            destination.shape
        )
        covered = positions[index]
    except (IndexError, TypeError, RuntimeError):
        return False
    flattened = covered.reshape(-1)
    if int(flattened.numel()) != int(destination.numel()):
        return False
    return int(torch.unique(flattened).numel()) == int(destination.numel())


def _tensor_is_integer_index(tensor: torch.Tensor) -> bool:
    """Return whether ``tensor`` has an integer dtype accepted for indexing.

    Parameters
    ----------
    tensor:
        Tensor index component.

    Returns
    -------
    bool
        True when the tensor dtype is an integer indexing dtype.
    """

    return not tensor.dtype.is_floating_point and not tensor.dtype.is_complex


def _index_put_destination_is_fully_overwritten(
    perturbed_tensor: torch.Tensor | None,
    layer: Op,
) -> bool:
    """Return whether an ``index_put`` call overwrites the perturbed destination.

    Parameters
    ----------
    perturbed_tensor:
        Tensor selected for perturbation.
    layer:
        Captured ``index_put``/``index_put_`` op.

    Returns
    -------
    bool
        True only when the perturbed tensor is the destination (``args[0]``), the
        call is non-accumulating, and the indexed positions cover the ENTIRE
        destination and are exactly written by the broadcast ``values`` (so the
        destination's prior value is wholly irrelevant). Returns False for any
        perturbed VALUE/INDEX parent, for the accumulating case, and for a
        partial overwrite (where un-indexed destination elements still flow
        through).
    """

    args = layer.saved_args
    if not isinstance(perturbed_tensor, torch.Tensor):
        return False
    if args is None or len(args) < 3:
        return False
    destination, indices, values = args[0], args[1], args[2]
    if not isinstance(destination, torch.Tensor) or not isinstance(values, torch.Tensor):
        return False
    # Narrow: the perturbed parent must be the DESTINATION, never the values/index.
    if not torch.equal(perturbed_tensor.to(destination.device), destination):
        return False
    # accumulate=True adds the value to the prior destination, so the prior value
    # is NOT irrelevant -- never exempt that case. accumulate may arrive as a
    # positional arg (index 3) or as a keyword.
    accumulate = False
    if len(args) > 3:
        accumulate = bool(args[3])
    elif "accumulate" in (layer.saved_kwargs or {}):
        accumulate = bool(layer.saved_kwargs["accumulate"])
    if accumulate:
        return False
    # index_put indices are an advanced-indexing tuple/list of LongTensors.
    if isinstance(indices, list):
        index = tuple(indices)
    elif isinstance(indices, tuple):
        index = indices
    else:
        index = (indices,)
    try:
        selected = destination[index]
    except (IndexError, TypeError, RuntimeError):
        return False
    # The written slice must broadcast the replacement exactly (every selected
    # element is overwritten, none left at its prior value).
    try:
        broadcast_shape = torch.broadcast_shapes(tuple(selected.shape), tuple(values.shape))
    except RuntimeError:
        return False
    if tuple(broadcast_shape) != tuple(selected.shape):
        return False
    # The exemption is only sound when the WHOLE destination is overwritten: any
    # un-indexed element keeps its prior value and so still influences the output
    # (the partial-overwrite false-exemption guard). Require the indexed region to
    # cover every destination element, with no duplicate indices inflating the
    # count -- duplicates would match the numel without covering everything.
    if not _index_put_indices_are_unique(index):
        return False
    if int(selected.numel()) != int(destination.numel()):
        return False
    return _index_positions_cover_destination_exactly(destination, index)


def _index_put_indices_are_unique(index: tuple[Any, ...]) -> bool:
    """Return whether advanced ``index_put`` indices address distinct positions.

    Duplicate indices would let ``selected.numel()`` reach ``destination.numel()``
    without actually covering every destination element, so the full-overwrite
    coverage check would be fooled. This conservatively requires each integer
    index tensor to hold unique values; a non-integer (e.g. boolean mask) or any
    structure it cannot verify returns False so the exemption is withheld.
    """

    for component in index:
        if not isinstance(component, torch.Tensor):
            return False
        if component.dtype == torch.bool:
            # A bool mask selects each True position once -> inherently unique.
            continue
        flattened = component.reshape(-1)
        if int(flattened.numel()) != int(torch.unique(flattened).numel()):
            return False
    return True


def _get_scatter_destination_dim_index(
    layer: Op,
) -> tuple[torch.Tensor, int, torch.Tensor] | None:
    """Return scatter destination, dim, and index tensors when they are replayable.

    Parameters
    ----------
    layer:
        Scatter operation being validated.

    Returns
    -------
    tuple[torch.Tensor, int, torch.Tensor] | None
        Destination tensor, scatter dimension, and index tensor, or ``None``
        when the call shape is unsupported or uses reduce semantics.
    """

    args = layer.saved_args
    kwargs = layer.saved_kwargs
    if len(args) < 1 or not isinstance(args[0], torch.Tensor):
        return None
    if kwargs.get("reduce") is not None:
        return None
    if len(args) > 4 and args[4] is not None:
        return None

    dest = args[0]
    dim = kwargs.get("dim", args[1] if len(args) > 1 else None)
    index = kwargs.get("index", args[2] if len(args) > 2 else None)
    if not isinstance(dim, int) or not isinstance(index, torch.Tensor):
        return None
    if dim < 0:
        dim = dest.ndim + dim
    if dim < 0 or dim >= dest.ndim:
        return None
    return dest, dim, index


def _scatter_index_fully_overwrites_dim(dest: torch.Tensor, dim: int, index: torch.Tensor) -> bool:
    """Return whether scatter index covers every destination slot along ``dim``.

    Parameters
    ----------
    dest:
        Scatter destination tensor.
    dim:
        Normalized scatter dimension.
    index:
        Scatter index tensor.

    Returns
    -------
    bool
        True when every slice orthogonal to ``dim`` contains each valid
        destination index, making the destination's prior values irrelevant.
    """

    if index.ndim != dest.ndim or index.shape[dim] < dest.shape[dim]:
        return False
    if any(index.shape[axis] < dest.shape[axis] for axis in range(dest.ndim) if axis != dim):
        return False
    n_positions = dest.shape[dim]
    if n_positions == 0:
        return False
    moved = index.detach().cpu().movedim(dim, -1).reshape(-1, index.shape[dim])
    required = set(range(n_positions))
    for row in moved:
        row_values = {int(value) for value in row.tolist() if 0 <= int(value) < n_positions}
        if not required.issubset(row_values):
            return False
    return True
