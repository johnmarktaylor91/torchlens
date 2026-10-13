"""Index-domain perturbation probes for index/target-consuming ops.

The gather/scatter/embedding/cross_entropy family reads an integer index whose
VALUES determine the output but whose valid domain is bounded by a sibling
argument's shape. These helpers locate that index slot, measure its domain, and
build in-domain perturbations (full rotation, single-entry move) for the replay
ladder. ``exemptions.py`` consumes the slot and domain facts for its degenerate-
domain and value-irrelevance decisions; ``core.py`` consumes the perturbations.
Split from ``validation/exemptions.py`` along this seam (R43 file-size
ratchet); the logic is unchanged.
"""

from typing import Any

import torch

from ..data_classes.op import Op

# ---------------------------------------------------------------------------
# Index-domain perturbation standard (F2 tightening).
#
# The ops below consume an integer index/target arg whose VALUES genuinely
# determine the output, but whose valid domain is bounded by a sibling arg's
# shape (random draws can go out of bounds and crash the kernel). The legacy
# treatment blanket-skipped perturbing those args, which could excuse a
# genuinely-missed dependency. The tightened standard perturbs them IN-DOMAIN:
# every in-range entry is rotated by one position (``(v + 1) % n``, a bijection
# on ``[0, n)``), out-of-range sentinels (e.g. ``cross_entropy`` ignore_index)
# are preserved, and only the provably degenerate domain (``n <= 1`` or no
# in-range entries) is exempted pre-execution.
# ---------------------------------------------------------------------------

_INDEX_DOMAIN_INT_DTYPES = frozenset(
    {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
)

# func_name -> (positional index-arg slot, kwarg spellings of the index arg).
_INDEX_DOMAIN_ARG_SPECS: dict[str, tuple[int, frozenset[str]]] = {
    # aten spelling: embedding(weight, indices, ...) -- domain = weight rows.
    "embedding": (1, frozenset({"indices", "input"})),
    # gather/index_select/scatter*(input, dim, index, ...) -- domain =
    # input.shape[dim].
    "gather": (2, frozenset({"index"})),
    "index_select": (2, frozenset({"index"})),
    "scatter": (2, frozenset({"index"})),
    "scatter_": (2, frozenset({"index"})),
    "scatter_add": (2, frozenset({"index"})),
    "scatter_add_": (2, frozenset({"index"})),
    "scatteradd": (2, frozenset({"index"})),
    # cross_entropy(input, target, ...) -- domain = the class dimension.
    "cross_entropy": (1, frozenset({"target"})),
    # nll_loss(input, target, ...) -- domain = the class dimension.
    "nll_loss": (1, frozenset({"target"})),
    # take_along_dim(input, indices, dim) -- domain = input.shape[dim], or
    # input.numel() when dim is None (indices into the flattened input).
    "take_along_dim": (1, frozenset({"indices"})),
    # index_add/index_copy/index_fill(input, dim, index, ...) and their in-place
    # forms -- domain = input.shape[dim], as for index_select. No exemption
    # applies to them: the scatter overwrite proof matches overwriting scatters
    # by name (``_destination_coverage._OVERWRITE_SCATTER_FUNC_NAMES``).
    "index_add": (2, frozenset({"index"})),
    "index_add_": (2, frozenset({"index"})),
    "index_copy": (2, frozenset({"index"})),
    "index_copy_": (2, frozenset({"index"})),
    "index_fill": (2, frozenset({"index"})),
    "index_fill_": (2, frozenset({"index"})),
}

#: Class-target ops whose ``ignore_index`` marks targets that contribute nothing.
_IGNORE_INDEX_OPS = frozenset({"cross_entropy", "nll_loss"})


def _parent_is_index_domain_arg(layer: Op, parent_label: str) -> bool:
    """Return whether ``parent_label`` occupies the op's index/target arg slot.

    Parameters
    ----------
    layer:
        Captured op whose parent-argument map is inspected.
    parent_label:
        Perturbed parent label.

    Returns
    -------
    bool
        True when the parent is registered at the index-arg position or one of
        its kwarg spellings. Position identity only -- never tensor equality.
    """

    spec = _INDEX_DOMAIN_ARG_SPECS.get(getattr(layer, "func_name", None) or "")
    if spec is None:
        return False
    index_pos, index_kwargs = spec
    parent_arg_positions = getattr(layer, "parent_arg_positions", {}) or {}
    if (parent_arg_positions.get("args", {}) or {}).get(index_pos) == parent_label:
        return True
    kwarg_map = parent_arg_positions.get("kwargs", {}) or {}
    return any(kwarg_map.get(name) == parent_label for name in index_kwargs)


def layer_has_index_domain_parent(layer: Op) -> bool:
    """Return whether any recorded parent occupies the layer's index-domain slot.

    Parameters
    ----------
    layer:
        Child op being validated.

    Returns
    -------
    bool
        True when a parent label sits at the op's index argument position or
        one of its kwarg spellings (``_parent_is_index_domain_arg``).
    """

    parent_arg_positions = getattr(layer, "parent_arg_positions", {}) or {}
    labels = set((parent_arg_positions.get("args", {}) or {}).values())
    labels |= set((parent_arg_positions.get("kwargs", {}) or {}).values())
    return any(
        isinstance(label, str) and _parent_is_index_domain_arg(layer, label) for label in labels
    )


def _index_domain_size(layer: Op) -> int | None:
    """Return the exclusive upper bound of the op's valid index domain.

    Parameters
    ----------
    layer:
        Captured index-consuming op.

    Returns
    -------
    int or None
        Number of valid index values (``weight`` rows for ``embedding``,
        ``input.shape[dim]`` for the gather/scatter family, the class-dim size
        for ``cross_entropy``), or ``None`` when the saved call shape cannot
        prove a bound (callers then stay strict).
    """

    func_name = getattr(layer, "func_name", None)
    args: tuple[Any, ...] = getattr(layer, "saved_args", None) or ()
    kwargs = getattr(layer, "saved_kwargs", None) or {}
    if func_name == "embedding":
        weight = kwargs.get("weight", args[0] if args else None)
        if isinstance(weight, torch.Tensor) and weight.ndim >= 1:
            return int(weight.shape[0])
        return None
    if func_name == "take_along_dim":
        return _take_along_dim_domain_size(args, kwargs)
    if func_name in _IGNORE_INDEX_OPS:
        logits = kwargs.get("input", args[0] if args else None)
        if not isinstance(logits, torch.Tensor) or logits.ndim < 1:
            return None
        return int(logits.shape[1]) if logits.ndim >= 2 else int(logits.shape[0])
    source = kwargs.get("input", args[0] if args else None)
    dim = kwargs.get("dim", args[1] if len(args) > 1 else None)
    if not isinstance(source, torch.Tensor) or not isinstance(dim, int):
        return None
    if dim < 0:
        dim = source.ndim + dim
    if dim < 0 or dim >= source.ndim:
        return None
    return int(source.shape[dim])


def _take_along_dim_domain_size(args: tuple[Any, ...], kwargs: dict[str, Any]) -> int | None:
    """Return ``take_along_dim``'s index domain from its saved call.

    Parameters
    ----------
    args, kwargs:
        Saved positional and keyword arguments ``(input, indices, dim)``.

    Returns
    -------
    int or None
        ``input.shape[dim]``, ``input.numel()`` when ``dim`` is ``None``, or
        ``None`` when the shape cannot prove a bound.
    """

    source = kwargs.get("input", args[0] if args else None)
    dim = kwargs.get("dim", args[2] if len(args) > 2 else None)
    if not isinstance(source, torch.Tensor):
        return None
    if dim is None:
        return int(source.numel())
    if not isinstance(dim, int) or not -source.ndim <= dim < source.ndim:
        return None
    return int(source.shape[dim])


def _in_domain_ignore_index(layer: Op, domain_size: int) -> int | None:
    """Return a class-target op's ``ignore_index`` when it lies inside the domain.

    Parameters
    ----------
    layer:
        Captured op.
    domain_size:
        Exclusive upper bound of the index domain.

    Returns
    -------
    int or None
        The ignore value when the op is ``cross_entropy``/``nll_loss`` and its
        ``ignore_index`` is in ``[0, domain_size)``; else ``None``. A target is
        never rotated onto or off that value: doing so would change which
        targets count, not just which class a counted target names.
    """

    if getattr(layer, "func_name", None) not in _IGNORE_INDEX_OPS:
        return None
    kwargs = getattr(layer, "saved_kwargs", None) or {}
    args: tuple[Any, ...] = getattr(layer, "saved_args", None) or ()
    # nll_loss(input, target, weight, size_average, ignore_index, ...);
    # cross_entropy shares the slot order.
    ignore = kwargs.get("ignore_index", args[4] if len(args) > 4 else -100)
    if isinstance(ignore, int) and 0 <= ignore < domain_size:
        return ignore
    return None


def _rotate_skipping(values: torch.Tensor, domain_size: int, skip: int | None) -> torch.Tensor:
    """Rotate indices by one position in ``[0, domain_size)``, never onto ``skip``.

    Parameters
    ----------
    values:
        In-domain integer indices (none equal to ``skip``).
    domain_size:
        Domain size, at least 2 (at least 3 when ``skip`` is set).
    skip:
        A value no rotated entry may take, or ``None``.

    Returns
    -------
    torch.Tensor
        ``(v + 1) % n``, stepping once more past ``skip``.
    """

    rotated = (values + 1).remainder(domain_size)
    if skip is None:
        return rotated
    return torch.where(rotated == skip, (rotated + 1).remainder(domain_size), rotated)


def _saved_index_domain_arg_value(layer: Op) -> Any:
    """Return the saved index/target argument value for an index-domain op.

    Parameters
    ----------
    layer:
        Captured index-consuming op.

    Returns
    -------
    Any
        The saved argument at the index slot (positional or kwarg spelling),
        or ``None`` when it cannot be located.
    """

    spec = _INDEX_DOMAIN_ARG_SPECS.get(getattr(layer, "func_name", None) or "")
    if spec is None:
        return None
    index_pos, index_kwargs = spec
    kwargs = getattr(layer, "saved_kwargs", None) or {}
    for name in index_kwargs:
        if name in kwargs:
            return kwargs[name]
    args: tuple[Any, ...] = getattr(layer, "saved_args", None) or ()
    if len(args) > index_pos:
        return args[index_pos]
    return None


def index_domain_rotation_values(
    layer: Op,
    parent_label: str,
    parent_values: torch.Tensor,
) -> torch.Tensor | None:
    """Return a domain-safe rotated index perturbation for ``parent_values``.

    Every in-domain entry is rotated by one valid position
    (``(v + 1) % n``, guaranteed distinct from ``v`` when ``n >= 2``);
    out-of-domain entries (e.g. ``ignore_index`` sentinels) are preserved so
    the perturbed call stays executable. Mirrors the ``one_hot`` precedent.

    Parameters
    ----------
    layer:
        Child op being replayed.
    parent_label:
        Parent label selected for perturbation.
    parent_values:
        Saved parent tensor values.

    Returns
    -------
    torch.Tensor or None
        Rotated in-domain indices, or ``None`` when this parent is not an
        integer index arg of an index-domain op or no in-domain rotation
        exists (callers fall through to the generic strategies).
    """

    if not _parent_is_index_domain_arg(layer, parent_label):
        return None
    if (
        not isinstance(parent_values, torch.Tensor)
        or parent_values.dtype not in _INDEX_DOMAIN_INT_DTYPES
    ):
        return None
    domain_size = _index_domain_size(layer)
    if domain_size is None or domain_size < 2:
        return None
    in_domain = (parent_values >= 0) & (parent_values < domain_size)
    ignore = _in_domain_ignore_index(layer, domain_size)
    if ignore is not None:
        if domain_size < 3:
            return None
        in_domain = in_domain & (parent_values != ignore)
    if not bool(in_domain.any()):
        return None
    rotated = _rotate_skipping(parent_values, domain_size, ignore)
    return torch.where(in_domain, rotated, parent_values)


def index_domain_single_entry_values(
    layer: Op,
    parent_label: str,
    parent_values: torch.Tensor,
) -> torch.Tensor | None:
    """Return the index tensor with only its FIRST in-domain entry rotated.

    The full rotation in ``index_domain_rotation_values`` is a permutation of
    the index domain, so any output that depends only on the HISTOGRAM of the
    indices (``scatter_add_`` of ones, ``bincount``-style edge counts) is
    unchanged when that histogram is uniform, for example two relations with
    32 edges each. Moving a single in-domain entry ``v`` to ``(v + 1) % n``
    changes the histogram whenever ``n >= 2`` while staying in-domain, so the
    replay remains executable. This is an extra PROBE for the retry ladder: a
    changed output proves the edge is real; a spurious index edge stays
    unchanged under it as under every other probe and still fails.

    Parameters
    ----------
    layer:
        Child op being replayed.
    parent_label:
        Parent label selected for perturbation.
    parent_values:
        Saved parent tensor values.

    Returns
    -------
    torch.Tensor or None
        The single-entry perturbation, or ``None`` when the parent is not an
        integer index arg of an index-domain op or has no in-domain entry.
    """

    if index_domain_rotation_values(layer, parent_label, parent_values) is None:
        return None
    domain_size = _index_domain_size(layer)
    if domain_size is None:
        return None
    flat = parent_values.reshape(-1).clone()
    in_domain = (flat >= 0) & (flat < domain_size)
    ignore = _in_domain_ignore_index(layer, domain_size)
    if ignore is not None:
        in_domain = in_domain & (flat != ignore)
    first = int(torch.nonzero(in_domain)[0, 0])
    flat[first : first + 1] = _rotate_skipping(flat[first : first + 1], domain_size, ignore)
    return flat.reshape(parent_values.shape)


def _saved_integer_index_tensor(layer: Op) -> torch.Tensor | None:
    """Return the saved index-domain argument when it is an integer tensor.

    Parameters
    ----------
    layer:
        Captured index-consuming op.

    Returns
    -------
    torch.Tensor | None
        The saved index tensor, or ``None`` when it is missing, not a tensor, or
        not one of the integer index dtypes.
    """

    saved_index = _saved_index_domain_arg_value(layer)
    if not isinstance(saved_index, torch.Tensor):
        return None
    if saved_index.dtype not in _INDEX_DOMAIN_INT_DTYPES:
        return None
    return saved_index
